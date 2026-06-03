"""MilvusStore: VectorStore backend using Milvus Lite for local persistence (provider='milvus')."""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from milvus_lite import CollectionSchema, DataType, FieldSchema, MilvusLite
from milvus_lite.exceptions import CollectionAlreadyExistsError

from libs.vector_store.base_vector_store import BaseVectorStore, QueryResult, VectorRecord
from libs.vector_store.vector_store_factory import register_provider

if TYPE_CHECKING:
    from core.settings import VectorStoreSettings

_DEFAULT_DATA_DIR = "data/db/milvus"
_VARCHAR_MAX = 65535
_ID_MAX_LENGTH = 512


class MilvusStore(BaseVectorStore):
    """
    VectorStore backed by Milvus Lite (local file-based persistence).

    Data is stored under `data_dir` (default: `data/db/milvus/`). Each
    collection gets its own sub-directory within `data_dir`.

    Scores returned by `query()` are COSINE similarity values in [-1, 1]
    where 1.0 = identical, 0.0 = orthogonal.  Results are sorted by
    descending score (most similar first), satisfying the BaseVectorStore
    contract.
    """

    def __init__(self, settings: "VectorStoreSettings", data_dir: str | None = None) -> None:
        self._settings = settings
        self._collection_name = (settings.collection_name or "knowledge_hub").lower()

        resolved_data_dir = (
            data_dir
            or getattr(settings, "data_dir", None)
            or _DEFAULT_DATA_DIR
        )
        os.makedirs(resolved_data_dir, exist_ok=True)
        self._data_dir = resolved_data_dir
        self._db = MilvusLite(resolved_data_dir)
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(self, dim: int):
        """Return the collection, creating it with the given dimension if needed."""
        if self._db.has_collection(self._collection_name):
            col = self._db.get_collection(self._collection_name)
            self._dim = dim
            return col

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=_ID_MAX_LENGTH),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=_VARCHAR_MAX),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=_VARCHAR_MAX),
        ])
        try:
            col = self._db.create_collection(self._collection_name, schema)
        except CollectionAlreadyExistsError:
            col = self._db.get_collection(self._collection_name)
        self._dim = dim
        return col

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> None:
        if not records:
            return

        dim = len(records[0].vector)
        col = self._get_or_create_collection(dim)

        data = [
            {
                "id": rec.id,
                "vector": rec.vector,
                "text": rec.text,
                "metadata_json": json.dumps(rec.metadata, ensure_ascii=False),
            }
            for rec in records
        ]
        col.upsert(data)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[QueryResult]:
        if not self._db.has_collection(self._collection_name):
            return []

        col = self._db.get_collection(self._collection_name)

        # Fetch extra candidates when filters are active so post-filter has enough
        fetch_k = top_k * 4 if filters else top_k

        raw = col.search(
            query_vectors=[vector],
            top_k=fetch_k,
            metric_type="COSINE",
            output_fields=["text", "metadata_json"],
        )

        results: list[QueryResult] = []
        for hit in raw[0]:
            entity = hit.get("entity", {})
            metadata: dict[str, Any] = {}
            try:
                metadata = json.loads(entity.get("metadata_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                pass

            if filters and not _matches_filters(metadata, filters):
                continue

            # COSINE distance ∈ [0, 2]: 0 = identical, 2 = opposite.
            # Convert to similarity score ∈ [-1, 1]: 1 = identical.
            score = 1.0 - float(hit["distance"])

            results.append(
                QueryResult(
                    id=str(hit["id"]),
                    score=score,
                    text=entity.get("text", ""),
                    metadata=metadata,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def close(self) -> None:
        """Release the Milvus Lite database handle."""
        self._db.close()


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(metadata.get(k) == v for k, v in filters.items())


register_provider("milvus", MilvusStore)
