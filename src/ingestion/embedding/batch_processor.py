"""BatchProcessor: split chunks into batches and drive dense/sparse encoding."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder

if TYPE_CHECKING:
    from core.settings import Settings


class BatchProcessor:
    """Coordinate dense + sparse encoding over batches of Chunks.

    Splits the input chunk list into fixed-size batches, runs DenseEncoder and
    SparseEncoder on each batch, merges the resulting vectors, and returns a
    single ordered list of ChunkRecords with both vectors populated.

    Batch timing is recorded per batch for trace/observability (Phase F).
    """

    def __init__(
        self,
        settings: "Settings",
        batch_size: int = 32,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._dense = dense_encoder or DenseEncoder(settings)
        self._sparse = sparse_encoder or SparseEncoder()

    def process(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        """Encode all chunks in batches and return ordered ChunkRecords.

        Each ChunkRecord has both dense_vector and sparse_vector set.

        Args:
            chunks: Input chunks to encode.
            trace:  Optional trace context (reserved for Phase F).

        Returns:
            One ChunkRecord per input Chunk in the same order.
        """
        if not chunks:
            return []

        results: list[ChunkRecord] = []
        for batch_idx, batch in enumerate(self._iter_batches(chunks)):
            t0 = time.monotonic()

            dense_records = self._dense.encode(batch, trace=trace)
            sparse_records = self._sparse.encode(batch, trace=trace)

            elapsed = time.monotonic() - t0

            # Merge: attach sparse_vector from sparse pass onto dense records
            merged = _merge_records(dense_records, sparse_records)
            results.extend(merged)

            if trace is not None and hasattr(trace, "record_stage"):
                trace.record_stage(
                    f"batch_{batch_idx}",
                    {"size": len(batch), "elapsed_s": elapsed},
                )

        return results

    def _iter_batches(self, chunks: list[Chunk]):
        """Yield successive fixed-size sub-lists."""
        for start in range(0, len(chunks), self._batch_size):
            yield chunks[start : start + self._batch_size]


def _merge_records(
    dense_records: list[ChunkRecord],
    sparse_records: list[ChunkRecord],
) -> list[ChunkRecord]:
    """Copy sparse_vector from sparse_records into dense_records in-place."""
    if len(dense_records) != len(sparse_records):
        raise ValueError(
            f"Dense/sparse record count mismatch: "
            f"{len(dense_records)} vs {len(sparse_records)}"
        )
    for dr, sr in zip(dense_records, sparse_records):
        dr.sparse_vector = sr.sparse_vector
    return dense_records
