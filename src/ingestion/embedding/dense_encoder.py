"""DenseEncoder: batch-encode Chunk texts into dense vectors via BaseEmbedding."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.types import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory

if TYPE_CHECKING:
    from core.settings import Settings


class DenseEncoder:
    """Wraps a BaseEmbedding backend to produce dense vectors for Chunks.

    Preserves chunk order and attaches the vector to ChunkRecord.dense_vector.
    """

    def __init__(self, settings: "Settings", embedding: BaseEmbedding | None = None) -> None:
        self._embedding = embedding or EmbeddingFactory.create(settings.embedding)

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        """Embed chunk texts and return ChunkRecords with dense_vector set.

        Returns:
            One ChunkRecord per input Chunk, same order.
        Raises:
            ValueError: if chunks is non-empty but embedding returns wrong count.
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = self._embedding.embed(texts, trace=trace)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"Embedding returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        records: list[ChunkRecord] = []
        for chunk, vector in zip(chunks, vectors):
            record = ChunkRecord.from_chunk(chunk)
            record.dense_vector = vector
            records.append(record)

        return records
