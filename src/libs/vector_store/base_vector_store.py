"""BaseVectorStore: abstract interface for all vector store backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorRecord:
    """A single record to upsert into the vector store."""
    id: str
    vector: list[float]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """A single result returned from a vector store query."""
    id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """All vector store implementations must subclass this."""

    @abstractmethod
    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> None:
        """Insert or update a batch of records.

        Args:
            records: List of VectorRecord objects to upsert.
            trace: Optional TraceContext (Phase F). Ignored until F is implemented.
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[QueryResult]:
        """Query the vector store by dense vector similarity.

        Args:
            vector: Query vector (must match indexed dimension).
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (key-value equality).
            trace: Optional TraceContext (Phase F).

        Returns:
            List of QueryResult sorted by descending score, len ≤ top_k.
        """
