"""BaseReranker: abstract interface for all reranker backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RerankCandidate:
    """A single candidate passed to and returned from a reranker."""
    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """All reranker implementations must subclass this."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        """Reorder candidates by relevance to the query.

        Args:
            query: The user query string.
            candidates: List of RerankCandidate to reorder.
            trace: Optional TraceContext (Phase F). Ignored until F is implemented.

        Returns:
            Reordered list of RerankCandidate (most relevant first).
            Length may be ≤ len(candidates) depending on implementation.
        """


class NoneReranker(BaseReranker):
    """Pass-through reranker: preserves original candidate order."""

    def __init__(self, settings: Any = None) -> None:
        pass

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        return list(candidates)
