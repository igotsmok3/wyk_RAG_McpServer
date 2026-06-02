"""BaseEmbedding: abstract interface for all embedding backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """All embedding implementations must subclass this and implement embed()."""

    @abstractmethod
    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        Args:
            texts: List of strings to embed.
            trace: Optional TraceContext (Phase F). Ignored until F is implemented.

        Returns:
            List of float vectors, one per input text.
        """
