"""BaseSplitter: abstract interface for all text splitting backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """All splitter implementations must subclass this and implement split_text()."""

    @abstractmethod
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        """Split a text string into a list of chunks.

        Args:
            text: The input text to split.
            trace: Optional TraceContext (Phase F). Ignored until F is implemented.

        Returns:
            List of text chunk strings. Empty list if text is empty.
        """
