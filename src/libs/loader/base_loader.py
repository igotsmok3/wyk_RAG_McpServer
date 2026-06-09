"""Abstract base class for document loaders."""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Document


class BaseLoader(ABC):
    """Load a document from a file path and return a Document object."""

    @abstractmethod
    def load(self, path: str) -> Document:
        """Load the file at *path* and return a Document.

        The returned Document.text may contain [IMAGE: {image_id}] placeholders.
        Document.metadata must contain at least ``source_path``.
        """
