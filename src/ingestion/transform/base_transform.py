"""BaseTransform: abstract interface for all ingestion transform steps."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from core.types import Chunk
from core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Each Transform receives a list of Chunks and returns a (possibly modified) list."""

    @abstractmethod
    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Apply the transform to *chunks* and return the result.

        Implementations must not mutate the original list; they may mutate
        individual Chunk objects in place or return new ones.
        A single failing chunk must not abort the entire batch.
        """
