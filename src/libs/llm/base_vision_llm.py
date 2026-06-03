"""BaseVisionLLM: abstract interface for all Vision LLM backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from libs.llm.base_llm import ChatResponse

if TYPE_CHECKING:
    pass


class BaseVisionLLM(ABC):
    """All Vision LLM implementations must subclass this and implement chat_with_image()."""

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a text prompt with an image and return a ChatResponse.

        Args:
            text: The text prompt to send alongside the image.
            image: Either a file path (str) or raw bytes (base64 source).
            trace: Optional TraceContext for observability.

        Returns:
            ChatResponse with at least .content populated.
        """

    def preprocess_image(self, image: str | bytes, max_size: int = 2048) -> bytes:
        """Extension point for image preprocessing (resize/compress).

        Subclasses may override to enforce max_size constraints or convert formats.
        Default implementation returns raw bytes unchanged.
        """
        if isinstance(image, bytes):
            return image
        with open(image, "rb") as f:
            return f.read()
