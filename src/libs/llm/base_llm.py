"""BaseLLM: abstract interface for all LLM backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ChatResponse:
    content: str
    model: str = ""
    usage: dict[str, Any] | None = None


class BaseLLM(ABC):
    """All LLM implementations must subclass this and implement chat()."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        """Send a list of messages and return a ChatResponse.

        Args:
            messages: List of dicts with 'role' and 'content' keys.

        Returns:
            ChatResponse with at least .content populated.
        """
