"""RecursiveSplitter: LangChain RecursiveCharacterTextSplitter backend (provider='recursive')."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import register_splitter

if TYPE_CHECKING:
    from core.settings import IngestionSettings

# Separators respect Markdown structure: headings, code fences, paragraphs, sentences, words.
_MARKDOWN_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n```",
    "\n\n",
    "\n",
    " ",
    "",
]


class RecursiveSplitter(BaseSplitter):
    """Text splitter backed by LangChain's RecursiveCharacterTextSplitter.

    Uses Markdown-aware separators so headings and code blocks are not split
    mid-structure.
    """

    def __init__(self, settings: "IngestionSettings") -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=_MARKDOWN_SEPARATORS,
            keep_separator=True,
        )

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        if not text:
            return []
        return self._splitter.split_text(text)


register_splitter("recursive", RecursiveSplitter)
