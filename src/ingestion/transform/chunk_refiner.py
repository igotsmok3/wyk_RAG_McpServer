"""ChunkRefiner: rule-based denoising + optional LLM enhancement."""
from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import List, Optional

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "config" / "prompts" / "chunk_refinement.txt"

# Patterns that identify pure noise lines (page markers, separator lines)
_PAGE_MARKER_RE = re.compile(
    r"^\s*(?:(?:─{4,}|─+|={4,}|-{4,}|_{4,}|\*{4,})\s*"
    r"|Page\s+\d+(?:\s*[|│]\s*.*?)?"
    r"|Footer\s*[:：].*"
    r"|Header\s*[:：].*"
    r"|(?:Confidential|Copyright|©).*"
    r")\s*$",
    re.IGNORECASE,
)


def _is_code_fence(line: str) -> bool:
    return line.strip().startswith("```")


def _is_separator_line(line: str) -> bool:
    """Return True if the line is a visual separator (dashes, underscores, etc.)."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(re.match(r"^[─\-=_*]{4,}\s*$", stripped))


def _remove_separator_blocks(text: str) -> str:
    """Remove blocks enclosed by separator lines (page headers/footers).

    Matches patterns like:
        ────────────────────────
        Short content (0–3 lines)
        ────────────────────────
    and removes the entire block including enclosing separators.
    """
    # Match: separator, optional short content (1-3 lines), separator
    return re.sub(
        r"[ \t]*[─\-=_*]{4,}[ \t]*\n(?:(?![ \t]*[─\-=_*]{4,}).{0,120}\n){0,3}[ \t]*[─\-=_*]{4,}[ \t]*(?:\n|$)",
        "\n",
        text,
    )


def _rule_based_refine(text: str) -> str:
    """Apply rule-based denoising while preserving code blocks and Markdown."""
    # --- Phase 0: protect code blocks from block-level cleanup ---
    code_blocks: list[str] = []
    placeholder_tmpl = "\x00CODE{}\x00"

    def _extract_code(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return placeholder_tmpl.format(len(code_blocks) - 1)

    text = re.sub(r"```[\s\S]*?```", _extract_code, text)

    # --- Phase 1: remove separator-enclosed blocks ---
    text = _remove_separator_blocks(text)

    # --- Phase 2: restore code blocks ---
    for i, block in enumerate(code_blocks):
        text = text.replace(placeholder_tmpl.format(i), block)

    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False

    for line in lines:
        if _is_code_fence(line):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        # Remove HTML comments
        line = re.sub(r"<!--.*?-->", "", line, flags=re.DOTALL)

        # Remove HTML tags (but NOT Markdown)
        line = re.sub(r"<[^>]+>", "", line)

        # Skip pure noise lines (separator lines, page numbers, etc.)
        if _PAGE_MARKER_RE.match(line) and line.strip():
            continue

        # Collapse multiple spaces to single space (preserve leading whitespace for lists)
        leading = re.match(r"^(\s*[-*+•]?\s*)", line)
        if leading:
            prefix = leading.group(1)
            rest = line[len(prefix):]
            rest = re.sub(r"  +", " ", rest)
            line = prefix + rest
        else:
            line = re.sub(r"  +", " ", line)

        result.append(line)

    # Join and collapse 3+ consecutive blank lines to max 2
    joined = "\n".join(result)
    joined = re.sub(r"\n{3,}", "\n\n", joined)

    # Strip leading/trailing blank lines
    return joined.strip()


class ChunkRefiner(BaseTransform):
    """Denoise and optionally LLM-enhance each Chunk's text.

    Rule-based cleaning always runs first.  If ``use_llm`` is enabled and an
    LLM instance is available, the rules output is forwarded to the LLM for
    further refinement.  On any LLM failure the rule result is used and
    ``metadata["refined_by"]`` is set to ``"rule"`` with a ``"refine_fallback_reason"``.
    """

    def __init__(
        self,
        settings: Settings,
        llm=None,
        prompt_path: Optional[str] = None,
    ) -> None:
        self._use_llm: bool = getattr(
            getattr(settings, "ingestion", None),
            "chunk_refiner",
            None,
        ) is not None and settings.ingestion.chunk_refiner.use_llm

        self._llm = llm
        if self._use_llm and self._llm is None:
            self._llm = self._try_create_llm(settings)

        self._prompt_template = self._load_prompt(prompt_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        refined: list[Chunk] = []
        for chunk in chunks:
            try:
                refined.append(self._refine_chunk(chunk, trace))
            except Exception as exc:  # noqa: BLE001
                logger.warning("ChunkRefiner: failed to refine chunk %s: %s", chunk.id, exc)
                kept = copy.copy(chunk)
                kept.metadata = dict(chunk.metadata)
                kept.metadata["refined_by"] = "none"
                kept.metadata["refine_error"] = str(exc)
                refined.append(kept)

        if trace is not None:
            trace.record_stage(
                "chunk_refiner",
                total=len(chunks),
                llm_refined=sum(
                    1 for c in refined if c.metadata.get("refined_by") == "llm"
                ),
                rule_refined=sum(
                    1 for c in refined if c.metadata.get("refined_by") == "rule"
                ),
            )
        return refined

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refine_chunk(self, chunk: Chunk, trace: Optional[TraceContext]) -> Chunk:
        rule_text = self._rule_based_refine(chunk.text)

        final_text = rule_text
        refined_by = "rule"
        fallback_reason: Optional[str] = None

        if self._use_llm and self._llm is not None:
            llm_result = self._llm_refine(rule_text, trace)
            if llm_result is not None:
                final_text = llm_result
                refined_by = "llm"
            else:
                fallback_reason = "llm_returned_none"

        new_chunk = copy.copy(chunk)
        new_chunk.text = final_text
        new_chunk.metadata = dict(chunk.metadata)
        new_chunk.metadata["refined_by"] = refined_by
        if fallback_reason:
            new_chunk.metadata["refine_fallback_reason"] = fallback_reason

        return new_chunk

    def _rule_based_refine(self, text: str) -> str:
        return _rule_based_refine(text)

    def _llm_refine(self, text: str, trace: Optional[TraceContext]) -> Optional[str]:
        if not text.strip():
            return None
        prompt = self._prompt_template.replace("{text}", text)
        try:
            response = self._llm.chat([{"role": "user", "content": prompt}])
            result = response.content.strip()
            return result if result else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChunkRefiner LLM refinement failed: %s", exc)
            return None

    def _load_prompt(self, prompt_path: Optional[str]) -> str:
        path = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT_PATH
        if path.exists():
            return path.read_text(encoding="utf-8")
        logger.warning("ChunkRefiner: prompt file not found at %s, using default", path)
        return (
            "You are an expert at cleaning and refining text chunks from documents.\n"
            "Given a text chunk, remove noise (headers, footers, formatting artifacts) "
            "while preserving the core content.\n"
            "Return only the refined text without any explanation.\n\nText:\n{text}"
        )

    @staticmethod
    def _try_create_llm(settings: Settings):
        try:
            import libs.llm  # noqa: F401 — triggers provider registration
            from libs.llm.llm_factory import LLMFactory
            return LLMFactory.create(settings.llm)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChunkRefiner: failed to create LLM (%s), falling back to rules", exc)
            return None
