"""LLMReranker: reranks candidates via an LLM call using a rerank prompt."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from libs.reranker.base_reranker import BaseReranker, RerankCandidate

if TYPE_CHECKING:
    from libs.llm.base_llm import BaseLLM
    from core.settings import RerankSettings


class RerankerFallbackError(Exception):
    """Raised when reranking fails and Core layer should fall back to fusion order."""


_DEFAULT_PROMPT_PATH = Path("config/prompts/rerank.txt")


def _load_prompt(prompt_path: Path | None = None) -> str:
    path = prompt_path or _DEFAULT_PROMPT_PATH
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise RerankerFallbackError(f"LLMReranker: cannot read prompt file '{path}': {e}") from e


class LLMReranker(BaseReranker):
    """Reranker that uses an LLM to rank candidates from most to least relevant.

    The LLM is prompted to return a JSON array of candidate IDs ordered by
    relevance. Any IDs missing from the response are appended at the end in
    their original order.
    """

    def __init__(
        self,
        settings: "RerankSettings",
        llm: "BaseLLM | None" = None,
        prompt_text: str | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        # prompt_text takes priority (test injection); otherwise load from file
        if prompt_text is not None:
            self._prompt_template = prompt_text
        else:
            self._prompt_template = _load_prompt(prompt_path)

    def _get_llm(self) -> "BaseLLM":
        if self._llm is not None:
            return self._llm
        # Lazy import to avoid circular deps
        from libs.llm.llm_factory import LLMFactory
        from core.settings import load_settings
        settings = load_settings()
        return LLMFactory.create(settings.llm)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []

        candidates_text = "\n".join(
            f'[{i}] id="{c.id}" text="{c.text[:200]}"'
            for i, c in enumerate(candidates)
        )
        prompt = self._prompt_template.format(
            query=query,
            candidates=candidates_text,
        )

        try:
            llm = self._get_llm()
            response = llm.chat([{"role": "user", "content": prompt}])
            raw = response.content.strip()
        except RerankerFallbackError:
            raise
        except Exception as e:
            raise RerankerFallbackError(
                f"LLMReranker: LLM call failed: {e}"
            ) from e

        ranked_ids = _parse_ranked_ids(raw)

        # Build id → candidate map
        id_map = {c.id: c for c in candidates}
        unknown = [rid for rid in ranked_ids if rid not in id_map]
        if unknown:
            raise ValueError(
                f"LLMReranker: response contains unknown candidate IDs: {unknown}. "
                f"Valid IDs: {list(id_map.keys())}"
            )

        seen = set(ranked_ids)
        tail = [c for c in candidates if c.id not in seen]
        return [id_map[rid] for rid in ranked_ids] + tail


def _parse_ranked_ids(raw: str) -> list[str]:
    """Extract a JSON array of string IDs from LLM response."""
    # Try full raw string first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
        # Valid JSON but wrong structure — don't try embedded extraction
        raise ValueError(
            f"LLMReranker: response is not a valid JSON array of strings. "
            f"Got: {raw!r}"
        )
    except json.JSONDecodeError:
        pass

    # Try to extract embedded JSON array from surrounding prose
    embedded = _extract_json_array(raw)
    if embedded is not None:
        try:
            parsed = json.loads(embedded)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"LLMReranker: response is not a valid JSON array of strings. "
        f"Got: {raw!r}"
    )


def _extract_json_array(text: str) -> str | None:
    """Return the first [...] substring found in text, or None."""
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        return text[start : end + 1]
    return None
