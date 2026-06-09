"""MetadataEnricher: rule-based + optional LLM-based metadata enrichment."""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "prompts" / "metadata_enrichment.txt"
)

# Match markdown headings: # Title, ## Title, etc.
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)
# Match IMAGE placeholder lines to skip when extracting title/summary
_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:[^\]]+\]")


def _rule_based_enrich(text: str) -> dict:
    """Extract title, summary, tags from text using heuristic rules."""
    clean_text = _IMAGE_PLACEHOLDER_RE.sub("", text).strip()

    # Title: first heading, or first non-empty line (truncated to 80 chars)
    title = ""
    heading_match = _HEADING_RE.search(clean_text)
    if heading_match:
        title = heading_match.group(1).strip()
    else:
        for line in clean_text.splitlines():
            line = line.strip()
            if line:
                title = line[:80]
                break

    # Summary: first 300 chars of text (non-heading content)
    no_headings = _HEADING_RE.sub("", clean_text).strip()
    summary = " ".join(no_headings.split())[:300]
    if len(no_headings) > 300:
        summary += "..."

    # Tags: extract capitalized words/phrases and long tokens as simple keywords
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{3,}\b", clean_text)
    # Frequency count, lowercase, deduplicate
    freq: dict[str, int] = {}
    for w in words:
        lw = w.lower()
        freq[lw] = freq.get(lw, 0) + 1
    # Sort by frequency, take top 5
    tags = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:5]]

    return {
        "title": title or "Untitled",
        "summary": summary or text[:100],
        "tags": tags,
    }


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Parse LLM output as JSON; return None on failure."""
    # Strip possible markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return None
        # Validate required fields
        if "title" not in parsed or "summary" not in parsed or "tags" not in parsed:
            return None
        if not isinstance(parsed["tags"], list):
            parsed["tags"] = []
        return {
            "title": str(parsed["title"]).strip() or "Untitled",
            "summary": str(parsed["summary"]).strip(),
            "tags": [str(t).strip() for t in parsed["tags"] if str(t).strip()],
        }
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


class MetadataEnricher(BaseTransform):
    """Enrich each Chunk's metadata with title, summary, and tags.

    Rule-based extraction always runs first as a fallback.  When ``use_llm``
    is enabled and an LLM instance is available, the LLM generates richer
    semantic metadata.  On LLM failure the rule result is kept and
    ``metadata["enriched_by"]`` is set to ``"rule"`` with a reason.
    """

    def __init__(
        self,
        settings: Settings,
        llm=None,
        prompt_path: Optional[str] = None,
    ) -> None:
        self._use_llm: bool = (
            getattr(getattr(settings, "ingestion", None), "metadata_enricher", None) is not None
            and settings.ingestion.metadata_enricher.use_llm
        )

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
        enriched: list[Chunk] = []
        for chunk in chunks:
            try:
                enriched.append(self._enrich_chunk(chunk, trace))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MetadataEnricher: failed to enrich chunk %s: %s", chunk.id, exc
                )
                kept = copy.copy(chunk)
                kept.metadata = dict(chunk.metadata)
                kept.metadata["enriched_by"] = "none"
                kept.metadata["enrich_error"] = str(exc)
                enriched.append(kept)

        if trace is not None:
            trace.record_stage(
                "metadata_enricher",
                total=len(chunks),
                llm_enriched=sum(
                    1 for c in enriched if c.metadata.get("enriched_by") == "llm"
                ),
                rule_enriched=sum(
                    1 for c in enriched if c.metadata.get("enriched_by") == "rule"
                ),
            )
        return enriched

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enrich_chunk(self, chunk: Chunk, trace: Optional[TraceContext]) -> Chunk:
        rule_meta = _rule_based_enrich(chunk.text)

        final_meta = rule_meta
        enriched_by = "rule"
        fallback_reason: Optional[str] = None

        if self._use_llm and self._llm is not None:
            llm_result = self._llm_enrich(chunk.text, trace)
            if llm_result is not None:
                final_meta = llm_result
                enriched_by = "llm"
            else:
                fallback_reason = "llm_returned_none"

        new_chunk = copy.copy(chunk)
        new_chunk.metadata = dict(chunk.metadata)
        new_chunk.metadata.update(final_meta)
        new_chunk.metadata["enriched_by"] = enriched_by
        if fallback_reason:
            new_chunk.metadata["enrich_fallback_reason"] = fallback_reason

        return new_chunk

    def _llm_enrich(self, text: str, trace: Optional[TraceContext]) -> Optional[dict]:
        if not text.strip():
            return None
        prompt = self._prompt_template.replace("{text}", text)
        try:
            response = self._llm.chat([{"role": "user", "content": prompt}])
            raw = response.content.strip()
            parsed = _parse_llm_json(raw)
            if parsed is None:
                logger.warning("MetadataEnricher: LLM output not valid JSON: %r", raw[:200])
            return parsed
        except Exception as exc:  # noqa: BLE001
            logger.warning("MetadataEnricher LLM enrichment failed: %s", exc)
            return None

    def _load_prompt(self, prompt_path: Optional[str]) -> str:
        path = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT_PATH
        if path.exists():
            return path.read_text(encoding="utf-8")
        logger.warning("MetadataEnricher: prompt file not found at %s, using default", path)
        return (
            "Analyze the following text and return a JSON object with fields: "
            "title (string), summary (string), tags (list of strings).\n"
            "Return ONLY valid JSON.\n\nText:\n{text}"
        )

    @staticmethod
    def _try_create_llm(settings: Settings):
        try:
            import libs.llm  # noqa: F401 — triggers provider registration
            from libs.llm.llm_factory import LLMFactory
            return LLMFactory.create(settings.llm)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MetadataEnricher: failed to create LLM (%s), falling back to rules", exc
            )
            return None
