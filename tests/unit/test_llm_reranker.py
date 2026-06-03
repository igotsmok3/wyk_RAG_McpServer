"""Tests for LLMReranker (B7.7)."""
import sys
import os
import json
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.llm_reranker import LLMReranker, RerankerFallbackError, _parse_ranked_ids
from libs.reranker.reranker_factory import RerankerFactory
from libs.llm.base_llm import ChatResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeRerankSettings:
    enabled: bool = True
    provider: str = "llm"
    model: str = ""
    top_k: int = 5


SIMPLE_PROMPT = "Query: {query}\nCandidates:\n{candidates}"


def _make_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = ChatResponse(content=content)
    return llm


def _candidates(ids=("c0", "c1", "c2")) -> list[RerankCandidate]:
    return [RerankCandidate(id=i, text=f"text for {i}") for i in ids]


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

def test_factory_creates_llm_reranker():
    s = FakeRerankSettings(enabled=True, provider="llm")
    r = RerankerFactory.create(s)
    assert isinstance(r, LLMReranker)


def test_factory_llm_backend_case_insensitive():
    s = FakeRerankSettings(enabled=True, provider="LLM")
    r = RerankerFactory.create(s)
    assert isinstance(r, LLMReranker)


# ---------------------------------------------------------------------------
# Normal reranking
# ---------------------------------------------------------------------------

def test_rerank_returns_ranked_order():
    llm = _make_llm('["c2", "c0", "c1"]')
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text=SIMPLE_PROMPT)
    cands = _candidates()
    result = r.rerank("query", cands)
    assert [c.id for c in result] == ["c2", "c0", "c1"]


def test_rerank_partial_response_appends_missing():
    """LLM returns subset of IDs; missing ones appended in original order."""
    llm = _make_llm('["c2"]')
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text=SIMPLE_PROMPT)
    cands = _candidates()
    result = r.rerank("query", cands)
    assert result[0].id == "c2"
    # c0 and c1 appended in original order
    assert [c.id for c in result[1:]] == ["c0", "c1"]


def test_rerank_empty_candidates_returns_empty():
    llm = _make_llm("[]")
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text=SIMPLE_PROMPT)
    assert r.rerank("query", []) == []


def test_rerank_passes_query_and_candidates_to_llm():
    llm = _make_llm('["c0"]')
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text="q={query} c={candidates}")
    cands = [RerankCandidate(id="c0", text="hello")]
    r.rerank("my query", cands)
    call_args = llm.chat.call_args[0][0]
    assert "my query" in call_args[0]["content"]


def test_rerank_response_with_extra_text_around_array():
    """LLM wraps array in prose — still parseable."""
    llm = _make_llm('Here are the results: ["c1", "c0", "c2"] ranked.')
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text=SIMPLE_PROMPT)
    result = r.rerank("q", _candidates())
    assert [c.id for c in result] == ["c1", "c0", "c2"]


def test_rerank_preserves_metadata():
    llm = _make_llm('["c0"]')
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text=SIMPLE_PROMPT)
    cands = [RerankCandidate(id="c0", text="t", score=0.9, metadata={"src": "doc.pdf"})]
    result = r.rerank("q", cands)
    assert result[0].metadata["src"] == "doc.pdf"
    assert result[0].score == 0.9


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_rerank_invalid_json_raises_value_error():
    llm = _make_llm("not json at all")
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text=SIMPLE_PROMPT)
    with pytest.raises(ValueError, match="valid JSON array"):
        r.rerank("q", _candidates())


def test_rerank_json_object_raises_value_error():
    llm = _make_llm('{"ids": ["c0"]}')
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text=SIMPLE_PROMPT)
    with pytest.raises(ValueError, match="valid JSON array"):
        r.rerank("q", _candidates())


def test_rerank_unknown_id_raises_value_error():
    llm = _make_llm('["c0", "unknown_id"]')
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text=SIMPLE_PROMPT)
    with pytest.raises(ValueError, match="unknown candidate IDs"):
        r.rerank("q", _candidates())


def test_rerank_llm_exception_raises_fallback_error():
    llm = MagicMock()
    llm.chat.side_effect = RuntimeError("connection timeout")
    r = LLMReranker(FakeRerankSettings(), llm=llm, prompt_text=SIMPLE_PROMPT)
    with pytest.raises(RerankerFallbackError, match="LLM call failed"):
        r.rerank("q", _candidates())


def test_rerank_missing_prompt_file_raises_fallback_error():
    from pathlib import Path
    with pytest.raises(RerankerFallbackError, match="cannot read prompt file"):
        LLMReranker(FakeRerankSettings(), prompt_path=Path("/nonexistent/rerank.txt"))


# ---------------------------------------------------------------------------
# _parse_ranked_ids unit tests
# ---------------------------------------------------------------------------

def test_parse_ranked_ids_plain_array():
    assert _parse_ranked_ids('["a", "b", "c"]') == ["a", "b", "c"]


def test_parse_ranked_ids_embedded_array():
    assert _parse_ranked_ids('Sure! ["x", "y"]') == ["x", "y"]


def test_parse_ranked_ids_empty_array():
    assert _parse_ranked_ids("[]") == []


def test_parse_ranked_ids_invalid_raises():
    with pytest.raises(ValueError, match="valid JSON array"):
        _parse_ranked_ids("not json")


def test_parse_ranked_ids_non_string_elements_raises():
    with pytest.raises(ValueError, match="valid JSON array"):
        _parse_ranked_ids("[1, 2, 3]")
