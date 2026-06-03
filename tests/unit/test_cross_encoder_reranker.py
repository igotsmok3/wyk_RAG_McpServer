"""Tests for CrossEncoderReranker (B7.8)."""
import sys
import os
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderScorer,
    SentenceTransformersCrossEncoderScorer,
)
from libs.reranker.llm_reranker import RerankerFallbackError
from libs.reranker.reranker_factory import RerankerFactory


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeRerankSettings:
    enabled: bool = True
    provider: str = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


class MockScorer(CrossEncoderScorer):
    """Deterministic mock scorer: returns preset scores or len-based scores."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = scores
        self.call_count = 0
        self.last_query: str | None = None
        self.last_texts: list[str] | None = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.call_count += 1
        self.last_query = query
        self.last_texts = texts
        if self._scores is not None:
            return self._scores
        # Default: score by index descending so last candidate ranks highest
        return [float(i) for i in range(len(texts))]


def _candidates(ids=("c0", "c1", "c2")) -> list[RerankCandidate]:
    return [RerankCandidate(id=i, text=f"text for {i}") for i in ids]


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

def test_factory_creates_cross_encoder_reranker():
    r = RerankerFactory.create(FakeRerankSettings())
    assert isinstance(r, CrossEncoderReranker)


def test_factory_cross_encoder_case_insensitive():
    s = FakeRerankSettings(provider="CROSS_ENCODER")
    r = RerankerFactory.create(s)
    assert isinstance(r, CrossEncoderReranker)


# ---------------------------------------------------------------------------
# CrossEncoderScorer abstract contract
# ---------------------------------------------------------------------------

def test_cross_encoder_scorer_is_abstract():
    with pytest.raises(TypeError):
        CrossEncoderScorer()  # type: ignore


# ---------------------------------------------------------------------------
# Normal reranking with mock scorer
# ---------------------------------------------------------------------------

def test_rerank_orders_by_score_descending():
    # Scores: c0=0.1, c1=0.9, c2=0.5 → expected order: c1, c2, c0
    scorer = MockScorer(scores=[0.1, 0.9, 0.5])
    r = CrossEncoderReranker(FakeRerankSettings(), scorer=scorer)
    result = r.rerank("query", _candidates())
    assert [c.id for c in result] == ["c1", "c2", "c0"]


def test_rerank_empty_candidates_returns_empty():
    scorer = MockScorer()
    r = CrossEncoderReranker(FakeRerankSettings(), scorer=scorer)
    assert r.rerank("q", []) == []
    assert scorer.call_count == 0


def test_rerank_single_candidate_returns_same():
    scorer = MockScorer(scores=[0.7])
    r = CrossEncoderReranker(FakeRerankSettings(), scorer=scorer)
    cands = [RerankCandidate(id="c0", text="hello")]
    result = r.rerank("q", cands)
    assert len(result) == 1
    assert result[0].id == "c0"


def test_rerank_passes_query_and_texts_to_scorer():
    scorer = MockScorer(scores=[0.5, 0.3])
    r = CrossEncoderReranker(FakeRerankSettings(), scorer=scorer)
    cands = [RerankCandidate(id="a", text="alpha"), RerankCandidate(id="b", text="beta")]
    r.rerank("my query", cands)
    assert scorer.last_query == "my query"
    assert scorer.last_texts == ["alpha", "beta"]


def test_rerank_preserves_metadata():
    scorer = MockScorer(scores=[0.5, 0.8])
    r = CrossEncoderReranker(FakeRerankSettings(), scorer=scorer)
    cands = [
        RerankCandidate(id="a", text="t", score=0.3, metadata={"src": "a.pdf"}),
        RerankCandidate(id="b", text="u", score=0.1, metadata={"src": "b.pdf"}),
    ]
    result = r.rerank("q", cands)
    assert result[0].id == "b"
    assert result[0].metadata["src"] == "b.pdf"


def test_rerank_deterministic_for_same_input():
    scorer = MockScorer(scores=[0.2, 0.8, 0.5])
    r = CrossEncoderReranker(FakeRerankSettings(), scorer=scorer)
    cands = _candidates()
    r1 = r.rerank("q", cands)
    scorer2 = MockScorer(scores=[0.2, 0.8, 0.5])
    r2 = CrossEncoderReranker(FakeRerankSettings(), scorer=scorer2).rerank("q", cands)
    assert [c.id for c in r1] == [c.id for c in r2]


# ---------------------------------------------------------------------------
# Fallback signal tests
# ---------------------------------------------------------------------------

def test_rerank_scorer_exception_raises_fallback_error():
    scorer = MockScorer()
    scorer.score = MagicMock(side_effect=RuntimeError("model load failed"))
    r = CrossEncoderReranker(FakeRerankSettings(), scorer=scorer)
    with pytest.raises(RerankerFallbackError, match="scorer failed"):
        r.rerank("q", _candidates())


def test_rerank_timeout_raises_fallback_error():
    import time

    class SlowScorer(CrossEncoderScorer):
        def score(self, query, texts):
            time.sleep(5)
            return [0.5] * len(texts)

    r = CrossEncoderReranker(FakeRerankSettings(), scorer=SlowScorer(), timeout=0.05)
    with pytest.raises(RerankerFallbackError, match="timed out"):
        r.rerank("q", _candidates())


def test_rerank_wrong_score_count_raises_fallback_error():
    # Scorer returns wrong number of scores
    scorer = MockScorer(scores=[0.5])  # 1 score for 3 candidates
    r = CrossEncoderReranker(FakeRerankSettings(), scorer=scorer)
    with pytest.raises(RerankerFallbackError, match="scorer returned"):
        r.rerank("q", _candidates())


# ---------------------------------------------------------------------------
# SentenceTransformersCrossEncoderScorer import guard
# ---------------------------------------------------------------------------

def test_sentence_transformers_scorer_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("mocked missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="sentence-transformers is required"):
        SentenceTransformersCrossEncoderScorer("any-model")
