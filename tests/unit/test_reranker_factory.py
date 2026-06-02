"""Tests for BaseReranker interface and RerankerFactory (B5)."""
import sys
import os
import pytest
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory, register_backend


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeRerankSettings:
    enabled: bool = True
    provider: str = "none"
    model: str = ""
    top_k: int = 5


class ReverseReranker(BaseReranker):
    """Test fake that reverses candidate order."""

    def __init__(self, settings):
        self.settings = settings

    def rerank(self, query, candidates, trace=None):
        return list(reversed(candidates))


def _candidates(n: int = 3) -> list[RerankCandidate]:
    return [
        RerankCandidate(id=f"c{i}", text=f"text {i}", score=float(i))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# BaseReranker abstract contract
# ---------------------------------------------------------------------------

def test_base_reranker_is_abstract():
    with pytest.raises(TypeError):
        BaseReranker()  # type: ignore


# ---------------------------------------------------------------------------
# RerankCandidate data shape
# ---------------------------------------------------------------------------

def test_rerank_candidate_shape():
    c = RerankCandidate(id="a", text="hello", score=0.9, metadata={"src": "doc.pdf"})
    assert c.id == "a"
    assert c.text == "hello"
    assert isinstance(c.score, float)
    assert c.metadata["src"] == "doc.pdf"


def test_rerank_candidate_defaults():
    c = RerankCandidate(id="a", text="hi")
    assert c.score == 0.0
    assert c.metadata == {}


# ---------------------------------------------------------------------------
# NoneReranker: preserves original order
# ---------------------------------------------------------------------------

def test_none_reranker_preserves_order():
    reranker = NoneReranker()
    cands = _candidates(4)
    result = reranker.rerank(query="test", candidates=cands)
    assert [r.id for r in result] == ["c0", "c1", "c2", "c3"]


def test_none_reranker_returns_copy():
    reranker = NoneReranker()
    cands = _candidates(2)
    result = reranker.rerank(query="q", candidates=cands)
    assert result is not cands


def test_none_reranker_empty_input():
    reranker = NoneReranker()
    assert reranker.rerank(query="q", candidates=[]) == []


def test_none_reranker_ignores_query():
    reranker = NoneReranker()
    cands = _candidates(2)
    r1 = reranker.rerank(query="foo", candidates=cands)
    r2 = reranker.rerank(query="bar", candidates=cands)
    assert [r.id for r in r1] == [r.id for r in r2]


# ---------------------------------------------------------------------------
# RerankerFactory routing
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def register_fakes():
    register_backend("reverse", ReverseReranker)
    yield


def test_factory_disabled_returns_none_reranker():
    s = FakeRerankSettings(enabled=False, provider="reverse")
    r = RerankerFactory.create(s)
    assert isinstance(r, NoneReranker)


def test_factory_none_backend_returns_none_reranker():
    s = FakeRerankSettings(enabled=True, provider="none")
    r = RerankerFactory.create(s)
    assert isinstance(r, NoneReranker)


def test_factory_creates_registered_backend():
    s = FakeRerankSettings(enabled=True, provider="reverse")
    r = RerankerFactory.create(s)
    assert isinstance(r, ReverseReranker)


def test_factory_provider_case_insensitive():
    s = FakeRerankSettings(enabled=True, provider="REVERSE")
    r = RerankerFactory.create(s)
    assert isinstance(r, ReverseReranker)


def test_factory_unknown_provider_raises():
    s = FakeRerankSettings(enabled=True, provider="nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        RerankerFactory.create(s)


def test_factory_unknown_provider_lists_known():
    s = FakeRerankSettings(enabled=True, provider="bad")
    with pytest.raises(ValueError, match="none"):
        RerankerFactory.create(s)


# ---------------------------------------------------------------------------
# NoneReranker doesn't change ordering — backend=none contract
# ---------------------------------------------------------------------------

def test_none_backend_does_not_change_ordering():
    s = FakeRerankSettings(enabled=True, provider="none")
    r = RerankerFactory.create(s)
    cands = [
        RerankCandidate(id="x", text="a", score=0.5),
        RerankCandidate(id="y", text="b", score=0.8),
        RerankCandidate(id="z", text="c", score=0.2),
    ]
    result = r.rerank("query", cands)
    assert [c.id for c in result] == ["x", "y", "z"]


def test_reverse_reranker_changes_order():
    s = FakeRerankSettings(enabled=True, provider="reverse")
    r = RerankerFactory.create(s)
    cands = _candidates(3)
    result = r.rerank("q", cands)
    assert [c.id for c in result] == ["c2", "c1", "c0"]
