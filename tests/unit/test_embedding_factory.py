"""Tests for BaseEmbedding interface and EmbeddingFactory routing (B2)."""
import sys
import os
import pytest
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory, register_provider


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

FAKE_DIM = 4


class FakeEmbedding(BaseEmbedding):
    def __init__(self, settings):
        self.settings = settings

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        return [[float(i)] * FAKE_DIM for i in range(len(texts))]


class AnotherFakeEmbedding(BaseEmbedding):
    def __init__(self, settings):
        self.settings = settings

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        return [[1.0] * FAKE_DIM for _ in texts]


@dataclass
class FakeEmbeddingSettings:
    provider: str
    model: str = "fake-embed"
    dimensions: int = FAKE_DIM
    api_key: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def register_fakes():
    register_provider("fake", FakeEmbedding)
    register_provider("another", AnotherFakeEmbedding)
    yield


# ---------------------------------------------------------------------------
# BaseEmbedding interface tests
# ---------------------------------------------------------------------------

def test_base_embedding_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbedding()  # type: ignore


# ---------------------------------------------------------------------------
# EmbeddingFactory routing tests
# ---------------------------------------------------------------------------

def test_factory_creates_fake_embedding():
    s = FakeEmbeddingSettings(provider="fake")
    emb = EmbeddingFactory.create(s)
    assert isinstance(emb, FakeEmbedding)


def test_factory_routes_by_provider():
    s = FakeEmbeddingSettings(provider="another")
    emb = EmbeddingFactory.create(s)
    assert isinstance(emb, AnotherFakeEmbedding)


def test_factory_provider_case_insensitive():
    s = FakeEmbeddingSettings(provider="FAKE")
    emb = EmbeddingFactory.create(s)
    assert isinstance(emb, FakeEmbedding)


def test_factory_unknown_provider_raises():
    s = FakeEmbeddingSettings(provider="nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        EmbeddingFactory.create(s)


def test_factory_error_lists_known_providers():
    s = FakeEmbeddingSettings(provider="nonexistent")
    with pytest.raises(ValueError, match="fake"):
        EmbeddingFactory.create(s)


def test_factory_empty_provider_raises():
    s = FakeEmbeddingSettings(provider="")
    with pytest.raises(ValueError, match="empty"):
        EmbeddingFactory.create(s)


# ---------------------------------------------------------------------------
# FakeEmbedding functional tests
# ---------------------------------------------------------------------------

def test_embed_returns_correct_count():
    s = FakeEmbeddingSettings(provider="fake")
    emb = EmbeddingFactory.create(s)
    texts = ["hello", "world", "test"]
    result = emb.embed(texts)
    assert len(result) == len(texts)


def test_embed_returns_vectors_of_expected_dim():
    s = FakeEmbeddingSettings(provider="fake")
    emb = EmbeddingFactory.create(s)
    result = emb.embed(["hello"])
    assert len(result[0]) == FAKE_DIM


def test_embed_stable_output():
    """Same input produces same output (deterministic)."""
    s = FakeEmbeddingSettings(provider="fake")
    emb = EmbeddingFactory.create(s)
    texts = ["foo", "bar"]
    assert emb.embed(texts) == emb.embed(texts)


def test_embed_accepts_trace_param():
    s = FakeEmbeddingSettings(provider="fake")
    emb = EmbeddingFactory.create(s)
    result = emb.embed(["hello"], trace=None)
    assert len(result) == 1


def test_embed_empty_list():
    s = FakeEmbeddingSettings(provider="fake")
    emb = EmbeddingFactory.create(s)
    result = emb.embed([])
    assert result == []
