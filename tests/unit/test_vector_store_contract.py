"""Contract tests for BaseVectorStore interface and VectorStoreFactory (B4)."""
import sys
import os
import pytest
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, QueryResult
from libs.vector_store.vector_store_factory import VectorStoreFactory, register_provider


# ---------------------------------------------------------------------------
# In-memory Fake store for contract testing
# ---------------------------------------------------------------------------

class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory implementation that satisfies the contract."""

    def __init__(self, settings):
        self.settings = settings
        self._store: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord], trace=None) -> None:
        for rec in records:
            self._store[rec.id] = rec

    def query(self, vector, top_k, filters=None, trace=None) -> list[QueryResult]:
        results = []
        for rec in self._store.values():
            if filters:
                if not all(rec.metadata.get(k) == v for k, v in filters.items()):
                    continue
            # Dot-product score (fake similarity)
            score = sum(a * b for a, b in zip(vector, rec.vector))
            results.append(QueryResult(id=rec.id, score=score, text=rec.text, metadata=rec.metadata))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


@dataclass
class FakeVectorStoreSettings:
    provider: str
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def register_fakes():
    register_provider("inmemory", InMemoryVectorStore)
    yield


@pytest.fixture
def store():
    s = FakeVectorStoreSettings(provider="inmemory")
    return VectorStoreFactory.create(s)


# ---------------------------------------------------------------------------
# Data shape / contract tests
# ---------------------------------------------------------------------------

def test_base_vector_store_is_abstract():
    with pytest.raises(TypeError):
        BaseVectorStore()  # type: ignore


def test_vector_record_shape():
    r = VectorRecord(id="r1", vector=[0.1, 0.2], text="hello", metadata={"k": "v"})
    assert r.id == "r1"
    assert len(r.vector) == 2
    assert r.text == "hello"
    assert r.metadata["k"] == "v"


def test_query_result_shape():
    qr = QueryResult(id="r1", score=0.9, text="hello", metadata={"k": "v"})
    assert qr.id == "r1"
    assert isinstance(qr.score, float)
    assert qr.text == "hello"


# ---------------------------------------------------------------------------
# VectorStoreFactory routing tests
# ---------------------------------------------------------------------------

def test_factory_creates_store():
    s = FakeVectorStoreSettings(provider="inmemory")
    vs = VectorStoreFactory.create(s)
    assert isinstance(vs, InMemoryVectorStore)


def test_factory_provider_case_insensitive():
    s = FakeVectorStoreSettings(provider="INMEMORY")
    vs = VectorStoreFactory.create(s)
    assert isinstance(vs, InMemoryVectorStore)


def test_factory_unknown_provider_raises():
    s = FakeVectorStoreSettings(provider="nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        VectorStoreFactory.create(s)


def test_factory_empty_provider_raises():
    s = FakeVectorStoreSettings(provider="")
    with pytest.raises(ValueError, match="empty"):
        VectorStoreFactory.create(s)


# ---------------------------------------------------------------------------
# upsert / query contract tests
# ---------------------------------------------------------------------------

def test_upsert_then_query_returns_result(store):
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0], text="doc a")])
    results = store.query(vector=[1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].id == "a"


def test_upsert_idempotent(store):
    rec = VectorRecord(id="a", vector=[1.0, 0.0], text="original")
    store.upsert([rec])
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0], text="updated")])
    results = store.query(vector=[1.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].text == "updated"


def test_query_respects_top_k(store):
    store.upsert([
        VectorRecord(id=f"r{i}", vector=[float(i), 0.0], text=f"doc {i}")
        for i in range(5)
    ])
    results = store.query(vector=[1.0, 0.0], top_k=2)
    assert len(results) <= 2


def test_query_results_sorted_by_score_desc(store):
    store.upsert([
        VectorRecord(id="low", vector=[0.1, 0.0]),
        VectorRecord(id="high", vector=[1.0, 0.0]),
    ])
    results = store.query(vector=[1.0, 0.0], top_k=2)
    assert results[0].score >= results[1].score


def test_query_result_has_text_and_metadata(store):
    store.upsert([VectorRecord(id="a", vector=[1.0], text="hello", metadata={"src": "test.pdf"})])
    results = store.query(vector=[1.0], top_k=1)
    assert results[0].text == "hello"
    assert results[0].metadata["src"] == "test.pdf"


def test_query_with_metadata_filter(store):
    store.upsert([
        VectorRecord(id="a", vector=[1.0, 0.0], metadata={"col": "A"}),
        VectorRecord(id="b", vector=[1.0, 0.0], metadata={"col": "B"}),
    ])
    results = store.query(vector=[1.0, 0.0], top_k=5, filters={"col": "A"})
    assert all(r.id == "a" for r in results)


def test_query_empty_store_returns_empty(store):
    results = store.query(vector=[1.0, 0.0], top_k=5)
    assert results == []
