"""Integration tests for MilvusStore: real upsert→query roundtrip using Milvus Lite."""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.vector_store.Milvus_store import MilvusStore
from libs.vector_store.base_vector_store import VectorRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory, register_provider


# ---------------------------------------------------------------------------
# Settings stub
# ---------------------------------------------------------------------------

@dataclass
class FakeVectorStoreSettings:
    provider: str = "milvus"
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "test_hub"
    data_dir: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """MilvusStore backed by a temp directory (Milvus Lite)."""
    settings = FakeVectorStoreSettings(
        collection_name="test_collection",
        data_dir=str(tmp_path),
    )
    store = MilvusStore(settings)
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(values: list[float]) -> list[float]:
    return values


def _records(n: int, dim: int = 4) -> list[VectorRecord]:
    return [
        VectorRecord(
            id=f"rec_{i}",
            vector=[float(i) / n] * dim,
            text=f"document text {i}",
            metadata={"index": i, "source": f"file_{i}.pdf", "collection": "main"},
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests: factory routing
# ---------------------------------------------------------------------------

def test_factory_creates_milvus_store(tmp_path):
    register_provider("milvus", MilvusStore)
    settings = FakeVectorStoreSettings(
        provider="milvus",
        collection_name="factory_test",
        data_dir=str(tmp_path),
    )
    store = MilvusStore(settings)
    assert isinstance(store, MilvusStore)
    store.close()


# ---------------------------------------------------------------------------
# Tests: basic upsert
# ---------------------------------------------------------------------------

def test_upsert_single_record(tmp_store):
    rec = VectorRecord(id="a1", vector=[1.0, 0.0, 0.0, 0.0], text="hello world")
    tmp_store.upsert([rec])
    results = tmp_store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].id == "a1"
    assert results[0].text == "hello world"


def test_upsert_multiple_records(tmp_store):
    recs = _records(5)
    tmp_store.upsert(recs)
    results = tmp_store.query(vector=[0.8, 0.8, 0.8, 0.8], top_k=5)
    assert len(results) == 5
    ids = {r.id for r in results}
    assert ids == {"rec_0", "rec_1", "rec_2", "rec_3", "rec_4"}


def test_upsert_empty_list_noop(tmp_store):
    tmp_store.upsert([])
    results = tmp_store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# Tests: idempotent upsert
# ---------------------------------------------------------------------------

def test_upsert_same_id_overwrites(tmp_store):
    tmp_store.upsert([VectorRecord(id="x", vector=[1.0, 0.0, 0.0, 0.0], text="original")])
    tmp_store.upsert([VectorRecord(id="x", vector=[1.0, 0.0, 0.0, 0.0], text="updated")])
    results = tmp_store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].text == "updated"


# ---------------------------------------------------------------------------
# Tests: vector query
# ---------------------------------------------------------------------------

def test_query_returns_closest_vector(tmp_store):
    tmp_store.upsert([
        VectorRecord(id="far",  vector=[0.0, 0.0, 0.0, 1.0], text="far"),
        VectorRecord(id="near", vector=[1.0, 0.0, 0.0, 0.0], text="near"),
    ])
    results = tmp_store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert results[0].id == "near"


def test_query_top_k_limits_results(tmp_store):
    tmp_store.upsert(_records(10))
    results = tmp_store.query(vector=[0.5] * 4, top_k=3)
    assert len(results) <= 3


def test_query_empty_store_returns_empty(tmp_store):
    results = tmp_store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_query_results_include_text_and_metadata(tmp_store):
    tmp_store.upsert([
        VectorRecord(
            id="m1",
            vector=[1.0, 0.0, 0.0, 0.0],
            text="some content",
            metadata={"source": "doc.pdf", "page": 3},
        )
    ])
    results = tmp_store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0].text == "some content"
    assert results[0].metadata["source"] == "doc.pdf"
    assert results[0].metadata["page"] == 3


# ---------------------------------------------------------------------------
# Tests: metadata filters
# ---------------------------------------------------------------------------

def test_query_filter_by_metadata_key(tmp_store):
    tmp_store.upsert([
        VectorRecord(id="a", vector=[1.0, 0.0, 0.0, 0.0], metadata={"col": "A"}),
        VectorRecord(id="b", vector=[1.0, 0.0, 0.0, 0.0], metadata={"col": "B"}),
    ])
    results = tmp_store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5, filters={"col": "A"})
    assert len(results) == 1
    assert results[0].id == "a"


def test_query_filter_excludes_non_matching(tmp_store):
    tmp_store.upsert([
        VectorRecord(id="x", vector=[0.5, 0.5, 0.0, 0.0], metadata={"type": "pdf"}),
        VectorRecord(id="y", vector=[0.5, 0.5, 0.0, 0.0], metadata={"type": "html"}),
        VectorRecord(id="z", vector=[0.5, 0.5, 0.0, 0.0], metadata={"type": "pdf"}),
    ])
    results = tmp_store.query(vector=[0.5, 0.5, 0.0, 0.0], top_k=10, filters={"type": "pdf"})
    ids = {r.id for r in results}
    assert ids == {"x", "z"}
    assert "y" not in ids


def test_query_no_filter_returns_all(tmp_store):
    tmp_store.upsert([
        VectorRecord(id="p", vector=[1.0, 0.0, 0.0, 0.0], metadata={"col": "A"}),
        VectorRecord(id="q", vector=[1.0, 0.0, 0.0, 0.0], metadata={"col": "B"}),
    ])
    results = tmp_store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Tests: persistence (data survives across instances)
# ---------------------------------------------------------------------------

def test_data_persists_across_store_instances(tmp_path):
    settings = FakeVectorStoreSettings(
        collection_name="persist_test",
        data_dir=str(tmp_path),
    )

    store1 = MilvusStore(settings)
    store1.upsert([VectorRecord(id="p1", vector=[1.0, 0.0, 0.0, 0.0], text="persisted")])
    store1.close()

    store2 = MilvusStore(settings)
    results = store2.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=1)
    store2.close()

    assert len(results) == 1
    assert results[0].id == "p1"
    assert results[0].text == "persisted"
