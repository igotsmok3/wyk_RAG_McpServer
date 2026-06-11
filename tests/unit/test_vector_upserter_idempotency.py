"""Tests for VectorUpserter (C12) — idempotency, ordering, error handling.

Tests verify:
  - Same chunk upserted twice produces the same ID (idempotency via stable ChunkRecord.id)
  - Content change produces a different ID
  - Batch upsert maintains insertion order
  - Missing dense_vector raises ValueError
  - Empty input is a no-op
"""
from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.types import ChunkRecord
from ingestion.storage.vector_upserter import VectorUpserter
from libs.vector_store.base_vector_store import VectorRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(chunk_id: str, vector: list[float], text: str = "hello") -> ChunkRecord:
    r = ChunkRecord(id=chunk_id, text=text, metadata={"source_path": "doc.pdf"})
    r.dense_vector = vector
    return r


def _make_upserter(mock_store: Any) -> VectorUpserter:
    """Create a VectorUpserter with an injected mock store (bypasses factory)."""
    upserter = VectorUpserter.__new__(VectorUpserter)
    upserter._store = mock_store
    return upserter


# ---------------------------------------------------------------------------
# Idempotency: same ChunkRecord → same VectorRecord.id
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_same_record_produces_same_id(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        record = _make_record("doc_0000_ab12cd34", [0.1, 0.2, 0.3])

        upserter.upsert([record])
        upserter.upsert([record])

        assert store.upsert.call_count == 2
        first_call_ids = [r.id for r in store.upsert.call_args_list[0][0][0]]
        second_call_ids = [r.id for r in store.upsert.call_args_list[1][0][0]]
        assert first_call_ids == second_call_ids

    def test_different_content_produces_different_id(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        record_a = _make_record("doc_0000_ab12cd34", [0.1, 0.2])
        record_b = _make_record("doc_0000_ef56gh78", [0.3, 0.4])

        upserter.upsert([record_a, record_b])

        passed_records: list[VectorRecord] = store.upsert.call_args[0][0]
        ids = [r.id for r in passed_records]
        assert ids[0] != ids[1]

    def test_vector_record_id_equals_chunk_record_id(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        record = _make_record("my-stable-id-001", [1.0, 2.0])
        upserter.upsert([record])

        passed: list[VectorRecord] = store.upsert.call_args[0][0]
        assert passed[0].id == "my-stable-id-001"


# ---------------------------------------------------------------------------
# Ordering: batch upsert must preserve input order
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_batch_order_preserved(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        records = [
            _make_record(f"doc_{i:04d}_xxxx", [float(i)] * 4)
            for i in range(5)
        ]
        upserter.upsert(records)

        passed: list[VectorRecord] = store.upsert.call_args[0][0]
        assert [r.id for r in passed] == [f"doc_{i:04d}_xxxx" for i in range(5)]

    def test_single_record_upsert(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        record = _make_record("single-id", [0.5, 0.6])
        upserter.upsert([record])

        store.upsert.assert_called_once()
        passed: list[VectorRecord] = store.upsert.call_args[0][0]
        assert len(passed) == 1
        assert passed[0].id == "single-id"


# ---------------------------------------------------------------------------
# VectorRecord field mapping
# ---------------------------------------------------------------------------

class TestFieldMapping:
    def test_vector_record_fields_match_chunk_record(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        record = ChunkRecord(
            id="test-id",
            text="chunk text",
            metadata={"source_path": "a/b.pdf", "chunk_index": 3},
        )
        record.dense_vector = [0.1, 0.2, 0.3]

        upserter.upsert([record])

        passed: list[VectorRecord] = store.upsert.call_args[0][0]
        vr = passed[0]
        assert vr.id == "test-id"
        assert vr.vector == [0.1, 0.2, 0.3]
        assert vr.text == "chunk text"
        assert vr.metadata["source_path"] == "a/b.pdf"
        assert vr.metadata["chunk_index"] == 3

    def test_metadata_is_copied_not_shared(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        original_meta = {"source_path": "doc.pdf"}
        record = ChunkRecord(id="id-1", text="text", metadata=original_meta)
        record.dense_vector = [1.0]

        upserter.upsert([record])

        passed: list[VectorRecord] = store.upsert.call_args[0][0]
        passed[0].metadata["new_key"] = "injected"
        assert "new_key" not in original_meta


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_dense_vector_raises_value_error(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        record = ChunkRecord(id="no-vec", text="text", metadata={})
        # dense_vector is None by default

        with pytest.raises(ValueError, match="missing dense_vector"):
            upserter.upsert([record])

        store.upsert.assert_not_called()

    def test_partial_missing_dense_vector_raises(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        good = _make_record("good-id", [1.0, 2.0])
        bad = ChunkRecord(id="bad-id", text="text", metadata={})

        with pytest.raises(ValueError, match="bad-id"):
            upserter.upsert([good, bad])

        store.upsert.assert_not_called()

    def test_empty_records_is_noop(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        upserter.upsert([])

        store.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# Trace forwarding
# ---------------------------------------------------------------------------

class TestTraceForwarding:
    def test_trace_is_passed_to_store(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        record = _make_record("id-1", [0.1])
        fake_trace = object()

        upserter.upsert([record], trace=fake_trace)

        _, kwargs = store.upsert.call_args
        assert kwargs.get("trace") is fake_trace

    def test_no_trace_passes_none(self):
        store = MagicMock()
        upserter = _make_upserter(store)

        record = _make_record("id-1", [0.1])
        upserter.upsert([record])

        _, kwargs = store.upsert.call_args
        assert kwargs.get("trace") is None
