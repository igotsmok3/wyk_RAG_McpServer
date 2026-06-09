"""Tests for BatchProcessor (C10) — uses mock encoders, no real API calls."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.types import Chunk, ChunkRecord
from ingestion.embedding.batch_processor import BatchProcessor, _merge_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(idx: int, text: str = "") -> Chunk:
    return Chunk(
        id=f"chunk_{idx:04d}",
        text=text or f"text for chunk {idx}",
        metadata={"source_path": "test.pdf", "chunk_index": idx},
    )


def _make_record(chunk: Chunk, dense: list[float] | None = None, sparse: dict | None = None) -> ChunkRecord:
    r = ChunkRecord.from_chunk(chunk)
    r.dense_vector = dense or [float(ord(chunk.id[-1])) / 100.0]
    r.sparse_vector = sparse or {"token": 1.0}
    return r


def _fake_dense_encoder(chunks: list[Chunk], **_) -> list[ChunkRecord]:
    """Returns records with a simple deterministic dense_vector."""
    return [_make_record(c, dense=[float(i) * 0.1 for i in range(4)]) for c in chunks]


def _fake_sparse_encoder(chunks: list[Chunk], **_) -> list[ChunkRecord]:
    """Returns records with a simple deterministic sparse_vector."""
    return [_make_record(c, sparse={f"t{i}": float(i + 1)}) for i, c in enumerate(chunks)]


def _build_processor(batch_size: int = 32) -> BatchProcessor:
    """Build a BatchProcessor with mock encoders (no settings needed)."""
    settings = MagicMock()

    dense_mock = MagicMock()
    dense_mock.encode.side_effect = _fake_dense_encoder

    sparse_mock = MagicMock()
    sparse_mock.encode.side_effect = _fake_sparse_encoder

    return BatchProcessor(
        settings=settings,
        batch_size=batch_size,
        dense_encoder=dense_mock,
        sparse_encoder=sparse_mock,
    )


# ---------------------------------------------------------------------------
# Batching logic
# ---------------------------------------------------------------------------

def test_five_chunks_batch2_yields_three_batches():
    """batch_size=2 on 5 chunks must split into 3 batches."""
    processor = _build_processor(batch_size=2)
    chunks = [_make_chunk(i) for i in range(5)]

    calls: list[int] = []

    original_encode = processor._dense.encode.side_effect

    def counting_encode(batch, **kw):
        calls.append(len(batch))
        return original_encode(batch, **kw)

    processor._dense.encode.side_effect = counting_encode
    processor._sparse.encode.side_effect = lambda b, **kw: _fake_sparse_encoder(b)

    processor.process(chunks)

    assert calls == [2, 2, 1], f"Expected [2, 2, 1] batch sizes, got {calls}"


def test_output_count_matches_input():
    processor = _build_processor(batch_size=2)
    chunks = [_make_chunk(i) for i in range(5)]
    results = processor.process(chunks)
    assert len(results) == 5


def test_exact_multiple_batch():
    """6 chunks with batch_size=3 → 2 batches of 3."""
    processor = _build_processor(batch_size=3)
    chunks = [_make_chunk(i) for i in range(6)]
    calls: list[int] = []

    original = processor._dense.encode.side_effect

    def track(batch, **kw):
        calls.append(len(batch))
        return original(batch, **kw)

    processor._dense.encode.side_effect = track
    results = processor.process(chunks)

    assert calls == [3, 3]
    assert len(results) == 6


def test_single_chunk():
    processor = _build_processor(batch_size=10)
    chunks = [_make_chunk(0, "hello")]
    results = processor.process(chunks)
    assert len(results) == 1


def test_empty_input_returns_empty():
    processor = _build_processor()
    assert processor.process([]) == []


# ---------------------------------------------------------------------------
# Order stability
# ---------------------------------------------------------------------------

def test_order_preserved():
    """Output order must match input order."""
    processor = _build_processor(batch_size=2)
    chunks = [_make_chunk(i, f"chunk text {i}") for i in range(5)]

    # Make dense encode return IDs deterministically
    def ordered_dense(batch, **_):
        return [ChunkRecord(id=c.id, text=c.text, metadata=c.metadata,
                            dense_vector=[float(idx) for idx in range(4)])
                for c in batch]

    processor._dense.encode.side_effect = ordered_dense
    processor._sparse.encode.side_effect = lambda b, **_: _fake_sparse_encoder(b)

    results = processor.process(chunks)
    result_ids = [r.id for r in results]
    expected_ids = [c.id for c in chunks]
    assert result_ids == expected_ids


# ---------------------------------------------------------------------------
# Vector population
# ---------------------------------------------------------------------------

def test_all_records_have_dense_vector():
    processor = _build_processor(batch_size=3)
    chunks = [_make_chunk(i) for i in range(5)]
    results = processor.process(chunks)
    for r in results:
        assert r.dense_vector is not None, f"Missing dense_vector for {r.id}"


def test_all_records_have_sparse_vector():
    processor = _build_processor(batch_size=3)
    chunks = [_make_chunk(i) for i in range(5)]
    results = processor.process(chunks)
    for r in results:
        assert r.sparse_vector is not None, f"Missing sparse_vector for {r.id}"


def test_sparse_vector_merged_from_sparse_encoder():
    """The sparse_vector on output records must come from the sparse encoder."""
    processor = _build_processor(batch_size=10)
    chunk = _make_chunk(0, "test text")

    processor._dense.encode.side_effect = lambda b, **_: [
        ChunkRecord(id=b[0].id, text=b[0].text, metadata={}, dense_vector=[1.0, 2.0])
    ]
    processor._sparse.encode.side_effect = lambda b, **_: [
        ChunkRecord(id=b[0].id, text=b[0].text, metadata={}, sparse_vector={"keyword": 3.14})
    ]

    results = processor.process([chunk])
    assert results[0].sparse_vector == {"keyword": 3.14}
    assert results[0].dense_vector == [1.0, 2.0]


# ---------------------------------------------------------------------------
# _merge_records helper
# ---------------------------------------------------------------------------

def test_merge_records_copies_sparse():
    chunk = _make_chunk(0)
    dr = ChunkRecord.from_chunk(chunk)
    dr.dense_vector = [0.1, 0.2]
    sr = ChunkRecord.from_chunk(chunk)
    sr.sparse_vector = {"word": 2.0}

    merged = _merge_records([dr], [sr])
    assert merged[0].sparse_vector == {"word": 2.0}
    assert merged[0].dense_vector == [0.1, 0.2]


def test_merge_records_mismatch_raises():
    dr = ChunkRecord(id="a", text="a")
    sr1 = ChunkRecord(id="b", text="b")
    sr2 = ChunkRecord(id="c", text="c")
    with pytest.raises(ValueError, match="mismatch"):
        _merge_records([dr], [sr1, sr2])


# ---------------------------------------------------------------------------
# Trace integration (smoke)
# ---------------------------------------------------------------------------

def test_trace_record_stage_called_per_batch():
    """If trace has record_stage, it should be called once per batch."""
    processor = _build_processor(batch_size=2)
    chunks = [_make_chunk(i) for i in range(5)]

    trace = MagicMock()
    trace.record_stage = MagicMock()

    processor.process(chunks, trace=trace)

    # 5 chunks / batch_size 2 = 3 batches
    assert trace.record_stage.call_count == 3


def test_trace_none_does_not_raise():
    """Passing trace=None should not raise."""
    processor = _build_processor(batch_size=3)
    chunks = [_make_chunk(i) for i in range(4)]
    processor.process(chunks, trace=None)
