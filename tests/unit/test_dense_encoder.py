"""Tests for DenseEncoder (C8) — uses real Qwen embedding, no mocks."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Register qwen embedding provider before any factory calls
import libs.embedding.qwen_embedding  # noqa: F401

from core.settings import load_settings
from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        id=f"chunk_{idx:04d}",
        text=text,
        metadata={"source_path": "test.pdf", "chunk_index": idx},
    )


SAMPLE_TEXTS = [
    "深度学习是机器学习的一个分支，通过多层神经网络来学习数据的表示。",
    "向量数据库专门用于存储和检索高维向量，广泛应用于语义搜索场景。",
    "检索增强生成（RAG）结合了检索系统和生成式语言模型的优势。",
    "自然语言处理技术使计算机能够理解、生成和操作人类语言。",
    "Transformer 架构是现代大语言模型的核心组件，基于注意力机制。",
]


@pytest.fixture(scope="module")
def settings():
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "settings.yaml"
    )
    return load_settings(cfg_path)


@pytest.fixture(scope="module")
def encoder(settings):
    return DenseEncoder(settings)


@pytest.fixture(scope="module")
def chunks():
    return [_make_chunk(i, text) for i, text in enumerate(SAMPLE_TEXTS)]


@pytest.fixture(scope="module")
def records(encoder, chunks):
    """Encode once, reuse across tests."""
    return encoder.encode(chunks)


# ---------------------------------------------------------------------------
# Structural / count tests
# ---------------------------------------------------------------------------

def test_output_count_matches_input(chunks, records):
    assert len(records) == len(chunks)


def test_all_records_are_chunk_record(records):
    for r in records:
        assert isinstance(r, ChunkRecord)


def test_all_records_have_dense_vector(records):
    for r in records:
        assert r.dense_vector is not None
        assert isinstance(r.dense_vector, list)
        assert len(r.dense_vector) > 0


def test_vector_dimension_consistent(records):
    dims = {len(r.dense_vector) for r in records}
    assert len(dims) == 1, f"Inconsistent dimensions across records: {dims}"


def test_vector_dimension_matches_config(settings, records):
    expected_dim = settings.embedding.dimensions
    actual_dim = len(records[0].dense_vector)
    assert actual_dim == expected_dim, (
        f"Expected dimension {expected_dim}, got {actual_dim}"
    )


def test_record_ids_match_chunk_ids(chunks, records):
    for chunk, record in zip(chunks, records):
        assert record.id == chunk.id


def test_record_text_matches_chunk_text(chunks, records):
    for chunk, record in zip(chunks, records):
        assert record.text == chunk.text


def test_metadata_preserved(chunks, records):
    for chunk, record in zip(chunks, records):
        assert record.metadata["source_path"] == chunk.metadata["source_path"]
        assert record.metadata["chunk_index"] == chunk.metadata["chunk_index"]


# ---------------------------------------------------------------------------
# Vector quality / semantic tests
# ---------------------------------------------------------------------------

def test_vectors_are_floats(records):
    for r in records:
        for v in r.dense_vector:
            assert isinstance(v, float)


def test_different_texts_produce_different_vectors(records):
    """Semantically different texts should yield distinct vectors."""
    v0 = records[0].dense_vector
    v1 = records[1].dense_vector
    assert v0 != v1, "Two different texts produced identical vectors"


def test_vectors_are_non_zero(records):
    for r in records:
        assert any(v != 0.0 for v in r.dense_vector), f"Zero vector for chunk {r.id}"


# ---------------------------------------------------------------------------
# Edge case tests (with dedicated encoder to avoid fixture contamination)
# ---------------------------------------------------------------------------

def test_empty_chunks_returns_empty(encoder):
    result = encoder.encode([])
    assert result == []


def test_single_chunk(encoder):
    chunk = _make_chunk(99, "单句向量化测试。")
    result = encoder.encode([chunk])
    assert len(result) == 1
    assert result[0].dense_vector is not None
    assert len(result[0].dense_vector) > 0


def test_order_preserved(encoder):
    texts = ["第一段", "第二段", "第三段"]
    input_chunks = [_make_chunk(i, t) for i, t in enumerate(texts)]
    result = encoder.encode(input_chunks)
    for i, (chunk, record) in enumerate(zip(input_chunks, result)):
        assert record.id == chunk.id, f"Order mismatch at index {i}"
        assert record.text == chunk.text
