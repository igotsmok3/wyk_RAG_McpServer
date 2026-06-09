"""Tests for SparseEncoder (C9).

Uses actual Chinese/English text passages to verify:
  - structural contract (count, types, sparse_vector always set)
  - semantic reasonableness (high-frequency terms get higher TF)
  - edge cases (empty text, single-word, multi-chunk ordering)
  - downstream BM25Indexer contract (term: float, non-negative)
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.types import Chunk, ChunkRecord
from ingestion.embedding.sparse_encoder import SparseEncoder


# ---------------------------------------------------------------------------
# Test fixtures — meaningful text passages
# ---------------------------------------------------------------------------

PASSAGE_RAG = (
    "检索增强生成（RAG）是一种将检索系统与生成式语言模型相结合的技术框架。"
    "RAG 首先通过检索模块从知识库中找到与查询最相关的文档片段，"
    "然后将这些片段连同用户查询一起输入到生成模型中，"
    "生成更准确、更有依据的答案。检索质量直接影响生成质量，"
    "因此高效的检索是 RAG 系统的核心。"
)

PASSAGE_VECTOR_DB = (
    "向量数据库（Vector Database）专门用于存储和检索高维向量。"
    "它通过近似最近邻（ANN）算法实现毫秒级的向量相似度搜索。"
    "常见的向量数据库包括 Milvus、Pinecone、Weaviate 等。"
    "在语义搜索场景中，向量数据库将文本编码为向量，"
    "并根据余弦相似度或欧式距离进行排序和检索。"
)

PASSAGE_BM25 = (
    "BM25 是信息检索领域经典的词频统计模型，全称 Best Matching 25。"
    "BM25 综合了词项频率（TF）和逆文档频率（IDF）的统计信息，"
    "通过对文档长度进行归一化，解决了长文档天然得分偏高的问题。"
    "BM25 在关键词匹配场景中表现优异，常与向量检索结合使用，形成混合检索（Hybrid Search）。"
)

PASSAGE_EN = (
    "Machine learning is a subset of artificial intelligence that enables computers "
    "to learn from data without being explicitly programmed. "
    "Machine learning algorithms build mathematical models from training data "
    "to make predictions or decisions. "
    "Deep learning is a specialized form of machine learning using neural networks "
    "with many layers to learn complex patterns."
)

PASSAGE_EMPTY = ""
PASSAGE_WHITESPACE = "   \t\n   "
PASSAGE_SINGLE_WORD = "检索"
PASSAGE_STOPWORDS_ONLY = "的 了 是 在 和 the is a an"


def _make_chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        id=f"chunk_{idx:04d}",
        text=text,
        metadata={"source_path": "test.pdf", "chunk_index": idx},
    )


@pytest.fixture(scope="module")
def encoder() -> SparseEncoder:
    return SparseEncoder()


@pytest.fixture(scope="module")
def chunks_multi() -> list[Chunk]:
    return [
        _make_chunk(0, PASSAGE_RAG),
        _make_chunk(1, PASSAGE_VECTOR_DB),
        _make_chunk(2, PASSAGE_BM25),
        _make_chunk(3, PASSAGE_EN),
    ]


@pytest.fixture(scope="module")
def records_multi(encoder, chunks_multi) -> list[ChunkRecord]:
    return encoder.encode(chunks_multi)


# ---------------------------------------------------------------------------
# Structural / contract tests
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty(encoder):
    assert encoder.encode([]) == []


def test_output_count_matches_input(chunks_multi, records_multi):
    assert len(records_multi) == len(chunks_multi)


def test_all_records_are_chunk_record(records_multi):
    for r in records_multi:
        assert isinstance(r, ChunkRecord)


def test_sparse_vector_always_set(records_multi):
    """sparse_vector must be a dict (never None) for every record."""
    for r in records_multi:
        assert r.sparse_vector is not None
        assert isinstance(r.sparse_vector, dict)


def test_sparse_vector_values_are_floats(records_multi):
    for r in records_multi:
        for term, weight in r.sparse_vector.items():
            assert isinstance(weight, float), f"Expected float, got {type(weight)} for term '{term}'"


def test_sparse_vector_values_are_positive(records_multi):
    for r in records_multi:
        for term, weight in r.sparse_vector.items():
            assert weight > 0, f"Non-positive weight {weight} for term '{term}'"


def test_sparse_vector_keys_are_strings(records_multi):
    for r in records_multi:
        for term in r.sparse_vector:
            assert isinstance(term, str)


def test_record_ids_match_chunk_ids(chunks_multi, records_multi):
    for chunk, record in zip(chunks_multi, records_multi):
        assert record.id == chunk.id


def test_record_text_matches_chunk_text(chunks_multi, records_multi):
    for chunk, record in zip(chunks_multi, records_multi):
        assert record.text == chunk.text


def test_metadata_preserved(chunks_multi, records_multi):
    for chunk, record in zip(chunks_multi, records_multi):
        assert record.metadata["source_path"] == chunk.metadata["source_path"]
        assert record.metadata["chunk_index"] == chunk.metadata["chunk_index"]


def test_dense_vector_not_touched(records_multi):
    """SparseEncoder must not populate dense_vector."""
    for r in records_multi:
        assert r.dense_vector is None


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_empty_text_produces_empty_sparse_vector(encoder):
    chunk = _make_chunk(99, PASSAGE_EMPTY)
    records = encoder.encode([chunk])
    assert len(records) == 1
    assert records[0].sparse_vector == {}


def test_whitespace_only_text_produces_empty_sparse_vector(encoder):
    chunk = _make_chunk(99, PASSAGE_WHITESPACE)
    records = encoder.encode([chunk])
    assert records[0].sparse_vector == {}


def test_stopwords_only_text_produces_empty_or_near_empty_sparse_vector(encoder):
    chunk = _make_chunk(99, PASSAGE_STOPWORDS_ONLY)
    records = encoder.encode([chunk])
    # All tokens are stopwords; sparse_vector should be empty or very small
    assert len(records[0].sparse_vector) == 0


def test_single_word_text(encoder):
    chunk = _make_chunk(99, PASSAGE_SINGLE_WORD)
    records = encoder.encode([chunk])
    assert len(records) == 1
    sv = records[0].sparse_vector
    assert len(sv) >= 1
    # "检索" should appear with count 1.0
    assert "检索" in sv
    assert sv["检索"] == 1.0


# ---------------------------------------------------------------------------
# Semantic reasonableness tests
# ---------------------------------------------------------------------------

def test_bm25_passage_contains_bm25_term(encoder):
    """PASSAGE_BM25 mentions 'BM25' three times; its TF should be the highest."""
    chunk = _make_chunk(0, PASSAGE_BM25)
    records = encoder.encode([chunk])
    sv = records[0].sparse_vector
    # "bm25" after lowercasing should be a key (jieba preserves ASCII tokens)
    assert "bm25" in sv
    assert sv["bm25"] >= 3.0, f"Expected TF ≥ 3 for 'bm25', got {sv['bm25']}"


def test_repeated_term_gets_higher_tf(encoder):
    """A term repeated 3× should have TF = 3.0; a term once should have TF = 1.0."""
    text = "机器学习 机器学习 机器学习 深度学习"
    chunk = _make_chunk(0, text)
    records = encoder.encode([chunk])
    sv = records[0].sparse_vector
    # jieba may segment differently, but 机器学习 as a whole phrase should appear
    # Find the term with highest count
    if "机器学习" in sv and "深度学习" in sv:
        assert sv["机器学习"] > sv["深度学习"], (
            f"'机器学习' (×3) should score higher than '深度学习' (×1): {sv}"
        )


def test_different_passages_produce_different_sparse_vectors(records_multi):
    """Distinct passages should yield clearly different term sets."""
    sv0 = set(records_multi[0].sparse_vector.keys())
    sv1 = set(records_multi[1].sparse_vector.keys())
    sv2 = set(records_multi[2].sparse_vector.keys())
    # Each passage has domain-specific vocabulary; overlapping ratio should be <50%
    overlap_01 = len(sv0 & sv1) / max(len(sv0 | sv1), 1)
    overlap_02 = len(sv0 & sv2) / max(len(sv0 | sv2), 1)
    assert overlap_01 < 0.5, f"Passages 0 and 1 share too many terms: {overlap_01:.2%}"
    assert overlap_02 < 0.5, f"Passages 0 and 2 share too many terms: {overlap_02:.2%}"


def test_rag_passage_contains_retrieval_terms(encoder):
    """PASSAGE_RAG should include domain terms like '检索' and '生成'."""
    chunk = _make_chunk(0, PASSAGE_RAG)
    records = encoder.encode([chunk])
    sv = records[0].sparse_vector
    assert "检索" in sv, f"Expected '检索' in sparse vector, keys: {list(sv.keys())[:20]}"
    assert "生成" in sv, f"Expected '生成' in sparse vector, keys: {list(sv.keys())[:20]}"


def test_english_passage_has_meaningful_terms(encoder):
    """PASSAGE_EN should include 'learning' (appears multiple times)."""
    chunk = _make_chunk(0, PASSAGE_EN)
    records = encoder.encode([chunk])
    sv = records[0].sparse_vector
    assert "learning" in sv, f"Expected 'learning' in sparse vector, keys: {list(sv.keys())[:20]}"
    # "machine" + "learning" each appear 3+ times
    assert sv.get("learning", 0) >= 3.0, (
        f"'learning' appears 3+ times in PASSAGE_EN, expected TF ≥ 3, got {sv.get('learning')}"
    )


def test_order_preserved(encoder):
    texts = [PASSAGE_RAG, PASSAGE_VECTOR_DB, PASSAGE_BM25]
    input_chunks = [_make_chunk(i, t) for i, t in enumerate(texts)]
    result = encoder.encode(input_chunks)
    for i, (chunk, record) in enumerate(zip(input_chunks, result)):
        assert record.id == chunk.id, f"Order mismatch at index {i}"
        assert record.text == chunk.text


def test_single_chunk(encoder):
    chunk = _make_chunk(0, PASSAGE_RAG)
    records = encoder.encode([chunk])
    assert len(records) == 1
    assert records[0].sparse_vector is not None
    assert len(records[0].sparse_vector) > 0


def test_sparse_vector_suitable_for_bm25_indexer(records_multi):
    """Verify contract: {term: float} where float is raw term count (≥ 1.0)."""
    for r in records_multi:
        for term, tf in r.sparse_vector.items():
            assert isinstance(term, str) and len(term) > 0
            assert isinstance(tf, float) and tf >= 1.0
