"""Tests for BM25Indexer (C11) — roundtrip, IDF accuracy, incremental update.

Corpus: realistic Chinese/English passages drawn from RAG/IR domain.
Tests verify:
  - build → load roundtrip produces identical results
  - IDF calculation correctness on known corpus
  - BM25 ranking respects term frequency and rarity
  - incremental update correctly extends the index
  - edge cases (empty corpus, no matches, unknown query terms)
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.types import Chunk, ChunkRecord
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer, BM25Result


# ---------------------------------------------------------------------------
# Test corpus — meaningful Chinese/English passages
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

PASSAGE_TRANSFORMER = (
    "Transformer 架构通过自注意力机制（Self-Attention）并行处理序列数据，"
    "大幅提升了自然语言处理任务的性能。BERT 和 GPT 都基于 Transformer 构建，"
    "分别用于编码理解和生成任务。预训练大模型已成为 NLP 领域的基础设施。"
)

PASSAGE_EN_ML = (
    "Machine learning is a subset of artificial intelligence that enables computers "
    "to learn from data without being explicitly programmed. "
    "Machine learning algorithms build mathematical models from training data "
    "to make predictions or decisions. "
    "Deep learning is a specialized form of machine learning using neural networks "
    "with many layers to learn complex patterns."
)

# A passage that overlaps partially with PASSAGE_RAG (for IDF testing)
PASSAGE_RETRIEVAL_FOCUSED = (
    "检索系统是信息系统的核心组件，负责从海量数据中快速定位相关信息。"
    "现代检索系统结合了稀疏检索（BM25）和稠密检索（向量检索）两种范式，"
    "通过混合检索策略提升召回率和精准度。"
)


def _make_chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        id=f"chunk_{idx:04d}",
        text=text,
        metadata={"source_path": "test_corpus.pdf", "chunk_index": idx},
    )


def _make_record_from_text(idx: int, text: str) -> ChunkRecord:
    """Convenience: use SparseEncoder to produce a realistic ChunkRecord."""
    encoder = SparseEncoder()
    chunk = _make_chunk(idx, text)
    return encoder.encode([chunk])[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder() -> SparseEncoder:
    return SparseEncoder()


@pytest.fixture(scope="module")
def corpus_chunks() -> list[Chunk]:
    passages = [
        PASSAGE_RAG,
        PASSAGE_VECTOR_DB,
        PASSAGE_BM25,
        PASSAGE_TRANSFORMER,
        PASSAGE_EN_ML,
        PASSAGE_RETRIEVAL_FOCUSED,
    ]
    return [_make_chunk(i, p) for i, p in enumerate(passages)]


@pytest.fixture(scope="module")
def corpus_records(encoder, corpus_chunks) -> list[ChunkRecord]:
    return encoder.encode(corpus_chunks)


@pytest.fixture
def tmp_index_dir(tmp_path) -> str:
    return str(tmp_path / "bm25_test_index")


@pytest.fixture
def built_indexer(corpus_records, tmp_index_dir) -> BM25Indexer:
    """A fresh BM25Indexer built from corpus_records."""
    idx = BM25Indexer(index_dir=tmp_index_dir)
    idx.build(corpus_records)
    return idx


# ---------------------------------------------------------------------------
# Structural / contract tests
# ---------------------------------------------------------------------------

def test_build_sets_doc_count(built_indexer, corpus_records):
    assert built_indexer.doc_count == len(corpus_records)


def test_is_loaded_after_build(built_indexer):
    assert built_indexer.is_loaded


def test_is_loaded_false_before_build(tmp_index_dir):
    idx = BM25Indexer(index_dir=tmp_index_dir + "_new")
    assert not idx.is_loaded


def test_query_returns_list(built_indexer):
    results = built_indexer.query({"检索": 2.0}, top_k=3)
    assert isinstance(results, list)


def test_query_results_are_bm25result(built_indexer):
    results = built_indexer.query({"检索": 1.0}, top_k=5)
    for r in results:
        assert isinstance(r, BM25Result)
        assert isinstance(r.chunk_id, str)
        assert isinstance(r.score, float)


def test_query_top_k_respected(built_indexer):
    results = built_indexer.query({"检索": 1.0}, top_k=2)
    assert len(results) <= 2


def test_query_scores_descending(built_indexer):
    results = built_indexer.query({"检索": 1.0, "向量": 1.0}, top_k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_empty_terms_returns_empty(built_indexer):
    assert built_indexer.query({}, top_k=5) == []


def test_query_unknown_term_returns_empty(built_indexer):
    results = built_indexer.query({"完全不存在的词xyzabc": 1.0}, top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# Roundtrip tests (build → save → load → query)
# ---------------------------------------------------------------------------

def test_save_creates_files(built_indexer, tmp_index_dir):
    built_indexer.save()
    assert os.path.exists(os.path.join(tmp_index_dir, "index.json"))
    assert os.path.exists(os.path.join(tmp_index_dir, "meta.json"))


def test_load_restores_doc_count(corpus_records, tmp_index_dir):
    # Build, save, then load into a fresh indexer
    idx1 = BM25Indexer(index_dir=tmp_index_dir)
    idx1.build(corpus_records)
    idx1.save()

    idx2 = BM25Indexer(index_dir=tmp_index_dir)
    idx2.load()
    assert idx2.doc_count == len(corpus_records)


def test_roundtrip_query_same_results(corpus_records, tmp_index_dir):
    """After save+load, query results must be identical to pre-save results."""
    idx1 = BM25Indexer(index_dir=tmp_index_dir)
    idx1.build(corpus_records)
    query = {"检索": 2.0, "向量": 1.0}
    results_before = idx1.query(query, top_k=5)
    idx1.save()

    idx2 = BM25Indexer(index_dir=tmp_index_dir)
    idx2.load()
    results_after = idx2.query(query, top_k=5)

    assert [r.chunk_id for r in results_before] == [r.chunk_id for r in results_after]
    for b, a in zip(results_before, results_after):
        assert abs(b.score - a.score) < 1e-9


def test_load_raises_if_no_index(tmp_path):
    idx = BM25Indexer(index_dir=str(tmp_path / "nonexistent"))
    with pytest.raises(FileNotFoundError):
        idx.load()


def test_query_before_build_raises(tmp_index_dir):
    idx = BM25Indexer(index_dir=tmp_index_dir + "_x")
    with pytest.raises(RuntimeError):
        idx.query({"检索": 1.0}, top_k=3)


# ---------------------------------------------------------------------------
# IDF accuracy tests (known corpus)
# ---------------------------------------------------------------------------

def test_idf_rare_term_higher_than_common_term(corpus_records, tmp_index_dir):
    """A term appearing in only 1 doc should have higher IDF than one in many docs."""
    idx = BM25Indexer(index_dir=tmp_index_dir)
    idx.build(corpus_records)

    # "bm25" appears in at most 2 docs; "检索" appears in multiple docs
    # IDF(rare) > IDF(common)
    idf_bm25 = idx._state.idf.get("bm25", None)
    idf_jianso = idx._state.idf.get("检索", None)

    if idf_bm25 is not None and idf_jianso is not None:
        assert idf_bm25 >= idf_jianso, (
            f"Expected IDF('bm25') >= IDF('检索'), got {idf_bm25:.4f} vs {idf_jianso:.4f}"
        )


def test_idf_formula_correctness():
    """Verify IDF = log((N - df + 0.5) / (df + 0.5)) on a synthetic 3-doc corpus."""
    # Build a tiny corpus where we know exact term frequencies
    records = [
        ChunkRecord(id="d0", text="", sparse_vector={"apple": 3.0, "fruit": 1.0}),
        ChunkRecord(id="d1", text="", sparse_vector={"apple": 1.0, "banana": 2.0}),
        ChunkRecord(id="d2", text="", sparse_vector={"banana": 1.0, "cherry": 1.0}),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25Indexer(index_dir=tmpdir)
        idx.build(records)

    N = 3
    # "apple" appears in 2 docs
    expected_idf_apple = math.log((N - 2 + 0.5) / (2 + 0.5))
    # "banana" appears in 2 docs
    expected_idf_banana = math.log((N - 2 + 0.5) / (2 + 0.5))
    # "cherry" appears in 1 doc
    expected_idf_cherry = math.log((N - 1 + 0.5) / (1 + 0.5))
    # "fruit" appears in 1 doc
    expected_idf_fruit = math.log((N - 1 + 0.5) / (1 + 0.5))

    assert abs(idx._state.idf["apple"] - expected_idf_apple) < 1e-9
    assert abs(idx._state.idf["banana"] - expected_idf_banana) < 1e-9
    assert abs(idx._state.idf["cherry"] - expected_idf_cherry) < 1e-9
    assert abs(idx._state.idf["fruit"] - expected_idf_fruit) < 1e-9


def test_idf_all_docs_contain_term_gives_negative_idf():
    """When a term appears in every document, IDF is negative (log < 1)."""
    records = [
        ChunkRecord(id="d0", text="", sparse_vector={"common": 1.0}),
        ChunkRecord(id="d1", text="", sparse_vector={"common": 2.0}),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25Indexer(index_dir=tmpdir)
        idx.build(records)

    # df=2, N=2 → log((2-2+0.5)/(2+0.5)) = log(0.5/2.5) < 0
    assert idx._state.idf["common"] < 0


# ---------------------------------------------------------------------------
# BM25 ranking semantic tests
# ---------------------------------------------------------------------------

def test_bm25_passage_ranks_high_for_bm25_query(built_indexer):
    """PASSAGE_BM25 (chunk_0002) should rank near the top for query term 'bm25'."""
    results = built_indexer.query({"bm25": 3.0}, top_k=3)
    chunk_ids = [r.chunk_id for r in results]
    assert "chunk_0002" in chunk_ids, f"Expected chunk_0002 in top-3, got: {chunk_ids}"


def test_retrieval_passage_ranks_high_for_jianso_query(built_indexer):
    """Passages about '检索' (RAG and retrieval-focused) should rank near the top."""
    results = built_indexer.query({"检索": 2.0}, top_k=3)
    chunk_ids = [r.chunk_id for r in results]
    # chunk_0000 (RAG) and chunk_0005 (retrieval-focused) both mention 检索 heavily
    assert any(cid in chunk_ids for cid in ("chunk_0000", "chunk_0005")), (
        f"Expected RAG or retrieval passage in top-3, got: {chunk_ids}"
    )


def test_ml_passage_ranks_high_for_learning_query(built_indexer):
    """PASSAGE_EN_ML (chunk_0004) should rank high for 'learning' query."""
    results = built_indexer.query({"learning": 3.0}, top_k=3)
    chunk_ids = [r.chunk_id for r in results]
    assert "chunk_0004" in chunk_ids, f"Expected chunk_0004 in top-3, got: {chunk_ids}"


def test_higher_tf_same_doc_boosts_score():
    """Higher TF in the same term yields a higher BM25 score when IDF > 0.

    IDF("kw") > 0  iff  df < N/2.  With N=5 and df=2, IDF = log(3.5/2.5) > 0,
    so BM25 scores are positively correlated with TF.
    """
    records = [
        ChunkRecord(id="hi_tf", text="", sparse_vector={"kw": 5.0}),
        ChunkRecord(id="lo_tf", text="", sparse_vector={"kw": 1.0}),
        # 3 padding docs without "kw" → N=5, df("kw")=2, IDF > 0
        ChunkRecord(id="d2", text="", sparse_vector={"other": 1.0}),
        ChunkRecord(id="d3", text="", sparse_vector={"other": 1.0}),
        ChunkRecord(id="d4", text="", sparse_vector={"other": 1.0}),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25Indexer(index_dir=tmpdir)
        idx.build(records)
        results = idx.query({"kw": 1.0}, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "hi_tf", (
        f"Chunk with TF=5 should rank above TF=1; got {results[0].chunk_id}"
    )
    assert results[0].score > results[1].score


def test_multi_term_query_boosts_matching_doc():
    """A document matching more query terms should outscore one matching fewer."""
    records = [
        ChunkRecord(id="full_match", text="", sparse_vector={"alpha": 1.0, "beta": 1.0, "gamma": 1.0}),
        ChunkRecord(id="partial_match", text="", sparse_vector={"alpha": 1.0}),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25Indexer(index_dir=tmpdir)
        idx.build(records)
        results = idx.query({"alpha": 1.0, "beta": 1.0, "gamma": 1.0}, top_k=2)

    assert results[0].chunk_id == "full_match", (
        f"Doc matching all 3 terms should rank first, got: {results[0].chunk_id}"
    )


# ---------------------------------------------------------------------------
# Incremental update tests
# ---------------------------------------------------------------------------

def test_update_increases_doc_count(corpus_records, tmp_index_dir):
    idx = BM25Indexer(index_dir=tmp_index_dir + "_upd")
    idx.build(corpus_records[:3])
    initial_count = idx.doc_count

    extra = [_make_record_from_text(10, PASSAGE_EN_ML)]
    idx.update(extra)

    assert idx.doc_count == initial_count + 1


def test_update_idempotent_for_same_records(corpus_records, tmp_index_dir):
    """Updating with already-indexed records must not change doc_count."""
    idx = BM25Indexer(index_dir=tmp_index_dir + "_idem")
    idx.build(corpus_records)
    count_before = idx.doc_count

    idx.update(corpus_records)  # all already indexed
    assert idx.doc_count == count_before


def test_update_makes_new_terms_queryable(tmp_index_dir):
    """After update, terms from new records should be retrievable."""
    base = [ChunkRecord(id="d0", text="", sparse_vector={"python": 1.0})]
    new_record = ChunkRecord(id="d1", text="", sparse_vector={"golang": 3.0})

    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25Indexer(index_dir=tmpdir)
        idx.build(base)
        assert idx.query({"golang": 1.0}, top_k=1) == []  # not yet indexed

        idx.update([new_record])
        results = idx.query({"golang": 1.0}, top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "d1"


def test_update_recalculates_idf(tmp_index_dir):
    """IDF values must change after adding documents that contain existing terms."""
    d0 = ChunkRecord(id="d0", text="", sparse_vector={"term": 1.0})
    d1 = ChunkRecord(id="d1", text="", sparse_vector={"other": 1.0})
    d2 = ChunkRecord(id="d2", text="", sparse_vector={"term": 2.0})  # adds df for "term"

    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25Indexer(index_dir=tmpdir)
        idx.build([d0, d1])
        idf_before = idx._state.idf["term"]

        idx.update([d2])
        idf_after = idx._state.idf["term"]

    # df("term") goes from 1 → 2 in a 3-doc corpus; IDF should decrease
    assert idf_after < idf_before, (
        f"IDF should decrease as df increases: before={idf_before:.4f}, after={idf_after:.4f}"
    )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_build_empty_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25Indexer(index_dir=tmpdir)
        idx.build([])
        assert idx.doc_count == 0
        assert idx.query({"检索": 1.0}, top_k=5) == []


def test_build_record_with_empty_sparse_vector():
    records = [
        ChunkRecord(id="empty_sv", text="", sparse_vector={}),
        ChunkRecord(id="normal", text="", sparse_vector={"term": 1.0}),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25Indexer(index_dir=tmpdir)
        idx.build(records)
        assert idx.doc_count == 2
        results = idx.query({"term": 1.0}, top_k=2)
        assert len(results) == 1
        assert results[0].chunk_id == "normal"


def test_top_k_zero_returns_empty(built_indexer):
    assert built_indexer.query({"检索": 1.0}, top_k=0) == []


def test_save_and_load_after_update(corpus_records, tmp_index_dir):
    """Full cycle: build → update → save → load → query still works."""
    extra = [_make_record_from_text(99, PASSAGE_TRANSFORMER)]

    idx1 = BM25Indexer(index_dir=tmp_index_dir + "_full")
    idx1.build(corpus_records)
    idx1.update(extra)
    idx1.save()

    idx2 = BM25Indexer(index_dir=tmp_index_dir + "_full")
    idx2.load()
    assert idx2.doc_count == len(corpus_records) + 1

    results = idx2.query({"transformer": 1.0}, top_k=3)
    assert any(r.chunk_id == "chunk_0099" for r in results)
