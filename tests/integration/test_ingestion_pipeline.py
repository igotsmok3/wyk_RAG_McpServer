"""Integration test for IngestionPipeline (C14).

Uses complex_technical_doc.pdf from fixtures.
External backends (embedding, vector store) are replaced with in-process fakes
so the test runs without a live Milvus/MySQL/LLM.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, List

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
COMPLEX_PDF = FIXTURES_DIR / "complex_technical_doc.pdf"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"


# ---------------------------------------------------------------------------
# Fake backends (no external I/O)
# ---------------------------------------------------------------------------

class FakeEmbedding:
    """Returns deterministic unit vectors of dimension 8 for any text list."""

    def embed(self, texts: List[str], trace: Any = None) -> List[List[float]]:
        return [[float(i % 8) / 8 for i in range(8)] for _ in texts]


class FakeVectorStore:
    """In-memory vector store that records every upsert call."""

    def __init__(self) -> None:
        self.upserted: list = []

    def upsert(self, records, trace: Any = None) -> None:
        self.upserted.extend(records)

    def query(self, vector, top_k, filters=None, trace=None):
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(tmp_bm25_dir: str):
    """Build a minimal Settings object suitable for testing."""
    from core.settings import (
        ChunkRefinerSettings,
        EmbeddingSettings,
        IngestionSettings,
        LLMSettings,
        MetadataEnricherSettings,
        ObservabilitySettings,
        Settings,
        VectorStoreSettings,
        VisionLLMSettings,
    )

    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=8),
        vector_store=VectorStoreSettings(provider="milvus"),
        vision_llm=VisionLLMSettings(enabled=False),
        observability=ObservabilitySettings(log_level="DEBUG"),
        ingestion=IngestionSettings(
            chunk_size=500,
            chunk_overlap=50,
            splitter="recursive",
            batch_size=16,
            chunk_refiner=ChunkRefinerSettings(use_llm=False),
            metadata_enricher=MetadataEnricherSettings(use_llm=False),
        ),
    )


def _build_pipeline(settings, fake_store: FakeVectorStore, tmp_dirs: dict):
    """Assemble a pipeline with all external deps replaced by fakes."""
    import libs.splitter.recursive_splitter  # noqa: F401 — registers 'recursive' splitter

    from ingestion.chunking.document_chunker import DocumentChunker
    from ingestion.embedding.batch_processor import BatchProcessor
    from ingestion.embedding.dense_encoder import DenseEncoder
    from ingestion.embedding.sparse_encoder import SparseEncoder
    from ingestion.pipeline import IngestionPipeline
    from ingestion.storage.bm25_indexer import BM25Indexer
    from ingestion.storage.vector_upserter import VectorUpserter
    from ingestion.transform.chunk_refiner import ChunkRefiner
    from ingestion.transform.image_captioner import ImageCaptioner
    from ingestion.transform.metadata_enricher import MetadataEnricher
    from libs.loader.file_integrity import SqliteIntegrityChecker
    from libs.loader.pdf_loader import PdfLoader

    integrity = SqliteIntegrityChecker(
        db_path=os.path.join(tmp_dirs["db"], "ingestion_history.db")
    )
    loader = PdfLoader(images_dir=tmp_dirs["images"])
    chunker = DocumentChunker(settings)

    fake_embedding = FakeEmbedding()
    dense_enc = DenseEncoder(settings, embedding=fake_embedding)
    sparse_enc = SparseEncoder()
    batch_proc = BatchProcessor(settings, batch_size=16, dense_encoder=dense_enc, sparse_encoder=sparse_enc)

    upserter = VectorUpserter(settings, vector_store=fake_store)
    bm25 = BM25Indexer(index_dir=tmp_dirs["bm25"])

    transforms = [
        ChunkRefiner(settings),
        MetadataEnricher(settings),
        ImageCaptioner(settings),
    ]

    return IngestionPipeline(
        settings,
        integrity_checker=integrity,
        loader=loader,
        chunker=chunker,
        transforms=transforms,
        batch_processor=batch_proc,
        vector_upserter=upserter,
        bm25_indexer=bm25,
        image_storage=None,  # skip MySQL for test
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dirs():
    with tempfile.TemporaryDirectory() as root:
        dirs = {
            "db": os.path.join(root, "db"),
            "bm25": os.path.join(root, "bm25"),
            "images": os.path.join(root, "images"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        yield dirs


@pytest.mark.skipif(not COMPLEX_PDF.exists(), reason="complex_technical_doc.pdf not found")
def test_pipeline_complex_pdf(tmp_dirs):
    """Full pipeline run on complex_technical_doc.pdf."""
    fake_store = FakeVectorStore()
    settings = _make_settings(tmp_dirs["bm25"])
    pipeline = _build_pipeline(settings, fake_store, tmp_dirs)

    result = pipeline.run(str(COMPLEX_PDF), collection="test_collection")

    # Not skipped
    assert not result.skipped
    assert result.doc_id != ""
    assert result.file_hash != ""

    # Chunks and records produced
    assert result.chunk_count > 0, "Expected at least one chunk"
    assert result.record_count == result.chunk_count, "Record count must match chunk count"

    # Vector store received all records
    assert len(fake_store.upserted) == result.record_count

    # BM25 index files written to disk
    bm25_index = os.path.join(tmp_dirs["bm25"], "index.json")
    bm25_meta = os.path.join(tmp_dirs["bm25"], "meta.json")
    assert os.path.exists(bm25_index), "BM25 index.json not found"
    assert os.path.exists(bm25_meta), "BM25 meta.json not found"


@pytest.mark.skipif(not COMPLEX_PDF.exists(), reason="complex_technical_doc.pdf not found")
def test_pipeline_skip_on_rerun(tmp_dirs):
    """Second run with same file is skipped (idempotent)."""
    fake_store = FakeVectorStore()
    settings = _make_settings(tmp_dirs["bm25"])
    pipeline = _build_pipeline(settings, fake_store, tmp_dirs)

    first = pipeline.run(str(COMPLEX_PDF), collection="test_collection")
    assert not first.skipped

    second = pipeline.run(str(COMPLEX_PDF), collection="test_collection")
    assert second.skipped
    # No additional upserts on second run
    assert len(fake_store.upserted) == first.record_count


@pytest.mark.skipif(not COMPLEX_PDF.exists(), reason="complex_technical_doc.pdf not found")
def test_pipeline_force_flag_reingests(tmp_dirs):
    """force=True bypasses skip logic and re-ingests."""
    fake_store = FakeVectorStore()
    settings = _make_settings(tmp_dirs["bm25"])
    pipeline = _build_pipeline(settings, fake_store, tmp_dirs)

    first = pipeline.run(str(COMPLEX_PDF), collection="test_collection")
    count_after_first = len(fake_store.upserted)

    second = pipeline.run(str(COMPLEX_PDF), collection="test_collection", force=True)
    assert not second.skipped
    assert len(fake_store.upserted) == count_after_first * 2


def test_pipeline_raises_on_missing_file(tmp_dirs):
    """Pipeline raises RuntimeError on a non-existent file."""
    fake_store = FakeVectorStore()
    settings = _make_settings(tmp_dirs["bm25"])
    pipeline = _build_pipeline(settings, fake_store, tmp_dirs)

    with pytest.raises((RuntimeError, FileNotFoundError)):
        pipeline.run("/non/existent/file.pdf")


@pytest.mark.skipif(not SIMPLE_PDF.exists(), reason="simple.pdf not found")
def test_pipeline_simple_pdf(tmp_dirs):
    """Regression: simple PDF still works after complex_technical_doc changes."""
    fake_store = FakeVectorStore()
    settings = _make_settings(tmp_dirs["bm25"])
    pipeline = _build_pipeline(settings, fake_store, tmp_dirs)

    result = pipeline.run(str(SIMPLE_PDF), collection="test_simple")

    assert not result.skipped
    assert result.chunk_count > 0
    assert len(fake_store.upserted) == result.record_count
