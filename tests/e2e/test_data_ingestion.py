"""E2E tests for scripts/ingest.py (C15).

Calls ``main()`` directly with injected fake backends so no live Milvus /
LLM / MySQL is required.  All persistent state (SQLite integrity DB, BM25
index) is written to pytest's ``tmp_path``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
SRC_DIR = str(Path(__file__).parent.parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
TEST_PDF = FIXTURES_DIR / "with_images.pdf"
COMPLEX_PDF = FIXTURES_DIR / "complex_technical_doc.pdf"
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


# ---------------------------------------------------------------------------
# Fake backends (identical to the integration test fakes)
# ---------------------------------------------------------------------------

class FakeEmbedding:
    def embed(self, texts: List[str], trace: Any = None) -> List[List[float]]:
        return [[float(i % 8) / 8 for i in range(8)] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list = []

    def upsert(self, records, trace: Any = None) -> None:
        self.upserted.extend(records)

    def query(self, vector, top_k, filters=None, trace=None):
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(tmp_root: Path):
    import libs.splitter.recursive_splitter  # noqa: F401 — registers 'recursive' splitter

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
        observability=ObservabilitySettings(log_level="WARNING"),
        ingestion=IngestionSettings(
            chunk_size=500,
            chunk_overlap=50,
            splitter="recursive",
            batch_size=16,
            chunk_refiner=ChunkRefinerSettings(use_llm=False),
            metadata_enricher=MetadataEnricherSettings(use_llm=False),
        ),
    )


def _make_pipeline_factory(fake_store: FakeVectorStore, tmp_root: Path):
    """Return a callable that constructs IngestionPipeline with fake backends."""

    def factory(settings):
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

        db_dir = tmp_root / "db"
        bm25_dir = tmp_root / "bm25"
        images_dir = tmp_root / "images"
        for d in (db_dir, bm25_dir, images_dir):
            d.mkdir(parents=True, exist_ok=True)

        integrity = SqliteIntegrityChecker(db_path=str(db_dir / "ingestion_history.db"))
        loader = PdfLoader(images_dir=str(images_dir))
        chunker = DocumentChunker(settings)
        dense_enc = DenseEncoder(settings, embedding=FakeEmbedding())
        sparse_enc = SparseEncoder()
        batch_proc = BatchProcessor(
            settings, batch_size=16, dense_encoder=dense_enc, sparse_encoder=sparse_enc
        )
        upserter = VectorUpserter(settings, vector_store=fake_store)
        bm25 = BM25Indexer(index_dir=str(bm25_dir))

        return IngestionPipeline(
            settings,
            integrity_checker=integrity,
            loader=loader,
            chunker=chunker,
            transforms=[
                ChunkRefiner(settings),
                MetadataEnricher(settings),
                ImageCaptioner(settings),
            ],
            batch_processor=batch_proc,
            vector_upserter=upserter,
            bm25_indexer=bm25,
            image_storage=None,
        )

    return factory


def _load_ingest_module():
    """Load scripts/ingest.py as a fresh module object each time."""
    spec = importlib.util.spec_from_file_location(
        "ingest_script", str(SCRIPTS_DIR / "ingest.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEST_PDF.exists(), reason="with_images.pdf not found")
def test_ingest_single_file(tmp_root, fake_store):
    """Basic ingest of a single PDF produces BM25 artifacts."""
    settings = _make_settings(tmp_root)
    factory = _make_pipeline_factory(fake_store, tmp_root)
    mod = _load_ingest_module()

    with (
        patch.object(mod, "load_settings", return_value=settings),
        patch.object(mod, "IngestionPipeline", side_effect=factory),
    ):
        rc = mod.main(["--path", str(TEST_PDF), "--collection", "test_e2e"])

    assert rc == 0
    bm25_index = tmp_root / "bm25" / "index.json"
    assert bm25_index.exists(), "BM25 index.json must be written after ingestion"


@pytest.mark.skipif(not TEST_PDF.exists(), reason="with_images.pdf not found")
def test_ingest_skips_on_rerun(tmp_root, fake_store):
    """Second run without --force is skipped; no duplicate upserts."""
    settings = _make_settings(tmp_root)
    factory = _make_pipeline_factory(fake_store, tmp_root)
    mod = _load_ingest_module()

    with (
        patch.object(mod, "load_settings", return_value=settings),
        patch.object(mod, "IngestionPipeline", side_effect=factory),
    ):
        rc1 = mod.main(["--path", str(TEST_PDF), "--collection", "test_e2e"])
        count_after_first = len(fake_store.upserted)
        rc2 = mod.main(["--path", str(TEST_PDF), "--collection", "test_e2e"])

    assert rc1 == 0
    assert rc2 == 0
    assert len(fake_store.upserted) == count_after_first, "No new upserts on second run"


@pytest.mark.skipif(not TEST_PDF.exists(), reason="with_images.pdf not found")
def test_ingest_force_reingests(tmp_root, fake_store):
    """--force causes re-ingestion even if already processed."""
    settings = _make_settings(tmp_root)
    factory = _make_pipeline_factory(fake_store, tmp_root)
    mod = _load_ingest_module()

    with (
        patch.object(mod, "load_settings", return_value=settings),
        patch.object(mod, "IngestionPipeline", side_effect=factory),
    ):
        rc1 = mod.main(["--path", str(TEST_PDF), "--collection", "test_e2e"])
        count_after_first = len(fake_store.upserted)
        rc2 = mod.main(["--path", str(TEST_PDF), "--collection", "test_e2e", "--force"])

    assert rc1 == 0
    assert rc2 == 0
    assert len(fake_store.upserted) == count_after_first * 2, "--force must re-upsert all records"


def test_ingest_missing_path_returns_error(tmp_root, fake_store):
    """Non-existent --path returns exit code 1."""
    settings = _make_settings(tmp_root)
    factory = _make_pipeline_factory(fake_store, tmp_root)
    mod = _load_ingest_module()

    with (
        patch.object(mod, "load_settings", return_value=settings),
        patch.object(mod, "IngestionPipeline", side_effect=factory),
    ):
        rc = mod.main(["--path", "/non/existent/file.pdf", "--collection", "test_e2e"])

    assert rc == 1


@pytest.mark.skipif(not TEST_PDF.exists(), reason="with_images.pdf not found")
def test_ingest_directory(tmp_root, fake_store):
    """Passing a directory ingests all PDFs inside it."""
    settings = _make_settings(tmp_root)
    factory = _make_pipeline_factory(fake_store, tmp_root)
    mod = _load_ingest_module()

    with (
        patch.object(mod, "load_settings", return_value=settings),
        patch.object(mod, "IngestionPipeline", side_effect=factory),
    ):
        rc = mod.main(["--path", str(FIXTURES_DIR), "--collection", "test_dir"])

    assert rc == 0
    assert len(fake_store.upserted) > 0, "Directory ingest must produce upserts"
