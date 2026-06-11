"""IngestionPipeline: serial execution of integrity→load→split→transform→encode→store."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.types import ImageRef
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader.file_integrity import SqliteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader

if TYPE_CHECKING:
    from core.settings import Settings
    from ingestion.storage.image_storage import ImageStorage
    from libs.loader.base_loader import BaseLoader
    from libs.loader.file_integrity import FileIntegrityChecker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Stats returned after a single pipeline run."""

    file_path: str
    file_hash: str
    skipped: bool = False
    doc_id: str = ""
    chunk_count: int = 0
    record_count: int = 0
    image_count: int = 0


class IngestionPipeline:
    """Serial ingestion pipeline: integrity → load → split → transform → encode → store.

    All sub-components are injectable so tests can swap out external backends
    (embedding, vector store, MySQL) without touching real infrastructure.

    Typical usage::

        pipeline = IngestionPipeline(settings)
        result = pipeline.run("path/to/doc.pdf", collection="my_kb")
    """

    def __init__(
        self,
        settings: "Settings",
        *,
        integrity_checker: "FileIntegrityChecker | None" = None,
        loader: "BaseLoader | None" = None,
        chunker: DocumentChunker | None = None,
        transforms: List[BaseTransform] | None = None,
        batch_processor: BatchProcessor | None = None,
        vector_upserter: VectorUpserter | None = None,
        bm25_indexer: BM25Indexer | None = None,
        image_storage: "ImageStorage | None" = None,
    ) -> None:
        self._settings = settings
        self._integrity_checker: "FileIntegrityChecker" = (
            integrity_checker or SqliteIntegrityChecker()
        )
        self._loader: "BaseLoader" = loader or PdfLoader()
        self._chunker = chunker or DocumentChunker(settings)
        self._transforms: List[BaseTransform] = (
            transforms
            if transforms is not None
            else [
                ChunkRefiner(settings),
                MetadataEnricher(settings),
                ImageCaptioner(settings),
            ]
        )
        self._batch_processor = batch_processor or BatchProcessor(
            settings, batch_size=settings.ingestion.batch_size
        )
        self._vector_upserter = vector_upserter or VectorUpserter(settings)
        self._bm25_indexer = bm25_indexer or BM25Indexer()
        self._image_storage: "ImageStorage | None" = image_storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        file_path: str,
        collection: str = "default",
        force: bool = False,
    ) -> PipelineResult:
        """Run the full ingestion pipeline for one file.

        Args:
            file_path:  Absolute or relative path to the source document.
            collection: Logical collection name used for image storage grouping.
            force:      Re-ingest even if the file hash was already marked success.

        Returns:
            PipelineResult with stage statistics.
            ``result.skipped=True`` when the file is already ingested and
            ``force=False``.

        Raises:
            RuntimeError: wraps any step failure; integrity state is set to
                ``failed`` before re-raising.
        """
        # ── Step 1: Integrity check ──────────────────────────────────────────
        file_hash = self._integrity_checker.compute_sha256(file_path)
        if not force and self._integrity_checker.should_skip(file_hash):
            logger.info(
                "[pipeline] skip %s (hash=%s…, already ingested)",
                file_path,
                file_hash[:8],
            )
            return PipelineResult(file_path=file_path, file_hash=file_hash, skipped=True)

        try:
            # ── Step 2: Load ─────────────────────────────────────────────────
            logger.info("[pipeline] load: %s", file_path)
            document = self._loader.load(file_path)
            logger.info(
                "[pipeline] load done: doc_id=%s text_len=%d",
                document.id,
                len(document.text),
            )

            # ── Step 3: Split ────────────────────────────────────────────────
            logger.info("[pipeline] split")
            chunks = self._chunker.split_document(document)
            logger.info("[pipeline] split done: %d chunks", len(chunks))

            # ── Step 4: Transform ────────────────────────────────────────────
            for transform in self._transforms:
                name = type(transform).__name__
                logger.info("[pipeline] transform: %s (%d chunks)", name, len(chunks))
                chunks = transform.transform(chunks)

            # ── Step 5: Encode ───────────────────────────────────────────────
            logger.info("[pipeline] encode: %d chunks", len(chunks))
            records = self._batch_processor.process(chunks)
            logger.info("[pipeline] encode done: %d records", len(records))

            # ── Step 6a: Vector store ─────────────────────────────────────────
            logger.info("[pipeline] store: upsert %d records to vector store", len(records))
            self._vector_upserter.upsert(records)

            # ── Step 6b: BM25 index ───────────────────────────────────────────
            logger.info("[pipeline] store: BM25 build + save")
            self._bm25_indexer.build(records)
            self._bm25_indexer.save()

            # ── Step 6c: Images (optional) ────────────────────────────────────
            image_count = 0
            if self._image_storage is not None:
                doc_images = document.metadata.get("images", [])
                for img in doc_images:
                    if isinstance(img, dict):
                        img = ImageRef.from_dict(img)
                    if os.path.exists(img.path):
                        self._image_storage.save(
                            image_id=img.id,
                            src_path=img.path,
                            collection=collection,
                            doc_hash=file_hash,
                            page_num=img.page,
                        )
                        image_count += 1

            # ── Mark success ──────────────────────────────────────────────────
            self._integrity_checker.mark_success(file_hash, file_path)

            logger.info(
                "[pipeline] done: %s — chunks=%d records=%d images=%d",
                file_path,
                len(chunks),
                len(records),
                image_count,
            )

            return PipelineResult(
                file_path=file_path,
                file_hash=file_hash,
                skipped=False,
                doc_id=document.id,
                chunk_count=len(chunks),
                record_count=len(records),
                image_count=image_count,
            )

        except Exception as exc:
            self._integrity_checker.mark_failed(file_hash, str(exc))
            raise RuntimeError(
                f"[pipeline] ingestion failed for '{file_path}': {exc}"
            ) from exc
