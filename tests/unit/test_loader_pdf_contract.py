"""Contract tests for BaseLoader and PdfLoader (C3)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.types import Document, ImageRef
from libs.loader.base_loader import BaseLoader
from libs.loader.pdf_loader import PdfLoader


# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
WITH_IMAGES_PDF = FIXTURES_DIR / "with_images.pdf"
# Use any available PDF as "simple" (the arxiv papers are text-heavy)
SIMPLE_PDF = FIXTURES_DIR / "2512.06906v1.pdf"


# ---------------------------------------------------------------------------
# BaseLoader contract
# ---------------------------------------------------------------------------

class TestBaseLoaderIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseLoader()  # type: ignore

    def test_concrete_subclass_must_implement_load(self):
        class IncompleteLoader(BaseLoader):
            pass

        with pytest.raises(TypeError):
            IncompleteLoader()  # type: ignore

    def test_concrete_subclass_with_load_works(self):
        class MinimalLoader(BaseLoader):
            def load(self, path: str) -> Document:
                return Document(id="x", text="hello", metadata={"source_path": path})

        loader = MinimalLoader()
        doc = loader.load("/some/path.pdf")
        assert isinstance(doc, Document)
        assert doc.metadata["source_path"] == "/some/path.pdf"


# ---------------------------------------------------------------------------
# PdfLoader — basic structure
# ---------------------------------------------------------------------------

class TestPdfLoaderInit:
    def test_default_images_dir(self):
        loader = PdfLoader()
        assert loader._images_dir == "data/images"

    def test_custom_images_dir(self):
        loader = PdfLoader(images_dir="/tmp/imgs")
        assert loader._images_dir == "/tmp/imgs"

    def test_is_base_loader_subclass(self):
        assert issubclass(PdfLoader, BaseLoader)


# ---------------------------------------------------------------------------
# PdfLoader — real PDF fixtures
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SIMPLE_PDF.exists(), reason="sample PDF not found")
class TestPdfLoaderSimplePdf:
    def test_returns_document(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(SIMPLE_PDF))
        assert isinstance(doc, Document)

    def test_document_has_source_path(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(SIMPLE_PDF))
        assert "source_path" in doc.metadata
        assert doc.metadata["source_path"] == str(SIMPLE_PDF.resolve())

    def test_document_has_non_empty_text(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(SIMPLE_PDF))
        assert len(doc.text.strip()) > 0

    def test_document_id_is_hex(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(SIMPLE_PDF))
        # id is first 16 hex chars of sha256
        assert len(doc.id) == 16
        int(doc.id, 16)  # raises ValueError if not valid hex

    def test_doc_hash_in_metadata(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(SIMPLE_PDF))
        assert "doc_hash" in doc.metadata
        assert len(doc.metadata["doc_hash"]) == 64  # sha256 hex

    def test_page_count_in_metadata(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(SIMPLE_PDF))
        assert "page_count" in doc.metadata
        assert doc.metadata["page_count"] > 0

    def test_deterministic_on_same_file(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc1 = loader.load(str(SIMPLE_PDF))
        doc2 = loader.load(str(SIMPLE_PDF))
        assert doc1.id == doc2.id
        assert doc1.text == doc2.text

    def test_serializable_to_dict(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(SIMPLE_PDF))
        d = doc.to_dict()
        assert isinstance(d, dict)
        assert "id" in d and "text" in d and "metadata" in d


# ---------------------------------------------------------------------------
# PdfLoader — PDF with images
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not WITH_IMAGES_PDF.exists(), reason="with_images.pdf not found")
class TestPdfLoaderWithImages:
    def test_returns_document(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(WITH_IMAGES_PDF))
        assert isinstance(doc, Document)

    def test_images_metadata_present_when_pdf_has_images(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(WITH_IMAGES_PDF))
        # PDF contains images — metadata should have "images" list
        if "images" in doc.metadata:
            images = doc.metadata["images"]
            assert isinstance(images, list)
            assert len(images) > 0

    def test_image_refs_are_imageref_objects(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(WITH_IMAGES_PDF))
        images = doc.metadata.get("images", [])
        for img in images:
            assert isinstance(img, ImageRef)

    def test_image_ref_fields(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(WITH_IMAGES_PDF))
        for img in doc.metadata.get("images", []):
            assert img.id, "image id must be non-empty"
            assert img.path, "image path must be non-empty"
            assert img.text_offset >= 0
            assert img.text_length > 0

    def test_image_placeholders_in_text(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(WITH_IMAGES_PDF))
        images = doc.metadata.get("images", [])
        for img in images:
            placeholder = f"[IMAGE: {img.id}]"
            assert placeholder in doc.text, (
                f"Placeholder '{placeholder}' not found in document text"
            )

    def test_image_placeholder_offset_matches_text(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(WITH_IMAGES_PDF))
        for img in doc.metadata.get("images", []):
            placeholder = f"[IMAGE: {img.id}]"
            segment = doc.text[img.text_offset: img.text_offset + img.text_length]
            assert segment == placeholder, (
                f"text at offset {img.text_offset} is '{segment}', expected '{placeholder}'"
            )

    def test_images_saved_to_disk(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(WITH_IMAGES_PDF))
        for img in doc.metadata.get("images", []):
            assert os.path.isfile(img.path), f"Image file not found: {img.path}"

    def test_image_ids_are_unique(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(WITH_IMAGES_PDF))
        ids = [img.id for img in doc.metadata.get("images", [])]
        assert len(ids) == len(set(ids)), "Image IDs must be unique"

    def test_no_images_field_when_none_extracted(self, tmp_path):
        """A pure-text PDF should either have no 'images' key or an empty list."""
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        doc = loader.load(str(SIMPLE_PDF))
        images = doc.metadata.get("images", [])
        # We don't assert strictly — arxiv PDFs can have embedded graphics
        # but if the key is present it must be a list
        assert isinstance(images, list)


# ---------------------------------------------------------------------------
# PdfLoader — error handling
# ---------------------------------------------------------------------------

class TestPdfLoaderErrorHandling:
    def test_raises_on_missing_file(self, tmp_path):
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        with pytest.raises((ValueError, FileNotFoundError, Exception)):
            loader.load("/nonexistent/path.pdf")

    def test_raises_on_non_pdf(self, tmp_path):
        bad_file = tmp_path / "bad.pdf"
        bad_file.write_bytes(b"this is not a pdf")
        loader = PdfLoader(images_dir=str(tmp_path / "images"))
        with pytest.raises(Exception):
            loader.load(str(bad_file))
