"""Tests for DocumentChunker (C4): Document → List[Chunk] adapter."""
import hashlib
import sys
import os
from dataclasses import dataclass, field

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.types import Chunk, Document, ImageRef, make_image_placeholder
from ingestion.chunking.document_chunker import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import register_splitter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSplitter(BaseSplitter):
    """Splits on '||' delimiter for deterministic test control."""

    def __init__(self, settings):
        pass

    def split_text(self, text: str, trace=None) -> list[str]:
        if not text:
            return []
        return text.split("||")


class SingleChunkSplitter(BaseSplitter):
    def __init__(self, settings):
        pass

    def split_text(self, text: str, trace=None) -> list[str]:
        return [text] if text else []


@dataclass
class FakeIngestionSettings:
    splitter: str = "fake_c4"
    chunk_size: int = 1000
    chunk_overlap: int = 0


@dataclass
class FakeSettings:
    ingestion: FakeIngestionSettings = field(default_factory=FakeIngestionSettings)


@pytest.fixture(autouse=True)
def register_fakes():
    register_splitter("fake_c4", FakeSplitter)
    register_splitter("single_c4", SingleChunkSplitter)


def make_settings(splitter: str = "fake_c4") -> FakeSettings:
    return FakeSettings(ingestion=FakeIngestionSettings(splitter=splitter))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expected_chunk_id(doc_id: str, index: int, text: str) -> str:
    hash_8 = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"{doc_id}_{index:04d}_{hash_8}"


# ---------------------------------------------------------------------------
# Basic splitting
# ---------------------------------------------------------------------------

def test_empty_document_returns_empty_list():
    chunker = DocumentChunker(make_settings())
    doc = Document(id="doc1", text="")
    assert chunker.split_document(doc) == []


def test_single_chunk_document():
    settings = FakeSettings(ingestion=FakeIngestionSettings(splitter="single_c4"))
    chunker = DocumentChunker(settings)
    doc = Document(id="doc1", text="hello world", metadata={"source_path": "/a.pdf"})
    chunks = chunker.split_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_multiple_chunks_produced():
    chunker = DocumentChunker(make_settings())
    doc = Document(id="doc1", text="part one||part two||part three")
    chunks = chunker.split_document(doc)
    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["part one", "part two", "part three"]


# ---------------------------------------------------------------------------
# Chunk ID uniqueness and determinism
# ---------------------------------------------------------------------------

def test_chunk_ids_are_unique_within_document():
    chunker = DocumentChunker(make_settings())
    doc = Document(id="doc1", text="aaa||bbb||ccc")
    chunks = chunker.split_document(doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_id_deterministic():
    chunker = DocumentChunker(make_settings())
    doc = Document(id="doc1", text="alpha||beta")
    chunks1 = chunker.split_document(doc)
    chunks2 = chunker.split_document(doc)
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


def test_chunk_id_format():
    chunker = DocumentChunker(make_settings())
    doc = Document(id="mydoc", text="only one chunk")
    settings = FakeSettings(ingestion=FakeIngestionSettings(splitter="single_c4"))
    chunker2 = DocumentChunker(settings)
    chunks = chunker2.split_document(doc)
    assert chunks[0].id == expected_chunk_id("mydoc", 0, "only one chunk")


def test_chunk_id_index_zero_padded():
    chunker = DocumentChunker(make_settings())
    doc = Document(id="d", text="||".join([f"chunk{i}" for i in range(10)]))
    chunks = chunker.split_document(doc)
    assert chunks[0].id.startswith("d_0000_")
    assert chunks[9].id.startswith("d_0009_")


# ---------------------------------------------------------------------------
# source_ref
# ---------------------------------------------------------------------------

def test_source_ref_points_to_parent_document():
    chunker = DocumentChunker(make_settings())
    doc = Document(id="parent-doc-id", text="a||b")
    chunks = chunker.split_document(doc)
    assert all(c.source_ref == "parent-doc-id" for c in chunks)


# ---------------------------------------------------------------------------
# Metadata inheritance
# ---------------------------------------------------------------------------

def test_metadata_inherits_source_path():
    settings = FakeSettings(ingestion=FakeIngestionSettings(splitter="single_c4"))
    chunker = DocumentChunker(settings)
    doc = Document(id="d1", text="text", metadata={"source_path": "/docs/file.pdf"})
    chunks = chunker.split_document(doc)
    assert chunks[0].metadata["source_path"] == "/docs/file.pdf"


def test_metadata_inherits_all_document_fields():
    settings = FakeSettings(ingestion=FakeIngestionSettings(splitter="single_c4"))
    chunker = DocumentChunker(settings)
    doc = Document(
        id="d1",
        text="text",
        metadata={"source_path": "/x.pdf", "doc_type": "pdf", "title": "My Doc"},
    )
    chunks = chunker.split_document(doc)
    meta = chunks[0].metadata
    assert meta["source_path"] == "/x.pdf"
    assert meta["doc_type"] == "pdf"
    assert meta["title"] == "My Doc"


def test_metadata_contains_chunk_index():
    chunker = DocumentChunker(make_settings())
    doc = Document(id="d1", text="a||b||c")
    chunks = chunker.split_document(doc)
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]


def test_chunk_index_starts_at_zero():
    settings = FakeSettings(ingestion=FakeIngestionSettings(splitter="single_c4"))
    chunker = DocumentChunker(settings)
    doc = Document(id="d1", text="only")
    chunks = chunker.split_document(doc)
    assert chunks[0].metadata["chunk_index"] == 0


# ---------------------------------------------------------------------------
# Image reference distribution
# ---------------------------------------------------------------------------

def _make_doc_with_images() -> Document:
    img1 = ImageRef(id="img_001", path="data/images/img_001.png", page=1,
                    text_offset=10, text_length=15)
    img2 = ImageRef(id="img_002", path="data/images/img_002.png", page=2,
                    text_offset=50, text_length=15)
    ph1 = make_image_placeholder("img_001")
    ph2 = make_image_placeholder("img_002")
    text = f"text before {ph1} middle||text with {ph2} end||plain text"
    return Document(id="doc1", text=text, metadata={"images": [img1, img2]})


def test_chunk_with_image_placeholder_gets_images_metadata():
    chunker = DocumentChunker(make_settings())
    doc = _make_doc_with_images()
    chunks = chunker.split_document(doc)
    # chunk 0 references img_001
    assert "images" in chunks[0].metadata
    assert chunks[0].metadata["image_refs"] == ["img_001"]


def test_chunk_without_placeholder_has_no_images_key():
    chunker = DocumentChunker(make_settings())
    doc = _make_doc_with_images()
    chunks = chunker.split_document(doc)
    # chunk 2 is "plain text" — no placeholder
    assert "images" not in chunks[2].metadata
    assert "image_refs" not in chunks[2].metadata


def test_image_distribution_is_subset_of_document_images():
    chunker = DocumentChunker(make_settings())
    doc = _make_doc_with_images()
    chunks = chunker.split_document(doc)
    assert len(chunks[0].metadata["images"]) == 1
    assert chunks[0].metadata["images"][0].id == "img_001"
    assert len(chunks[1].metadata["images"]) == 1
    assert chunks[1].metadata["images"][0].id == "img_002"


def test_chunk_image_refs_list_matches_placeholders():
    chunker = DocumentChunker(make_settings())
    doc = _make_doc_with_images()
    chunks = chunker.split_document(doc)
    assert chunks[1].metadata["image_refs"] == ["img_002"]


def test_document_without_images_no_images_key_in_chunks():
    settings = FakeSettings(ingestion=FakeIngestionSettings(splitter="single_c4"))
    chunker = DocumentChunker(settings)
    doc = Document(id="d1", text="no images here", metadata={"source_path": "/x.pdf"})
    chunks = chunker.split_document(doc)
    assert "images" not in chunks[0].metadata
    assert "image_refs" not in chunks[0].metadata


def test_image_refs_with_dict_format_in_metadata():
    """Document.metadata["images"] may be raw dicts (from JSON deserialization)."""
    chunker = DocumentChunker(make_settings())
    ph = make_image_placeholder("img_dict")
    doc = Document(
        id="d1",
        text=f"chunk with {ph}",
        metadata={
            "images": [{"id": "img_dict", "path": "/p.png", "text_offset": 0, "text_length": 15}]
        },
    )
    settings = FakeSettings(ingestion=FakeIngestionSettings(splitter="single_c4"))
    chunker2 = DocumentChunker(settings)
    chunks = chunker2.split_document(doc)
    assert chunks[0].metadata["image_refs"] == ["img_dict"]


# ---------------------------------------------------------------------------
# Type contract
# ---------------------------------------------------------------------------

def test_output_is_list_of_chunk_objects():
    chunker = DocumentChunker(make_settings())
    doc = Document(id="d1", text="a||b")
    chunks = chunker.split_document(doc)
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunks_are_serializable():
    settings = FakeSettings(ingestion=FakeIngestionSettings(splitter="single_c4"))
    chunker = DocumentChunker(settings)
    doc = Document(id="d1", text="hello", metadata={"source_path": "/f.pdf"})
    chunks = chunker.split_document(doc)
    d = chunks[0].to_dict()
    assert d["id"] == chunks[0].id
    assert d["text"] == "hello"
    assert d["metadata"]["source_path"] == "/f.pdf"
