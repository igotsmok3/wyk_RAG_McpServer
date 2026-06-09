"""Unit tests for core/types.py — Document, Chunk, ChunkRecord, ImageRef."""
import json

import pytest

from src.core.types import (
    Chunk,
    ChunkRecord,
    Document,
    ImageRef,
    make_image_placeholder,
)


# ---------------------------------------------------------------------------
# ImageRef
# ---------------------------------------------------------------------------

class TestImageRef:
    def test_to_dict_roundtrip(self):
        ref = ImageRef(id="img1", path="data/images/c1/img1.png", page=2, text_offset=10, text_length=18, position={"x": 0, "y": 100})
        d = ref.to_dict()
        restored = ImageRef.from_dict(d)
        assert restored.id == ref.id
        assert restored.path == ref.path
        assert restored.page == ref.page
        assert restored.text_offset == ref.text_offset
        assert restored.text_length == ref.text_length
        assert restored.position == ref.position

    def test_optional_fields_default(self):
        ref = ImageRef(id="x", path="p")
        assert ref.page is None
        assert ref.text_offset == 0
        assert ref.text_length == 0
        assert ref.position == {}

    def test_from_dict_missing_optional(self):
        ref = ImageRef.from_dict({"id": "a", "path": "b"})
        assert ref.page is None
        assert ref.position == {}


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class TestDocument:
    def _make_doc(self) -> Document:
        images = [
            ImageRef(id="doc1_0_abc", path="data/images/col/doc1_0_abc.png", page=1, text_offset=5, text_length=20)
        ]
        return Document(
            id="doc1",
            text="Hello [IMAGE: doc1_0_abc] world",
            metadata={"source_path": "/tmp/test.pdf", "images": images},
        )

    def test_metadata_has_source_path(self):
        doc = self._make_doc()
        assert "source_path" in doc.metadata

    def test_to_dict_serializes_images(self):
        doc = self._make_doc()
        d = doc.to_dict()
        assert isinstance(d["metadata"]["images"], list)
        assert d["metadata"]["images"][0]["id"] == "doc1_0_abc"

    def test_from_dict_restores_image_refs(self):
        doc = self._make_doc()
        restored = Document.from_dict(doc.to_dict())
        assert isinstance(restored.metadata["images"][0], ImageRef)
        assert restored.metadata["images"][0].id == "doc1_0_abc"

    def test_to_json_is_valid_json(self):
        doc = self._make_doc()
        parsed = json.loads(doc.to_json())
        assert parsed["id"] == "doc1"

    def test_roundtrip_preserves_text(self):
        doc = self._make_doc()
        assert Document.from_dict(doc.to_dict()).text == doc.text

    def test_no_images_metadata(self):
        doc = Document(id="d2", text="plain", metadata={"source_path": "/a.pdf"})
        assert "images" not in doc.metadata
        d = doc.to_dict()
        assert d["metadata"]["source_path"] == "/a.pdf"


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def _make_chunk(self) -> Chunk:
        return Chunk(
            id="doc1_0000_ab123456",
            text="Hello world",
            metadata={"source_path": "/tmp/test.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=11,
            source_ref="doc1",
        )

    def test_source_ref_links_to_parent(self):
        chunk = self._make_chunk()
        assert chunk.source_ref == "doc1"

    def test_to_dict_all_fields(self):
        chunk = self._make_chunk()
        d = chunk.to_dict()
        assert d["id"] == chunk.id
        assert d["start_offset"] == 0
        assert d["end_offset"] == 11
        assert d["source_ref"] == "doc1"
        assert d["metadata"]["chunk_index"] == 0

    def test_from_dict_roundtrip(self):
        chunk = self._make_chunk()
        restored = Chunk.from_dict(chunk.to_dict())
        assert restored.id == chunk.id
        assert restored.start_offset == chunk.start_offset
        assert restored.source_ref == chunk.source_ref

    def test_to_json_valid(self):
        chunk = self._make_chunk()
        parsed = json.loads(chunk.to_json())
        assert parsed["source_ref"] == "doc1"

    def test_images_serialized_in_metadata(self):
        ref = ImageRef(id="img1", path="p", text_offset=5, text_length=18)
        chunk = Chunk(
            id="c1",
            text="text [IMAGE: img1] more",
            metadata={"images": [ref], "image_refs": ["img1"]},
        )
        d = chunk.to_dict()
        assert d["metadata"]["images"][0]["id"] == "img1"

    def test_chunk_without_images_no_images_field(self):
        chunk = Chunk(id="c2", text="plain text", metadata={"source_path": "/a.pdf"})
        d = chunk.to_dict()
        assert "images" not in d["metadata"]

    def test_optional_source_ref_defaults_none(self):
        chunk = Chunk(id="c3", text="x")
        assert chunk.source_ref is None

    def test_offsets_default_zero(self):
        chunk = Chunk(id="c4", text="y")
        assert chunk.start_offset == 0
        assert chunk.end_offset == 0


# ---------------------------------------------------------------------------
# ChunkRecord
# ---------------------------------------------------------------------------

class TestChunkRecord:
    def test_from_chunk(self):
        chunk = Chunk(id="c1", text="hello", metadata={"source_path": "/a.pdf"}, source_ref="doc1")
        record = ChunkRecord.from_chunk(chunk)
        assert record.id == "c1"
        assert record.text == "hello"
        assert record.metadata["source_path"] == "/a.pdf"
        assert record.dense_vector is None
        assert record.sparse_vector is None

    def test_with_vectors(self):
        record = ChunkRecord(
            id="r1",
            text="data",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"word": 0.5},
        )
        d = record.to_dict()
        assert d["dense_vector"] == [0.1, 0.2, 0.3]
        assert d["sparse_vector"] == {"word": 0.5}

    def test_to_dict_roundtrip(self):
        record = ChunkRecord(id="r2", text="t", metadata={"k": "v"}, dense_vector=[1.0])
        restored = ChunkRecord.from_dict(record.to_dict())
        assert restored.id == "r2"
        assert restored.dense_vector == [1.0]

    def test_to_json_valid(self):
        record = ChunkRecord(id="r3", text="t")
        parsed = json.loads(record.to_json())
        assert parsed["id"] == "r3"

    def test_vectors_none_by_default(self):
        record = ChunkRecord(id="r4", text="t")
        assert record.dense_vector is None
        assert record.sparse_vector is None


# ---------------------------------------------------------------------------
# Image placeholder helper
# ---------------------------------------------------------------------------

class TestMakeImagePlaceholder:
    def test_format(self):
        assert make_image_placeholder("abc123") == "[IMAGE: abc123]"

    def test_length_matches_spec(self):
        image_id = "doc1_1_abc12345"
        placeholder = make_image_placeholder(image_id)
        assert placeholder == f"[IMAGE: {image_id}]"
        assert len(placeholder) == len(f"[IMAGE: {image_id}]")
