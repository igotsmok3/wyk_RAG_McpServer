"""Core data types shared across ingestion, retrieval, and MCP tools."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImageRef:
    """Reference to an image extracted from a document."""
    id: str
    path: str
    page: Optional[int] = None
    text_offset: int = 0
    text_length: int = 0
    position: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "page": self.page,
            "text_offset": self.text_offset,
            "text_length": self.text_length,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageRef":
        return cls(
            id=d["id"],
            path=d["path"],
            page=d.get("page"),
            text_offset=d.get("text_offset", 0),
            text_length=d.get("text_length", 0),
            position=d.get("position", {}),
        )


@dataclass
class Document:
    """Loaded document before chunking.

    text may contain [IMAGE: {image_id}] placeholders where images appear.
    metadata["images"] holds the corresponding ImageRef list.
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        meta = dict(self.metadata)
        if "images" in meta and isinstance(meta["images"], list):
            meta["images"] = [
                img.to_dict() if isinstance(img, ImageRef) else img
                for img in meta["images"]
            ]
        return {"id": self.id, "text": self.text, "metadata": meta}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        meta = dict(d.get("metadata", {}))
        if "images" in meta and isinstance(meta["images"], list):
            meta["images"] = [
                ImageRef.from_dict(img) if isinstance(img, dict) else img
                for img in meta["images"]
            ]
        return cls(id=d["id"], text=d["text"], metadata=meta)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Chunk:
    """Text chunk produced by splitting a Document.

    source_ref points to the parent Document.id.
    metadata inherits Document metadata and adds chunk-specific fields.
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        meta = dict(self.metadata)
        if "images" in meta and isinstance(meta["images"], list):
            meta["images"] = [
                img.to_dict() if isinstance(img, ImageRef) else img
                for img in meta["images"]
            ]
        return {
            "id": self.id,
            "text": self.text,
            "metadata": meta,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        meta = dict(d.get("metadata", {}))
        if "images" in meta and isinstance(meta["images"], list):
            meta["images"] = [
                ImageRef.from_dict(img) if isinstance(img, dict) else img
                for img in meta["images"]
            ]
        return cls(
            id=d["id"],
            text=d["text"],
            metadata=meta,
            start_offset=d.get("start_offset", 0),
            end_offset=d.get("end_offset", 0),
            source_ref=d.get("source_ref"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class ChunkRecord:
    """Storage/retrieval carrier for a Chunk with optional vectors.

    dense_vector and sparse_vector are populated by the encoding stage.
    sparse_vector maps term to weight.
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "dense_vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkRecord":
        return cls(
            id=d["id"],
            text=d["text"],
            metadata=d.get("metadata", {}),
            dense_vector=d.get("dense_vector"),
            sparse_vector=d.get("sparse_vector"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkRecord":
        return cls(id=chunk.id, text=chunk.text, metadata=dict(chunk.metadata))


IMAGE_PLACEHOLDER_PREFIX = "[IMAGE: "
IMAGE_PLACEHOLDER_SUFFIX = "]"


def make_image_placeholder(image_id: str) -> str:
    return f"{IMAGE_PLACEHOLDER_PREFIX}{image_id}{IMAGE_PLACEHOLDER_SUFFIX}"
