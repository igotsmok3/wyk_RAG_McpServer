"""DocumentChunker: adapts libs.splitter to produce business Chunk objects from a Document."""
from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, List

from core.types import Chunk, Document, ImageRef
from libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from core.settings import Settings

_IMAGE_PATTERN = re.compile(r"\[IMAGE: ([^\]]+)\]")


class DocumentChunker:
    """Business adapter: Document → List[Chunk].

    Uses libs.splitter for raw text splitting, then adds:
    - deterministic chunk IDs
    - metadata inheritance from the parent Document
    - chunk_index tracking
    - source_ref linking back to Document.id
    - per-chunk image reference distribution
    """

    def __init__(self, settings: "Settings") -> None:
        self._splitter = SplitterFactory.create(settings.ingestion)

    def split_document(self, document: Document) -> List[Chunk]:
        texts = self._splitter.split_text(document.text)
        chunks: List[Chunk] = []
        for index, text in enumerate(texts):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            metadata = self._inherit_metadata(document, index, text)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata=metadata,
                    source_ref=document.id,
                )
            )
        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        hash_8 = hashlib.sha256(text.encode()).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{hash_8}"

    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str) -> dict:
        meta = dict(document.metadata)
        meta["chunk_index"] = chunk_index

        doc_images: list[ImageRef] = [
            img if isinstance(img, ImageRef) else ImageRef.from_dict(img)
            for img in document.metadata.get("images", [])
        ]

        if doc_images:
            image_map = {img.id: img for img in doc_images}
            referenced_ids = _IMAGE_PATTERN.findall(chunk_text)
            if referenced_ids:
                chunk_images = [image_map[iid] for iid in referenced_ids if iid in image_map]
                meta["images"] = chunk_images
                meta["image_refs"] = [img.id for img in chunk_images]
            else:
                meta.pop("images", None)
                meta.pop("image_refs", None)
        else:
            meta.pop("images", None)

        return meta
