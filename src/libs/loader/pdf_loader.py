"""PDF loader using PyMuPDF.

Extracts text page-by-page, inserts [IMAGE: {image_id}] placeholders at each
image position, saves extracted images to data/images/{doc_hash}/, and
records ImageRef metadata in Document.metadata["images"].
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List

import pymupdf  # PyMuPDF

from core.types import Document, ImageRef, make_image_placeholder
from libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class PdfLoader(BaseLoader):
    """Load a PDF file into a Document with text and embedded images.

    Args:
        images_dir: Root directory for saving extracted images.
                    Images are saved under ``{images_dir}/{doc_hash}/``.
    """

    def __init__(self, images_dir: str = "data/images") -> None:
        self._images_dir = images_dir

    def load(self, path: str) -> Document:
        doc_hash = _file_sha256(path)
        doc_id = doc_hash[:16]

        text_parts: List[str] = []
        image_refs: List[ImageRef] = []

        image_dir = os.path.join(self._images_dir, doc_hash)
        os.makedirs(image_dir, exist_ok=True)

        global_image_seq = 0
        current_offset = 0  # character offset in accumulated text

        try:
            pdf = pymupdf.open(path)
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{path}': {exc}") from exc

        page_count = pdf.page_count
        with pdf:
            for page_num, page in enumerate(pdf):
                page_text = page.get_text("text")

                # Collect image xrefs on this page
                image_list = page.get_images(full=True)
                page_image_refs: List[ImageRef] = []

                for img_info in image_list:
                    xref = img_info[0]
                    image_id = f"{doc_hash[:8]}_{page_num}_{global_image_seq}"
                    image_filename = f"{image_id}.png"
                    image_path = os.path.join(image_dir, image_filename)

                    try:
                        base_image = pdf.extract_image(xref)
                        img_bytes = base_image["image"]
                        ext = base_image.get("ext", "png")

                        # Save as PNG regardless of source format via pixmap
                        pix = pymupdf.Pixmap(pdf, xref)
                        if pix.colorspace and pix.colorspace.n > 3:
                            # Convert CMYK → RGB before saving
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        pix.save(image_path)

                    except Exception as exc:
                        logger.warning(
                            "Failed to extract image xref=%d page=%d: %s",
                            xref, page_num, exc,
                        )
                        global_image_seq += 1
                        continue

                    # Build rects for position info (best-effort)
                    rect_info: Dict[str, Any] = {}
                    try:
                        img_rects = page.get_image_rects(xref)
                        if img_rects:
                            r = img_rects[0]
                            rect_info = {
                                "x0": r.x0, "y0": r.y0,
                                "x1": r.x1, "y1": r.y1,
                            }
                    except Exception:
                        pass

                    placeholder = make_image_placeholder(image_id)
                    page_image_refs.append(
                        ImageRef(
                            id=image_id,
                            path=image_path,
                            page=page_num,
                            text_offset=0,  # will update after appending
                            text_length=len(placeholder),
                            position=rect_info,
                        )
                    )
                    global_image_seq += 1

                # Build this page's text segment:
                # If the page has images, append placeholders at the end of the
                # page text (we cannot know exact intra-page offsets from text
                # extraction alone, so we append after the page block).
                page_block = page_text
                if not page_block.endswith("\n"):
                    page_block += "\n"

                for ref in page_image_refs:
                    placeholder = make_image_placeholder(ref.id)
                    ref.text_offset = current_offset + len(page_block)
                    page_block += placeholder + "\n"
                    image_refs.append(ref)

                text_parts.append(page_block)
                current_offset += len(page_block)

        full_text = "".join(text_parts)

        metadata: Dict[str, Any] = {
            "source_path": os.path.abspath(path),
            "doc_hash": doc_hash,
            "page_count": page_count,
        }
        if image_refs:
            metadata["images"] = image_refs

        return Document(id=doc_id, text=full_text, metadata=metadata)
