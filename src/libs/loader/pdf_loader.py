"""PDF loader: MarkItDown for PDF→Markdown, PyMuPDF for image extraction.

Pipeline per document:
  1. MarkItDown  — converts PDF to canonical Markdown text
  2. PyMuPDF     — extracts embedded images, saves as PNG
  3. Combine     — appends [IMAGE: {image_id}] placeholders to Markdown text,
                   records precise text_offset for each placeholder
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pymupdf  # PyMuPDF
from markitdown import MarkItDown

from core.types import Document, ImageRef, make_image_placeholder
from libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

_markitdown = MarkItDown()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _convert_to_markdown(path: str) -> Tuple[str, Optional[str]]:
    """Use MarkItDown to convert PDF → Markdown text.

    Returns (markdown_text, title).
    """
    result = _markitdown.convert(path)
    text = result.markdown or result.text_content or ""
    title = getattr(result, "title", None)
    return text, title


def _extract_images(
    path: str,
    doc_hash: str,
    image_dir: str,
) -> List[ImageRef]:
    """Use PyMuPDF to extract embedded images from the PDF.

    Images are saved as PNG files under *image_dir*.
    Returns a list of ImageRef (text_offset=0, will be filled by caller).
    """
    refs: List[ImageRef] = []
    seq = 0

    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        logger.warning("PyMuPDF could not open '%s' for image extraction: %s", path, exc)
        return refs

    os.makedirs(image_dir, exist_ok=True)

    with pdf:
        for page_num, page in enumerate(pdf):
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                image_id = f"{doc_hash[:8]}_{page_num}_{seq}"
                image_path = os.path.join(image_dir, f"{image_id}.png")

                try:
                    pix = pymupdf.Pixmap(pdf, xref)
                    if pix.colorspace and pix.colorspace.n > 3:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    pix.save(image_path)
                except Exception as exc:
                    logger.warning(
                        "Failed to extract image xref=%d page=%d: %s", xref, page_num, exc
                    )
                    seq += 1
                    continue

                position: Dict[str, Any] = {}
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        r = rects[0]
                        position = {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}
                except Exception:
                    pass

                placeholder = make_image_placeholder(image_id)
                refs.append(
                    ImageRef(
                        id=image_id,
                        path=image_path,
                        page=page_num,
                        text_offset=0,       # filled below
                        text_length=len(placeholder),
                        position=position,
                    )
                )
                seq += 1

    return refs


class PdfLoader(BaseLoader):
    """Load a PDF into a Document with Markdown text and extracted images.

    Text extraction uses MarkItDown (PDF → canonical Markdown).
    Image extraction uses PyMuPDF (saves PNG files, records ImageRef metadata).
    Image placeholders are appended at the end of the Markdown text so that
    text_offset values are deterministic and accurate.

    Args:
        images_dir: Root directory for saved images.
                    Files are written to ``{images_dir}/{doc_hash}/``.
    """

    def __init__(self, images_dir: str = "data/images") -> None:
        self._images_dir = images_dir

    def load(self, path: str) -> Document:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        # Validate PDF magic bytes before handing off to MarkItDown
        with open(path, "rb") as fh:
            header = fh.read(5)
        if header != b"%PDF-":
            raise ValueError(f"Not a valid PDF file (bad magic bytes): {path}")

        doc_hash = _file_sha256(path)
        doc_id = doc_hash[:16]

        # 1. MarkItDown: PDF → Markdown
        try:
            markdown_text, title = _convert_to_markdown(path)
        except Exception as exc:
            raise ValueError(f"MarkItDown failed on '{path}': {exc}") from exc

        if not markdown_text.endswith("\n"):
            markdown_text += "\n"

        # 2. PyMuPDF: extract images
        image_dir = os.path.join(self._images_dir, doc_hash)
        image_refs = _extract_images(path, doc_hash, image_dir)

        # 3. Append image placeholders to Markdown text, record offsets
        base_offset = len(markdown_text)
        for ref in image_refs:
            placeholder = make_image_placeholder(ref.id)
            ref.text_offset = base_offset
            markdown_text += placeholder + "\n"
            base_offset += len(placeholder) + 1  # +1 for "\n"

        # 4. Build metadata
        page_count: Optional[int] = None
        try:
            with pymupdf.open(path) as pdf:
                page_count = pdf.page_count
        except Exception:
            pass

        metadata: Dict[str, Any] = {
            "source_path": os.path.abspath(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "page_count": page_count,
        }
        if title:
            metadata["title"] = title
        if image_refs:
            metadata["images"] = image_refs

        return Document(id=doc_id, text=markdown_text, metadata=metadata)
