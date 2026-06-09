"""ImageCaptioner: optional Vision LLM captioning with graceful degradation."""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import List, Optional

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ImageRef
from ingestion.transform.base_transform import BaseTransform

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = (
    Path(__file__).parent.parent.parent.parent / "config" / "prompts" / "image_captioning.txt"
)

_DEFAULT_PROMPT = (
    "You are an expert at describing images from technical documents. "
    "Provide a clear and concise description of this image. "
    "Focus on the main content, structure, and key information visible. "
    "Keep it under 150 words."
)


class ImageCaptioner(BaseTransform):
    """Generate captions for images referenced in Chunk metadata.

    When vision_llm is enabled and a chunk contains image_refs, the captioner
    calls the Vision LLM for each image path and writes captions back to
    ``chunk.metadata["image_captions"]`` (dict of image_id → caption string).

    Degradation contract:
    - vision_llm disabled or unavailable → mark ``has_unprocessed_images=True``,
      keep ``image_refs`` intact, no captions generated.
    - vision_llm call fails for an image → same mark, caption skipped for that
      image; other images in the same chunk continue processing.
    - chunk has no image_refs → pass through unchanged.
    """

    def __init__(
        self,
        settings: Settings,
        vision_llm=None,
        prompt_path: Optional[str] = None,
    ) -> None:
        vision_cfg = getattr(settings, "vision_llm", None)
        self._enabled: bool = bool(vision_cfg and vision_cfg.enabled)

        self._vision_llm = vision_llm
        if self._enabled and self._vision_llm is None:
            self._vision_llm = self._try_create_vision_llm(settings)

        self._prompt = self._load_prompt(prompt_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self, chunks: List[Chunk], trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        results: List[Chunk] = []
        captioned = 0
        degraded = 0

        for chunk in chunks:
            try:
                processed = self._process_chunk(chunk, trace)
                results.append(processed)
                if processed.metadata.get("image_captions"):
                    captioned += 1
                elif chunk.metadata.get("image_refs") and processed.metadata.get(
                    "has_unprocessed_images"
                ):
                    degraded += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ImageCaptioner: unexpected error on chunk %s: %s", chunk.id, exc
                )
                fallback = copy.copy(chunk)
                fallback.metadata = dict(chunk.metadata)
                if fallback.metadata.get("image_refs"):
                    fallback.metadata["has_unprocessed_images"] = True
                    degraded += 1
                results.append(fallback)

        if trace is not None:
            trace.record_stage(
                "image_captioner",
                total=len(chunks),
                captioned=captioned,
                degraded=degraded,
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_chunk(self, chunk: Chunk, trace: Optional[TraceContext]) -> Chunk:
        image_refs: List[str] = chunk.metadata.get("image_refs", [])

        new_chunk = copy.copy(chunk)
        new_chunk.metadata = dict(chunk.metadata)

        if not image_refs:
            return new_chunk

        if not self._enabled or self._vision_llm is None:
            new_chunk.metadata["has_unprocessed_images"] = True
            return new_chunk

        image_map = self._build_image_map(chunk)
        captions: dict[str, str] = {}
        failed: List[str] = []

        for image_id in image_refs:
            path = image_map.get(image_id)
            if path is None:
                logger.warning(
                    "ImageCaptioner: no path registered for image_id=%s in chunk %s",
                    image_id,
                    chunk.id,
                )
                failed.append(image_id)
                continue

            if not os.path.exists(path):
                logger.warning(
                    "ImageCaptioner: image file not found: %s (chunk=%s)", path, chunk.id
                )
                failed.append(image_id)
                continue

            try:
                caption = self._caption_image(path, trace)
                captions[image_id] = caption
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ImageCaptioner: failed to caption image %s: %s", image_id, exc
                )
                failed.append(image_id)

        if captions:
            new_chunk.metadata["image_captions"] = captions
        if failed:
            new_chunk.metadata["has_unprocessed_images"] = True

        return new_chunk

    def _caption_image(self, path: str, trace: Optional[TraceContext]) -> str:
        response = self._vision_llm.chat_with_image(
            text=self._prompt,
            image=path,
            trace=trace,
        )
        return response.content.strip()

    def _build_image_map(self, chunk: Chunk) -> dict[str, str]:
        """Return image_id → file_path from chunk metadata["images"]."""
        images = chunk.metadata.get("images", [])
        result: dict[str, str] = {}
        for img in images:
            if isinstance(img, ImageRef):
                result[img.id] = img.path
            elif isinstance(img, dict):
                img_id = img.get("id", "")
                img_path = img.get("path", "")
                if img_id:
                    result[img_id] = img_path
        return result

    def _load_prompt(self, prompt_path: Optional[str]) -> str:
        path = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT_PATH
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        logger.warning(
            "ImageCaptioner: prompt file not found at %s, using default", path
        )
        return _DEFAULT_PROMPT

    @staticmethod
    def _try_create_vision_llm(settings: Settings):
        try:
            import libs.llm  # noqa: F401 — triggers provider registration
            import libs.llm.qwen_vision_llm  # noqa: F401
            import libs.llm.azure_vision_llm  # noqa: F401
            from libs.llm.llm_factory import LLMFactory
            return LLMFactory.create_vision_llm(settings.vision_llm)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ImageCaptioner: failed to create Vision LLM (%s), all images will be "
                "marked as unprocessed",
                exc,
            )
            return None
