"""Unit tests for ImageCaptioner – all using mocks, no real Vision LLM calls."""
from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from core.settings import (
    IngestionSettings,
    Settings,
    VisionLLMSettings,
    LLMSettings,
    EmbeddingSettings,
    VectorStoreSettings,
)
from core.types import Chunk, ImageRef
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.base_llm import ChatResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(vision_enabled: bool = True, provider: str = "qwen") -> Settings:
    return Settings(
        llm=LLMSettings(provider="qwen", model="qwen-plus", api_key="fake"),
        embedding=EmbeddingSettings(provider="openai"),
        vector_store=VectorStoreSettings(provider="milvus"),
        vision_llm=VisionLLMSettings(
            enabled=vision_enabled,
            provider=provider,
            model="qwen-vl-plus",
            api_key="fake-key",
        ),
    )


def _fake_vision_llm(caption: str = "A test image caption."):
    mock = MagicMock()
    mock.chat_with_image.return_value = ChatResponse(
        content=caption, model="qwen-vl-plus", usage=None
    )
    return mock


def _chunk_with_images(image_paths: list[str]) -> Chunk:
    """Build a Chunk that references images stored at the given paths."""
    images = [
        ImageRef(
            id=f"img_{i:03d}",
            path=p,
            page=1,
            text_offset=i * 20,
            text_length=15,
        )
        for i, p in enumerate(image_paths)
    ]
    image_ids = [img.id for img in images]
    text = " ".join(f"[IMAGE: {img_id}]" for img_id in image_ids) + " Some surrounding text."
    return Chunk(
        id="chunk_001",
        text=text,
        metadata={
            "source_path": "test.pdf",
            "images": images,
            "image_refs": image_ids,
        },
    )


def _create_test_image(path: str, size: tuple[int, int] = (64, 64)) -> None:
    """Write a minimal PNG to *path*."""
    img = PILImage.new("RGB", size, color=(100, 150, 200))
    img.save(path, format="PNG")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_image(tmp_path: Path) -> str:
    """Return the path to a real tiny PNG image."""
    p = str(tmp_path / "test_image.png")
    _create_test_image(p)
    return p


# ---------------------------------------------------------------------------
# Tests: enabled mode (mock Vision LLM)
# ---------------------------------------------------------------------------

class TestEnabledMode:
    def test_caption_written_to_metadata(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=True)
        mock_vlm = _fake_vision_llm("A bar chart showing quarterly revenue.")

        captioner = ImageCaptioner(settings, vision_llm=mock_vlm)
        result = captioner.transform([chunk])

        assert len(result) == 1
        c = result[0]
        assert "image_captions" in c.metadata
        assert c.metadata["image_captions"]["img_000"] == "A bar chart showing quarterly revenue."

    def test_vision_llm_called_once_per_image(self, tmp_image: str, tmp_path: Path) -> None:
        img2 = str(tmp_path / "img2.png")
        _create_test_image(img2)
        chunk = _chunk_with_images([tmp_image, img2])
        settings = _make_settings(vision_enabled=True)
        mock_vlm = _fake_vision_llm("description")

        captioner = ImageCaptioner(settings, vision_llm=mock_vlm)
        captioner.transform([chunk])

        assert mock_vlm.chat_with_image.call_count == 2

    def test_vision_llm_called_with_image_path(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=True)
        mock_vlm = _fake_vision_llm()

        captioner = ImageCaptioner(settings, vision_llm=mock_vlm)
        captioner.transform([chunk])

        call_kwargs = mock_vlm.chat_with_image.call_args
        assert call_kwargs.kwargs.get("image") == tmp_image or call_kwargs.args[1] == tmp_image

    def test_chunk_without_image_refs_passes_through(self) -> None:
        chunk = Chunk(id="c1", text="plain text", metadata={"source_path": "doc.pdf"})
        settings = _make_settings(vision_enabled=True)
        mock_vlm = _fake_vision_llm()

        captioner = ImageCaptioner(settings, vision_llm=mock_vlm)
        result = captioner.transform([chunk])

        assert result[0].metadata == chunk.metadata
        mock_vlm.chat_with_image.assert_not_called()

    def test_has_unprocessed_images_not_set_on_success(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=True)
        captioner = ImageCaptioner(settings, vision_llm=_fake_vision_llm())
        result = captioner.transform([chunk])

        assert result[0].metadata.get("has_unprocessed_images") is None

    def test_original_metadata_not_mutated(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        original_meta = copy.deepcopy(chunk.metadata)
        settings = _make_settings(vision_enabled=True)

        captioner = ImageCaptioner(settings, vision_llm=_fake_vision_llm())
        captioner.transform([chunk])

        assert chunk.metadata.keys() == original_meta.keys()
        assert "image_captions" not in chunk.metadata

    def test_image_refs_retained_in_enabled_mode(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=True)
        captioner = ImageCaptioner(settings, vision_llm=_fake_vision_llm())
        result = captioner.transform([chunk])

        assert result[0].metadata["image_refs"] == chunk.metadata["image_refs"]


# ---------------------------------------------------------------------------
# Tests: degradation mode – vision disabled via config
# ---------------------------------------------------------------------------

class TestDisabledMode:
    def test_disabled_sets_has_unprocessed_images(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=False)

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert result[0].metadata.get("has_unprocessed_images") is True

    def test_disabled_no_captions_generated(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=False)

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert "image_captions" not in result[0].metadata

    def test_disabled_image_refs_preserved(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=False)

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert result[0].metadata["image_refs"] == chunk.metadata["image_refs"]

    def test_no_image_refs_no_flag_when_disabled(self) -> None:
        chunk = Chunk(id="c1", text="text", metadata={})
        settings = _make_settings(vision_enabled=False)

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert result[0].metadata.get("has_unprocessed_images") is None


# ---------------------------------------------------------------------------
# Tests: degradation mode – Vision LLM raises exception
# ---------------------------------------------------------------------------

class TestExceptionDegradation:
    def test_llm_exception_marks_unprocessed(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=True)
        mock_vlm = MagicMock()
        mock_vlm.chat_with_image.side_effect = RuntimeError("API error")

        captioner = ImageCaptioner(settings, vision_llm=mock_vlm)
        result = captioner.transform([chunk])

        assert result[0].metadata.get("has_unprocessed_images") is True
        assert "image_captions" not in result[0].metadata

    def test_llm_exception_does_not_raise(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=True)
        mock_vlm = MagicMock()
        mock_vlm.chat_with_image.side_effect = ConnectionError("timeout")

        captioner = ImageCaptioner(settings, vision_llm=mock_vlm)
        result = captioner.transform([chunk])  # must not raise
        assert len(result) == 1

    def test_one_image_fails_others_succeed(self, tmp_image: str, tmp_path: Path) -> None:
        img_ok = str(tmp_path / "ok.png")
        _create_test_image(img_ok)
        chunk = _chunk_with_images([tmp_image, img_ok])

        settings = _make_settings(vision_enabled=True)
        mock_vlm = MagicMock()

        def side_effect(text, image, **kwargs):
            if image == tmp_image:
                raise RuntimeError("fail on first image")
            return ChatResponse(content="caption for ok image", model="m", usage=None)

        mock_vlm.chat_with_image.side_effect = side_effect

        captioner = ImageCaptioner(settings, vision_llm=mock_vlm)
        result = captioner.transform([chunk])

        c = result[0]
        assert c.metadata.get("has_unprocessed_images") is True
        assert "image_captions" in c.metadata
        assert c.metadata["image_captions"].get("img_001") == "caption for ok image"

    def test_missing_image_file_marks_unprocessed(self, tmp_path: Path) -> None:
        nonexistent = str(tmp_path / "no_such_file.png")
        chunk = _chunk_with_images([nonexistent])
        settings = _make_settings(vision_enabled=True)
        mock_vlm = _fake_vision_llm()

        captioner = ImageCaptioner(settings, vision_llm=mock_vlm)
        result = captioner.transform([chunk])

        assert result[0].metadata.get("has_unprocessed_images") is True
        mock_vlm.chat_with_image.assert_not_called()

    def test_multiple_chunks_one_failing(self, tmp_image: str) -> None:
        good_chunk = _chunk_with_images([tmp_image])
        bad_chunk = Chunk(
            id="bad",
            text="[IMAGE: broken]",
            metadata={"image_refs": ["broken"], "images": []},
        )

        settings = _make_settings(vision_enabled=True)
        captioner = ImageCaptioner(settings, vision_llm=_fake_vision_llm("ok caption"))
        result = captioner.transform([good_chunk, bad_chunk])

        assert "image_captions" in result[0].metadata
        assert result[1].metadata.get("has_unprocessed_images") is True

    def test_vision_llm_factory_failure_degrades(self, tmp_image: str) -> None:
        chunk = _chunk_with_images([tmp_image])
        settings = _make_settings(vision_enabled=True, provider="nonexistent_provider")

        # No injected vision_llm; factory will fail → degrade silently
        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert result[0].metadata.get("has_unprocessed_images") is True
        assert "image_captions" not in result[0].metadata


# ---------------------------------------------------------------------------
# Tests: prompt loading
# ---------------------------------------------------------------------------

class TestPromptLoading:
    def test_custom_prompt_path_used(self, tmp_path: Path, tmp_image: str) -> None:
        prompt_file = tmp_path / "custom_prompt.txt"
        custom_text = "Describe this image briefly."
        prompt_file.write_text(custom_text, encoding="utf-8")

        settings = _make_settings(vision_enabled=True)
        mock_vlm = _fake_vision_llm()
        captioner = ImageCaptioner(settings, vision_llm=mock_vlm, prompt_path=str(prompt_file))
        chunk = _chunk_with_images([tmp_image])
        captioner.transform([chunk])

        call_kwargs = mock_vlm.chat_with_image.call_args
        text_arg = call_kwargs.kwargs.get("text") or call_kwargs.args[0]
        assert text_arg == custom_text

    def test_default_prompt_used_when_file_missing(self, tmp_image: str) -> None:
        settings = _make_settings(vision_enabled=True)
        mock_vlm = _fake_vision_llm()
        captioner = ImageCaptioner(
            settings, vision_llm=mock_vlm, prompt_path="/nonexistent/path.txt"
        )
        chunk = _chunk_with_images([tmp_image])
        captioner.transform([chunk])

        call_kwargs = mock_vlm.chat_with_image.call_args
        text_arg = call_kwargs.kwargs.get("text") or call_kwargs.args[0]
        assert len(text_arg) > 0
