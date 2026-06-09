"""Integration tests for ImageCaptioner with real Qwen Vision LLM (DashScope).

These tests require DASHSCOPE_API_KEY to be set and make real API calls.
Run with: pytest tests/integration/test_image_captioner_real.py -v -s
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image as PILImage

from core.settings import load_settings
from core.types import Chunk, ImageRef
from ingestion.transform.image_captioner import ImageCaptioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_image(path: str, text_label: str = "TEST") -> None:
    """Create a real PNG with some visible structure for vision testing."""
    from PIL import ImageDraw, ImageFont
    img = PILImage.new("RGB", (200, 150), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw a simple bar-chart-like shape
    draw.rectangle([20, 100, 60, 130], fill=(70, 130, 180))
    draw.rectangle([80, 70, 120, 130], fill=(255, 160, 50))
    draw.rectangle([140, 50, 180, 130], fill=(100, 200, 100))
    draw.text((10, 10), text_label, fill=(0, 0, 0))
    draw.text((10, 135), "bar chart", fill=(100, 100, 100))
    img.save(path, format="PNG")


def _chunk_with_real_image(image_path: str) -> Chunk:
    img_ref = ImageRef(
        id="real_img_001",
        path=image_path,
        page=1,
        text_offset=0,
        text_length=20,
    )
    return Chunk(
        id="chunk_real_001",
        text=f"[IMAGE: real_img_001] This section discusses chart data.",
        metadata={
            "source_path": "test_document.pdf",
            "images": [img_ref],
            "image_refs": ["real_img_001"],
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRealVisionLLMEnabled:
    """Tests that use the real Qwen Vision LLM."""

    @pytest.mark.skipif(
        not os.environ.get("DASHSCOPE_API_KEY"),
        reason="DASHSCOPE_API_KEY not set"
    )
    def test_real_caption_generated(self, tmp_path: Path) -> None:
        """Real Vision LLM should generate a non-empty caption for a test image."""
        img_path = str(tmp_path / "chart.png")
        _create_test_image(img_path, "Revenue Chart")

        settings = load_settings("config/settings.yaml")
        assert settings.vision_llm.enabled, "vision_llm must be enabled in settings.yaml"

        chunk = _chunk_with_real_image(img_path)
        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        c = result[0]
        print("\n=== Real Vision LLM Caption Result ===")
        print(f"  image_captions: {c.metadata.get('image_captions')}")
        print(f"  has_unprocessed_images: {c.metadata.get('has_unprocessed_images')}")
        print("======================================\n")

        assert "image_captions" in c.metadata, "Caption should be generated in enabled mode"
        caption = c.metadata["image_captions"].get("real_img_001", "")
        assert isinstance(caption, str), "Caption must be a string"
        assert len(caption) > 10, f"Caption too short: {repr(caption)}"
        assert c.metadata.get("has_unprocessed_images") is None, \
            "Should not be marked as unprocessed when captioning succeeded"

    @pytest.mark.skipif(
        not os.environ.get("DASHSCOPE_API_KEY"),
        reason="DASHSCOPE_API_KEY not set"
    )
    def test_real_caption_content_is_descriptive(self, tmp_path: Path) -> None:
        """Caption should contain meaningful descriptive text, not an error message."""
        img_path = str(tmp_path / "chart2.png")
        _create_test_image(img_path, "Quarterly Report")

        settings = load_settings("config/settings.yaml")
        chunk = _chunk_with_real_image(img_path)
        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        caption = result[0].metadata.get("image_captions", {}).get("real_img_001", "")
        print(f"\n  Caption text: {caption!r}\n")
        # Should not be an empty or error response
        assert len(caption) > 20, f"Expected descriptive caption, got: {repr(caption)}"

    @pytest.mark.skipif(
        not os.environ.get("DASHSCOPE_API_KEY"),
        reason="DASHSCOPE_API_KEY not set"
    )
    def test_image_refs_still_present_after_captioning(self, tmp_path: Path) -> None:
        """image_refs metadata should be preserved after captioning."""
        img_path = str(tmp_path / "chart3.png")
        _create_test_image(img_path)

        settings = load_settings("config/settings.yaml")
        chunk = _chunk_with_real_image(img_path)
        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert result[0].metadata["image_refs"] == ["real_img_001"]


class TestRealVisionLLMDegradation:
    """Tests that verify graceful degradation with the real LLM infrastructure."""

    @pytest.mark.skipif(
        not os.environ.get("DASHSCOPE_API_KEY"),
        reason="DASHSCOPE_API_KEY not set"
    )
    def test_invalid_model_degrades_gracefully(self, tmp_path: Path) -> None:
        """Using an invalid model name should trigger degradation, not a crash."""
        from core.settings import VisionLLMSettings
        from libs.llm.qwen_vision_llm import QwenVisionLLM

        img_path = str(tmp_path / "chart_invalid.png")
        _create_test_image(img_path)

        settings = load_settings("config/settings.yaml")
        chunk = _chunk_with_real_image(img_path)

        # Inject a Vision LLM with an invalid model name to force API error
        bad_settings = VisionLLMSettings(
            enabled=True,
            provider="qwen",
            model="nonexistent-model-xyz-12345",  # deliberately invalid
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_image_size=2048,
        )
        bad_vision_llm = QwenVisionLLM(bad_settings)

        captioner = ImageCaptioner(settings, vision_llm=bad_vision_llm)
        result = captioner.transform([chunk])  # must not raise

        c = result[0]
        print("\n=== Degradation Mode (invalid model) ===")
        print(f"  has_unprocessed_images: {c.metadata.get('has_unprocessed_images')}")
        print(f"  image_captions: {c.metadata.get('image_captions')}")
        print("========================================\n")

        assert c.metadata.get("has_unprocessed_images") is True, \
            "Should mark images as unprocessed when VisionLLM fails"
        assert "image_captions" not in c.metadata, \
            "No captions should be written on failure"
        assert c.metadata["image_refs"] == ["real_img_001"], \
            "image_refs must be preserved for downstream retry"

    @pytest.mark.skipif(
        not os.environ.get("DASHSCOPE_API_KEY"),
        reason="DASHSCOPE_API_KEY not set"
    )
    def test_vision_disabled_in_config_no_api_calls(self, tmp_path: Path) -> None:
        """When vision_llm.enabled=False, no API calls should be made."""
        from unittest.mock import patch

        img_path = str(tmp_path / "chart_disabled.png")
        _create_test_image(img_path)

        settings = load_settings("config/settings.yaml")
        settings.vision_llm.enabled = False  # Override for this test

        chunk = _chunk_with_real_image(img_path)

        with patch("libs.llm.qwen_vision_llm.QwenVisionLLM.chat_with_image") as mock_api:
            captioner = ImageCaptioner(settings)
            result = captioner.transform([chunk])
            mock_api.assert_not_called()

        c = result[0]
        assert c.metadata.get("has_unprocessed_images") is True
        assert "image_captions" not in c.metadata

        # Restore (not strictly needed since settings is local)
        settings.vision_llm.enabled = True
