"""Smoke tests for AzureVisionLLM (B9). All HTTP calls are mocked."""
from __future__ import annotations

import base64
import io
import sys
import os
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_vision_response(content: str = "a cat", model: str = "gpt-4o") -> MagicMock:
    usage = MagicMock()
    usage.model_dump.return_value = {"prompt_tokens": 10, "completion_tokens": 5}
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = usage
    return resp


def _make_tiny_jpeg() -> bytes:
    """Create a minimal 10x10 white JPEG in memory."""
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_large_png(size: int = 4096) -> bytes:
    """Create a large square PNG that exceeds max_image_size."""
    from PIL import Image
    img = Image.new("RGB", (size, size), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class VisionCfg:
    provider: str = "azure"
    enabled: bool = True
    model: str = "gpt-4o"
    deployment_name: str = "gpt-4o"
    azure_endpoint: str = "https://my-resource.openai.azure.com/"
    api_version: str = "2024-02-01"
    api_key: str = "sk-test"
    base_url: str = ""
    max_image_size: int = 2048


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

class TestAzureVisionFactory:
    def test_factory_routes_to_azure_vision(self):
        from libs.llm.azure_vision_llm import AzureVisionLLM
        cfg = VisionCfg()
        with patch("libs.llm.azure_vision_llm.AzureOpenAI"):
            vlm = LLMFactory.create_vision_llm(cfg)
        assert isinstance(vlm, AzureVisionLLM)

    def test_factory_creates_instance_with_settings(self):
        cfg = VisionCfg(model="gpt-4o", deployment_name="my-gpt4o")
        with patch("libs.llm.azure_vision_llm.AzureOpenAI"):
            vlm = LLMFactory.create_vision_llm(cfg)
        assert vlm._settings.deployment_name == "my-gpt4o"

    def test_missing_endpoint_raises(self):
        cfg = VisionCfg(azure_endpoint="")
        with patch("libs.llm.azure_vision_llm.AzureOpenAI"):
            with pytest.raises(ValueError, match="azure_endpoint"):
                LLMFactory.create_vision_llm(cfg)

    def test_disabled_raises(self):
        cfg = VisionCfg(enabled=False)
        with pytest.raises(ValueError, match="enabled is False"):
            LLMFactory.create_vision_llm(cfg)

    def test_default_api_version_used(self):
        cfg = VisionCfg(api_version="")
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            LLMFactory.create_vision_llm(cfg)
            _, kwargs = MockCls.call_args
        assert kwargs.get("api_version") == "2024-02-01"


# ---------------------------------------------------------------------------
# Normal call
# ---------------------------------------------------------------------------

class TestAzureVisionChatWithImage:
    def test_chat_with_image_path_returns_response(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(_make_tiny_jpeg())

        cfg = VisionCfg()
        fake_resp = _fake_vision_response(content="a white square", model="gpt-4o")
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            vlm = LLMFactory.create_vision_llm(cfg)
            resp = vlm.chat_with_image("describe this", str(img_path))

        assert isinstance(resp, ChatResponse)
        assert resp.content == "a white square"
        assert resp.model == "gpt-4o"
        assert resp.usage is not None

    def test_chat_with_image_bytes_returns_response(self):
        cfg = VisionCfg()
        fake_resp = _fake_vision_response(content="a cat")
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            vlm = LLMFactory.create_vision_llm(cfg)
            resp = vlm.chat_with_image("what is this?", _make_tiny_jpeg())

        assert resp.content == "a cat"

    def test_message_contains_image_url_with_base64(self, tmp_path):
        img_path = tmp_path / "img.jpg"
        img_bytes = _make_tiny_jpeg()
        img_path.write_bytes(img_bytes)

        cfg = VisionCfg()
        fake_resp = _fake_vision_response()
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            vlm = LLMFactory.create_vision_llm(cfg)
            vlm.chat_with_image("describe", str(img_path))
            call_kwargs = mock_client.chat.completions.create.call_args[1]

        messages = call_kwargs["messages"]
        content = messages[0]["content"]
        types = [c["type"] for c in content]
        assert "text" in types
        assert "image_url" in types
        image_url = next(c for c in content if c["type"] == "image_url")
        assert image_url["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_uses_deployment_name(self):
        cfg = VisionCfg(deployment_name="my-vision-deployment", model="")
        fake_resp = _fake_vision_response()
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            vlm = LLMFactory.create_vision_llm(cfg)
            vlm.chat_with_image("test", _make_tiny_jpeg())
            call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-vision-deployment"

    def test_empty_text_raises(self):
        cfg = VisionCfg()
        with patch("libs.llm.azure_vision_llm.AzureOpenAI"):
            vlm = LLMFactory.create_vision_llm(cfg)
        with pytest.raises(ValueError, match="text must not be empty"):
            vlm.chat_with_image("", _make_tiny_jpeg())

    def test_missing_deployment_raises(self):
        cfg = VisionCfg(deployment_name="", model="")
        with patch("libs.llm.azure_vision_llm.AzureOpenAI"):
            vlm = LLMFactory.create_vision_llm(cfg)
        with pytest.raises(ValueError, match="deployment_name or model"):
            vlm.chat_with_image("describe", _make_tiny_jpeg())

    def test_accepts_trace_parameter(self):
        cfg = VisionCfg()
        fake_resp = _fake_vision_response()
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            vlm = LLMFactory.create_vision_llm(cfg)
            resp = vlm.chat_with_image("test", _make_tiny_jpeg(), trace=object())
        assert resp.content == "a cat"


# ---------------------------------------------------------------------------
# Image compression
# ---------------------------------------------------------------------------

class TestImageCompression:
    def test_small_image_not_resized(self, tmp_path):
        img_path = tmp_path / "small.jpg"
        img_bytes = _make_tiny_jpeg()
        img_path.write_bytes(img_bytes)

        cfg = VisionCfg(max_image_size=2048)
        with patch("libs.llm.azure_vision_llm.AzureOpenAI"):
            vlm = LLMFactory.create_vision_llm(cfg)

        result = vlm.preprocess_image(str(img_path), max_size=2048)
        from PIL import Image
        img = Image.open(io.BytesIO(result))
        assert img.size == (10, 10)

    def test_large_image_compressed_within_max_size(self):
        large_bytes = _make_large_png(size=4096)
        cfg = VisionCfg(max_image_size=512)
        with patch("libs.llm.azure_vision_llm.AzureOpenAI"):
            vlm = LLMFactory.create_vision_llm(cfg)

        result = vlm.preprocess_image(large_bytes, max_size=512)
        from PIL import Image
        img = Image.open(io.BytesIO(result))
        assert img.size[0] <= 512
        assert img.size[1] <= 512

    def test_large_image_auto_compressed_in_chat(self):
        large_bytes = _make_large_png(size=4096)
        cfg = VisionCfg(max_image_size=256)
        fake_resp = _fake_vision_response()

        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            vlm = LLMFactory.create_vision_llm(cfg)
            vlm.chat_with_image("describe", large_bytes)
            call_kwargs = mock_client.chat.completions.create.call_args[1]

        image_url = next(
            c for c in call_kwargs["messages"][0]["content"]
            if c["type"] == "image_url"
        )
        b64_data = image_url["image_url"]["url"].split(",", 1)[1]
        compressed = base64.b64decode(b64_data)
        from PIL import Image
        img = Image.open(io.BytesIO(compressed))
        assert img.size[0] <= 256
        assert img.size[1] <= 256


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestAzureVisionErrors:
    def test_connection_error_wrapped(self):
        from openai import APIConnectionError
        cfg = VisionCfg()
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            vlm = LLMFactory.create_vision_llm(cfg)
        with pytest.raises(ConnectionError, match="azure-vision"):
            vlm.chat_with_image("test", _make_tiny_jpeg())

    def test_timeout_error_wrapped(self):
        from openai import APITimeoutError
        cfg = VisionCfg()
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = APITimeoutError(
                request=MagicMock()
            )
            vlm = LLMFactory.create_vision_llm(cfg)
        with pytest.raises(TimeoutError, match="azure-vision"):
            vlm.chat_with_image("test", _make_tiny_jpeg())

    def test_auth_failure_wrapped(self):
        from openai import APIStatusError
        cfg = VisionCfg()
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.chat.completions.create.side_effect = APIStatusError(
                "Unauthorized",
                response=mock_response,
                body={"message": "Unauthorized"},
            )
            vlm = LLMFactory.create_vision_llm(cfg)
        with pytest.raises(RuntimeError, match="azure-vision"):
            vlm.chat_with_image("test", _make_tiny_jpeg())

    def test_auth_error_includes_status_code(self):
        from openai import APIStatusError
        cfg = VisionCfg()
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_client.chat.completions.create.side_effect = APIStatusError(
                "Forbidden",
                response=mock_response,
                body={"message": "Forbidden"},
            )
            vlm = LLMFactory.create_vision_llm(cfg)
        with pytest.raises(RuntimeError, match="403"):
            vlm.chat_with_image("test", _make_tiny_jpeg())
