"""Tests for BaseVisionLLM interface and LLMFactory.create_vision_llm() routing (B8)."""
import sys
import os
import pytest
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.base_llm import ChatResponse
from libs.llm.llm_factory import LLMFactory, register_vision_provider


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, settings):
        self.settings = settings

    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        image_label = image if isinstance(image, str) else f"<bytes:{len(image)}>"
        return ChatResponse(
            content=f"caption for {image_label}: {text}",
            model="fake-vision",
        )


class AnotherFakeVisionLLM(BaseVisionLLM):
    def __init__(self, settings):
        self.settings = settings

    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        return ChatResponse(content="another-vision", model="another-vision-model")


@dataclass
class FakeVisionSettings:
    provider: str
    enabled: bool = True
    model: str = "fake-vision"
    max_image_size: int = 2048
    api_key: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def register_fakes():
    register_vision_provider("fake_vision", FakeVisionLLM)
    register_vision_provider("another_vision", AnotherFakeVisionLLM)
    yield


# ---------------------------------------------------------------------------
# BaseVisionLLM interface tests
# ---------------------------------------------------------------------------

def test_base_vision_llm_is_abstract():
    with pytest.raises(TypeError):
        BaseVisionLLM()  # type: ignore


def test_base_vision_llm_requires_chat_with_image():
    class Incomplete(BaseVisionLLM):
        pass
    with pytest.raises(TypeError):
        Incomplete()  # type: ignore


# ---------------------------------------------------------------------------
# LLMFactory.create_vision_llm() routing tests
# ---------------------------------------------------------------------------

def test_factory_creates_fake_vision_llm():
    s = FakeVisionSettings(provider="fake_vision")
    vlm = LLMFactory.create_vision_llm(s)
    assert isinstance(vlm, FakeVisionLLM)


def test_factory_routes_by_provider():
    s = FakeVisionSettings(provider="another_vision")
    vlm = LLMFactory.create_vision_llm(s)
    assert isinstance(vlm, AnotherFakeVisionLLM)


def test_factory_provider_case_insensitive():
    s = FakeVisionSettings(provider="FAKE_VISION")
    vlm = LLMFactory.create_vision_llm(s)
    assert isinstance(vlm, FakeVisionLLM)


def test_factory_unknown_provider_raises():
    s = FakeVisionSettings(provider="nonexistent_vision")
    with pytest.raises(ValueError, match="nonexistent_vision"):
        LLMFactory.create_vision_llm(s)


def test_factory_error_lists_known_providers():
    s = FakeVisionSettings(provider="nonexistent_vision")
    with pytest.raises(ValueError, match="fake_vision"):
        LLMFactory.create_vision_llm(s)


def test_factory_empty_provider_raises():
    s = FakeVisionSettings(provider="")
    with pytest.raises(ValueError, match="empty"):
        LLMFactory.create_vision_llm(s)


def test_factory_disabled_raises():
    s = FakeVisionSettings(provider="fake_vision", enabled=False)
    with pytest.raises(ValueError, match="enabled is False"):
        LLMFactory.create_vision_llm(s)


# ---------------------------------------------------------------------------
# FakeVisionLLM functional tests
# ---------------------------------------------------------------------------

def test_chat_with_image_path_returns_response():
    s = FakeVisionSettings(provider="fake_vision")
    vlm = LLMFactory.create_vision_llm(s)
    resp = vlm.chat_with_image("describe this", "/tmp/test.jpg")
    assert isinstance(resp, ChatResponse)
    assert "test.jpg" in resp.content
    assert "describe this" in resp.content


def test_chat_with_image_bytes_returns_response():
    s = FakeVisionSettings(provider="fake_vision")
    vlm = LLMFactory.create_vision_llm(s)
    image_data = b"\x89PNG fake image bytes"
    resp = vlm.chat_with_image("what is this?", image_data)
    assert isinstance(resp, ChatResponse)
    assert "what is this?" in resp.content


def test_chat_with_image_accepts_trace():
    s = FakeVisionSettings(provider="fake_vision")
    vlm = LLMFactory.create_vision_llm(s)
    resp = vlm.chat_with_image("test", b"bytes", trace=object())
    assert resp.content


def test_fake_vision_llm_receives_settings():
    s = FakeVisionSettings(provider="fake_vision", model="test-vision-model")
    vlm = LLMFactory.create_vision_llm(s)
    assert vlm.settings.model == "test-vision-model"


# ---------------------------------------------------------------------------
# BaseVisionLLM.preprocess_image() extension point tests
# ---------------------------------------------------------------------------

def test_preprocess_image_bytes_passthrough():
    s = FakeVisionSettings(provider="fake_vision")
    vlm = LLMFactory.create_vision_llm(s)
    raw = b"fake image bytes"
    result = vlm.preprocess_image(raw)
    assert result == raw


def test_preprocess_image_file_reads_content(tmp_path):
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"PNG content")
    s = FakeVisionSettings(provider="fake_vision")
    vlm = LLMFactory.create_vision_llm(s)
    result = vlm.preprocess_image(str(img_path))
    assert result == b"PNG content"
