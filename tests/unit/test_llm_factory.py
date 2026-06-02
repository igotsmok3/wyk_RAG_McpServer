"""Tests for BaseLLM interface and LLMFactory routing (B1)."""
import sys
import os
import pytest
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.llm.llm_factory import LLMFactory, register_provider


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLLM(BaseLLM):
    def __init__(self, settings):
        self.settings = settings

    def chat(self, messages: list[dict]) -> ChatResponse:
        last = messages[-1]["content"] if messages else ""
        return ChatResponse(content=f"fake:{last}", model="fake-model")


class AnotherFakeLLM(BaseLLM):
    def __init__(self, settings):
        self.settings = settings

    def chat(self, messages: list[dict]) -> ChatResponse:
        return ChatResponse(content="another", model="another-model")


@dataclass
class FakeLLMSettings:
    provider: str
    model: str = "fake"
    api_key: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def register_fakes():
    """Register fake providers before each test."""
    register_provider("fake", FakeLLM)
    register_provider("another", AnotherFakeLLM)
    yield


# ---------------------------------------------------------------------------
# BaseLLM interface tests
# ---------------------------------------------------------------------------

def test_base_llm_is_abstract():
    with pytest.raises(TypeError):
        BaseLLM()  # type: ignore


def test_chat_response_has_content():
    resp = ChatResponse(content="hello")
    assert resp.content == "hello"


def test_chat_response_defaults():
    resp = ChatResponse(content="x")
    assert resp.model == ""
    assert resp.usage is None


# ---------------------------------------------------------------------------
# LLMFactory routing tests
# ---------------------------------------------------------------------------

def test_factory_creates_fake_llm():
    s = FakeLLMSettings(provider="fake")
    llm = LLMFactory.create(s)
    assert isinstance(llm, FakeLLM)


def test_factory_routes_by_provider():
    s = FakeLLMSettings(provider="another")
    llm = LLMFactory.create(s)
    assert isinstance(llm, AnotherFakeLLM)


def test_factory_provider_case_insensitive():
    s = FakeLLMSettings(provider="FAKE")
    llm = LLMFactory.create(s)
    assert isinstance(llm, FakeLLM)


def test_factory_unknown_provider_raises():
    s = FakeLLMSettings(provider="nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        LLMFactory.create(s)


def test_factory_error_lists_known_providers():
    s = FakeLLMSettings(provider="nonexistent")
    with pytest.raises(ValueError, match="fake"):
        LLMFactory.create(s)


def test_factory_empty_provider_raises():
    s = FakeLLMSettings(provider="")
    with pytest.raises(ValueError, match="empty"):
        LLMFactory.create(s)


# ---------------------------------------------------------------------------
# FakeLLM functional tests
# ---------------------------------------------------------------------------

def test_fake_llm_chat_returns_response():
    s = FakeLLMSettings(provider="fake")
    llm = LLMFactory.create(s)
    resp = llm.chat([{"role": "user", "content": "hello"}])
    assert isinstance(resp, ChatResponse)
    assert "hello" in resp.content


def test_fake_llm_receives_settings():
    s = FakeLLMSettings(provider="fake", model="test-model")
    llm = LLMFactory.create(s)
    assert llm.settings.model == "test-model"
