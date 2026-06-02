"""Tests for OllamaLLM (B7.2). All HTTP calls are mocked."""
from __future__ import annotations

import sys
import os
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.llm.base_llm import ChatResponse
from libs.llm.llm_factory import LLMFactory


@dataclass
class LLMCfg:
    provider: str
    model: str = "llama3"
    api_key: str = ""
    base_url: str = ""
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = ""
    temperature: float = 0.0
    max_tokens: int = 512


def _fake_response(content: str = "ok", model: str = "llama3", status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.ok = status < 400
    resp.status_code = status
    resp.text = content
    resp.json.return_value = {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    return resp


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

class TestOllamaFactory:
    def test_factory_routes_to_ollama(self):
        from libs.llm.ollama_llm import OllamaLLM
        cfg = LLMCfg(provider="ollama")
        with patch("libs.llm.ollama_llm.requests.post", return_value=_fake_response()):
            llm = LLMFactory.create(cfg)
        assert isinstance(llm, OllamaLLM)

    def test_factory_case_insensitive(self):
        from libs.llm.ollama_llm import OllamaLLM
        cfg = LLMCfg(provider="Ollama")
        llm = LLMFactory.create(cfg)
        assert isinstance(llm, OllamaLLM)

    def test_empty_model_raises(self):
        cfg = LLMCfg(provider="ollama", model="")
        with pytest.raises(ValueError, match="model"):
            LLMFactory.create(cfg)


# ---------------------------------------------------------------------------
# Normal responses
# ---------------------------------------------------------------------------

class TestOllamaChat:
    def test_chat_returns_chat_response(self):
        cfg = LLMCfg(provider="ollama", model="llama3")
        with patch("libs.llm.ollama_llm.requests.post", return_value=_fake_response("Hello!", "llama3")) as mock_post:
            llm = LLMFactory.create(cfg)
            resp = llm.chat([{"role": "user", "content": "hi"}])
        assert isinstance(resp, ChatResponse)
        assert resp.content == "Hello!"
        assert resp.model == "llama3"
        assert resp.usage is not None

    def test_chat_posts_to_api_chat_endpoint(self):
        cfg = LLMCfg(provider="ollama", base_url="http://localhost:11434")
        with patch("libs.llm.ollama_llm.requests.post", return_value=_fake_response()) as mock_post:
            llm = LLMFactory.create(cfg)
            llm.chat([{"role": "user", "content": "test"}])
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:11434/api/chat"

    def test_chat_uses_default_base_url_when_empty(self):
        cfg = LLMCfg(provider="ollama", base_url="")
        with patch("libs.llm.ollama_llm.requests.post", return_value=_fake_response()) as mock_post:
            llm = LLMFactory.create(cfg)
            llm.chat([{"role": "user", "content": "test"}])
        call_url = mock_post.call_args[0][0]
        assert "localhost:11434" in call_url

    def test_chat_uses_custom_base_url(self):
        cfg = LLMCfg(provider="ollama", base_url="http://remote-host:11434")
        with patch("libs.llm.ollama_llm.requests.post", return_value=_fake_response()) as mock_post:
            llm = LLMFactory.create(cfg)
            llm.chat([{"role": "user", "content": "test"}])
        call_url = mock_post.call_args[0][0]
        assert "remote-host:11434" in call_url

    def test_chat_sends_correct_payload(self):
        cfg = LLMCfg(provider="ollama", model="mistral", temperature=0.5, max_tokens=100)
        with patch("libs.llm.ollama_llm.requests.post", return_value=_fake_response()) as mock_post:
            llm = LLMFactory.create(cfg)
            llm.chat([{"role": "user", "content": "test"}])
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "mistral"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.5

    def test_usage_extracted_from_response(self):
        cfg = LLMCfg(provider="ollama")
        with patch("libs.llm.ollama_llm.requests.post", return_value=_fake_response()):
            llm = LLMFactory.create(cfg)
            resp = llm.chat([{"role": "user", "content": "test"}])
        assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_no_usage_when_counts_absent(self):
        fake = MagicMock()
        fake.ok = True
        fake.status_code = 200
        fake.json.return_value = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "reply"},
        }
        cfg = LLMCfg(provider="ollama")
        with patch("libs.llm.ollama_llm.requests.post", return_value=fake):
            llm = LLMFactory.create(cfg)
            resp = llm.chat([{"role": "user", "content": "test"}])
        assert resp.usage is None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestOllamaValidation:
    def test_empty_messages_raises(self):
        cfg = LLMCfg(provider="ollama")
        llm = LLMFactory.create(cfg)
        with pytest.raises(ValueError, match="ollama"):
            llm.chat([])

    def test_malformed_message_missing_role_raises(self):
        cfg = LLMCfg(provider="ollama")
        llm = LLMFactory.create(cfg)
        with pytest.raises(ValueError, match="ollama"):
            llm.chat([{"content": "no role here"}])

    def test_malformed_message_missing_content_raises(self):
        cfg = LLMCfg(provider="ollama")
        llm = LLMFactory.create(cfg)
        with pytest.raises(ValueError, match="ollama"):
            llm.chat([{"role": "user"}])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestOllamaErrors:
    def test_connection_error_wrapped(self):
        import requests as req_mod
        cfg = LLMCfg(provider="ollama")
        llm = LLMFactory.create(cfg)
        with patch(
            "libs.llm.ollama_llm.requests.post",
            side_effect=req_mod.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(ConnectionError, match="ollama"):
                llm.chat([{"role": "user", "content": "hi"}])

    def test_timeout_error_wrapped(self):
        import requests as req_mod
        cfg = LLMCfg(provider="ollama")
        llm = LLMFactory.create(cfg)
        with patch(
            "libs.llm.ollama_llm.requests.post",
            side_effect=req_mod.exceptions.Timeout("timed out"),
        ):
            with pytest.raises(TimeoutError, match="ollama"):
                llm.chat([{"role": "user", "content": "hi"}])

    def test_http_error_status_wrapped(self):
        cfg = LLMCfg(provider="ollama")
        bad_resp = MagicMock()
        bad_resp.ok = False
        bad_resp.status_code = 404
        bad_resp.text = "model not found"
        llm = LLMFactory.create(cfg)
        with patch("libs.llm.ollama_llm.requests.post", return_value=bad_resp):
            with pytest.raises(RuntimeError, match="ollama"):
                llm.chat([{"role": "user", "content": "hi"}])

    def test_http_error_does_not_leak_config(self):
        cfg = LLMCfg(provider="ollama", base_url="http://secret-host:11434")
        bad_resp = MagicMock()
        bad_resp.ok = False
        bad_resp.status_code = 500
        bad_resp.text = "internal error"
        llm = LLMFactory.create(cfg)
        with patch("libs.llm.ollama_llm.requests.post", return_value=bad_resp):
            with pytest.raises(RuntimeError) as exc_info:
                llm.chat([{"role": "user", "content": "hi"}])
        assert "api_key" not in str(exc_info.value).lower()

    def test_generic_request_exception_wrapped(self):
        import requests as req_mod
        cfg = LLMCfg(provider="ollama")
        llm = LLMFactory.create(cfg)
        with patch(
            "libs.llm.ollama_llm.requests.post",
            side_effect=req_mod.exceptions.RequestException("unexpected"),
        ):
            with pytest.raises(RuntimeError, match="ollama"):
                llm.chat([{"role": "user", "content": "hi"}])
