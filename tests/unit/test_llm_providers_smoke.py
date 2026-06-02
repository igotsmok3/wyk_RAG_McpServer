"""Smoke tests for OpenAI-compatible LLM providers (B7.1).

All HTTP calls are mocked — no real network requests.
"""
import sys
import os
import pytest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.llm.base_llm import ChatResponse
from libs.llm.llm_factory import LLMFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_openai_response(content: str = "ok", model: str = "gpt-4") -> MagicMock:
    usage = MagicMock()
    usage.model_dump.return_value = {"prompt_tokens": 5, "completion_tokens": 3}
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = usage
    return resp


@dataclass
class LLMCfg:
    provider: str
    model: str = "test-model"
    api_key: str = "sk-test"
    base_url: str = ""
    azure_endpoint: str = "https://my-resource.openai.azure.com/"
    deployment_name: str = "gpt-4"
    api_version: str = "2024-02-01"
    temperature: float = 0.0
    max_tokens: int = 512


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class TestOpenAILLM:
    def test_factory_routes_to_openai(self):
        from libs.llm.openai_llm import OpenAILLM
        cfg = LLMCfg(provider="openai")
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = LLMFactory.create(cfg)
        assert isinstance(llm, OpenAILLM)

    def test_chat_returns_chat_response(self):
        cfg = LLMCfg(provider="openai", model="gpt-4")
        fake_resp = _fake_openai_response(content="Hello!", model="gpt-4")
        with patch("libs.llm.openai_llm.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            llm = LLMFactory.create(cfg)
            resp = llm.chat([{"role": "user", "content": "hi"}])
        assert isinstance(resp, ChatResponse)
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4"
        assert resp.usage is not None

    def test_chat_passes_model_and_params(self):
        cfg = LLMCfg(provider="openai", model="gpt-4o", temperature=0.7, max_tokens=256)
        fake_resp = _fake_openai_response()
        with patch("libs.llm.openai_llm.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            llm = LLMFactory.create(cfg)
            llm.chat([{"role": "user", "content": "test"}])
            call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 256

    def test_chat_empty_messages_raises(self):
        cfg = LLMCfg(provider="openai")
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = LLMFactory.create(cfg)
        with pytest.raises(ValueError, match="openai"):
            llm.chat([])

    def test_chat_malformed_message_raises(self):
        cfg = LLMCfg(provider="openai")
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = LLMFactory.create(cfg)
        with pytest.raises(ValueError, match="openai"):
            llm.chat([{"role": "user"}])  # missing "content"

    def test_connection_error_wrapped(self):
        from openai import APIConnectionError
        cfg = LLMCfg(provider="openai")
        with patch("libs.llm.openai_llm.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            llm = LLMFactory.create(cfg)
        with pytest.raises(ConnectionError, match="openai"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_api_error_wrapped(self):
        from openai import APIStatusError
        cfg = LLMCfg(provider="openai")
        with patch("libs.llm.openai_llm.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.chat.completions.create.side_effect = APIStatusError(
                "Unauthorized", response=mock_response, body={"message": "Unauthorized"}
            )
            llm = LLMFactory.create(cfg)
        with pytest.raises(RuntimeError, match="openai"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_uses_custom_base_url(self):
        cfg = LLMCfg(provider="openai", base_url="https://custom.endpoint/v1")
        with patch("libs.llm.openai_llm.OpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            LLMFactory.create(cfg)
            _, kwargs = MockCls.call_args
        assert kwargs.get("base_url") == "https://custom.endpoint/v1"


# ---------------------------------------------------------------------------
# Azure provider
# ---------------------------------------------------------------------------

class TestAzureLLM:
    def test_factory_routes_to_azure(self):
        from libs.llm.azure_llm import AzureLLM
        cfg = LLMCfg(provider="azure")
        with patch("libs.llm.azure_llm.AzureOpenAI"):
            llm = LLMFactory.create(cfg)
        assert isinstance(llm, AzureLLM)

    def test_chat_returns_chat_response(self):
        cfg = LLMCfg(provider="azure", deployment_name="my-gpt4")
        fake_resp = _fake_openai_response(content="Azure reply", model="gpt-4")
        with patch("libs.llm.azure_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            llm = LLMFactory.create(cfg)
            resp = llm.chat([{"role": "user", "content": "hi"}])
        assert resp.content == "Azure reply"

    def test_uses_deployment_name_as_model(self):
        cfg = LLMCfg(provider="azure", deployment_name="dep-gpt4", model="")
        fake_resp = _fake_openai_response()
        with patch("libs.llm.azure_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            llm = LLMFactory.create(cfg)
            llm.chat([{"role": "user", "content": "test"}])
            call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "dep-gpt4"

    def test_missing_endpoint_raises(self):
        cfg = LLMCfg(provider="azure", azure_endpoint="")
        with patch("libs.llm.azure_llm.AzureOpenAI"):
            with pytest.raises(ValueError, match="azure_endpoint"):
                LLMFactory.create(cfg)

    def test_empty_messages_raises(self):
        cfg = LLMCfg(provider="azure")
        with patch("libs.llm.azure_llm.AzureOpenAI"):
            llm = LLMFactory.create(cfg)
        with pytest.raises(ValueError, match="azure"):
            llm.chat([])

    def test_malformed_message_raises(self):
        cfg = LLMCfg(provider="azure")
        with patch("libs.llm.azure_llm.AzureOpenAI"):
            llm = LLMFactory.create(cfg)
        with pytest.raises(ValueError, match="azure"):
            llm.chat([{"content": "no role"}])

    def test_connection_error_wrapped(self):
        from openai import APIConnectionError
        cfg = LLMCfg(provider="azure")
        with patch("libs.llm.azure_llm.AzureOpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            llm = LLMFactory.create(cfg)
        with pytest.raises(ConnectionError, match="azure"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_default_api_version_used(self):
        cfg = LLMCfg(provider="azure", api_version="")
        with patch("libs.llm.azure_llm.AzureOpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            LLMFactory.create(cfg)
            _, kwargs = MockCls.call_args
        assert kwargs.get("api_version") == "2024-02-01"


# ---------------------------------------------------------------------------
# DeepSeek provider
# ---------------------------------------------------------------------------

class TestDeepSeekLLM:
    def test_factory_routes_to_deepseek(self):
        from libs.llm.deepseek_llm import DeepSeekLLM
        cfg = LLMCfg(provider="deepseek")
        with patch("libs.llm.deepseek_llm.OpenAI"):
            llm = LLMFactory.create(cfg)
        assert isinstance(llm, DeepSeekLLM)

    def test_chat_returns_chat_response(self):
        cfg = LLMCfg(provider="deepseek", model="deepseek-chat")
        fake_resp = _fake_openai_response(content="DeepSeek reply", model="deepseek-chat")
        with patch("libs.llm.deepseek_llm.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.return_value = fake_resp
            llm = LLMFactory.create(cfg)
            resp = llm.chat([{"role": "user", "content": "hi"}])
        assert resp.content == "DeepSeek reply"

    def test_uses_default_base_url_when_empty(self):
        cfg = LLMCfg(provider="deepseek", base_url="")
        with patch("libs.llm.deepseek_llm.OpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            LLMFactory.create(cfg)
            _, kwargs = MockCls.call_args
        assert kwargs.get("base_url") == "https://api.deepseek.com/v1"

    def test_uses_custom_base_url(self):
        cfg = LLMCfg(provider="deepseek", base_url="https://custom-deepseek.com/v1")
        with patch("libs.llm.deepseek_llm.OpenAI") as MockCls:
            MockCls.return_value = MagicMock()
            LLMFactory.create(cfg)
            _, kwargs = MockCls.call_args
        assert kwargs.get("base_url") == "https://custom-deepseek.com/v1"

    def test_empty_messages_raises(self):
        cfg = LLMCfg(provider="deepseek")
        with patch("libs.llm.deepseek_llm.OpenAI"):
            llm = LLMFactory.create(cfg)
        with pytest.raises(ValueError, match="deepseek"):
            llm.chat([])

    def test_malformed_message_raises(self):
        cfg = LLMCfg(provider="deepseek")
        with patch("libs.llm.deepseek_llm.OpenAI"):
            llm = LLMFactory.create(cfg)
        with pytest.raises(ValueError, match="deepseek"):
            llm.chat([{"role": "user"}])

    def test_connection_error_wrapped(self):
        from openai import APIConnectionError
        cfg = LLMCfg(provider="deepseek")
        with patch("libs.llm.deepseek_llm.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = APIConnectionError(
                request=MagicMock()
            )
            llm = LLMFactory.create(cfg)
        with pytest.raises(ConnectionError, match="deepseek"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_api_error_wrapped(self):
        from openai import APIStatusError
        cfg = LLMCfg(provider="deepseek")
        with patch("libs.llm.deepseek_llm.OpenAI") as MockCls:
            mock_client = MagicMock()
            MockCls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_client.chat.completions.create.side_effect = APIStatusError(
                "Rate limited", response=mock_response, body={"message": "Rate limited"}
            )
            llm = LLMFactory.create(cfg)
        with pytest.raises(RuntimeError, match="deepseek"):
            llm.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Factory routing coverage
# ---------------------------------------------------------------------------

class TestFactoryRouting:
    def test_all_three_providers_registered(self):
        from libs.llm.openai_llm import OpenAILLM
        from libs.llm.azure_llm import AzureLLM
        from libs.llm.deepseek_llm import DeepSeekLLM

        for provider, expected_cls in [
            ("openai", OpenAILLM),
            ("deepseek", DeepSeekLLM),
        ]:
            cfg = LLMCfg(provider=provider)
            with patch(f"libs.llm.{provider}_llm.OpenAI"):
                llm = LLMFactory.create(cfg)
            assert isinstance(llm, expected_cls), f"Expected {expected_cls} for {provider}"

        cfg = LLMCfg(provider="azure")
        with patch("libs.llm.azure_llm.AzureOpenAI"):
            llm = LLMFactory.create(cfg)
        assert isinstance(llm, AzureLLM)

    def test_provider_name_case_insensitive(self):
        from libs.llm.openai_llm import OpenAILLM
        cfg = LLMCfg(provider="OpenAI")
        with patch("libs.llm.openai_llm.OpenAI"):
            llm = LLMFactory.create(cfg)
        assert isinstance(llm, OpenAILLM)
