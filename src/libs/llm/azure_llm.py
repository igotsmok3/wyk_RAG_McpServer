"""AzureLLM: Azure OpenAI LLM backend (provider='azure')."""
from __future__ import annotations

from typing import TYPE_CHECKING

from openai import APIConnectionError, APIError, APITimeoutError, AzureOpenAI

from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.llm.llm_factory import register_provider

if TYPE_CHECKING:
    from core.settings import LLMSettings

_DEFAULT_API_VERSION = "2024-02-01"


class AzureLLM(BaseLLM):
    """LLM backed by the Azure OpenAI Chat Completions API."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        if not settings.azure_endpoint:
            raise ValueError("[azure] azure_endpoint must be set in config")
        self._client = AzureOpenAI(
            api_key=settings.api_key,
            azure_endpoint=settings.azure_endpoint,
            api_version=settings.api_version or _DEFAULT_API_VERSION,
        )

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        self._validate_messages(messages)
        deployment = self._settings.deployment_name or self._settings.model
        if not deployment:
            raise ValueError("[azure] deployment_name or model must be configured")
        try:
            resp = self._client.chat.completions.create(
                model=deployment,
                messages=messages,  # type: ignore[arg-type]
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
            )
        except APIConnectionError as e:
            raise ConnectionError(f"[azure] Connection failed: {e}") from e
        except APITimeoutError as e:
            raise TimeoutError(f"[azure] Request timed out: {e}") from e
        except APIError as e:
            raise RuntimeError(
                f"[azure] API error (status={e.status_code}): {e.message}"
            ) from e

        choice = resp.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=resp.usage.model_dump() if resp.usage else None,
        )

    def _validate_messages(self, messages: list[dict[str, str]]) -> None:
        if not messages:
            raise ValueError("[azure] messages must not be empty")
        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"[azure] message[{i}] must have 'role' and 'content' keys"
                )


register_provider("azure", AzureLLM)
