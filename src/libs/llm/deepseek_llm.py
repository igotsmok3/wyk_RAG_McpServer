"""DeepSeekLLM: DeepSeek LLM backend (OpenAI-compatible, provider='deepseek')."""
from __future__ import annotations

from typing import TYPE_CHECKING

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.llm.llm_factory import register_provider

if TYPE_CHECKING:
    from core.settings import LLMSettings

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekLLM(BaseLLM):
    """LLM backed by the DeepSeek API (OpenAI-compatible endpoint)."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url or _DEEPSEEK_BASE_URL,
        )

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        self._validate_messages(messages)
        try:
            resp = self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
            )
        except APIConnectionError as e:
            raise ConnectionError(f"[deepseek] Connection failed: {e}") from e
        except APITimeoutError as e:
            raise TimeoutError(f"[deepseek] Request timed out: {e}") from e
        except APIError as e:
            raise RuntimeError(
                f"[deepseek] API error (status={e.status_code}): {e.message}"
            ) from e

        choice = resp.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=resp.usage.model_dump() if resp.usage else None,
        )

    def _validate_messages(self, messages: list[dict[str, str]]) -> None:
        if not messages:
            raise ValueError("[deepseek] messages must not be empty")
        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"[deepseek] message[{i}] must have 'role' and 'content' keys"
                )


register_provider("deepseek", DeepSeekLLM)
