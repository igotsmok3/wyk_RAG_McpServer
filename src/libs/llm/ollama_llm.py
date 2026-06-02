"""OllamaLLM: local Ollama HTTP backend (provider='ollama')."""
from __future__ import annotations

from typing import TYPE_CHECKING

import requests

from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.llm.llm_factory import register_provider

if TYPE_CHECKING:
    from core.settings import LLMSettings

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaLLM(BaseLLM):
    """LLM backed by a local Ollama HTTP server."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._base_url = (settings.base_url or _DEFAULT_BASE_URL).rstrip("/")
        if not settings.model:
            raise ValueError("[ollama] model must not be empty — check config/settings.yaml")

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        self._validate_messages(messages)
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._settings.temperature,
            },
        }
        if self._settings.max_tokens:
            payload["options"]["num_predict"] = self._settings.max_tokens  # type: ignore[index]

        url = f"{self._base_url}/api/chat"
        try:
            resp = requests.post(url, json=payload, timeout=60)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"[ollama] Connection failed to {self._base_url}: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"[ollama] Request timed out (url={url}): {e}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[ollama] HTTP request failed: {e}") from e

        if not resp.ok:
            raise RuntimeError(
                f"[ollama] API error (status={resp.status_code}): {resp.text[:200]}"
            )

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        model = data.get("model", self._settings.model)
        usage = None
        if "prompt_eval_count" in data or "eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
        return ChatResponse(content=content, model=model, usage=usage)

    def _validate_messages(self, messages: list[dict[str, str]]) -> None:
        if not messages:
            raise ValueError("[ollama] messages must not be empty")
        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"[ollama] message[{i}] must have 'role' and 'content' keys"
                )


register_provider("ollama", OllamaLLM)
