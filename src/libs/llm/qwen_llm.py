"""QwenLLM: DashScope Qwen via OpenAI-compatible endpoint (provider='qwen')."""
from __future__ import annotations

from libs.llm.llm_factory import register_provider
from libs.llm.openai_llm import OpenAILLM


class QwenLLM(OpenAILLM):
    """Qwen models accessed through DashScope's OpenAI-compatible API."""


register_provider("qwen", QwenLLM)
