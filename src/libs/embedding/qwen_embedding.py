"""QwenEmbedding: DashScope Qwen via OpenAI-compatible endpoint (provider='qwen')."""
from __future__ import annotations

from libs.embedding.embedding_factory import register_provider
from libs.embedding.openai_embedding import OpenAIEmbedding


class QwenEmbedding(OpenAIEmbedding):
    """Qwen embedding models via DashScope's OpenAI-compatible API."""


register_provider("qwen", QwenEmbedding)
