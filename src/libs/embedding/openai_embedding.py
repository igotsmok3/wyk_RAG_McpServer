"""OpenAIEmbedding: OpenAI Embeddings API backend (provider='openai')."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import register_provider

if TYPE_CHECKING:
    from core.settings import EmbeddingSettings


class OpenAIEmbedding(BaseEmbedding):
    """Embedding backed by the OpenAI Embeddings API."""

    def __init__(self, settings: "EmbeddingSettings") -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url or None,
        )

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        if not texts:
            raise ValueError("[openai] texts must not be empty")
        try:
            resp = self._client.embeddings.create(
                model=self._settings.model,
                input=texts,
            )
        except APITimeoutError as e:
            raise TimeoutError(f"[openai] Request timed out: {e}") from e
        except APIConnectionError as e:
            raise ConnectionError(f"[openai] Connection failed: {e}") from e
        except APIError as e:
            raise RuntimeError(
                f"[openai] API error (status={e.status_code}): {e.message}"
            ) from e

        # API returns embeddings in index order
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


register_provider("openai", OpenAIEmbedding)
