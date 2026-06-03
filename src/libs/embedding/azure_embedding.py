"""AzureEmbedding: Azure OpenAI Embeddings backend (provider='azure')."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, APIError, APITimeoutError, AzureOpenAI

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import register_provider

if TYPE_CHECKING:
    from core.settings import EmbeddingSettings

_DEFAULT_API_VERSION = "2024-02-01"


class AzureEmbedding(BaseEmbedding):
    """Embedding backed by the Azure OpenAI Embeddings API."""

    def __init__(self, settings: "EmbeddingSettings") -> None:
        self._settings = settings
        if not settings.azure_endpoint:
            raise ValueError("[azure] azure_endpoint must be set in config")
        self._client = AzureOpenAI(
            api_key=settings.api_key,
            azure_endpoint=settings.azure_endpoint,
            api_version=settings.api_version or _DEFAULT_API_VERSION,
        )

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        if not texts:
            raise ValueError("[azure] texts must not be empty")
        deployment = self._settings.deployment_name or self._settings.model
        if not deployment:
            raise ValueError("[azure] deployment_name or model must be configured")
        try:
            resp = self._client.embeddings.create(
                model=deployment,
                input=texts,
            )
        except APIConnectionError as e:
            raise ConnectionError(f"[azure] Connection failed: {e}") from e
        except APITimeoutError as e:
            raise TimeoutError(f"[azure] Request timed out: {e}") from e
        except APIError as e:
            raise RuntimeError(
                f"[azure] API error (status={e.status_code}): {e.message}"
            ) from e

        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


register_provider("azure", AzureEmbedding)
