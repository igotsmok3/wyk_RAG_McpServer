"""OllamaEmbedding: local Ollama HTTP backend for embeddings (provider='ollama')."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import requests

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import register_provider

if TYPE_CHECKING:
    from core.settings import EmbeddingSettings

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaEmbedding(BaseEmbedding):
    """Embedding backed by a local Ollama HTTP server (/api/embed endpoint)."""

    def __init__(self, settings: "EmbeddingSettings") -> None:
        self._settings = settings
        self._base_url = (settings.base_url or _DEFAULT_BASE_URL).rstrip("/")
        if not settings.model:
            raise ValueError("[ollama] model must not be empty — check config/settings.yaml")

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        if not texts:
            raise ValueError("[ollama] texts must not be empty")

        url = f"{self._base_url}/api/embed"
        payload = {
            "model": self._settings.model,
            "input": texts,
        }
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
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise RuntimeError(
                f"[ollama] Unexpected response format — 'embeddings' key missing: {str(data)[:200]}"
            )
        return embeddings


register_provider("ollama", OllamaEmbedding)
