"""EmbeddingFactory: create BaseEmbedding instances from Settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.settings import EmbeddingSettings

# Provider registry: maps provider name → class(settings) -> BaseEmbedding
_REGISTRY: dict[str, type[BaseEmbedding]] = {}


def register_provider(name: str, cls: type[BaseEmbedding]) -> None:
    """Register an embedding implementation under a provider name."""
    _REGISTRY[name.lower()] = cls


class EmbeddingFactory:
    @staticmethod
    def create(settings: "EmbeddingSettings") -> BaseEmbedding:
        """Instantiate the correct BaseEmbedding based on settings.provider.

        Raises:
            ValueError: If the provider is unknown or empty.
        """
        provider = (settings.provider or "").lower()
        if not provider:
            raise ValueError("embedding.provider is empty — check config/settings.yaml")

        cls = _REGISTRY.get(provider)
        if cls is None:
            known = sorted(_REGISTRY.keys())
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. "
                f"Registered providers: {known}"
            )
        return cls(settings)
