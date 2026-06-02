"""VectorStoreFactory: create BaseVectorStore instances from Settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

from libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from core.settings import VectorStoreSettings

# Provider registry: maps provider name → class(settings) -> BaseVectorStore
_REGISTRY: dict[str, type[BaseVectorStore]] = {}


def register_provider(name: str, cls: type[BaseVectorStore]) -> None:
    """Register a vector store implementation under a provider name."""
    _REGISTRY[name.lower()] = cls


class VectorStoreFactory:
    @staticmethod
    def create(settings: "VectorStoreSettings") -> BaseVectorStore:
        """Instantiate the correct BaseVectorStore based on settings.provider.

        Raises:
            ValueError: If the provider is unknown or empty.
        """
        provider = (settings.provider or "").lower()
        if not provider:
            raise ValueError("vector_store.provider is empty — check config/settings.yaml")

        cls = _REGISTRY.get(provider)
        if cls is None:
            known = sorted(_REGISTRY.keys())
            raise ValueError(
                f"Unknown vector store provider: '{provider}'. "
                f"Registered providers: {known}"
            )
        return cls(settings)
