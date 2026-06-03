"""RerankerFactory: create BaseReranker instances from Settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

from libs.reranker.base_reranker import BaseReranker, NoneReranker
from libs.reranker.llm_reranker import LLMReranker

if TYPE_CHECKING:
    from core.settings import RerankSettings

# Provider registry: maps backend name → class(settings) -> BaseReranker
_REGISTRY: dict[str, type[BaseReranker]] = {
    "none": NoneReranker,
    "llm": LLMReranker,
}


def register_backend(name: str, cls: type[BaseReranker]) -> None:
    """Register a reranker implementation under a backend name."""
    _REGISTRY[name.lower()] = cls


class RerankerFactory:
    @staticmethod
    def create(settings: "RerankSettings") -> BaseReranker:
        """Instantiate the correct BaseReranker based on settings.provider.

        Returns NoneReranker when reranking is disabled or provider is 'none'.

        Raises:
            ValueError: If the provider is unknown (and not 'none').
        """
        if not settings.enabled:
            return NoneReranker(settings)

        provider = (settings.provider or "none").lower()
        cls = _REGISTRY.get(provider)
        if cls is None:
            known = sorted(_REGISTRY.keys())
            raise ValueError(
                f"Unknown reranker backend: '{provider}'. "
                f"Registered backends: {known}"
            )
        return cls(settings)
