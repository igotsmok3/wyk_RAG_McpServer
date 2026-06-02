"""LLMFactory: create BaseLLM instances from Settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

from libs.llm.base_llm import BaseLLM

if TYPE_CHECKING:
    from core.settings import LLMSettings

# Provider registry: maps provider name → callable(settings) -> BaseLLM
_REGISTRY: dict[str, type[BaseLLM]] = {}


def register_provider(name: str, cls: type[BaseLLM]) -> None:
    """Register a LLM implementation under a provider name."""
    _REGISTRY[name.lower()] = cls


class LLMFactory:
    @staticmethod
    def create(settings: "LLMSettings") -> BaseLLM:
        """Instantiate the correct BaseLLM based on settings.provider.

        Raises:
            ValueError: If the provider is unknown.
        """
        provider = (settings.provider or "").lower()
        if not provider:
            raise ValueError("llm.provider is empty — check config/settings.yaml")

        cls = _REGISTRY.get(provider)
        if cls is None:
            known = sorted(_REGISTRY.keys())
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Registered providers: {known}"
            )
        return cls(settings)
