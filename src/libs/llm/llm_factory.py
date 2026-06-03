"""LLMFactory: create BaseLLM and BaseVisionLLM instances from Settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

from libs.llm.base_llm import BaseLLM
from libs.llm.base_vision_llm import BaseVisionLLM

if TYPE_CHECKING:
    from core.settings import LLMSettings, VisionLLMSettings

# Provider registry: maps provider name → callable(settings) -> BaseLLM
_REGISTRY: dict[str, type[BaseLLM]] = {}

# Vision provider registry: maps provider name → callable(settings) -> BaseVisionLLM
_VISION_REGISTRY: dict[str, type[BaseVisionLLM]] = {}


def register_provider(name: str, cls: type[BaseLLM]) -> None:
    """Register a LLM implementation under a provider name."""
    _REGISTRY[name.lower()] = cls


def register_vision_provider(name: str, cls: type[BaseVisionLLM]) -> None:
    """Register a Vision LLM implementation under a provider name."""
    _VISION_REGISTRY[name.lower()] = cls


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

    @staticmethod
    def create_vision_llm(settings: "VisionLLMSettings") -> BaseVisionLLM:
        """Instantiate the correct BaseVisionLLM based on settings.provider.

        Raises:
            ValueError: If vision_llm is disabled, provider is empty, or unknown.
        """
        if not getattr(settings, "enabled", True):
            raise ValueError(
                "vision_llm.enabled is False — enable it in config/settings.yaml"
            )

        provider = (settings.provider or "").lower()
        if not provider:
            raise ValueError(
                "vision_llm.provider is empty — check config/settings.yaml"
            )

        cls = _VISION_REGISTRY.get(provider)
        if cls is None:
            known = sorted(_VISION_REGISTRY.keys())
            raise ValueError(
                f"Unknown Vision LLM provider: '{provider}'. "
                f"Registered providers: {known}"
            )
        return cls(settings)
