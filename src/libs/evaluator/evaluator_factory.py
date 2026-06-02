"""EvaluatorFactory: create BaseEvaluator instances from Settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    from core.settings import EvaluationSettings

# Provider registry: maps backend name → class
_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "custom": CustomEvaluator,
}


def register_backend(name: str, cls: type[BaseEvaluator]) -> None:
    """Register an evaluator implementation under a provider name."""
    _REGISTRY[name.lower()] = cls


class EvaluatorFactory:
    @staticmethod
    def create(settings: "EvaluationSettings") -> BaseEvaluator:
        """Instantiate the correct BaseEvaluator based on settings.provider.

        Raises:
            ValueError: If the provider is unknown.
        """
        provider = (settings.provider or "custom").lower()
        cls = _REGISTRY.get(provider)
        if cls is None:
            known = sorted(_REGISTRY.keys())
            raise ValueError(
                f"Unknown evaluator provider: '{provider}'. "
                f"Registered providers: {known}"
            )
        metrics = getattr(settings, "metrics", None) or []
        # Filter to only metrics supported by CustomEvaluator
        if cls is CustomEvaluator:
            supported = CustomEvaluator.SUPPORTED_METRICS
            metrics = [m for m in metrics if m in supported] or None
        return cls(settings=settings, metrics=metrics if metrics else None)
