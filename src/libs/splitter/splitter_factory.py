"""SplitterFactory: create BaseSplitter instances from Settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

from libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from core.settings import IngestionSettings

# Provider registry: maps splitter type name → class(settings) -> BaseSplitter
_REGISTRY: dict[str, type[BaseSplitter]] = {}


def register_splitter(name: str, cls: type[BaseSplitter]) -> None:
    """Register a splitter implementation under a type name."""
    _REGISTRY[name.lower()] = cls


class SplitterFactory:
    @staticmethod
    def create(settings: "IngestionSettings") -> BaseSplitter:
        """Instantiate the correct BaseSplitter based on settings.splitter.

        Args:
            settings: IngestionSettings with a .splitter field (e.g. "recursive").

        Raises:
            ValueError: If the splitter type is unknown or empty.
        """
        splitter_type = (settings.splitter or "").lower()
        if not splitter_type:
            raise ValueError("ingestion.splitter is empty — check config/settings.yaml")

        cls = _REGISTRY.get(splitter_type)
        if cls is None:
            known = sorted(_REGISTRY.keys())
            raise ValueError(
                f"Unknown splitter type: '{splitter_type}'. "
                f"Registered types: {known}"
            )
        return cls(settings)
