"""BaseEvaluator: abstract interface for all evaluator backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalInput:
    """Single evaluation sample."""
    query: str
    retrieved_ids: list[str]
    golden_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation metrics for one or more samples."""
    metrics: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """All evaluator implementations must subclass this."""

    @abstractmethod
    def evaluate(
        self,
        samples: list[EvalInput],
        trace: Any | None = None,
    ) -> EvalResult:
        """Compute metrics over a list of evaluation samples.

        Args:
            samples: List of EvalInput (query + retrieved_ids + golden_ids).
            trace: Optional TraceContext (Phase F). Ignored until F is implemented.

        Returns:
            EvalResult with aggregated metrics.
        """
