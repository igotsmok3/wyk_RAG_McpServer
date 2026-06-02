"""CustomEvaluator: lightweight hit_rate and MRR metrics."""
from __future__ import annotations

from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator, EvalInput, EvalResult


class CustomEvaluator(BaseEvaluator):
    """Compute hit_rate and MRR from retrieved_ids vs golden_ids."""

    SUPPORTED_METRICS = {"hit_rate", "mrr"}

    def __init__(self, settings: Any = None, metrics: list[str] | None = None) -> None:
        if metrics is None:
            self._metrics = ["hit_rate", "mrr"]
        else:
            unknown = set(metrics) - self.SUPPORTED_METRICS
            if unknown:
                raise ValueError(
                    f"Unsupported metrics: {sorted(unknown)}. "
                    f"Supported: {sorted(self.SUPPORTED_METRICS)}"
                )
            self._metrics = list(metrics)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        samples: list[EvalInput],
        trace: Any | None = None,
    ) -> EvalResult:
        if not samples:
            return EvalResult(metrics={m: 0.0 for m in self._metrics}, sample_count=0)

        aggregated: dict[str, float] = {}
        if "hit_rate" in self._metrics:
            aggregated["hit_rate"] = self._mean(
                self._hit_rate(s) for s in samples
            )
        if "mrr" in self._metrics:
            aggregated["mrr"] = self._mean(
                self._reciprocal_rank(s) for s in samples
            )

        return EvalResult(metrics=aggregated, sample_count=len(samples))

    # ------------------------------------------------------------------
    # Per-sample helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_rate(sample: EvalInput) -> float:
        """1.0 if any golden_id appears in retrieved_ids, else 0.0."""
        golden = set(sample.golden_ids)
        return 1.0 if any(rid in golden for rid in sample.retrieved_ids) else 0.0

    @staticmethod
    def _reciprocal_rank(sample: EvalInput) -> float:
        """1/rank of the first golden_id in retrieved_ids, or 0.0."""
        golden = set(sample.golden_ids)
        for rank, rid in enumerate(sample.retrieved_ids, start=1):
            if rid in golden:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _mean(values) -> float:
        total = 0.0
        count = 0
        for v in values:
            total += v
            count += 1
        return total / count if count > 0 else 0.0
