"""Tests for BaseEvaluator, CustomEvaluator, and EvaluatorFactory (B6)."""
import sys
import os
import pytest
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from libs.evaluator.base_evaluator import BaseEvaluator, EvalInput, EvalResult
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory, register_backend


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeEvalSettings:
    enabled: bool = True
    provider: str = "custom"
    metrics: list[str] = field(default_factory=lambda: ["hit_rate", "mrr"])


def _sample(retrieved: list[str], golden: list[str], query: str = "q") -> EvalInput:
    return EvalInput(query=query, retrieved_ids=retrieved, golden_ids=golden)


# ---------------------------------------------------------------------------
# BaseEvaluator abstract contract
# ---------------------------------------------------------------------------

def test_base_evaluator_is_abstract():
    with pytest.raises(TypeError):
        BaseEvaluator()  # type: ignore


# ---------------------------------------------------------------------------
# EvalInput / EvalResult data shape
# ---------------------------------------------------------------------------

def test_eval_input_shape():
    inp = EvalInput(
        query="what is RAG?",
        retrieved_ids=["a", "b", "c"],
        golden_ids=["b"],
        metadata={"collection": "test"},
    )
    assert inp.query == "what is RAG?"
    assert inp.retrieved_ids == ["a", "b", "c"]
    assert inp.golden_ids == ["b"]
    assert inp.metadata["collection"] == "test"


def test_eval_input_defaults():
    inp = EvalInput(query="q", retrieved_ids=[], golden_ids=[])
    assert inp.metadata == {}


def test_eval_result_shape():
    result = EvalResult(metrics={"hit_rate": 0.8, "mrr": 0.5}, sample_count=10)
    assert result.metrics["hit_rate"] == 0.8
    assert result.sample_count == 10
    assert result.metadata == {}


# ---------------------------------------------------------------------------
# CustomEvaluator: hit_rate
# ---------------------------------------------------------------------------

def test_hit_rate_hit():
    ev = CustomEvaluator(metrics=["hit_rate"])
    result = ev.evaluate([_sample(["a", "b", "c"], ["b"])])
    assert result.metrics["hit_rate"] == 1.0


def test_hit_rate_miss():
    ev = CustomEvaluator(metrics=["hit_rate"])
    result = ev.evaluate([_sample(["a", "b", "c"], ["z"])])
    assert result.metrics["hit_rate"] == 0.0


def test_hit_rate_multiple_golden():
    ev = CustomEvaluator(metrics=["hit_rate"])
    result = ev.evaluate([_sample(["a", "b"], ["x", "a"])])
    assert result.metrics["hit_rate"] == 1.0


def test_hit_rate_empty_retrieved():
    ev = CustomEvaluator(metrics=["hit_rate"])
    result = ev.evaluate([_sample([], ["a"])])
    assert result.metrics["hit_rate"] == 0.0


def test_hit_rate_averaged_over_samples():
    ev = CustomEvaluator(metrics=["hit_rate"])
    samples = [
        _sample(["a", "b"], ["a"]),   # hit
        _sample(["c", "d"], ["a"]),   # miss
    ]
    result = ev.evaluate(samples)
    assert result.metrics["hit_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CustomEvaluator: mrr
# ---------------------------------------------------------------------------

def test_mrr_first_position():
    ev = CustomEvaluator(metrics=["mrr"])
    result = ev.evaluate([_sample(["a", "b", "c"], ["a"])])
    assert result.metrics["mrr"] == pytest.approx(1.0)


def test_mrr_second_position():
    ev = CustomEvaluator(metrics=["mrr"])
    result = ev.evaluate([_sample(["a", "b", "c"], ["b"])])
    assert result.metrics["mrr"] == pytest.approx(0.5)


def test_mrr_third_position():
    ev = CustomEvaluator(metrics=["mrr"])
    result = ev.evaluate([_sample(["a", "b", "c"], ["c"])])
    assert result.metrics["mrr"] == pytest.approx(1.0 / 3)


def test_mrr_miss():
    ev = CustomEvaluator(metrics=["mrr"])
    result = ev.evaluate([_sample(["a", "b", "c"], ["z"])])
    assert result.metrics["mrr"] == pytest.approx(0.0)


def test_mrr_uses_first_hit_rank():
    ev = CustomEvaluator(metrics=["mrr"])
    # Both "b" (rank 2) and "c" (rank 3) are golden; first hit is rank 2
    result = ev.evaluate([_sample(["a", "b", "c"], ["b", "c"])])
    assert result.metrics["mrr"] == pytest.approx(0.5)


def test_mrr_averaged_over_samples():
    ev = CustomEvaluator(metrics=["mrr"])
    samples = [
        _sample(["a", "b"], ["a"]),   # rr = 1.0
        _sample(["a", "b"], ["b"]),   # rr = 0.5
    ]
    result = ev.evaluate(samples)
    assert result.metrics["mrr"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# CustomEvaluator: combined metrics
# ---------------------------------------------------------------------------

def test_combined_metrics():
    ev = CustomEvaluator()  # default: hit_rate + mrr
    result = ev.evaluate([_sample(["a", "b"], ["b"])])
    assert "hit_rate" in result.metrics
    assert "mrr" in result.metrics


def test_sample_count_reported():
    ev = CustomEvaluator()
    samples = [_sample(["a"], ["a"]), _sample(["b"], ["b"])]
    result = ev.evaluate(samples)
    assert result.sample_count == 2


# ---------------------------------------------------------------------------
# CustomEvaluator: edge cases
# ---------------------------------------------------------------------------

def test_empty_samples_returns_zeros():
    ev = CustomEvaluator()
    result = ev.evaluate([])
    assert result.metrics["hit_rate"] == 0.0
    assert result.metrics["mrr"] == 0.0
    assert result.sample_count == 0


def test_results_are_deterministic():
    ev = CustomEvaluator()
    samples = [
        _sample(["a", "b", "c"], ["b"]),
        _sample(["x", "y"], ["z"]),
    ]
    r1 = ev.evaluate(samples)
    r2 = ev.evaluate(samples)
    assert r1.metrics == r2.metrics


def test_unsupported_metric_raises():
    with pytest.raises(ValueError, match="faithfulness"):
        CustomEvaluator(metrics=["faithfulness"])


# ---------------------------------------------------------------------------
# EvaluatorFactory routing
# ---------------------------------------------------------------------------

def test_factory_creates_custom_evaluator():
    s = FakeEvalSettings(provider="custom")
    ev = EvaluatorFactory.create(s)
    assert isinstance(ev, CustomEvaluator)


def test_factory_provider_case_insensitive():
    s = FakeEvalSettings(provider="CUSTOM")
    ev = EvaluatorFactory.create(s)
    assert isinstance(ev, CustomEvaluator)


def test_factory_unknown_provider_raises():
    s = FakeEvalSettings(provider="nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        EvaluatorFactory.create(s)


def test_factory_unknown_provider_lists_known():
    s = FakeEvalSettings(provider="bad")
    with pytest.raises(ValueError, match="custom"):
        EvaluatorFactory.create(s)


def test_factory_filters_unsupported_metrics():
    """Settings may include 'faithfulness' (future Ragas metric); factory filters it."""
    s = FakeEvalSettings(provider="custom", metrics=["hit_rate", "mrr", "faithfulness"])
    ev = EvaluatorFactory.create(s)
    result = ev.evaluate([_sample(["a"], ["a"])])
    assert "hit_rate" in result.metrics
    assert "mrr" in result.metrics
    assert "faithfulness" not in result.metrics


def test_factory_registers_custom_backend():
    class FakeEvaluator(BaseEvaluator):
        def __init__(self, settings=None, metrics=None):
            pass
        def evaluate(self, samples, trace=None):
            return EvalResult(metrics={"fake": 1.0}, sample_count=len(samples))

    register_backend("fake", FakeEvaluator)
    s = FakeEvalSettings(provider="fake")
    ev = EvaluatorFactory.create(s)
    assert isinstance(ev, FakeEvaluator)


def test_factory_created_evaluator_produces_valid_results():
    s = FakeEvalSettings(provider="custom", metrics=["hit_rate", "mrr"])
    ev = EvaluatorFactory.create(s)
    samples = [
        EvalInput(query="test", retrieved_ids=["doc1", "doc2"], golden_ids=["doc2"]),
    ]
    result = ev.evaluate(samples)
    assert result.sample_count == 1
    assert result.metrics["hit_rate"] == pytest.approx(1.0)
    assert result.metrics["mrr"] == pytest.approx(0.5)
