"""CrossEncoderReranker: reranks candidates using a cross-encoder scoring model."""
from __future__ import annotations

import concurrent.futures
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from libs.reranker.llm_reranker import RerankerFallbackError

if TYPE_CHECKING:
    from core.settings import RerankSettings

_DEFAULT_TIMEOUT = 30.0  # seconds


class CrossEncoderScorer(ABC):
    """Abstract interface for cross-encoder scoring backends."""

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score each (query, text) pair. Returns a list of floats, same length as texts."""


class SentenceTransformersCrossEncoderScorer(CrossEncoderScorer):
    """Scorer backed by sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            ) from e
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [[query, t] for t in texts]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


class CrossEncoderReranker(BaseReranker):
    """Reranker that scores query-candidate pairs via a cross-encoder model.

    Uses a thread-pool to enforce a per-call timeout. On timeout or any scorer
    failure, raises RerankerFallbackError so Core layer D6 can fall back to the
    fusion-ranked order.
    """

    def __init__(
        self,
        settings: "RerankSettings",
        scorer: CrossEncoderScorer | None = None,
        timeout: float | None = None,
    ) -> None:
        self._settings = settings
        self._timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT
        if scorer is not None:
            self._scorer = scorer
        else:
            model = settings.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self._scorer = SentenceTransformersCrossEncoderScorer(model)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        if not candidates:
            return []

        texts = [c.text for c in candidates]

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._scorer.score, query, texts)
                scores = future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as e:
            raise RerankerFallbackError(
                f"CrossEncoderReranker: scoring timed out after {self._timeout}s"
            ) from e
        except Exception as e:
            raise RerankerFallbackError(
                f"CrossEncoderReranker: scorer failed: {e}"
            ) from e

        if len(scores) != len(candidates):
            raise RerankerFallbackError(
                f"CrossEncoderReranker: scorer returned {len(scores)} scores "
                f"for {len(candidates)} candidates"
            )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [c for c, _ in ranked]
