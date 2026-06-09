"""SparseEncoder: compute BM25 term-frequency statistics for Chunks.

Produces a sparse_vector = {term: raw_tf_count} per Chunk, which is the
input contract expected by BM25Indexer (C11).  IDF and length-normalisation
are deferred to the indexer so that cross-document statistics remain correct.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import jieba

from core.types import Chunk, ChunkRecord

# ---------------------------------------------------------------------------
# Minimal stopword set (Chinese + English function words)
# ---------------------------------------------------------------------------
_ZH_STOPWORDS: frozenset[str] = frozenset(
    "的 了 是 在 和 有 我 他 她 它 们 这 那 个 一 不 也 都 与 为 以 及 等"
    " 中 上 下 对 于 可 但 到 就 从 被 将 让 由 其 所 之 此 该 并 已 则 因"
    " 而 或 如 若 只 还 着 过 又 再 些 很 更 最 能 会 应 该 还是 但是 所以"
    " 如果 虽然 因为 那么 然而".split()
)
_EN_STOPWORDS: frozenset[str] = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might must can could of in on at to "
    "for with by from as into through during about above below between "
    "and or but not no nor so yet both either neither than though "
    "this that these those it its i me my we our you your he him his "
    "she her they them their what which who whom when where why how "
    "all each every few more most other some such only own same so "
    "than too very just because if while".split()
)
_STOPWORDS = _ZH_STOPWORDS | _EN_STOPWORDS

# Minimum token length to keep
_MIN_TOKEN_LEN = 1

# Pattern matching purely numeric / punctuation tokens
_NOISE_RE = re.compile(r"^[\d\s\W]+$")


def _tokenize(text: str) -> list[str]:
    """Tokenize text with jieba for Chinese and whitespace-split for English."""
    text = text.strip()
    if not text:
        return []
    tokens: list[str] = []
    for tok in jieba.cut(text):
        tok = tok.strip().lower()
        if not tok:
            continue
        if len(tok) < _MIN_TOKEN_LEN:
            continue
        if _NOISE_RE.match(tok):
            continue
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


class SparseEncoder:
    """Compute BM25 term-frequency statistics for a list of Chunks.

    Returns one ChunkRecord per Chunk with sparse_vector populated as
    {term: raw_term_frequency_count}.  The counts are floats so they satisfy
    the Dict[str, float] contract on ChunkRecord.

    Empty text produces an empty sparse_vector (not None).
    """

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        """Compute sparse (BM25-TF) vectors for each chunk.

        Returns:
            One ChunkRecord per input Chunk, preserving order.
            sparse_vector is always set (empty dict for empty/stop-word-only text).
        """
        if not chunks:
            return []

        records: list[ChunkRecord] = []
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            tf: dict[str, float] = {term: float(cnt) for term, cnt in Counter(tokens).items()}
            record = ChunkRecord.from_chunk(chunk)
            record.sparse_vector = tf
            records.append(record)

        return records
