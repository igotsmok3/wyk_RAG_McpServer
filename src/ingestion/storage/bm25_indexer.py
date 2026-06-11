"""BM25Indexer: build and persist BM25 inverted index from SparseEncoder output.

Consumes ChunkRecords with sparse_vector = {term: raw_tf} produced by C9,
calculates IDF, builds an inverted index, and persists it to data/db/bm25/.

Supports full rebuild and incremental update.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.types import ChunkRecord

_INDEX_FILE = "index.json"
_META_FILE = "meta.json"

# Standard BM25 hyper-parameters
_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75


@dataclass
class BM25Result:
    """Single result returned by BM25Indexer.query()."""
    chunk_id: str
    score: float


@dataclass
class _IndexState:
    """In-memory representation of the BM25 inverted index."""
    # term -> list of {chunk_id, tf, doc_length}
    postings: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # term -> IDF (recomputed after each build/update)
    idf: Dict[str, float] = field(default_factory=dict)
    # chunk_id -> doc_length (number of tokens in the chunk)
    doc_lengths: Dict[str, int] = field(default_factory=dict)
    # total number of indexed documents
    doc_count: int = 0
    # average document length
    avg_dl: float = 0.0


class BM25Indexer:
    """Build and query a BM25 inverted index from SparseEncoder output.

    Typical usage::

        indexer = BM25Indexer()
        indexer.build(records)          # build from list[ChunkRecord]
        indexer.save()                  # persist to disk
        indexer.load()                  # restore from disk
        results = indexer.query({"检索": 2.0, "生成": 1.0}, top_k=5)
    """

    def __init__(
        self,
        index_dir: str = "data/db/bm25",
        k1: float = _DEFAULT_K1,
        b: float = _DEFAULT_B,
    ) -> None:
        self._index_dir = index_dir
        self._k1 = k1
        self._b = b
        self._state: Optional[_IndexState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, records: list[ChunkRecord], trace: Any = None) -> None:
        """Full rebuild: discard any prior state and index all records.

        Args:
            records: ChunkRecords with sparse_vector populated by SparseEncoder.
        """
        self._state = _IndexState()
        self._add_records(records, self._state)
        self._recompute_idf(self._state)

    def update(self, records: list[ChunkRecord], trace: Any = None) -> None:
        """Incremental update: append new records and recalculate global IDF.

        Records whose chunk_id already exists in the index are skipped to
        preserve idempotency.  A full IDF recalculation is performed after
        insertion so cross-document statistics remain accurate.

        Args:
            records: New ChunkRecords to add.
        """
        if self._state is None:
            self._state = _IndexState()

        existing_ids = set(self._state.doc_lengths.keys())
        new_records = [r for r in records if r.id not in existing_ids]
        if not new_records:
            return

        self._add_records(new_records, self._state)
        self._recompute_idf(self._state)

    def query(
        self,
        query_terms: Dict[str, float],
        top_k: int,
        trace: Any = None,
    ) -> list[BM25Result]:
        """Return top_k chunks ranked by BM25 score.

        Args:
            query_terms: {term: weight} dict (e.g. from SparseEncoder).
                         Only the term keys are used; query-side weights are
                         treated as 1.0 each (standard BM25 formulation).
            top_k:       Maximum results to return.

        Returns:
            List of BM25Result sorted by score descending (may be < top_k if
            the corpus is small).
        """
        if self._state is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        if not query_terms or top_k <= 0:
            return []

        scores: Dict[str, float] = {}
        avg_dl = self._state.avg_dl or 1.0

        for term in query_terms:
            if term not in self._state.postings:
                continue
            idf = self._state.idf.get(term, 0.0)
            for posting in self._state.postings[term]:
                cid = posting["chunk_id"]
                tf = posting["tf"]
                dl = posting["doc_length"]
                norm = tf * (self._k1 + 1) / (
                    tf + self._k1 * (1 - self._b + self._b * dl / avg_dl)
                )
                scores[cid] = scores.get(cid, 0.0) + idf * norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [BM25Result(chunk_id=cid, score=sc) for cid, sc in ranked[:top_k]]

    def save(self) -> None:
        """Persist the current index to index_dir."""
        if self._state is None:
            raise RuntimeError("Nothing to save. Call build() first.")
        os.makedirs(self._index_dir, exist_ok=True)
        index_path = os.path.join(self._index_dir, _INDEX_FILE)
        meta_path = os.path.join(self._index_dir, _META_FILE)

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({"postings": self._state.postings, "idf": self._state.idf}, f, ensure_ascii=False)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "doc_lengths": self._state.doc_lengths,
                    "doc_count": self._state.doc_count,
                    "avg_dl": self._state.avg_dl,
                },
                f,
                ensure_ascii=False,
            )

    def load(self) -> None:
        """Restore index from index_dir."""
        index_path = os.path.join(self._index_dir, _INDEX_FILE)
        meta_path = os.path.join(self._index_dir, _META_FILE)

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"BM25 index not found at {self._index_dir}. Call build() and save() first."
            )

        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._state = _IndexState(
            postings=idx["postings"],
            idf=idx["idf"],
            doc_lengths=meta["doc_lengths"],
            doc_count=meta["doc_count"],
            avg_dl=meta["avg_dl"],
        )

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    @property
    def doc_count(self) -> int:
        return self._state.doc_count if self._state else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_records(records: list[ChunkRecord], state: _IndexState) -> None:
        """Append records to postings and update doc_lengths / doc_count / avg_dl."""
        for record in records:
            sv = record.sparse_vector or {}
            doc_length = int(sum(sv.values()))  # total token count as proxy for length
            state.doc_lengths[record.id] = doc_length
            state.doc_count += 1

            for term, tf in sv.items():
                if term not in state.postings:
                    state.postings[term] = []
                state.postings[term].append(
                    {"chunk_id": record.id, "tf": tf, "doc_length": doc_length}
                )

        if state.doc_count > 0:
            state.avg_dl = sum(state.doc_lengths.values()) / state.doc_count

    @staticmethod
    def _recompute_idf(state: _IndexState) -> None:
        """Recalculate IDF for all terms using Robertson-Sparck Jones formula.

        IDF(term) = log((N - df + 0.5) / (df + 0.5))
        """
        N = state.doc_count
        state.idf = {}
        for term, postings in state.postings.items():
            df = len(postings)
            state.idf[term] = math.log((N - df + 0.5) / (df + 0.5))
