"""Reciprocal Rank Fusion (RRF): combine several rankings into one.

Cormack, Clarke & Buettcher (SIGIR 2009):

    RRF(d) = sum over rankings r of  1 / (k + rank_r(d)),   k = 60

Only ranks are used, never raw scores, so methods with very different score
scales (BM25F around 0-10, Query Likelihood log-probabilities, TF-IDF cosine
0-1) can be combined without any normalisation or training. A paragraph only
gets credit from a ranking in which it actually matches the query.
"""

from __future__ import annotations

import numpy as np

from .base import Retriever
from .records import Paragraph


def rrf_scores(score_lists: list[np.ndarray], masks: list[np.ndarray], k: float = 60.0) -> np.ndarray:
    """Fuse per-method score arrays (same paragraph order) with RRF."""
    if not score_lists:
        raise ValueError("Need at least one ranking to fuse.")
    n = len(score_lists[0])
    fused = np.zeros(n)
    for scores, mask in zip(score_lists, masks):
        scores = np.asarray(scores, dtype=np.float64)
        order = np.lexsort((np.arange(n), -scores))  # rank 1 = best; ties by corpus order
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(1, n + 1)
        fused += np.where(mask, 1.0 / (k + ranks), 0.0)
    return fused


class RRFRetriever(Retriever):
    def __init__(self, retrievers: list[Retriever], k: float = 60.0) -> None:
        super().__init__()
        if not retrievers:
            raise ValueError("RRF needs at least one retriever.")
        self.retrievers, self.k = list(retrievers), k
        self.name = "rrf(" + ",".join(r.name for r in self.retrievers) + ")"

    def _fit(self, paragraphs: list[Paragraph]) -> None:
        for r in self.retrievers:
            r.fit(paragraphs)

    def score(self, query: str) -> np.ndarray:
        return rrf_scores(
            [r.score(query) for r in self.retrievers],
            [r.candidates(query) for r in self.retrievers],
            self.k,
        )
