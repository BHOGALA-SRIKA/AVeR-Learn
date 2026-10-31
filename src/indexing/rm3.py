"""RM3 pseudo-relevance feedback for BM25 and BM25F (Lavrenko & Croft 2001; Abdul-Jaleel et al. 2004).

Idea: assume the top few paragraphs of a first search are relevant, find the
words that are most typical of them, and search again with those words added
to the question. It needs no model and no glossary: the expansion words come
from the corpus itself.

Steps for a query q:

1. First search with the inner method (BM25 or BM25F). Keep the top
   ``fb_docs`` paragraphs that match.
2. Relevance model: P(t | R) = sum over those paragraphs d of
   P(t | d) * weight(d), with P(t | d) = tf(t, d) / len(d) and weight(d) the
   paragraph's first-search score divided by the sum of those scores.
3. Keep the ``fb_terms`` most likely words and renormalise them to sum to 1.
4. Mix with the original query (its words weighted equally, summing to 1):

       w(t) = orig_weight * P(t | q) + (1 - orig_weight) * P(t | R)

5. Second search with the weighted query. BM25 and BM25F add up one
   contribution per word, so a weighted query is exact: sum of w(t) * score_t.

Caution from the literature: early pseudo-relevance feedback failed on small
collections, and it can drift when the first search is wrong. On a 28-
paragraph pilot, fb_docs = 10 is a third of the corpus. Tune on dev questions
and report it as an ablation. Common defaults (Anserini): fb_docs = 10,
fb_terms = 10, orig_weight = 0.5. Dev grid: fb_docs in {3, 5, 10},
fb_terms in {5, 10, 20}, orig_weight in {0.3, 0.5, 0.7}.

Raw scores, higher is better, 0 when nothing matches. Scores are on a
different scale from plain BM25 (weights sum to 1), so do not mix them.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.text.tokenize import tokenize

from .base import Retriever
from .records import Paragraph


class RM3Retriever(Retriever):
    def __init__(
        self,
        inner: Retriever,
        fb_docs: int = 10,
        fb_terms: int = 10,
        orig_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if not hasattr(inner, "score_weighted") or not hasattr(inner, "field_tokens"):
            raise TypeError("RM3 needs a BM25 or BM25F retriever (weighted-query scoring).")
        if fb_docs < 1 or fb_terms < 1 or not 0 <= orig_weight <= 1:
            raise ValueError("Need fb_docs >= 1, fb_terms >= 1 and 0 <= orig_weight <= 1.")
        self.inner, self.fb_docs, self.fb_terms, self.orig_weight = inner, fb_docs, fb_terms, orig_weight
        self.name = inner.name + "+rm3"

    def params(self) -> dict:
        return {
            "inner": self.inner.name,
            **self.inner.params(),
            "fb_docs": self.fb_docs,
            "fb_terms": self.fb_terms,
            "orig_weight": self.orig_weight,
        }

    def _fit(self, paragraphs: list[Paragraph]) -> None:
        self.inner.fit(paragraphs)
        self._doc_tf = []
        for p in paragraphs:
            counts = Counter(self.inner.field_tokens(p))
            self._doc_tf.append((counts, sum(counts.values())))

    def query_weights(self, query: str) -> dict[str, float]:
        """The expanded, weighted query (useful for error analysis)."""
        q_counts = Counter(tokenize(query))
        if not q_counts:
            return {}
        first = self.inner.score(query)
        order = np.lexsort((np.arange(len(first)), -first))
        top = [i for i in order[: self.fb_docs] if first[i] > 0]
        q_total = sum(q_counts.values())
        weights = {t: self.orig_weight * c / q_total for t, c in q_counts.items()}
        if not top:
            return {t: c / q_total for t, c in q_counts.items()}

        doc_w = first[top] / first[top].sum()
        relevance: Counter[str] = Counter()
        for i, wd in zip(top, doc_w):
            counts, length = self._doc_tf[i]
            if length:
                for t, c in counts.items():
                    relevance[t] += wd * c / length
        best = sorted(relevance.items(), key=lambda kv: (-kv[1], kv[0]))[: self.fb_terms]
        total = sum(v for _, v in best) or 1.0
        for t, v in best:
            weights[t] = weights.get(t, 0.0) + (1 - self.orig_weight) * v / total
        return {t: float(w) for t, w in weights.items()}

    def score(self, query: str) -> np.ndarray:
        return self.inner.score_weighted(self.query_weights(query))
