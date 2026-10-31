"""Formal BM25F: field-aware BM25 (Robertson, Zaragoza & Taylor 2004).

Each paragraph has three fields: body, heading (nearest heading) and chapter.
BM25F first combines a word's counts across the fields, each field with its
own weight and its own length normalisation, and only then applies the BM25
saturation once:

    B_f(d)     = 1 - b_f + b_f * len_f(d) / avg_len_f
    tf~(t, d)  = sum over fields f of  w_f * tf_f(t, d) / B_f(d)
    score(q,d) = sum over query terms t of  idf(t) * tf~(t, d) / (k1 + tf~(t, d))
    idf(t)     = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5)),  df over the whole paragraph

Why "combine first": if heading and body were scored separately and the
scores added, a word appearing in both would be counted as two independent
pieces of evidence and could never saturate. Combining first keeps one
saturation per word. Adding separate per-field scores (the current repo code
and the Implementation Spec's main formula) is a different, simpler model and
is useful as an ablation, not as BM25F.

With weights {"body": 1} only, this is exactly Lucene BM25 (tested).

Raw scores in [0, upper_bound(q)); higher is better; 0 = no query word in any field.
Dev grid: w_heading, w_chapter in {1, 2, 3, 5} (w_body = 1); b_f in {0.3, 0.5, 0.75};
k1 in {0.9, 1.2, 1.5}.

Scores are computed eagerly at fit time into a sparse paragraph x term matrix
(the same idea as BM25S), so a query is just a sum of a few columns.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .base import Retriever
from .records import FIELDS, Paragraph
from src.text.tokenize import tokenize

DEFAULT_WEIGHTS = {"body": 1.0, "heading": 2.0, "chapter": 1.0}
DEFAULT_B = {"body": 0.75, "heading": 0.5, "chapter": 0.5}


class BM25FRetriever(Retriever):
    name = "bm25f"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        b: dict[str, float] | None = None,
        k1: float = 1.2,
    ) -> None:
        super().__init__()
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.b = {**DEFAULT_B, **(b or {})}
        self.k1 = k1
        unknown = set(self.weights) - set(FIELDS)
        if unknown:
            raise ValueError(f"Unknown fields {unknown}; expected some of {FIELDS}")

    def params(self) -> dict:
        return {"k1": self.k1, "weights": self.weights, "b": {f: self.b[f] for f in self.weights}}

    def _fit(self, paragraphs: list[Paragraph]) -> None:
        fields = [f for f, w in self.weights.items() if w > 0]
        if not fields:
            raise ValueError("At least one field needs a positive weight.")

        field_tokens = {f: [tokenize(p.field_text(f)) for p in paragraphs] for f in fields}
        vocab: dict[str, int] = {}
        for docs in field_tokens.values():
            for doc in docs:
                for t in doc:
                    vocab.setdefault(t, len(vocab))
        self.vocab = vocab
        n_docs, n_terms = len(paragraphs), max(len(vocab), 1)

        combined = sparse.csr_matrix((n_docs, n_terms), dtype=np.float64)
        presence = sparse.csr_matrix((n_docs, n_terms), dtype=np.float64)
        for f in fields:
            tf = _count_matrix(field_tokens[f], vocab, n_terms)
            lengths = np.asarray(tf.sum(axis=1)).ravel()
            avg = lengths.mean() if lengths.mean() > 0 else 1.0
            norm = 1.0 - self.b[f] + self.b[f] * lengths / avg
            combined = combined + sparse.diags(self.weights[f] / norm) @ tf
            presence = presence + tf

        df = np.asarray((presence > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        # Eager scoring: replace each tf~ by idf(t) * tf~ / (k1 + tf~).
        scores = combined.tocsc()
        scores.data = scores.data / (self.k1 + scores.data)
        scores = scores @ sparse.diags(self.idf)
        self._scores = scores.tocsc()

    def score(self, query: str) -> np.ndarray:
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return np.zeros(len(self.pids))
        return np.asarray(self._scores[:, cols].sum(axis=1)).ravel()

    def score_weighted(self, weights: dict[str, float]) -> np.ndarray:
        """BM25F for a weighted query of already-tokenized terms (used by RM3)."""
        pairs = [(self.vocab[t], w) for t, w in weights.items() if t in self.vocab and w]
        if not pairs:
            return np.zeros(len(self.pids))
        cols, w = zip(*pairs)
        return np.asarray(self._scores[:, list(cols)] @ np.asarray(w, dtype=np.float64)).ravel()

    def field_tokens(self, p: Paragraph) -> list[str]:
        return [t for f, wf in self.weights.items() if wf > 0 for t in tokenize(p.field_text(f))]

    def upper_bound(self, query: str) -> float:
        """Sum of query-term IDFs: each term contributes idf * x with x < 1."""
        return float(sum(self.idf[self.vocab[t]] for t in tokenize(query) if t in self.vocab))


def _count_matrix(docs: list[list[str]], vocab: dict[str, int], n_terms: int) -> sparse.csr_matrix:
    rows, cols = [], []
    for i, doc in enumerate(docs):
        for t in doc:
            rows.append(i)
            cols.append(vocab[t])
    data = np.ones(len(rows), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(docs), n_terms))  # duplicates summed
