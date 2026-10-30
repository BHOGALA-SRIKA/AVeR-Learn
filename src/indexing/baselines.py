"""Lexical baselines: BM25 (bm25s, Lucene), TF-IDF, character TF-IDF, Query Likelihood.

Every class scores every paragraph for a query (see base.Retriever) and uses
the shared tokenizer in src/text/tokenize.py. Higher scores are better.

BM25Retriever (Lucene variant, via the `bm25s` library)
---------------------------------------------------------
    score(q, d) = sum over query terms t of  idf(t) * tf / (tf + k1 * (1 - b + b * len(d) / avg_len))
    idf(t)      = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

- tf: how often t appears in d; len(d): tokens in d; avg_len: corpus average
- k1: how fast repeated words stop adding score; b: how much long paragraphs are penalised
- The Lucene IDF never goes negative. In rank_bm25's BM25Okapi (the original
  repo code, kept in src/indexing/bm25_index.py as the reference baseline), a
  word in over half the paragraphs, like "system" in an OS chapter, gets a
  negative IDF replaced by a small floor, so it barely counts.
- fields=("body",) is plain BM25. fields=("chapter", "heading", "body") is the
  cheap "heading words added to the text" baseline; it is NOT BM25F.
- Raw scores, in [0, upper_bound(q)). Paragraphs sharing no word score 0.
- Dev grid: k1 in {0.9, 1.2, 1.5}, b in {0.5, 0.75, 0.9}.

TfidfRetriever / CharTfidfRetriever (scikit-learn)
----------------------------------------------------
- Word TF-IDF: shared tokenizer, sublinear tf (1 + log tf), cosine similarity.
- Character TF-IDF: 3-5 character n-grams inside word boundaries ("char_wb")
  over unstemmed, stopword-free words. Matches word variants and partial words
  stemming misses ("multiprogrammed" vs "multiprogramming"). Still vectorless
  by the team definition: corpus statistics only, no neural model.
- Raw cosine similarity in [0, 1].

QueryLikelihoodRetriever (Zhai & Lafferty 2001)
-------------------------------------------------
Ranks paragraphs by how likely their word distribution is to produce the query,
smoothed with the whole-corpus distribution p(t | C):

    Dirichlet: score(q, d) = sum over t of  log( (tf(t, d) + mu * p(t|C)) / (len(d) + mu) )
    JM:        score(q, d) = sum over t of  log( (1 - lam) * tf(t, d) / len(d) + lam * p(t|C) )

- Scores are log-probabilities: negative, higher is still better. Every
  paragraph gets a score; candidates() marks those containing a query word.
- Query words absent from the corpus are skipped (same constant for every
  paragraph, so they cannot change the ranking).
- Zhai & Lafferty found wordy queries need more smoothing; student questions
  are wordy, so tune on dev: mu in {50, 100, 250, 500, 1000, 2000};
  lam in {0.1, 0.3, 0.5, 0.7}.
"""

from __future__ import annotations

import math
from collections import Counter

import bm25s
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.text.tokenize import Tokenizer, tokenize

from .base import Retriever
from .records import Paragraph


def _joined(p: Paragraph, fields: tuple[str, ...]) -> str:
    return "\n".join(p.field_text(f) for f in fields)


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, fields: tuple[str, ...] = ("body",)) -> None:
        super().__init__()
        self.k1, self.b, self.fields = k1, b, tuple(fields)
        if self.fields != ("body",):
            self.name = "bm25[" + "+".join(self.fields) + "]"

    def params(self) -> dict:
        return {"k1": self.k1, "b": self.b, "fields": list(self.fields), "variant": "lucene"}

    def _fit(self, paragraphs: list[Paragraph]) -> None:
        corpus_tokens = [tokenize(_joined(p, self.fields)) for p in paragraphs]
        self._bm25 = bm25s.BM25(method="lucene", k1=self.k1, b=self.b)
        self._bm25.index(corpus_tokens, show_progress=False)
        n = len(corpus_tokens)
        df = Counter(t for doc in corpus_tokens for t in set(doc))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: str) -> np.ndarray:
        tokens = [t for t in tokenize(query) if t in self.idf]
        if not tokens:
            return np.zeros(len(self.pids))
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

    def score_weighted(self, weights: dict[str, float]) -> np.ndarray:
        """BM25 for a weighted query of already-tokenized terms: sum of w_t * score_t.

        BM25 adds up one contribution per query term, so this is exact. Used by RM3.
        """
        scores = np.zeros(len(self.pids))
        for t, w in weights.items():
            if t in self.idf and w:
                scores += w * np.asarray(self._bm25.get_scores([t]), dtype=np.float64)
        return scores

    def field_tokens(self, p: Paragraph) -> list[str]:
        return tokenize(_joined(p, self.fields))

    def upper_bound(self, query: str) -> float:
        """Largest score any paragraph could get: the sum of query-term IDFs.

        Each term contributes idf * tf / (tf + k1 * norm) < idf. Dividing a
        score by this gives the [0, 1] "absolute scale" EMTE uses (team Step 8,
        V2). The team document's bound has an extra (k1 + 1) factor; that
        belongs to the Robertson form of BM25 and is not needed for Lucene.
        """
        return float(sum(self.idf.get(t, 0.0) for t in tokenize(query)))


_unstemmed = Tokenizer(stem=False)


def _join_unstemmed(text: str) -> str:
    return " ".join(_unstemmed(text))


class _CosineRetriever(Retriever):
    fields: tuple[str, ...] = ("body",)

    def _make_vectorizer(self) -> TfidfVectorizer:
        raise NotImplementedError

    def _fit(self, paragraphs: list[Paragraph]) -> None:
        self._vec = self._make_vectorizer()
        self._matrix = self._vec.fit_transform([_joined(p, self.fields) for p in paragraphs])

    def score(self, query: str) -> np.ndarray:
        q = self._vec.transform([query])
        if q.nnz == 0:
            return np.zeros(len(self.pids))
        return np.asarray((self._matrix @ q.T).todense()).ravel()


class TfidfRetriever(_CosineRetriever):
    name = "tfidf"

    def __init__(self, fields: tuple[str, ...] = ("body",), sublinear_tf: bool = True) -> None:
        super().__init__()
        self.fields, self.sublinear_tf = tuple(fields), sublinear_tf

    def params(self) -> dict:
        return {"fields": list(self.fields), "sublinear_tf": self.sublinear_tf}

    def _make_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=self.sublinear_tf
        )


class CharTfidfRetriever(_CosineRetriever):
    name = "char_tfidf"

    def __init__(self, fields: tuple[str, ...] = ("body",), ngram_range: tuple[int, int] = (3, 5)) -> None:
        super().__init__()
        self.fields, self.ngram_range = tuple(fields), tuple(ngram_range)

    def params(self) -> dict:
        return {"fields": list(self.fields), "ngram_range": list(self.ngram_range)}

    def _make_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.ngram_range, preprocessor=_join_unstemmed, sublinear_tf=True
        )


class QueryLikelihoodRetriever(Retriever):
    name = "ql_dirichlet"

    def __init__(
        self,
        mu: float = 250.0,
        smoothing: str = "dirichlet",
        lam: float = 0.7,
        fields: tuple[str, ...] = ("body",),
    ) -> None:
        super().__init__()
        if smoothing not in {"dirichlet", "jm"}:
            raise ValueError("smoothing must be 'dirichlet' or 'jm'")
        if smoothing == "dirichlet" and mu <= 0:
            raise ValueError("mu must be positive")
        if smoothing == "jm" and not 0 < lam < 1:
            raise ValueError("lam must be between 0 and 1")
        self.mu, self.smoothing, self.lam, self.fields = mu, smoothing, lam, tuple(fields)
        if smoothing == "jm":
            self.name = "ql_jm"

    def params(self) -> dict:
        p = {"smoothing": self.smoothing, "fields": list(self.fields)}
        p.update({"mu": self.mu} if self.smoothing == "dirichlet" else {"lam": self.lam})
        return p

    def _fit(self, paragraphs: list[Paragraph]) -> None:
        vocab: dict[str, int] = {}
        rows, cols = [], []
        for i, p in enumerate(paragraphs):
            for t in tokenize(_joined(p, self.fields)):
                rows.append(i)
                cols.append(vocab.setdefault(t, len(vocab)))
        self.vocab = vocab
        shape = (len(paragraphs), max(len(vocab), 1))
        self._tf = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)  # duplicates summed
        self._len = np.asarray(self._tf.sum(axis=1), dtype=np.float64).ravel()
        cf = np.asarray(self._tf.sum(axis=0), dtype=np.float64).ravel()
        self._p_coll = cf / (cf.sum() or 1.0)

    def _query_terms(self, query: str) -> Counter[str]:
        return Counter(t for t in tokenize(query) if t in self.vocab)

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.pids))
        for t, c in self._query_terms(query).items():
            j = self.vocab[t]
            tf = self._tf[:, j].toarray().ravel()
            p = self._p_coll[j]
            if self.smoothing == "dirichlet":
                scores += c * (np.log(tf + self.mu * p) - np.log(self._len + self.mu))
            else:
                safe_len = np.where(self._len > 0, self._len, 1.0)
                scores += c * np.log((1 - self.lam) * tf / safe_len + self.lam * p)
        return scores

    def candidates(self, query: str) -> np.ndarray:
        terms = self._query_terms(query)
        if not terms:
            return np.zeros(len(self.pids), dtype=bool)
        cols = [self.vocab[t] for t in terms]
        return np.asarray(self._tf[:, cols].sum(axis=1)).ravel() > 0
