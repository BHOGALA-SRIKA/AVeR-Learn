"""LexicalIndex: the Hemanth (Member 2) -> Srika (Member 3) contract (Interface Agreement, section 6).

    from src.indexing.lexical import LexicalIndex

    index = LexicalIndex(method="bm25f").build(paragraph_records)
    scores = index.score("what are interrupts")      # {"OS_U1_P001": 2.45, ...}
    index.save("indexes/bm25f.pkl")
    index = LexicalIndex().load("indexes/bm25f.pkl")

Score contract, for every method:

- ``score`` returns a score for EVERY paragraph passed to ``build``, in corpus
  order. Nothing is left uncomputed. Higher is always better.
- Scores are raw (not normalised). Ranges differ by method, so never add
  scores from two methods; use RRF (method "rrf") or normalise explicitly.
- A paragraph sharing no query word gets the method's "no match" value: 0 for
  BM25, BM25F and TF-IDF; a low but non-zero log-probability for Query
  Likelihood. ``matches(query)`` tells which paragraphs really match.
- Parameters, tokenizer version and paragraph count: ``index.info()``.

Methods (``METHODS``):

    bm25           BM25, Lucene variant (bm25s), paragraph text only
    bm25_headings  BM25 with heading and chapter words added to the text (simple baseline)
    bm25f          formal BM25F over body, heading and chapter fields
    tfidf          word TF-IDF, cosine
    char_tfidf     character 3-5-gram TF-IDF, cosine
    ql_dirichlet   Query Likelihood, Dirichlet smoothing
    ql_jm          Query Likelihood, Jelinek-Mercer smoothing
    rrf            Reciprocal Rank Fusion of bm25f + ql_dirichlet + char_tfidf
    bm25_rm3       BM25 + RM3 pseudo-relevance feedback (expansion words from top paragraphs)
    bm25f_rm3      BM25F + RM3 pseudo-relevance feedback

Any method can run with ``expand=True`` (OS glossary query expansion).
The original rank_bm25 implementation stays in src/indexing/bm25_index.py as
the reference baseline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from src.text.tokenize import TOKENIZER_VERSION

from .base import Hit, Retriever
from .baselines import BM25Retriever, CharTfidfRetriever, QueryLikelihoodRetriever, TfidfRetriever
from .bm25f import BM25FRetriever
from .expansion import ExpandedRetriever
from .fusion import RRFRetriever
from .records import Paragraph, read_paragraphs, to_paragraphs
from .rm3 import RM3Retriever

HEADING_FIELDS = ("chapter", "heading", "body")
INDEX_FORMAT = "aver-lexical-index-1"

METHODS: dict[str, Callable[..., Retriever]] = {
    "bm25": lambda **kw: BM25Retriever(**kw),
    "bm25_headings": lambda **kw: BM25Retriever(fields=HEADING_FIELDS, **kw),
    "bm25f": lambda **kw: BM25FRetriever(**kw),
    "tfidf": lambda **kw: TfidfRetriever(**kw),
    "char_tfidf": lambda **kw: CharTfidfRetriever(**kw),
    "ql_dirichlet": lambda **kw: QueryLikelihoodRetriever(smoothing="dirichlet", **kw),
    "ql_jm": lambda **kw: QueryLikelihoodRetriever(smoothing="jm", **kw),
    "rrf": lambda **kw: RRFRetriever(
        [
            BM25FRetriever(),
            QueryLikelihoodRetriever(fields=HEADING_FIELDS),
            CharTfidfRetriever(fields=HEADING_FIELDS),
        ],
        **kw,
    ),
    "bm25_rm3": lambda fb_docs=10, fb_terms=10, orig_weight=0.5, **kw: RM3Retriever(
        BM25Retriever(**kw), fb_docs=fb_docs, fb_terms=fb_terms, orig_weight=orig_weight
    ),
    "bm25f_rm3": lambda fb_docs=10, fb_terms=10, orig_weight=0.5, **kw: RM3Retriever(
        BM25FRetriever(**kw), fb_docs=fb_docs, fb_terms=fb_terms, orig_weight=orig_weight
    ),
}


def build_retriever(method: str, expand: bool = False, **params: Any) -> Retriever:
    if method not in METHODS:
        raise KeyError(f"Unknown method {method!r}. Choose from: {', '.join(METHODS)}")
    retriever = METHODS[method](**params)
    return ExpandedRetriever(retriever) if expand else retriever


class LexicalIndex:
    def __init__(self, method: str = "bm25f", expand: bool = False, **params: Any) -> None:
        if method not in METHODS:
            raise KeyError(f"Unknown method {method!r}. Choose from: {', '.join(METHODS)}")
        self.method, self.expand, self.params = method, expand, params
        self._retriever: Retriever | None = None
        self._nodes: dict[str, dict[str, Any]] = {}
        self._level_index: dict[int, tuple[Retriever, dict[str, int]]] = {}

    # ------------------------------------------------------------- building

    def _new_retriever(self) -> Retriever:
        return build_retriever(self.method, self.expand, **self.params)

    def build(self, paragraphs: Iterable[Any] | str | Path, tree: Any = None) -> "LexicalIndex":
        """Build the index.

        paragraphs: paragraph records (dicts in the agreed schema), Paragraph
            objects, or a path to paragraphs.jsonl / paragraphs.json.
        tree: optional, needed only for score_nodes on non-paragraph nodes.
            A list of node dicts, a dict id -> node, or a dict/object with "nodes".
        """
        if isinstance(paragraphs, (str, Path)):
            items = read_paragraphs(paragraphs)
        else:
            items = to_paragraphs(paragraphs)
        self._retriever = self._new_retriever().fit(items)
        self._nodes = _node_map(tree) if tree is not None else {}
        self._level_index = {}
        return self

    def _require(self) -> Retriever:
        if self._retriever is None:
            raise RuntimeError("Call build() or load() first.")
        return self._retriever

    # -------------------------------------------------------------- scoring

    @property
    def paragraph_ids(self) -> list[str]:
        return list(self._require().pids)

    def score(self, query: str) -> dict[str, float]:
        """Paragraph ID -> raw score for every paragraph. Higher is better."""
        return self._require().score_dict(query)

    def scores_array(self, query: str) -> np.ndarray:
        """Same scores as a NumPy array in ``paragraph_ids`` order."""
        return np.asarray(self._require().score(query), dtype=np.float64)

    def search(self, query: str, k: int | None = 10) -> list[Hit]:
        """Ranked paragraphs (ties broken by corpus order)."""
        return self._require().search(query, k)

    def matches(self, query: str) -> dict[str, bool]:
        """Paragraph ID -> whether it actually matches the query."""
        r = self._require()
        return dict(zip(r.pids, map(bool, r.candidates(query))))

    def upper_bound(self, query: str) -> float:
        """Upper bound on any paragraph's score (BM25 and BM25F only), for
        EMTE's absolute scale: score / upper_bound(query) is in [0, 1)."""
        r = self._require()
        inner = r.inner if isinstance(r, ExpandedRetriever) else r
        if not hasattr(inner, "upper_bound"):
            raise NotImplementedError(f"{self.method} has no score upper bound")
        q = r.glossary.expand(query) if isinstance(r, ExpandedRetriever) else query
        return inner.upper_bound(q)

    def score_nodes(self, query: str, node_ids: list[str]) -> dict[str, float]:
        """Scores for any nodes: paragraphs, subsections, sections, chapters.

        Paragraph IDs get their paragraph score. A non-paragraph node is scored
        as one document made of its title plus all paragraph text below it,
        against the other nodes of the SAME level (so IDF and length
        normalisation are computed per level, as INEX-style element retrieval
        and section-then-paragraph need). Compare scores within a level only.
        Requires ``tree`` to have been passed to ``build``.
        """
        para = self.score(query)
        out: dict[str, float] = {}
        by_level: dict[int, list[str]] = {}
        for nid in node_ids:
            if nid in para:
                out[nid] = para[nid]
                continue
            node = self._nodes.get(nid)
            if node is None:
                raise KeyError(
                    f"Unknown node id {nid!r}. Pass the document tree to build() to score non-paragraph nodes."
                )
            by_level.setdefault(int(node.get("level", -1)), []).append(nid)
        for level, ids in by_level.items():
            retriever, position = self._level_retriever(level)
            scores = retriever.score(query)
            out.update({nid: float(scores[position[nid]]) for nid in ids})
        return {nid: out[nid] for nid in node_ids}

    def _level_retriever(self, level: int) -> tuple[Retriever, dict[str, int]]:
        if level not in self._level_index:
            level_nodes = [n for n in self._nodes.values() if int(n.get("level", -1)) == level]
            docs = []
            for n in level_nodes:
                body = "\n".join(_leaf_texts(self._nodes, n))
                title = str(n.get("title") or "")
                crumbs = [c.strip() for c in str(n.get("breadcrumb") or "").split(">") if c.strip()]
                docs.append(
                    Paragraph(
                        pid=str(n["id"]),
                        body=body or title,
                        heading=title,
                        chapter=crumbs[0] if crumbs else title,
                    )
                )
            retriever = self._new_retriever().fit(docs)
            self._level_index[level] = (retriever, {d.pid: i for i, d in enumerate(docs)})
        return self._level_index[level]

    # ---------------------------------------------------------- persistence

    def info(self) -> dict[str, Any]:
        r = self._require()
        inner = r.inner if isinstance(r, ExpandedRetriever) else r
        return {
            "method": self.method,
            "expand": self.expand,
            "params": {**inner.params(), **self.params},
            "tokenizer_version": TOKENIZER_VERSION,
            "n_paragraphs": len(r.pids),
            "score_direction": "higher is better",
            "score_scale": "raw",
        }

    def save(self, path: str | Path) -> None:
        """Save the built index to one file (pickle). Only load files your team made."""
        self._require()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"format": INDEX_FORMAT, "tokenizer_version": TOKENIZER_VERSION, "state": self.__dict__}, fh)

    def load(self, path: str | Path) -> "LexicalIndex":
        """Load a saved index into this object and return it."""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("format") != INDEX_FORMAT:
            raise ValueError(f"{path} is not a lexical index of format {INDEX_FORMAT}")
        if payload.get("tokenizer_version") != TOKENIZER_VERSION:
            raise ValueError(
                f"{path} was built with tokenizer {payload.get('tokenizer_version')!r}; "
                f"current is {TOKENIZER_VERSION!r}. Rebuild the index."
            )
        self.__dict__.update(payload["state"])
        return self


def _node_map(tree: Any) -> dict[str, dict[str, Any]]:
    nodes = getattr(tree, "nodes", tree)
    if isinstance(nodes, dict) and "nodes" in nodes:
        nodes = nodes["nodes"]
    if isinstance(nodes, dict):
        values = list(nodes.values())
    else:
        values = list(nodes)
    return {str(n["id"]): dict(n) for n in values}


def _leaf_texts(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> list[str]:
    """Paragraph texts below a node, in document order."""
    out: list[tuple[int, str]] = []
    stack = [node]
    while stack:
        n = stack.pop()
        children = [nodes[c] for c in n.get("children") or [] if c in nodes]
        if not children:
            if n.get("text"):
                out.append((int(n.get("doc_order", 0)), str(n["text"])))
        stack.extend(children)
    return [t for _, t in sorted(out)]
