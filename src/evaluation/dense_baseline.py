"""Embedding (dense) retrieval: the comparison point for the vectorless methods (owner: Member 4).

The project claims that lexical, vectorless retrieval is good enough while
being lighter. To test that claim fairly we need the usual alternative: a
small sentence-embedding model.

- ``dense``:  BAAI/bge-small-en-v1.5 (33M parameters, 384 dimensions, ~64 MB),
  run on the CPU through ONNX Runtime (fastembed). Each paragraph (heading +
  text) becomes a vector; a question is scored by cosine similarity.
- ``hybrid``: Reciprocal Rank Fusion of BM25 and dense (Hemanth's RRF code),
  the standard "best of both" baseline.

The model is downloaded once into ``models/fastembed`` and then loaded with
the Hugging Face hub set to offline, so evaluation never touches the network.

Building a dense index includes loading the model, because a student opening
a PDF pays that cost too.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.indexing.base import Hit, Retriever
from src.indexing.baselines import BM25Retriever
from src.indexing.fusion import RRFRetriever
from src.indexing.lexical import LexicalIndex
from src.indexing.records import Paragraph, read_paragraphs, to_paragraphs

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "fastembed"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_METHODS = ("dense", "hybrid")


def model_bytes(model_dir: Path = MODEL_DIR) -> int:
    return sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) if model_dir.exists() else 0


class DenseRetriever(Retriever):
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        super().__init__()
        self.model_name = model
        self.name = "dense"

    def params(self) -> dict:
        return {"model": self.model_name}

    def _fit(self, paragraphs: list[Paragraph]) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download during an experiment
        from fastembed import TextEmbedding

        self._model = TextEmbedding(self.model_name, cache_dir=str(MODEL_DIR))
        texts = [f"{p.heading}. {p.body}" if p.heading else p.body for p in paragraphs]
        self.vectors = _unit(np.array(list(self._model.passage_embed(texts))))

    def score(self, query: str) -> np.ndarray:
        q = _unit(np.array(list(self._model.query_embed(query))))[0]
        return self.vectors @ q

    def candidates(self, query: str) -> np.ndarray:
        return np.ones(len(self.pids), dtype=bool)  # every paragraph has a similarity


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


class DenseIndex:
    """Same calls as LexicalIndex (build / search / save), so the harness treats both alike."""

    def __init__(self, method: str = "dense", **params: Any) -> None:
        if method not in DENSE_METHODS:
            raise KeyError(f"Unknown dense method {method!r}")
        self.method, self.params = method, params

    def build(self, paragraphs: Iterable[Any] | str | Path) -> "DenseIndex":
        items = read_paragraphs(paragraphs) if isinstance(paragraphs, (str, Path)) else to_paragraphs(paragraphs)
        dense = DenseRetriever(**self.params)
        self._retriever = dense if self.method == "dense" else RRFRetriever([BM25Retriever(), dense])
        self._dense = dense
        self._retriever.fit(items)
        return self

    def search(self, query: str, k: int | None = 10) -> list[Hit]:
        return self._retriever.search(query, k)

    def save(self, path: str | Path) -> None:
        """Saves what must be stored per PDF: the paragraph vectors (and BM25 for hybrid).
        The model file is shared by all PDFs and reported separately (model_bytes)."""
        with Path(path).open("wb") as fh:
            pickle.dump({"pids": self._dense.pids, "vectors": self._dense.vectors.astype(np.float32),
                         "bm25": self._retriever.retrievers[0] if self.method == "hybrid" else None}, fh)


def make_index(method: str, expand: bool = False, **params: Any):
    """LexicalIndex for Hemanth's methods, DenseIndex for the embedding baselines."""
    if method in DENSE_METHODS:
        if expand:
            raise ValueError("Glossary expansion is a lexical technique; run dense methods without it.")
        return DenseIndex(method, **params)
    return LexicalIndex(method, expand=expand, **params)
