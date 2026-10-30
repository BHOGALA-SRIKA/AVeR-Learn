"""Common interface shared by every lexical retriever.

Each retriever is fitted once on a list of paragraphs and then scores *every*
paragraph for a query. Scoring everything (instead of returning only a top-k)
is cheap for a course-sized corpus, and it is what Srika's selection
methods (Adaptive-k, HiChunk, EMTE) need, because they look at the whole
score distribution.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .records import Paragraph

# Bump this when the tokenizer or an index layout changes, so old saved
# indexes are rejected instead of silently giving different scores.
SAVE_FORMAT = "aver-lexical-1"


@dataclass(frozen=True)
class Hit:
    pid: str
    score: float
    rank: int


class Retriever(ABC):
    name: str = "retriever"

    def __init__(self) -> None:
        self.paragraphs: list[Paragraph] = []
        self.pids: list[str] = []

    def fit(self, paragraphs: list[Paragraph]) -> "Retriever":
        if not paragraphs:
            raise ValueError("Cannot fit a retriever on an empty corpus.")
        self.paragraphs = list(paragraphs)
        self.pids = [p.pid for p in self.paragraphs]
        self._fit(self.paragraphs)
        return self

    @abstractmethod
    def _fit(self, paragraphs: list[Paragraph]) -> None: ...

    @abstractmethod
    def score(self, query: str) -> np.ndarray:
        """Return one score per paragraph, in the order passed to `fit`."""

    def params(self) -> dict:
        """Parameters of this method, recorded with every saved index and run."""
        return {}

    def score_dict(self, query: str) -> dict[str, float]:
        return dict(zip(self.pids, map(float, self.score(query))))

    def candidates(self, query: str) -> np.ndarray:
        """Boolean mask of paragraphs that actually match the query.

        Default: a positive score. Methods whose scores are never zero (such as
        Query Likelihood) override this. Rank fusion uses it so that paragraphs
        sharing no words with the query do not get fusion credit.
        """
        return np.asarray(self.score(query)) > 0

    def save(self, path: str | Path) -> None:
        """Save the fitted retriever (index included) to one file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"format": SAVE_FORMAT, "retriever": self}, fh)

    @staticmethod
    def load(path: str | Path) -> "Retriever":
        """Load a retriever saved with `save`. Only load files your team made:
        pickle files can run code when loaded."""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("format") != SAVE_FORMAT:
            raise ValueError(f"{path} was saved with a different index format")
        return payload["retriever"]

    def search(self, query: str, k: int | None = 10) -> list[Hit]:
        """Rank paragraphs by score. Ties are broken by corpus order."""
        if not self.pids:
            raise RuntimeError(f"{self.name}: call fit() before search().")
        scores = np.asarray(self.score(query), dtype=np.float64)
        order = np.lexsort((np.arange(len(scores)), -scores))
        if k is not None:
            order = order[:k]
        return [Hit(self.pids[i], float(scores[i]), r) for r, i in enumerate(order, start=1)]
