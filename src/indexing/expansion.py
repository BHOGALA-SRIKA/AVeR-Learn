"""Glossary-based query expansion: close the gap between student words and book words.

BM25 only matches words that literally appear. A student may write "DMA"
while the book says "direct memory access", or "software interrupt" while the
notes say "trap". A small course glossary fixes this without any neural model,
fully offline:

- abbreviations: "dma" -> also search "direct memory access", and the reverse
- synonyms:      "software interrupt" -> also search "trap exception"

The glossary (data/glossary/os_glossary.json) is FROZEN before the final test and
holds only textbook terms. Entries written in students' own words ("switched on")
were removed after review: some matched smoke-test questions word for word.

The original query is always kept; expansion words are only added. Whether
expansion helps must be measured on dev questions (it can also add noise),
so it is a wrapper that any retriever can be run with or without.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import Retriever
from .records import Paragraph
from src.text.tokenize import normalize

DEFAULT_GLOSSARY = Path(__file__).resolve().parents[2] / "data" / "glossary" / "os_glossary.json"


class Glossary:
    def __init__(self, abbreviations: dict[str, str], synonyms: dict[str, list[str]]):
        self.rules: list[tuple[re.Pattern[str], list[str]]] = []
        for short, long in abbreviations.items():
            self._add(short, [long])
            self._add(long, [short])
        for phrase, extra in synonyms.items():
            self._add(phrase, list(extra))

    def _add(self, phrase: str, extra: list[str]) -> None:
        phrase = normalize(phrase).strip()
        if phrase:
            pattern = re.compile(r"(?<![\w/])" + re.escape(phrase) + r"(?![\w/])")
            self.rules.append((pattern, extra))

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_GLOSSARY) -> "Glossary":
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data.get("abbreviations", {}), data.get("synonyms", {}))

    def expansions(self, query: str) -> list[str]:
        text = normalize(query)
        added: list[str] = []
        for pattern, extra in self.rules:
            if pattern.search(text):
                added.extend(e for e in extra if e not in added)
        return added

    def expand(self, query: str) -> str:
        extra = self.expansions(query)
        return query if not extra else query + " " + " ".join(extra)


@lru_cache(maxsize=4)
def load_glossary(path: str = str(DEFAULT_GLOSSARY)) -> Glossary:
    return Glossary.from_file(path)


class ExpandedRetriever(Retriever):
    """Run any retriever on the glossary-expanded query."""

    def __init__(self, inner: Retriever, glossary: Glossary | None = None) -> None:
        super().__init__()
        self.inner = inner
        self.glossary = glossary or load_glossary()
        self.name = inner.name + "+exp"

    def _fit(self, paragraphs: list[Paragraph]) -> None:
        self.inner.fit(paragraphs)

    def score(self, query: str) -> np.ndarray:
        return self.inner.score(self.glossary.expand(query))

    def candidates(self, query: str) -> np.ndarray:
        return self.inner.candidates(self.glossary.expand(query))
