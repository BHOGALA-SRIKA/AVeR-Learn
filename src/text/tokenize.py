"""Shared tokenizer for every lexical retrieval method.

Every method (BM25, BM25F, TF-IDF, Query Likelihood) must use the same
tokenizer, so that differences in results come from the scoring method and
not from text processing. Queries and paragraphs go through exactly the same
steps:

1. Unicode NFKC normalisation and lowercasing.
2. Rejoin words hyphenated across PDF line breaks ("inter-\nrupt" -> "interrupt").
3. Split into word tokens. Slash compounds such as "i/o" are kept whole,
   because "I/O" is a key operating-systems term.
4. Remove stopwords: the Lucene English list plus question words such as
   "what", "why", "how", which appear in student questions but never help
   find the answer.
5. Stem with the Snowball English stemmer ("scheduling" -> "schedul",
   "interrupts" -> "interrupt").

Owner: Hemanth (Member 2). Interface (Architecture and Interface Agreement, section 5):
``tokenize(text: str) -> list[str]``. This is for matching only; token
budgets use the separate shared token counter (src/utils/token_counter.py).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import Stemmer

# Lucene's default English stopword set (the list Elasticsearch/Lucene use).
LUCENE_STOPWORDS = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with""".split()
)

# Words that appear in student questions but carry no topic information.
QUESTION_STOPWORDS = frozenset(
    """what which who whom whose when where why how
    do does did done can could would should shall may might must
    explain describe define definition meant mean means
    please tell give list briefly short brief detail""".split()
)

DEFAULT_STOPWORDS = LUCENE_STOPWORDS | QUESTION_STOPWORDS

# Change this whenever any rule below changes, then rebuild every saved index.
# Freeze it before the final benchmark run (Interface Agreement, section 5).
TOKENIZER_VERSION = "1.0: nfkc, lowercase, hyphen-join, i/o-kept, lucene+question stopwords, snowball-english"

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
# A token is letters/digits, optionally joined by "/" (keeps "i/o", "tcp/ip").
_TOKEN = re.compile(r"[^\W_]+(?:/[^\W_]+)*")


def normalize(text: str) -> str:
    """NFKC-normalise, lowercase and rejoin hyphenated line breaks."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return text.lower()


class Tokenizer:
    """Configurable tokenizer. Use `tokenize()` below for the shared default."""

    def __init__(
        self,
        stopwords: frozenset[str] | None = DEFAULT_STOPWORDS,
        stem: bool = True,
        language: str = "english",
    ) -> None:
        self.stopwords = stopwords or frozenset()
        self.stem = stem
        self._stemmer = Stemmer.Stemmer(language) if stem else None

    def __call__(self, text: str) -> list[str]:
        words = _TOKEN.findall(normalize(text))
        words = [w for w in words if w not in self.stopwords]
        if self._stemmer is not None:
            words = self._stemmer.stemWords(words)
        return words


@lru_cache(maxsize=1)
def default_tokenizer() -> Tokenizer:
    return Tokenizer()


def tokenize(text: str) -> list[str]:
    """Tokenize with the shared default settings. Use this everywhere."""
    return default_tokenizer()(text)
