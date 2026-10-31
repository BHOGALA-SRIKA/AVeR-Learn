"""Shared token counter for context budgets (owner: Hemanth, Member 2).

Budgets such as "1,024 evidence tokens" must be counted the same way by every
module: Bhavya's ``n_tokens`` / ``n_tokens_breadcrumb`` in the tree, Srika's
selectors and EMTE, and Sreeram's delivered-token metrics. This is the one
place that counting happens (Architecture and Interface Agreement, sections 5
and 14).

This is NOT the retrieval tokenizer (src/text/tokenize.py). That one removes
stopwords and stems words for matching; budgets must count every token the
answer-generating model will actually read.

Counters
--------
``approx-v1`` (default, no extra files or libraries)
    Counts word pieces the way subword tokenizers roughly do: each run of
    letters is one token, plus one more for every 8 letters beyond the first
    8 (long words are split by real tokenizers); every digit is one token
    (Qwen-family tokenizers split numbers into single digits); every
    punctuation mark or symbol is one token. Deterministic and fast, but an
    approximation of the generator's real count.

``hf:<path to tokenizer.json>`` (exact, once the generator is chosen)
    Uses the model's own tokenizer file through the ``tokenizers`` library
    (pip install tokenizers), fully offline. This only splits text into
    tokens; it runs no neural network. Switch to it when the team fixes the
    Phase 1 generator (e.g. Qwen3-4B), and rebuild all token counts.

Usage
-----
    from src.utils.token_counter import count_tokens, counter_name

    count_tokens("An interrupt transfers control to the ISR.")   # -> 10
    counter_name()                                               # -> "approx-v1"; record it in every run

    from src.utils.token_counter import use_counter, HFTokenizerCounter
    use_counter(HFTokenizerCounter("models/qwen3/tokenizer.json"))

Every experiment must record ``counter_name()``, because token budgets from
two different counters are not comparable.
"""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path

_PIECE = re.compile(r"[^\W\d_]+|\d|[^\w\s]|_", re.UNICODE)
_LONG_WORD = 8


class TokenCounter:
    name = "base"

    def count(self, text: str) -> int:
        raise NotImplementedError

    def __call__(self, text: str) -> int:
        return self.count(text)


class ApproxCounter(TokenCounter):
    """Library-free approximation of a subword tokenizer (see module docstring)."""

    name = "approx-v1"

    def count(self, text: str) -> int:
        text = unicodedata.normalize("NFKC", text or "")
        total = 0
        for piece in _PIECE.findall(text):
            if piece[0].isalpha():
                total += 1 + max(0, math.ceil((len(piece) - _LONG_WORD) / _LONG_WORD))
            else:
                total += 1
        return total


class HFTokenizerCounter(TokenCounter):
    """Exact counts with a model's tokenizer.json, via the `tokenizers` library."""

    def __init__(self, tokenizer_file: str | Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError("HFTokenizerCounter needs: pip install tokenizers") from exc
        path = Path(tokenizer_file)
        self._tok = Tokenizer.from_file(str(path))
        self.name = f"hf:{path.parent.name or path.stem}"

    def count(self, text: str) -> int:
        return len(self._tok.encode(text or "", add_special_tokens=False).ids)


_default: TokenCounter = ApproxCounter()


def use_counter(counter: TokenCounter) -> None:
    """Switch the shared counter for this process. Record counter_name() in results."""
    global _default
    _default = counter


def counter_name() -> str:
    return _default.name


def count_tokens(text: str) -> int:
    """Number of budget tokens in `text`, using the shared counter."""
    return _default.count(text)


def count_breadcrumb(breadcrumb: str | list[str] | None, sep: str = " > ") -> int:
    """Tokens of a breadcrumb as it will be shown to the generator.

    Accepts the agreed string form ("Unit 1 > Interrupts") or a list of parts.
    """
    if not breadcrumb:
        return 0
    text = sep.join(breadcrumb) if isinstance(breadcrumb, (list, tuple)) else str(breadcrumb)
    return count_tokens(text)
