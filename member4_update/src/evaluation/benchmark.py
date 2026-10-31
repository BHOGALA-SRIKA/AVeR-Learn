"""Benchmark questions, gold-quote matching and evidence coverage (owner: Member 4).

Benchmark format (data/benchmark/questions.json): a list of questions, each with
``qid``, ``question``, ``split`` ("dev" or "test"), ``out_of_scope`` and
``gold_quotes``: exact text copied from the PDF, each with a grade
(2 = essential, 1 = helpful) and ``source_pdf``.

Why quotes instead of paragraph IDs: paragraph IDs change whenever the PDF
loader or the chunk size changes, which would silently break every judgement.
Quotes are matched to whatever paragraphs the current pipeline produces, so
the same benchmark can compare different chunkings fairly.

Matching rule: a paragraph "contains" part of a quote when they share runs of
at least ``MIN_RUN`` consecutive words. It is relevant to the quote when those
runs cover at least half the quote, or at least ``MIN_WORDS`` words (a long
quote split across two chunks makes both chunks relevant). The paragraph's
grade is the highest grade of the quotes it contains.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.indexing.records import Paragraph

MIN_RUN = 3
MIN_WORDS = 8
MIN_FRACTION = 0.5
COVERED = 0.8  # a quote counts as delivered when 80% of its words are in the delivered text

_WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    """Lowercased words, with PDF ligatures and odd Unicode normalised."""
    return _WORD.findall(unicodedata.normalize("NFKC", text or "").lower())


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        questions = json.load(fh)
    seen: set[str] = set()
    for q in questions:
        if q["qid"] in seen:
            raise ValueError(f"Duplicate qid {q['qid']}")
        seen.add(q["qid"])
        if not q.get("out_of_scope") and not q.get("gold_quotes"):
            raise ValueError(f"{q['qid']} is in scope but has no gold_quotes")
    return questions


def covered_positions(quote: Sequence[str], para: Sequence[str]) -> set[int]:
    """Indexes of quote words found in ``para`` inside runs of >= MIN_RUN words."""
    covered: set[int] = set()
    matcher = SequenceMatcher(None, quote, para, autojunk=False)
    for block in matcher.get_matching_blocks():
        if block.size >= MIN_RUN:
            covered.update(range(block.a, block.a + block.size))
    return covered


def _same_source(quote: dict[str, Any], paragraph: Paragraph) -> bool:
    source = Path(quote.get("source_pdf") or "").stem
    doc = paragraph.meta.get("document_id")
    return not source or not doc or source == doc


@dataclass
class Judgements:
    qrels: dict[str, dict[str, int]]          # qid -> {pid: grade}
    unmatched: list[tuple[str, str]]          # (qid, quote) found in no paragraph: fix the quote


def full_text(p: Paragraph) -> str:
    """What the generator reads for a retrieved unit: its heading, then its text."""
    return f"{p.heading}\n{p.body}" if p.heading else p.body


def build_qrels(questions: Iterable[dict[str, Any]], paragraphs: Sequence[Paragraph]) -> Judgements:
    para_words = [words(full_text(p)) for p in paragraphs]
    qrels: dict[str, dict[str, int]] = {}
    unmatched: list[tuple[str, str]] = []
    for q in questions:
        grades: dict[str, int] = {}
        for gq in q.get("gold_quotes") or []:
            qw = words(gq["quote"])
            found = False
            for p, pw in zip(paragraphs, para_words):
                if not _same_source(gq, p):
                    continue
                n = len(covered_positions(qw, pw))
                if n >= min(MIN_WORDS, len(qw)) or n >= MIN_FRACTION * len(qw):
                    grades[p.pid] = max(grades.get(p.pid, 0), int(gq["grade"]))
                    found = True
            if not found:
                unmatched.append((q["qid"], gq["quote"]))
        qrels[q["qid"]] = grades
    return Judgements(qrels, unmatched)


def deliver(ranked: Sequence[str], texts: dict[str, str], budget: int, count: Callable[[str], int]) -> list[str]:
    """The paragraphs a generator would get: top-ranked first, until the next one no longer fits."""
    delivered, used = [], 0
    for pid in ranked:
        n = count(texts[pid])
        if used + n > budget:
            break
        delivered.append(pid)
        used += n
    return delivered


def evidence_coverage(
    question: dict[str, Any], delivered: Sequence[str], texts: dict[str, str]
) -> dict[str, float]:
    """How much of the gold evidence is inside the delivered paragraphs.

    - ``coverage``:      mean, over essential quotes, of the fraction of their words delivered
    - ``all_essential``: 1 if every essential quote is at least 80% delivered, else 0
    Essential = grade 2; if a question has none, its grade-1 quotes are used.
    """
    quotes = [g for g in question.get("gold_quotes") or [] if g["grade"] >= 2]
    quotes = quotes or list(question.get("gold_quotes") or [])
    if not quotes:
        raise ValueError(f"{question['qid']}: coverage needs gold quotes")
    delivered_words = [words(texts[pid]) for pid in delivered]
    fractions = []
    for gq in quotes:
        qw = words(gq["quote"])
        covered: set[int] = set()
        for pw in delivered_words:
            covered |= covered_positions(qw, pw)
        fractions.append(len(covered) / len(qw))
    return {
        "coverage": sum(fractions) / len(fractions),
        "all_essential": float(all(f >= COVERED for f in fractions)),
    }
