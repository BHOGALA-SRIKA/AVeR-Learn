"""Check an MCQ written by the SLM (or anyone) against the notes before a student sees it.

Small models make mistakes that ruin an MCQ: an answer that is not in the notes,
a wrong answer key, a "wrong" option that is actually also correct, and broken
format. ``validate_mcq`` catches these with lexical checks only - no second model.

Checks (each failed check is listed in ``problems``):

- format: question text, exactly 4 different options, answer letter A-D
- evidence: the quoted evidence really is in the notes
- answer key: the chosen answer is backed by the quoted evidence itself
  (catches "waiting for an event -> Running")
- options: no wrong option is a part of the answer ("switches process" vs
  "switches process and saves state"), is stated in the evidence, or is a fact
  the notes give about the concept the question asks about (catches "What does
  the dispatcher do?" with three true dispatcher functions as "wrong" options)
- give-away: the answer is not much longer than the wrong options

Model output is cleaned first: "A. " prefixes on options and citation tails such
as "(unit1_topic3-s19, p6)" after the quote are removed.

Measured against a hand check of 18 qwen2.5:1.5b MCQs, see experiments/slm_eval.py.
"""

from __future__ import annotations

import re

from src.indexing.concepts import ConceptSet, content_tokens, key, plain
from src.mcq.context import Passage
from src.text import tokenize

LETTERS = "ABCD"
ORDER_WORDS = {"first", "last", "second", "third", "next", "final", "order", "before", "after",
               "begins", "starts", "sequence"}
_PREFIX = re.compile(r"^\s*(?:\(?[A-Da-d][.):]\s*)+")
_CITATION = re.compile(r"\s*[\(\[][^()\[\]]*\b(?:unit\d|p\d+|pid|page|evidence_quote|source)\b[^()\[\]]*[\)\]]\s*\.?\s*$",
                       re.I)


def clean_option(text: str) -> str:
    """'A. A. Save current CPU state' -> 'Save current CPU state'."""
    return _PREFIX.sub("", str(text)).strip()


_DASH_CITATION = re.compile(r"\s*[–—-]+\s*(?:unit|topic|p\.?\s*\d|page|source|notes)\b.*$", re.I)
_QUOTED = re.compile(r"[‘“\"']([^‘’“”\"]{8,}?)[’”\"'](?=\s|$|[.,;:)–—-])")
_WRAP = "\"'‘’“”"


def clean_quote(text: str) -> str:
    """The quoted notes text without what the model wrapped around it:
    "The notes state: 'Child terminates'" and "'Process has finished execution' -
    Unit 1, Topic 3, p3" both become the bare quote; citation tails are dropped."""
    text = str(text or "").strip()
    for _ in range(3):
        text = _CITATION.sub("", text).strip()
        text = _DASH_CITATION.sub("", text).strip()
    inner = [m.group(1).strip() for m in _QUOTED.finditer(text)]
    inner = [q for q in inner if len(q.split()) >= 2]
    if inner:
        text = max(inner, key=len)
    return text.strip(_WRAP).strip()


def _seq(text: str) -> list[str]:
    return tokenize(plain(text))


def phrase_coverage(part: str, whole: str) -> float:
    """Share of ``part``'s word pairs (in order) that also occur in ``whole``.

    Stricter than shared words: "Perform a CPU switch" is not in "... performs a
    context switch", although every word is.
    """
    a, b = _seq(part), _seq(whole)
    if len(a) < 2:
        return 1.0 if a and a[0] in b else 0.0
    pairs_b = set(zip(b, b[1:]))
    pairs_a = list(zip(a, a[1:]))
    return sum(p in pairs_b for p in pairs_a) / len(pairs_a)


def _find_evidence(quote: str, passages: list[Passage], min_coverage: float = 0.9) -> Passage | None:
    """The passage that contains the quote: word for word after normalising, or with
    at least ``min_coverage`` of its word pairs in order (so a changed subject, as in
    "An interrupt is ..." for the notes' "A trap is ...", is rejected)."""
    if not quote:
        return None
    kq = plain(quote)
    for p in passages:
        if kq and kq in plain(p.text):
            return p
    best = max(passages, key=lambda p: phrase_coverage(quote, p.text), default=None)
    if best is not None and phrase_coverage(quote, best.text) >= min_coverage:
        return best
    return None


def _label_before(quote: str, passage: Passage, n: int = 2) -> list[str]:
    """The ``n`` words in the notes right before the quote (a list item's label)."""
    q, t = _seq(quote), _seq(passage.text)
    if len(q) < 2:
        return []
    k = min(len(q), 6)  # match the quote's opening words, not just the first two
    for i in range(len(t) - k + 1):
        if t[i:i + k] == q[:k]:
            return t[max(0, i - n):i]
    return []


def _answer_backed(answer: str, quote: str, ev: Passage, coverage: float,
                   concepts: ConceptSet | None = None) -> bool:
    a = set(tokenize(answer))
    if not a:
        return False
    backing = set(_seq(quote)) | set(tokenize(ev.heading)) | set(_label_before(quote, ev))
    if concepts is not None:  # a "Key Points" section belongs to the concept before it
        for s in concepts.sections:
            if ev.pid in s.pids:
                backing |= set(tokenize(" ".join([s.name, *s.aliases])))
    return len(a & backing) / len(a) >= coverage


def _part_of(wrong: str, answer: str) -> bool:
    """True when the wrong option says nothing beyond the answer ("Preemptive" vs
    "Uses preemptive scheduling"). Same words in a different order ("parent creates
    child" vs "child creates parent") is a different statement."""
    tw, ta = content_tokens(wrong), content_tokens(answer)
    if not tw or not ta:
        return False
    if tw == ta:
        return key(wrong) == key(answer)
    return tw < ta


def _named(question: str, name: str, aliases: set[str]) -> bool:
    low = f" {key(question)} "
    return any(n and len(n) > 2 and f" {key(n)} " in low for n in {name, *aliases})


def _true_fact_about_subject(option: str, question: str, answer: str, concepts: ConceptSet) -> str | None:
    """Name of the concept the question asks about, if the notes state ``option`` as a
    fact about it (so the "wrong" option is actually correct too)."""
    ot = content_tokens(option)
    if len(ot) < 2:
        return None
    # template questions start with 'In "<section>", ...' - that is context, not the subject
    question = re.sub(r'^\s*In\s+"[^"]+",\s*', "", question)

    def covers(text: str) -> bool:
        """The option states this line: it has the line's words, not just a name inside it."""
        tt = content_tokens(text)
        return bool(tt) and len(ot & tt) / len(ot) >= 0.9 and len(ot & tt) / len(tt) >= 0.6

    # an option that is the definition of some list term is about that term
    for d in concepts.definitions:
        if covers(d.definition) and len(content_tokens(d.definition)) <= len(ot) + 3:
            aliases = concepts.aliases.names(d.term)
            if _named(question, d.term, aliases) and not covers(answer):
                return d.term
            return None
    for s in concepts.sections:
        if _named(question, s.name, s.aliases | concepts.aliases.names(s.name)):
            if any(covers(f) for f in s.facts):
                return s.name
    return None


def validate_mcq(mcq: dict, passages: list[Passage], concepts: ConceptSet | None = None,
                 answer_coverage: float = 0.7) -> dict:
    """{"ok": bool, "problems": [...], "evidence_pid": str | None} for one MCQ dict.

    ``mcq`` uses the SLM output format: question, options (4), answer (letter),
    evidence_quote. ``passages`` are the ones the SLM was given; ``concepts``
    (``ContextRetriever.concepts``) enables the "also a true fact" check.
    """
    problems: list[str] = []
    question = (mcq.get("question") or "").strip()
    options = [clean_option(o) for o in (mcq.get("options") or [])]
    letter = str(mcq.get("answer") or "").strip().upper()[:1]

    if not question:
        problems.append("no question text")
    if len(options) != 4:
        problems.append(f"needs 4 options, got {len(options)}")
    if any(not o for o in options):
        problems.append("empty option")
    if len({key(o) for o in options}) != len(options):
        problems.append("repeated options")
    if letter not in LETTERS[: len(options)]:
        problems.append(f"answer letter {letter!r} is not one of the options")
        return {"ok": False, "problems": problems, "evidence_pid": None}

    answer = options[LETTERS.index(letter)]
    wrong = [o for i, o in enumerate(options) if i != LETTERS.index(letter)]

    quote = clean_quote(mcq.get("evidence_quote"))
    ev = _find_evidence(quote, passages)
    if ev is None:
        problems.append("evidence quote is not in the notes")
    # the answer key must be backed by the quoted evidence itself, or by the list
    # label written right before it ("Ready" before "Process is waiting to be assigned")
    elif not _answer_backed(answer, quote, ev, answer_coverage, concepts):
        problems.append("answer is not backed by the quoted evidence (wrong answer key?)")

    # "What is the first step ...?": the other steps are in the evidence on purpose
    order_question = bool(ORDER_WORDS & set(key(question).split()))
    for o in wrong:
        if _part_of(o, answer):
            problems.append(f"wrong option is part of the answer, so also correct: {o!r}")
        elif quote and not order_question and len(content_tokens(o)) >= 2 \
                and phrase_coverage(o, quote) >= 0.8 and phrase_coverage(answer, o) < 0.8:
            problems.append(f"wrong option is also stated in the evidence: {o!r}")
        elif concepts is not None and not order_question:
            subject = _true_fact_about_subject(o, question, answer, concepts)
            if subject:
                problems.append(f"wrong option is a true fact about {subject}: {o!r}")

    lens = sorted(len(o.split()) for o in wrong)
    if lens and len(answer.split()) >= 4:
        ratio = len(answer.split()) / max(lens[len(lens) // 2], 1)
        if ratio >= 2.5:
            problems.append("answer is much longer than the other options (gives it away)")

    return {"ok": not problems, "problems": problems, "evidence_pid": ev.pid if ev else None}
