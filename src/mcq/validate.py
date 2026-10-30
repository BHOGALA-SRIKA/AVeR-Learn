"""Check an MCQ written by the SLM (or anyone) against the notes before a student sees it.

Small models make three kinds of mistake that ruin an MCQ: an answer that is not
in the notes (made up), a wrong option that is actually also correct, and broken
format (3 options, repeated options, answer letter missing). ``validate_mcq``
catches these with lexical checks only - no second model needed.

Checks (each failed check is listed in ``problems``):

- format: question text, exactly 4 different options, answer letter A-D
- evidence: the quoted evidence really is in the notes (it is found in a passage)
- answer: the answer's content words are in the evidence or its passage
- options: no wrong option means the same as the answer, and no wrong option is
  itself stated in the evidence (then it would be correct too)
- give-away: the answer is not much longer or shorter than the wrong options
"""

from __future__ import annotations

import re

from src.indexing.concepts import content_tokens, key, overlap, parse_lines
from src.indexing.distractors import answer_supported
from src.mcq.context import Passage

LETTERS = "ABCD"


def _plain(text: str) -> str:
    """Passage text without bullet marks or numbering, as one comparable string."""
    lines = [ln.text for ln in parse_lines(text)]
    # also drop bullet marks written inline, e.g. "Ready o Process is waiting ..."
    return key(re.sub(r"(?:^|\s)[o•▪◦](?=\s)", " ", " ".join(lines)))


def _find_evidence(quote: str, passages: list[Passage], min_coverage: float = 0.9) -> Passage | None:
    """The passage that contains the quote (exact after normalising, else by word coverage)."""
    if not quote:
        return None
    kq = _plain(quote)
    for p in passages:
        if kq and kq in _plain(p.text):
            return p
    q = content_tokens(kq)
    if not q:
        return None
    plain = {p.pid: content_tokens(_plain(p.text)) for p in passages}
    best = max(passages, key=lambda p: len(q & plain[p.pid]), default=None)
    if best is not None and len(q & plain[best.pid]) / len(q) >= min_coverage:
        return best
    return None


def validate_mcq(mcq: dict, passages: list[Passage], answer_coverage: float = 0.7) -> dict:
    """{"ok": bool, "problems": [...], "evidence_pid": str | None} for one MCQ dict.

    ``mcq`` uses the SLM output format: question, options (4), answer (letter),
    evidence_quote. ``passages`` are the ones the SLM was given.
    """
    problems: list[str] = []
    question = (mcq.get("question") or "").strip()
    options = [str(o).strip() for o in (mcq.get("options") or [])]
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

    quote = (mcq.get("evidence_quote") or "").strip()
    ev = _find_evidence(quote, passages)
    if ev is None:
        problems.append("evidence quote is not in the notes")
        support_text = " ".join(p.text for p in passages)
    else:
        support_text = f"{quote} {ev.text}"

    # the answer must come from the notes; for "which term" questions the answer is
    # a name, which can sit in the question/heading rather than the quote
    stem_and_heading = f"{question} {ev.heading if ev else ''}"
    if not (answer_supported(answer, f"{quote} {ev.text if ev else ''}", answer_coverage)
            or answer_supported(answer, f"{support_text} {stem_and_heading}", 1.0)):
        problems.append("answer is not supported by the notes")

    for o in wrong:
        if overlap(o, answer) >= 0.6:
            problems.append(f"wrong option means the same as the answer: {o!r}")
        elif quote and len(content_tokens(o)) >= 3 and answer_supported(o, quote, 0.9) \
                and not answer_supported(answer, o, 0.9):
            problems.append(f"wrong option is also stated in the evidence: {o!r}")

    lens = sorted(len(o.split()) for o in wrong)
    if lens and len(answer.split()) >= 4:
        ratio = len(answer.split()) / max(lens[len(lens) // 2], 1)
        if ratio >= 2.5:
            problems.append("answer is much longer than the other options (gives it away)")

    return {"ok": not problems, "problems": problems, "evidence_pid": ev.pid if ev else None}
