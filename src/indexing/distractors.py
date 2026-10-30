"""Wrong options (distractors) and safety checks for MCQs - vectorless.

How wrong options are chosen, best first:

1. **Siblings** - other members of the same list in the notes (for "Throughput",
   the other scheduling criteria). Believable and clearly wrong.
2. **Lexical neighbours** - items from other parts of the notes that the lexical
   index (BM25) finds closest to the question's topic. Similar wording makes them
   believable; they come from a different concept, so they are wrong.

Every candidate must pass ``is_safe`` so that no wrong option is also correct:

- it is not the answer or an alias of the answer ("CPU scheduler" for
  "short-term scheduler");
- it does not mean the same as the answer (content-word ``overlap`` below
  ``max_overlap``, e.g. "Preemptive" vs "Uses preemptive scheduling" is rejected);
- for statement options, it does not repeat anything the notes say about the
  question's concept, and it does not name that concept;
- it does not repeat another chosen option.

``answer_supported`` checks that an answer is actually in the evidence text. It
is used for the template MCQs here and can check generated (e.g. LLM) answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.indexing.concepts import AliasTable, content_tokens, key, overlap
from src.indexing.lexical import LexicalIndex
from src.indexing.records import Paragraph

MAX_OVERLAP = 0.6


@dataclass(frozen=True)
class Candidate:
    text: str  # what is shown as the option
    concept: str  # which concept it really belongs to
    source: str  # "sibling" or "lexical"
    pid: str
    score: float = 0.0


def answer_supported(answer: str, evidence: str, min_coverage: float = 0.8) -> bool:
    """True when at least ``min_coverage`` of the answer's content words are in the evidence."""
    ta = content_tokens(answer)
    if not ta:
        return False
    return len(ta & content_tokens(evidence)) / len(ta) >= min_coverage


def is_safe(
    candidate: Candidate,
    *,
    answer: str,
    answer_concept: str,
    chosen: list[str],
    aliases: AliasTable,
    true_statements: list[str] = (),
    max_overlap: float = MAX_OVERLAP,
) -> tuple[bool, str]:
    """(ok, reason). ``true_statements`` = everything the notes say about the answer concept."""
    text = candidate.text
    if not content_tokens(text):
        return False, "empty"
    if key(text) == key(answer) or aliases.same(text, answer):
        return False, "same as answer"
    if aliases.same(candidate.concept, answer_concept):
        return False, "alias of answer concept"
    if overlap(text, answer) >= max_overlap:
        return False, "means the same as the answer"
    for t in true_statements:
        if overlap(text, t) >= max_overlap:
            return False, "true for the asked concept"
    names = {answer_concept, *aliases.names(answer_concept)}
    low = f" {key(text)} "
    if any(len(n) > 3 and f" {key(n)} " in low for n in names):
        return False, "names the asked concept"
    for c in chosen:
        if key(c) == key(text) or overlap(text, c) >= 0.8:
            return False, "repeats another option"
    return True, "ok"


class NeighbourFinder:
    """BM25 index over candidate options, to find lexical neighbours of a topic."""

    def __init__(self, candidates: list[Candidate], contexts: list[str] | None = None,
                 chapters: list[str] | None = None, method: str = "bm25") -> None:
        """``contexts[i]`` is extra text indexed for candidate i (e.g. a term's definition);
        ``chapters[i]`` lets ``rank`` put candidates from the question's chapter first."""
        self.candidates = candidates
        self.chapters = chapters or [""] * len(candidates)
        contexts = contexts or [""] * len(candidates)
        self._index = LexicalIndex(method=method)
        self._index.build([
            Paragraph(pid=str(i), body=f"{c.concept} {c.text} {ctx}", heading=c.concept)
            for i, (c, ctx) in enumerate(zip(candidates, contexts))
        ])

    def rank(self, query: str, k: int = 50, chapter: str | None = None) -> list[Candidate]:
        hits = self._index.search(query, k=k)
        if chapter is not None:  # stable: same chapter first, then by score
            hits = sorted(hits, key=lambda h: self.chapters[int(h.pid)] != chapter)
        return [
            Candidate(**{**self.candidates[int(h.pid)].__dict__, "source": "lexical", "score": float(h.score)})
            for h in hits
        ]


def pick_distractors(
    *,
    answer: str,
    answer_concept: str,
    siblings: list[Candidate],
    neighbours: list[Candidate],
    aliases: AliasTable,
    true_statements: list[str] = (),
    n: int = 3,
) -> tuple[list[Candidate], list[tuple[str, str]]]:
    """Up to ``n`` safe distractors (siblings first) and the rejected (text, reason) pairs."""
    chosen: list[Candidate] = []
    rejected: list[tuple[str, str]] = []
    for cand in [*siblings, *neighbours]:
        if len(chosen) == n:
            break
        ok, why = is_safe(
            cand, answer=answer, answer_concept=answer_concept, chosen=[c.text for c in chosen],
            aliases=aliases, true_statements=true_statements,
        )
        if ok:
            chosen.append(cand)
        elif why not in ("same as answer",):
            rejected.append((cand.text, why))
    return chosen, rejected
