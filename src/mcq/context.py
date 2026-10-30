"""Vectorless RAG for the SLM: find the notes text for a topic and build the prompt.

Flow for one request ("make 3 MCQs on process states"):

1. ``ContextRetriever.retrieve(topic)`` ranks the sections with the lexical index
   (BM25F + OS glossary, no embeddings) and packs the best ones into a token
   budget, so a small model's context window is not wasted.
2. If the topic is not in the notes (e.g. "semaphores" in Unit 1) it returns no
   passages - the caller must not ask the SLM to make questions from nothing.
3. ``ContextRetriever.option_hints(passages)`` lists related terms from the notes
   (sibling concepts such as the other process states). Small models are weak at
   writing good wrong options; these hints give them believable ones.
4. ``build_prompt`` writes the instruction for the SLM, asking for JSON with an
   exact quote from the notes for every answer, so ``validate`` can check it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.indexing.concepts import content_tokens, extract_concepts, key
from src.indexing.lexical import LexicalIndex
from src.indexing.records import Paragraph
from src.utils.token_counter import count_tokens

DIFFICULTY_GUIDE = {
    "easy": "Ask about one definition or fact stated directly in the notes.",
    "medium": "Ask about a fact or step; use wrong options from the same topic.",
    "hard": "Ask the student to tell apart closely related concepts (use the related terms as wrong options).",
}


@dataclass
class Passage:
    pid: str
    heading: str
    chapter: str
    text: str
    score: float
    source_pdf: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    def cite(self) -> str:
        where = f"{self.source_pdf}, p{self.page_start}" if self.source_pdf else self.pid
        return f"[{self.pid}] {self.chapter} > {self.heading} ({where})"


class ContextRetriever:
    def __init__(self, paragraphs: list[Paragraph], method: str = "bm25f", expand: bool = True) -> None:
        self.paragraphs = list(paragraphs)
        self.by_pid = {p.pid: p for p in self.paragraphs}
        self.index = LexicalIndex(method=method, expand=expand)
        self.index.build(self.paragraphs)
        self.concepts = extract_concepts(self.paragraphs)

    def coverage(self, topic: str, text: str) -> float:
        """Share of the topic's content words that occur in ``text``."""
        q = content_tokens(topic)
        return len(q & content_tokens(text)) / len(q) if q else 0.0

    def retrieve(self, topic: str, budget_tokens: int = 600, max_passages: int = 4,
                 min_coverage: float | None = None) -> list[Passage]:
        """Best sections for the topic within ``budget_tokens``; [] if the notes do not cover it.

        The best section must contain the topic's words: all of them for a one- or
        two-word topic ("virtual memory" is rejected even though "memory" occurs),
        two thirds for longer topics. Set ``min_coverage`` to override.
        """
        if min_coverage is None:
            min_coverage = 1.0 if len(content_tokens(topic)) <= 2 else 0.67
        hits = [h for h in self.index.search(topic, k=max_passages * 3) if h.score > 0]
        if not hits:
            return []
        best = self.by_pid[hits[0].pid]
        if self.coverage(topic, f"{best.heading} {best.body}") < min_coverage:
            return []  # only weak, partial matches: treat the topic as not in the notes
        # sections whose heading names the topic first, then by lexical score
        q = content_tokens(topic)
        hits.sort(key=lambda h: -len(q & content_tokens(self.by_pid[h.pid].heading)))
        out, used = [], 0
        for h in hits:
            p = self.by_pid[h.pid]
            text = f"{p.heading}\n{p.body}"
            n = count_tokens(text)
            if out and used + n > budget_tokens:
                continue
            out.append(Passage(p.pid, p.heading, p.chapter, p.body, float(h.score),
                               p.meta.get("source_pdf"), p.meta.get("page_start"), p.meta.get("page_end")))
            used += n
            if len(out) == max_passages:
                break
        return out

    def option_hints(self, passages: list[Passage], limit: int = 12) -> list[str]:
        """Terms from the same lists / sibling concepts as the passages (for wrong options)."""
        pids = {p.pid for p in passages}
        groups = self.concepts.groups()
        hints: list[str] = []
        for d in self.concepts.definitions:
            if d.pid in pids:
                hints.extend(g.term for g in groups[d.group])
        for s in self.concepts.sections:
            if pids & set(s.pids):
                head = key(s.name).split()[-1:]
                hints.extend(t.name for t in self.concepts.sections if key(t.name).split()[-1:] == head)
        seen, out = set(), []
        for h in hints:
            if key(h) not in seen:
                seen.add(key(h))
                out.append(h)
        return out[:limit]


def build_prompt(topic: str, passages: list[Passage], n: int = 3, difficulty: str = "medium",
                 hints: list[str] | None = None) -> str:
    """Instruction for the small language model. Expects JSON back (see validate.py)."""
    notes = "\n\n".join(f"{p.cite()}\n{p.text}" for p in passages)
    hint_line = f"Related terms you may use as wrong options: {', '.join(hints)}\n" if hints else ""
    return (
        "You are writing multiple-choice questions for a student, using ONLY the course notes below.\n"
        f"Topic: {topic}\nNumber of questions: {n}\nDifficulty: {difficulty} - "
        f"{DIFFICULTY_GUIDE.get(difficulty, DIFFICULTY_GUIDE['medium'])}\n{hint_line}"
        "Rules:\n"
        "- Exactly 4 options, exactly one correct.\n"
        "- The correct answer must be stated in the notes. Do not use outside knowledge.\n"
        "- Wrong options must be clearly wrong according to the notes, but believable.\n"
        "- evidence_quote must be copied exactly from the notes and must support the answer.\n"
        "Reply with only a JSON list, each item:\n"
        '{"question": "...", "options": ["...", "...", "...", "..."], "answer": "A", '
        '"explanation": "...", "evidence_quote": "...", "source": "[pid]"}\n\n'
        f"NOTES:\n{notes}\n"
    )
