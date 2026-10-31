"""Vectorless RAG for the SLM: find the notes text for a topic and build the prompt.

Flow for one request ("make 3 MCQs on process states"):

1. ``ContextRetriever.retrieve(topic)`` ranks the sections with the lexical index
   (BM25F, optional OS glossary, no embeddings) and packs the best ones into a token
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

from src.indexing.concepts import content_tokens, extract_concepts, key, plain
from src.indexing.lexical import LexicalIndex
from src.indexing.records import Paragraph
from src.utils.token_counter import count_tokens

# words that shape a question rather than name a topic
QUESTION_WORDS = content_tokens(
    "difference differences different between compare comparison versus vs similar distinguish "
    "contrast mean meaning explain describe define definition role purpose use")

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
    def __init__(self, paragraphs: list[Paragraph], method: str = "bm25f", expand: bool = False) -> None:
        """``expand``: OS glossary query expansion. Off by default: after the glossary was
        frozen to textbook terms it changed little, and on Sreeram's benchmark it slightly
        lowered BM25F (0.821 -> 0.813, old glossary). Decide on dev questions only."""
        self.paragraphs = list(paragraphs)
        self.by_pid = {p.pid: p for p in self.paragraphs}
        self.index = LexicalIndex(method=method, expand=expand)
        self.index.build(self.paragraphs)
        self.concepts = extract_concepts(self.paragraphs)
        self.plain_text = {p.pid: plain(p.body) for p in self.paragraphs}
        self.vocab = set().union(*(content_tokens(f"{p.heading} {p.body}") for p in self.paragraphs)) \
            if self.paragraphs else set()

    def coverage(self, topic: str, text: str) -> float:
        """Share of the topic's content words that occur in ``text``."""
        q = content_tokens(topic)
        return len(q & content_tokens(text)) / len(q) if q else 0.0

    def in_notes(self, topic: str, best_text: str) -> bool:
        """Is the topic covered by the notes at all?

        - words the notes never use ("semaphore", "SaaS") are the main signal:
          a short topic (1-3 words) must not contain any; a longer question may
          have up to 40% of its words missing (students' own wording varies)
        - the best section must also contain the topic: all of a 1-2 word topic,
          two thirds of a 3-word topic, a third of a longer question
        """
        q = content_tokens(topic) - QUESTION_WORDS  # "difference", "explain" are not the topic
        if not q:
            return False
        unknown = q - self.vocab
        if len(q) <= 3 and unknown:
            return False
        if len(q) > 3 and len(unknown) / len(q) > 0.4:
            return False
        return self._covers(q, best_text)

    @staticmethod
    def _covers(q: set[str], text: str) -> bool:
        need = 1.0 if len(q) <= 2 else 0.67 if len(q) == 3 else 0.34
        return len(q & content_tokens(text)) / len(q) >= need

    def about(self, topic: str, pid: str) -> bool:
        """Is this section actually about the topic (not just ranked for it)?
        Same rule as ``in_notes``: it must contain the topic's words."""
        q = content_tokens(topic) - QUESTION_WORDS
        p = self.by_pid[pid]
        return bool(q) and self._covers(q, f"{p.heading} {p.body}")

    def rank(self, topic: str, k: int = 12) -> list[tuple[str, float]]:
        """(pid, score) best first, or [] when the topic is not in the notes.

        Order: sections whose heading contains the whole topic, then sections that
        contain the topic as an exact phrase ("CPU Utilization" in Scheduling
        Criteria), then the rest - each group by lexical (BM25F) score.
        """
        hits = [h for h in self.index.search(topic, k=k) if h.score > 0]
        if not hits:
            return []
        q, phrase = content_tokens(topic), key(topic)
        core = q - QUESTION_WORDS  # "difference between program and process" -> program, process
        short = len(q) <= 4

        def tier(pid: str) -> int:
            p = self.by_pid[pid]
            if short and core and core <= content_tokens(p.heading):
                return 0
            if short and phrase and f" {phrase} " in f" {self.plain_text[pid]} ":
                return 1
            return 2

        ranked = sorted(((h.pid, float(h.score)) for h in hits), key=lambda x: tier(x[0]))
        best = self.by_pid[ranked[0][0]]
        if not self.in_notes(topic, f"{best.heading} {best.body}"):
            return []
        return ranked

    def retrieve(self, topic: str, budget_tokens: int = 600, max_passages: int = 4) -> list[Passage]:
        """Best sections for the topic within ``budget_tokens``; [] if the notes do not cover it."""
        ranked = self.rank(topic, k=max_passages * 3)
        out, used = [], 0
        for pid, score in ranked:
            p = self.by_pid[pid]
            text = f"{p.heading}\n{p.body}"
            n = count_tokens(text)
            if out and used + n > budget_tokens:
                continue
            out.append(Passage(p.pid, p.heading, p.chapter, p.body, score,
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


# JSON schema for the reply; local runtimes such as Ollama can force the model to follow it
MCQ_SCHEMA = {
    "type": "object",
    "properties": {"mcqs": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
            "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
            "explanation": {"type": "string"},
            "evidence_quote": {"type": "string"},
        },
        "required": ["question", "options", "answer", "explanation", "evidence_quote"],
    }}},
    "required": ["mcqs"],
}


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
        "Reply with only JSON in this form:\n"
        '{"mcqs": [{"question": "...", "options": ["...", "...", "...", "..."], "answer": "A", '
        '"explanation": "...", "evidence_quote": "...", "source": "[pid]"}]}\n\n'
        f"NOTES:\n{notes}\n"
    )
