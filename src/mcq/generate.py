"""Template MCQs built only from the notes - a working offline baseline.

Question kinds (every answer is a line in the notes):

- ``identify`` - 'In "Common Process States", which term matches this description:
                 "Process is waiting to be assigned to a CPU"?'  answer "Ready";
                 wrong options = the other states. Also used for a section concept:
                 'Which of these does the notes describe as: "Gives control of the CPU
                 to the process selected by the short-term scheduler"?'  answer "Dispatcher".
- ``define``   - 'In "Scheduling Criteria", what does "Throughput" refer to?'
                 answer = its definition, wrong options = the other criteria's definitions.
- ``fact``     - 'Which of the following is correct about the long-term scheduler?'
                 Only made when the concept has real siblings (short-, medium-term
                 scheduler): the wrong options are facts about those siblings, so they
                 are believable and clearly false for the asked one.

Each MCQ has a ``difficulty`` for adaptive learning, from how close the wrong
options are to the answer:

- ``hard``   - all wrong options are siblings from the same list or sibling concepts
               (Ready vs Waiting vs Running)
- ``medium`` - some wrong options were found by the lexical index in the same chapter
- ``easy``   - some wrong options come from other chapters (easier to rule out)

The wording is plain on purpose. A language model can later rewrite the question
text; the answer, options and evidence stay fixed and checked. Every MCQ is
marked for human review.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field

from src.indexing.concepts import ConceptSet, content_tokens, extract_concepts, key, overlap
from src.indexing.distractors import Candidate, NeighbourFinder, answer_supported, pick_distractors
from src.indexing.lexical import LexicalIndex
from src.indexing.records import Paragraph

N_OPTIONS = 4
LETTERS = "ABCD"
LEVELS = ("hard", "medium", "easy")
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}
PRONOUN_START = {"this", "these", "it", "its", "they", "their", "such", "that", "those"}
# head nouns that name a category rather than one thing, so "X scheduling" and
# "Y scheduling" are not exclusive siblings (preemptive scheduling IS process scheduling)
GENERIC_HEADS = {
    "scheduling", "process", "processes", "system", "systems", "options", "option", "criteria",
    "concept", "concepts", "structure", "operation", "operations", "organization", "view",
    "information", "details", "model", "types", "flow", "storage", "memory",
}


@dataclass
class MCQ:
    mcq_id: str
    kind: str
    difficulty: str
    concept: str
    question: str
    options: list[str]
    answer: str  # letter
    answer_text: str
    evidence: dict
    distractor_sources: list[dict]
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# helpers


def _lower_name(name: str) -> str:
    """'Long-Term Scheduler' -> 'the long-term scheduler'; keeps abbreviations."""
    words = [w if (w.isupper() and len(w) > 1) or "/" in w else w.lower() for w in name.split()]
    text = " ".join(words)
    last = key(name).split()[-1] if key(name) else ""
    if last.endswith(("ing", "s", "tion")) or name.isupper():
        return text
    return "the " + text


def _shuffle(options: list[str], seed: str) -> list[str]:
    rng = random.Random(int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16))
    out = options[:]
    rng.shuffle(out)
    return out


def _length_ratio(answer: str, others: list[str]) -> float:
    lens = sorted(len(o.split()) for o in others)
    median = lens[len(lens) // 2] if lens else 1
    return round(len(answer.split()) / max(median, 1), 2)


def _mentions(text: str, names: set[str]) -> bool:
    """True if ``text`` contains one of ``names`` (all of a name's content words)."""
    low = f" {key(text)} "
    tt = content_tokens(text)
    for n in names:
        if not n:
            continue
        if f" {key(n)} " in low:
            return True
        nt = content_tokens(n)
        if len(nt) >= 2 and nt <= tt:
            return True
    return False


TITLE_HEADS = GENERIC_HEADS | {"goals", "handling", "timeline", "child", "model", "example", "examples",
                               "need", "issues", "problems", "solution", "options", "observations"}
TITLE_WORDS = {"the", "for", "of", "vs", "and", "in", "an", "a", "using", "with", "to"}


def _concept_like(name: str) -> bool:
    """A name a teacher would use as an answer ("Dispatcher", "Context Switch"),
    not a section title ("Parent Terminating Child", "Operating System Goals")."""
    w = key(name).split()
    return 1 <= len(w) <= 3 and not (TITLE_WORDS & set(w)) and w[-1] not in TITLE_HEADS


def _usable_statement(text: str, min_words: int = 3, max_words: int = 25) -> bool:
    words = text.split()
    return (min_words <= len(words) <= max_words and not ({w.lower() for w in words} & PRONOUN_START)
            and not text.endswith((":", "?")))


def _difficulty(distractors: list[Candidate], chapter: str, chapter_of: dict[str, str]) -> str:
    if all(d.source == "sibling" for d in distractors):
        return "hard"
    if all(d.source == "sibling" or chapter_of.get(key(d.concept)) == chapter for d in distractors):
        return "medium"
    return "easy"


def _evidence(p: Paragraph, quote: str) -> dict:
    return {
        "pid": p.pid, "heading": p.heading, "chapter": p.chapter, "quote": quote,
        "source_pdf": p.meta.get("source_pdf"), "page_start": p.meta.get("page_start"),
        "page_end": p.meta.get("page_end"),
    }


def _make(kind, difficulty, concept, question, answer, distractors, rejected, para, quote) -> MCQ:
    texts = [answer] + [d.text for d in distractors]
    options = _shuffle(texts, question + answer)
    ratio = _length_ratio(answer, [d.text for d in distractors])
    return MCQ(
        mcq_id="", kind=kind, difficulty=difficulty, concept=concept, question=question, options=options,
        answer=LETTERS[options.index(answer)], answer_text=answer,
        evidence=_evidence(para, quote),
        distractor_sources=[{"option": d.text, "really_about": d.concept, "how_found": d.source, "pid": d.pid}
                            for d in distractors],
        checks={
            "answer_in_notes": answer_supported(answer, para.body.replace("\n", " ") + " " + quote),
            "options_distinct": len({key(t) for t in texts}) == len(texts),
            "answer_length_vs_others": ratio,
            # only matters when options are sentences; short names vary naturally
            "length_gives_answer_away": kind != "identify" and (ratio >= 2.0 or ratio <= 0.4),
            "rejected_options": [{"option": t, "reason": r} for t, r in rejected[:5]],
            "human_review": "needed",
        },
    )


# ---------------------------------------------------------------------------
# main


def generate_mcqs(paragraphs: list[Paragraph], per_concept: int = 1,
                  levels: tuple[str, ...] = LEVELS) -> list[MCQ]:
    """MCQs for every usable concept in the sections, in document order.

    ``levels`` limits the difficulties returned, e.g. ("hard",) for a strong student.
    With ``per_concept=1`` the hardest valid question for each concept is kept.
    """
    paragraphs = list(paragraphs)
    by_pid = {p.pid: p for p in paragraphs}
    cs: ConceptSet = extract_concepts(paragraphs)
    groups = cs.groups()

    # everything the notes say about each name (definitions and section facts)
    said: dict[str, list[str]] = {}
    chapter_of: dict[str, str] = {}
    for d in cs.definitions:
        said.setdefault(key(d.term), []).append(d.definition)
        chapter_of.setdefault(key(d.term), d.chapter)
    for s in cs.sections:
        for n in {s.name, *s.aliases}:
            said.setdefault(key(n), []).extend(s.facts)
            chapter_of.setdefault(key(n), s.chapter)

    def says(name: str) -> list[str]:
        return [t for n in cs.aliases.names(name) | {key(name)} for t in said.get(n, [])]

    # pools of possible wrong options
    names, seen = [], set()
    for n, chap in [(d.term, d.chapter) for d in cs.definitions] + \
            [(s.name, s.chapter) for s in cs.sections if _concept_like(s.name)]:
        if key(n) not in seen:
            seen.add(key(n))
            names.append((n, chap))
    name_finder = NeighbourFinder(
        [Candidate(n, n, "lexical", "") for n, _ in names],
        contexts=[" ".join(says(n)) for n, _ in names], chapters=[c for _, c in names])
    def_finder = NeighbourFinder(
        [Candidate(d.definition, d.term, "lexical", d.pid) for d in cs.definitions],
        chapters=[d.chapter for d in cs.definitions])

    def not_fitting(description: str, answer_name: str):
        def ok(c: Candidate) -> bool:
            if cs.aliases.same(c.concept, answer_name):
                return False
            if _mentions(description, {c.concept, *cs.aliases.names(c.concept)}):
                return False  # the description talks about this option, so it could be read as right
            return not any(overlap(t, description) >= 0.6 for t in says(c.concept))
        return ok

    made: list[MCQ] = []

    def identify(answer_name, description, siblings, heading, chapter, para, quote, question):
        if _mentions(description, {answer_name, *cs.aliases.names(answer_name)}):
            return None  # the description would give the answer away
        fit = not_fitting(description, answer_name)
        sibs = [c for c in siblings if fit(c)]
        neigh = [c for c in name_finder.rank(f"{heading} {description}", k=60, chapter=chapter) if fit(c)]
        chosen, rejected = pick_distractors(answer=answer_name, answer_concept=answer_name,
                                            siblings=sibs, neighbours=neigh, aliases=cs.aliases)
        if len(chosen) < N_OPTIONS - 1:
            return None
        return _make("identify", _difficulty(chosen, chapter, chapter_of), answer_name, question,
                     answer_name, chosen, rejected, para, quote)

    # 1. definitions from lists
    for d in cs.definitions:
        para = by_pid[d.pid]
        group = [g for g in groups[d.group] if g is not d]
        quote = f"{d.term}: {d.definition}"
        mcq = None
        if len(d.term.split()) <= 4 and len(d.definition.split()) >= 3:
            q = f'In "{d.heading}", which term matches this description: "{d.definition}"?'
            mcq = identify(d.term, d.definition, [Candidate(g.term, g.term, "sibling", g.pid) for g in group],
                           d.heading, d.chapter, para, quote, q)
        if mcq is None and len(d.definition.split()) >= 2:
            sibs = [Candidate(g.definition, g.term, "sibling", g.pid) for g in group]
            neigh = [c for c in def_finder.rank(f"{d.heading} {d.term}", chapter=d.chapter)
                     if not cs.aliases.same(c.concept, d.term)]
            chosen, rejected = pick_distractors(
                answer=d.definition, answer_concept=d.term, siblings=sibs, neighbours=neigh,
                aliases=cs.aliases, true_statements=[x for x in says(d.term) if x != d.definition])
            if len(chosen) == N_OPTIONS - 1:
                q = f'In "{d.heading}", what does "{d.term}" refer to?'
                mcq = _make("define", _difficulty(chosen, d.chapter, chapter_of), d.term, q, d.definition,
                            chosen, rejected, para, quote)
        if mcq:
            made.append(mcq)

    # 2. section concepts
    def head(name: str) -> str:
        w = key(name).split()
        return w[-1] if len(w) >= 2 else ""

    for s in cs.sections:
        para = by_pid[s.pids[0]]
        own_names = {s.name, *s.aliases}
        statements = [f for f in s.facts if _usable_statement(f) and not _mentions(f, own_names)]
        # 2a. "which is correct about X" - only with real sibling concepts
        h = head(s.name)
        siblings = [t for t in cs.sections if t is not s and h and h not in GENERIC_HEADS
                    and head(t.name) == h and not cs.aliases.same(t.name, s.name)]
        if siblings and statements:
            sib_facts = [Candidate(f, t.name, "sibling", t.pids[0]) for t in siblings for f in t.facts
                         if _usable_statement(f) and not _mentions(f, {t.name, *t.aliases})]
            best = None
            for fact in statements:
                chosen, rejected = pick_distractors(
                    answer=fact, answer_concept=s.name, siblings=sib_facts, neighbours=[],
                    aliases=cs.aliases, true_statements=[f for f in says(s.name) if f != fact])
                if len(chosen) < N_OPTIONS - 1:
                    continue
                score = abs(1 - _length_ratio(fact, [c.text for c in chosen]))
                if best is None or score < best[0]:
                    best = (score, fact, chosen, rejected)
            if best:
                _, fact, chosen, rejected = best
                ev = next((by_pid[pid] for pid in s.pids if fact.split(":")[0] in by_pid[pid].body), para)
                q = f"Which of the following is correct about {_lower_name(s.name)}?"
                made.append(_make("fact", "hard", s.name, q, fact, chosen, rejected, ev, fact))
        # 2b. "which concept is this" - only when the heading names a concept
        #     ("Dispatcher", "Context Switch"), not a title ("Need for an Operating System")
        if not _concept_like(s.name):
            continue
        for fact in statements:
            if len(fact.split()) < 4 or ":" in fact:
                continue
            q = f'Which of the following do the notes describe as: "{fact}"?'
            ev = next((by_pid[pid] for pid in s.pids if fact.split(":")[0] in by_pid[pid].body), para)
            mcq = identify(s.name, fact, [], s.name, s.chapter, ev, fact, q)
            if mcq:
                made.append(mcq)
                break

    # keep the best questions per concept
    made = [m for m in made if m.difficulty in levels and m.checks["answer_in_notes"]
            and m.checks["options_distinct"] and not m.checks["length_gives_answer_away"]]
    kind_rank = {"fact": 0, "identify": 1, "define": 2}
    made.sort(key=lambda m: (LEVEL_RANK[m.difficulty], kind_rank[m.kind]))
    out, count = [], {}
    for m in made:
        k = next((c for c in count if cs.aliases.same(c, m.concept)), key(m.concept))
        if count.get(k, 0) < per_concept:
            count[k] = count.get(k, 0) + 1
            out.append(m)

    order = {p.pid: i for i, p in enumerate(paragraphs)}
    out.sort(key=lambda m: (order.get(m.evidence["pid"], 0), m.evidence["quote"]))
    for i, m in enumerate(out, 1):
        m.mcq_id = f"MCQ-{i:03d}"
    return out


def generate_for_topic(paragraphs: list[Paragraph], topic: str, n: int = 5,
                       method: str = "bm25f", levels: tuple[str, ...] = LEVELS) -> list[MCQ]:
    """MCQs about a topic the student typed, from the sections that best match it."""
    paragraphs = list(paragraphs)
    index = LexicalIndex(method=method, expand=True)
    index.build(paragraphs)
    by_pid = {p.pid: p for p in paragraphs}
    hits = [h for h in index.search(topic, k=len(paragraphs)) if h.score > 0]
    q = content_tokens(topic)
    # sections whose heading names the topic come first ("Common Process States" for
    # "process states"), then by lexical score
    heading_cov = {h.pid: len(q & content_tokens(by_pid[h.pid].heading)) / max(len(q), 1) for h in hits}
    ranked = sorted((h.pid for h in hits), key=lambda pid: -heading_cov[pid])
    rank = {pid: r for r, pid in enumerate(ranked)}
    mcqs = [m for m in generate_mcqs(paragraphs, levels=levels) if m.evidence["pid"] in rank]
    mcqs.sort(key=lambda m: (rank[m.evidence["pid"]], LEVEL_RANK[m.difficulty]))
    return mcqs[:n]
