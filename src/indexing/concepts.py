"""Find the concepts in course notes that MCQs can be asked about (vectorless).

Everything here is rule-based and runs offline on CPU. It works on the
``Paragraph`` records the lexical index already uses (one record per section:
``heading`` = section title, ``body`` = section text, line breaks kept).

Two kinds of concept are found:

1. **Definitions** - a term and what the notes say it is, taken from lists such as

       2. Throughput
          o Number of processes completed per unit time

   or "CPU burst -> process is executing instructions", or "Hardware - Provides ...".
   Terms from the same list form a *group* (e.g. the five scheduling criteria).
   Siblings in a group are the best wrong options for each other: they are real
   course content, so they look believable, but they are clearly wrong.

2. **Section concepts** - a section whose heading names a concept ("Dispatcher",
   "Long-Term Scheduler (Job Scheduler)") with its facts (the bullet points).
   Sections with a generic heading ("Key Points", "Solution", "Examples") are
   added to the concept just before them.

Aliases come from headings and text such as "Short-Term Scheduler (CPU Scheduler)",
"Process Control Block (PCB)" or "also called Task Control Block". Names that are
aliases of each other are treated as one concept, so "CPU scheduler" is never
offered as a wrong option for a question about the short-term scheduler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.indexing.records import Paragraph
from src.text import tokenize

GENERIC_HEADINGS = {
    "introduction", "key points", "solution", "examples", "example", "observations",
    "important scenarios", "summary", "notes", "note", "overview",
}
# first words that mark a list item as an instruction or quantity, not a concept name
NOT_A_TERM_START = {
    "maximize", "minimize", "manage", "must", "do", "does", "want", "desire", "expect",
    "when", "if", "what", "how", "why", "each", "all", "some", "most", "only", "it",
    "this", "these", "time", "uses", "use", "can", "may", "occurs", "selects",
}
_BULLET = re.compile(r"^\s*[•▪●\-\*]\s*")
_SUB = re.compile(r"^\s*(?:o(?=[A-Z0-9(])|o\s+|◦\s*|-\s{2,})")
_NUMBER = re.compile(r"^\s*\d+\s*[.)]\s*")
_ARROW = re.compile(r"\s*(?:→|->)\s*")
_DASH_DEF = re.compile(r"^(?P<term>[^-–:]{2,60}?)\s+[-–]\s+(?P<def>.{3,})$")
_COLON_DEF = re.compile(r"^(?P<term>[^:]{2,60}?):\s+(?P<def>.{3,})$")
_PAREN = re.compile(r"\(([^()]{1,60})\)")
_ALSO_CALLED = re.compile(r"\((?:also\s+)?(?:called|known as)\s+([^()]+?)\)", re.I)


# ---------------------------------------------------------------------------
# data classes


@dataclass
class Line:
    level: int  # 0 plain text, 1 bullet or numbered item, 2 sub-bullet
    text: str
    numbered: bool = False


@dataclass
class Definition:
    term: str
    definition: str
    group: str  # definitions with the same group came from the same list
    pid: str
    heading: str
    chapter: str
    style: str  # "list", "arrow", "dash" or "colon"
    meta: dict = field(default_factory=dict)


@dataclass
class SectionConcept:
    name: str
    aliases: set[str]
    facts: list[str]
    pids: list[str]
    chapter: str
    meta: dict = field(default_factory=dict)


@dataclass
class ConceptSet:
    definitions: list[Definition]
    sections: list[SectionConcept]
    aliases: "AliasTable"

    def groups(self) -> dict[str, list[Definition]]:
        out: dict[str, list[Definition]] = {}
        for d in self.definitions:
            out.setdefault(d.group, []).append(d)
        return out


# ---------------------------------------------------------------------------
# small text helpers


def key(text: str) -> str:
    """Comparison key for names: lowercase, no punctuation, single spaces."""
    text = re.sub(r"[^\w\s/]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def overlap(a: str, b: str) -> float:
    """Shared content words divided by the size of the smaller set (0..1).

    1.0 means one text says nothing the other does not ("Preemptive" vs
    "Uses preemptive scheduling"); used to catch options that mean the same.
    """
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def strip_parens(name: str) -> str:
    return re.sub(r"\s+", " ", _PAREN.sub("", name)).strip(" :-–")


def _is_abbreviation(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9/]{1,7}s?", s.strip()))


# ---------------------------------------------------------------------------
# aliases


class AliasTable:
    """Union-find over name keys; ``same(a, b)`` is True for aliases."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def _find(self, k: str) -> str:
        self._parent.setdefault(k, k)
        while self._parent[k] != k:
            self._parent[k] = self._parent[self._parent[k]]
            k = self._parent[k]
        return k

    def link(self, a: str, b: str) -> None:
        ra, rb = self._find(key(a)), self._find(key(b))
        if ra != rb:
            self._parent[rb] = ra

    def same(self, a: str, b: str) -> bool:
        ka, kb = key(a), key(b)
        if not ka or not kb:
            return False
        if ka == kb or self._find(ka) == self._find(kb):
            return True
        # "short-term scheduler" vs "short term scheduler"
        return ka.replace(" ", "") == kb.replace(" ", "")

    def names(self, a: str) -> set[str]:
        root = self._find(key(a))
        return {k for k in list(self._parent) if self._find(k) == root}

    def add_from_text(self, text: str) -> None:
        """Record aliases written as "Name (ALIAS)" or "Name (also called Other Name)"."""
        for m in _ALSO_CALLED.finditer(text):
            before = text[: m.start()]
            name = _last_name_before(before)
            if name:
                self.link(name, m.group(1))
        for m in _PAREN.finditer(text):
            inner = m.group(1).strip()
            name = _last_name_before(text[: m.start()])
            if not name:
                continue
            if _is_abbreviation(inner) and _initials_match(name, inner):
                self.link(name, inner)
            elif "," not in inner and _same_head_noun(name, inner):
                self.link(name, inner)


def _last_name_before(before: str) -> str:
    """The noun phrase right before a parenthesis: "... by a Process Control Block" ->
    "Process Control Block" (capitalised words), else the last 1-4 words."""
    before = strip_parens(before).rstrip()
    caps = re.search(r"((?:[A-Z][\w/-]*\s*){1,5})$", before)
    if caps:
        return caps.group(1).strip()
    words = re.findall(r"[\w/-]+", before)
    return " ".join(words[-3:])


def _initials_match(name: str, abbr: str) -> bool:
    letters = "".join(w[0] for w in re.findall(r"[A-Za-z]+", name)).upper()
    return abbr.rstrip("s").upper() in letters


def _same_head_noun(a: str, b: str) -> bool:
    wa, wb = key(a).split(), key(b).split()
    return bool(wa and wb) and wa[-1] == wb[-1] and len(wb) <= 4 and wa != wb


# ---------------------------------------------------------------------------
# line structure


def parse_lines(body: str) -> list[Line]:
    """Split a section body into lines with a bullet level, joining wrapped lines."""
    out: list[Line] = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s:
            continue
        if _SUB.match(s):
            out.append(Line(2, _SUB.sub("", s, count=1).strip()))
        elif _NUMBER.match(s):
            out.append(Line(1, _NUMBER.sub("", s, count=1).strip(), numbered=True))
        elif _BULLET.match(s):
            out.append(Line(1, _BULLET.sub("", s, count=1).strip()))
        elif out and (s[0].islower() or s[0] == "("):
            out[-1].text += " " + s  # a wrapped line continues the previous one
        else:
            out.append(Line(0, s))
    return out


def _clean_term(term: str) -> str:
    return term.strip().strip(":").strip()


def _looks_like_term(term: str) -> bool:
    words = term.split()
    if not 1 <= len(words) <= 6:
        return False
    if term.endswith(":") or term.endswith("?"):
        return False
    if words[0].lower() in NOT_A_TERM_START:
        return False
    return bool(re.match(r"[A-Za-z(]", term))


# ---------------------------------------------------------------------------
# extraction


def _definitions_in(p: Paragraph, concept_name: str) -> list[Definition]:
    lines = parse_lines(p.body)
    found: list[Definition] = []

    def add(term, definition, group, style):
        term, definition = _clean_term(term), definition.strip().rstrip(";")
        m = re.match(r"^(?:an?|the)\s+(?P<t>.+?)\s+is\s+(?:an?\s+|the\s+)?(?P<d>.+)$", term, re.I)
        if m:  # "A program is a passive entity" + "Stored on disk ..."
            term, definition = m.group("t").capitalize(), m.group("d") + "; " + definition
        if _looks_like_term(term) and len(definition.split()) >= 1:
            found.append(Definition(term, definition, group, p.pid, concept_name, p.chapter, style, dict(p.meta)))

    # 1. list items with sub-bullets
    i, list_no = 0, 0
    while i < len(lines):
        ln = lines[i]
        if ln.level == 1:
            subs = []
            j = i + 1
            while j < len(lines) and lines[j].level == 2:
                subs.append(lines[j].text)
                j += 1
            m = _DASH_DEF.match(ln.text)
            c = _COLON_DEF.match(ln.text)
            extra = (" " + ", ".join(subs)) if subs else ""
            if m and (ln.numbered or subs):  # "Hardware - Provides basic computing resources such as:" + subs
                add(m.group("term"), m.group("def") + extra, f"{p.pid}#list{list_no}", "dash")
            elif c and subs and len(c.group("term").split()) <= 4:  # "Handheld devices: optimized for:" + subs
                add(c.group("term"), c.group("def") + extra, f"{p.pid}#colon", "colon")
            elif subs and not ln.text.endswith(":"):
                add(ln.text, "; ".join(subs), f"{p.pid}#list{list_no}", "list")
            i = j
        else:
            if ln.level == 0:
                list_no += 1  # plain text ends a list
            i += 1

    # 2. "• X → Y" items (bullets only; numbered arrows are state changes)
    for ln in lines:
        if ln.level == 1 and not ln.numbered and _ARROW.search(ln.text):
            parts = _ARROW.split(ln.text, maxsplit=1)
            if len(parts) == 2 and len(parts[0].split()) <= 5 and not _ARROW.search(parts[1]):
                add(parts[0], parts[1], f"{p.pid}#arrow", "arrow")

    # 3. "• Term: definition" bullets
    for ln in lines:
        if ln.level == 1:
            m = _COLON_DEF.match(ln.text)
            if m and len(m.group("term").split()) <= 4 and not m.group("def").endswith(":"):
                add(m.group("term"), m.group("def"), f"{p.pid}#colon", "colon")

    # keep groups with at least two members (a lone item has no siblings)
    by_group: dict[str, list[Definition]] = {}
    for d in found:
        by_group.setdefault(d.group, []).append(d)
    return [d for g in by_group.values() if len(g) >= 2 for d in g]


def _facts_in(p: Paragraph) -> list[str]:
    """Bullet points of a section as stand-alone statements."""
    lines = parse_lines(p.body)
    facts: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        subs = []
        j = i + 1
        while j < len(lines) and lines[j].level == 2:
            subs.append(lines[j].text)
            j += 1
        text = ln.text
        if subs:
            text = text.rstrip(":") + ": " + ", ".join(subs)
        elif text.endswith(":"):
            i = j
            continue  # an intro line with nothing under it
        n = len(text.split())
        if 3 <= n <= 30 and not text.endswith("?"):
            facts.append(text)
        i = j
    return facts


DETAIL_WORDS = {"functions", "function", "details", "contents", "content", "of", "the", "a",
                "types", "features", "components", "parts", "steps"}


def _is_detail_of(heading: str, previous: str | None) -> bool:
    """True for "Dispatcher Functions" after "Dispatcher", "Contents of Process
    Control Block" after "Process Control Block"."""
    if not previous:
        return False
    h, prev = key(concept_name(heading)), key(previous)
    if not prev or prev not in h or h == prev:
        return False
    rest = set(h.replace(prev, " ").split())
    return bool(rest) and rest <= DETAIL_WORDS


def concept_name(heading: str) -> str:
    return strip_parens(heading)


def _is_concept_heading(heading: str) -> bool:
    name = concept_name(heading)
    words = name.split()
    if not 1 <= len(words) <= 5:
        return False
    if key(name) in GENERIC_HEADINGS or "–" in name or " - " in name:
        return False
    return words[0].lower() not in {"what", "why", "how", "when", "which", "one", "four", "two"}


def extract_concepts(paragraphs: list[Paragraph]) -> ConceptSet:
    """All definitions, section concepts and aliases in the given sections (in order)."""
    aliases = AliasTable()
    for p in paragraphs:
        aliases.add_from_text(p.heading)
        aliases.add_from_text(p.body.replace("\n", " "))

    definitions: list[Definition] = []
    merged: dict[str, SectionConcept] = {}
    order: list[str] = []
    last_concept: str | None = None
    for p in paragraphs:
        generic = key(p.heading) in GENERIC_HEADINGS or _is_detail_of(p.heading, last_concept)
        name = last_concept if (generic and last_concept) else concept_name(p.heading)
        definitions.extend(_definitions_in(p, name))
        if not generic:
            last_concept = name
        if not (generic and last_concept) and not _is_concept_heading(p.heading):
            continue
        canon = next((k for k in order if aliases.same(k, name)), None)
        if canon is None:
            canon = name
            order.append(canon)
            merged[canon] = SectionConcept(canon, set(), [], [], p.chapter, dict(p.meta))
        sc = merged[canon]
        for m in _PAREN.finditer(p.heading):
            if aliases.same(m.group(1), canon):
                sc.aliases.add(m.group(1).strip())
        if name != canon:
            sc.aliases.add(name)
        sc.facts.extend(f for f in _facts_in(p) if f not in sc.facts)
        sc.pids.append(p.pid)
    sections = [merged[k] for k in order if len(merged[k].facts) >= 1]
    return ConceptSet(definitions, sections, aliases)
