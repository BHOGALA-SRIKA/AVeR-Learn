"""Read Bhavya's document tree (tree.json) as sections for retrieval and MCQs.

The MCQ pipeline works on sections: a heading plus its body text with bullet
marks, e.g. heading "Dispatcher" and body "• Gives control of the CPU ...".
Bhavya's ingestion (feature/member1-ingestion) stores one record per line, which
matches the Interface Agreement, so this module joins the lines back into sections.

It also repairs three patterns seen in the first version of tree.json (Unit 1
notes, 2026-10-30), so retrieval works now and keeps working after they are fixed:

1. Bullet marks are separate paragraphs ("•") that come right AFTER the line they
   belong to. They are attached to that line.
2. Bold bullet items are stored as section headings ("Performs context switch",
   "Process state"). A "heading" that is followed by a "•" mark, starts with "o ",
   ends with ":" or is longer than 10 words is treated as a line of the section
   before it, not as a new section.
3. A wrapped sentence is stored as a heading ("... only one process can execute
   at a time. All" followed by "other processes must wait ..."). A "heading" whose
   next line starts in lower case is joined to that line.
4. Footer page numbers ("4") are stored as paragraphs; they are dropped.

``sections_from_tree`` returns ``Paragraph`` records (pid = the section node id),
so everything else (LexicalIndex, ContextRetriever, generate_mcqs) is unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.indexing.records import Paragraph

BULLET_MARKS = {"•", "●", "▪", "◦", "-", "–", "o"}
PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d{1,3}$", re.I)
MAX_HEADING_WORDS = 10


def _load(tree: str | Path | dict) -> dict:
    if isinstance(tree, dict):
        return tree
    return json.loads(Path(tree).read_text(encoding="utf-8"))


def _nodes(tree: dict) -> list[dict[str, Any]]:
    nodes = tree["nodes"]
    nodes = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    return sorted(nodes, key=lambda n: n.get("doc_order", 0))


def _text(node: dict) -> str:
    return " ".join(str(node.get("title") or node.get("text") or "").split())


def sections_from_tree(tree: str | Path | dict, max_heading_words: int = MAX_HEADING_WORDS) -> list[Paragraph]:
    """Sections (heading + body) from one document's tree.json, in reading order."""
    t = _load(tree)
    nodes = _nodes(t)
    source = t.get("source_file") or t.get("document_id")

    # 1. one entry per line in reading order; "•" marks the line before it as a bullet
    items: list[dict[str, Any]] = []
    for n in nodes:
        kind = n.get("kind")
        if kind in ("document",):
            continue
        text = _text(n)
        if not text:
            continue
        if kind == "paragraph" and text in BULLET_MARKS:
            if items:
                items[-1]["bullet"] = True
            continue
        if kind == "paragraph" and PAGE_NUMBER.match(text):
            continue  # a page number from the footer, not content
        items.append({"node": n, "text": text, "kind": kind, "bullet": False})

    # 2. decide which "headings" are real
    chapter = ""
    for i, it in enumerate(items):
        nxt = items[i + 1] if i + 1 < len(items) else None
        if it["kind"] == "chapter":
            chapter = it["text"]
            it["role"] = "chapter"
            continue
        it["chapter"] = chapter
        if it["kind"] in ("section", "subsection"):
            text = it["text"]
            fake = (
                it["bullet"]
                or text.startswith("o ")
                or text.endswith(":")
                or len(text.split()) > max_heading_words
                or (nxt is not None and nxt["text"][:1].islower())
            )
            it["role"] = "line" if fake else "heading"
        else:
            it["role"] = "line"

    # 3. group lines under the real headings
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for it in items:
        role = it["role"]
        if role == "chapter":
            continue
        if role == "heading":
            n = it["node"]
            current = {"id": n["id"], "heading": it["text"], "chapter": it["chapter"], "lines": [],
                       "pages": [n.get("page_start"), n.get("page_end")], "node_ids": [n["id"]]}
            sections.append(current)
            continue
        if current is None:  # text before the first heading
            n = it["node"]
            current = {"id": n.get("parent_id") or n["id"], "heading": it["chapter"], "chapter": it["chapter"],
                       "lines": [], "pages": [n.get("page_start"), n.get("page_end")], "node_ids": []}
            sections.append(current)
        line = it["text"]
        if current["lines"] and line[:1].islower() and not line.startswith("o "):
            current["lines"][-1] += " " + line  # wrapped sentence
        else:
            current["lines"].append(("• " if it["bullet"] and not line.startswith("o ") else "") + line)
        n = it["node"]
        current["node_ids"].append(n["id"])
        pages = [p for p in (n.get("page_start"), n.get("page_end")) if p]
        if pages:
            current["pages"] = [min([p for p in current["pages"] if p] + pages),
                                max([p for p in current["pages"] if p] + pages)]

    out = []
    for s in sections:
        body = "\n".join(s["lines"]).strip()
        if not body:
            continue
        out.append(Paragraph(
            pid=s["id"], body=body, heading=s["heading"], chapter=s["chapter"],
            meta={"source_pdf": source, "page_start": s["pages"][0], "page_end": s["pages"][1],
                  "node_ids": s["node_ids"]},
        ))
    return out


def sections_from_trees(paths: list[str | Path]) -> list[Paragraph]:
    out: list[Paragraph] = []
    for p in paths:
        out.extend(sections_from_tree(p))
    return out
