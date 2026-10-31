"""Build the evaluation corpus from Bhavya's document trees (Member 4).

    python experiments/build_corpus.py

Reads data/processed/unit1_topic*/tree.json (Member 1), turns each tree into
sections with Hemanth's ``sections_from_tree`` (Member 2: heading + body, bullet
marks and wrapped lines repaired), and writes one record per section to
data/processed/corpus_unit1_sections.jsonl.

Why sections and not tree.json's paragraph nodes: in the Unit 1 trees a
"paragraph" is one line or bullet (median 3 words, some just "•"). Sections are
what the MCQ pipeline actually retrieves (src/mcq/context.py), so they are the
fair unit to evaluate. Headings stay in their own field, so BM25F can use them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.indexing.tree_sections import sections_from_tree  # noqa: E402

DOCS = ["unit1_topic1", "unit1_topic2", "unit1_topic3"]
OUT = REPO / "data" / "processed" / "corpus_unit1_sections.jsonl"


def main() -> None:
    records = []
    for doc in DOCS:
        sections = sections_from_tree(REPO / "data" / "processed" / doc / "tree.json")
        for s in sections:
            records.append({
                "paragraph_id": s.pid,
                "document_id": doc,
                "section_title": s.heading,
                "chapter_title": s.chapter,
                "page_number": s.meta.get("page_start"),
                "text": s.body,
            })
        print(f"{doc}: {len(sections)} sections")
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    no_heading = sum(not r["section_title"] for r in records)
    footers = sum("pes.edu" in r["text"] for r in records)
    print(f"Wrote {len(records)} sections to {OUT} ({no_heading} without a heading, {footers} with page footers)")


if __name__ == "__main__":
    main()
