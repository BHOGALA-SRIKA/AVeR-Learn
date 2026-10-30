"""Turn paragraph records into the three text fields the lexical methods index.

Every paragraph becomes:

- ``body``    - the paragraph text
- ``heading`` - its nearest heading (last breadcrumb part), e.g. "Interrupt Handling"
- ``chapter`` - its top-level heading (first breadcrumb part)

Accepted record shapes (dicts):

1. Agreed schema (Architecture and Interface Agreement, section 4): ``id``,
   ``kind``, ``text`` and ``breadcrumb`` as a string such as
   "Unit 1 > Introduction > Processes". Non-paragraph nodes are skipped.
2. The current ``data/processed/paragraphs.json`` (``paragraph_id``,
   ``heading_path``), so the code runs today before Bhavya's tree lands.
3. The draft plan format (``chapter_title``, ``section_title``).

Integration note for Bhavya (Member 1): if the breadcrumb starts with the document
title (level 0), set ``BREADCRUMB_SKIP_ROOT = True`` so ``chapter`` is the
chapter title rather than the document title.

``read_paragraphs`` is only a small file reader for this package's tests and
demo. In the pipeline, records come from Bhavya's
``src.ingestion.tree.load_paragraphs`` and are passed to ``LexicalIndex.build``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

FIELDS = ("body", "heading", "chapter")
BREADCRUMB_SEP = ">"
BREADCRUMB_SKIP_ROOT = False


@dataclass(frozen=True)
class Paragraph:
    pid: str
    body: str
    heading: str = ""
    chapter: str = ""
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def field_text(self, name: str) -> str:
        return getattr(self, name)


def _breadcrumb_parts(breadcrumb: Any) -> list[str]:
    if isinstance(breadcrumb, str):
        parts = [p.strip() for p in breadcrumb.split(BREADCRUMB_SEP)]
    elif isinstance(breadcrumb, (list, tuple)):
        parts = [str(p).strip() for p in breadcrumb]
    else:
        return []
    parts = [p for p in parts if p]
    if BREADCRUMB_SKIP_ROOT and len(parts) > 1:
        parts = parts[1:]
    return parts


def to_paragraph(record: Any) -> Paragraph | None:
    """Convert one record to a Paragraph; None for non-paragraph or empty records."""
    if isinstance(record, Paragraph):
        return record
    text = (record.get("text") or "").strip()
    pid = record.get("id") or record.get("paragraph_id")
    if not pid or not text:
        return None
    kind = record.get("kind")
    if kind is not None and kind != "paragraph":
        return None

    parts = _breadcrumb_parts(record.get("breadcrumb"))
    if parts:
        heading, chapter = parts[-1], parts[0]
    else:
        heading = record.get("section_title") or record.get("heading_path") or record.get("title") or ""
        chapter = record.get("chapter_title") or ""

    meta = {k: v for k, v in record.items() if k not in {"text", "children"}}
    return Paragraph(pid=str(pid), body=text, heading=heading, chapter=chapter, meta=meta)


def to_paragraphs(records: Iterable[Any]) -> list[Paragraph]:
    """Convert records, keep corpus order, and reject duplicate IDs."""
    paragraphs = [p for p in map(to_paragraph, records) if p is not None]
    if not paragraphs:
        raise ValueError("No paragraph records with text were found.")
    seen: set[str] = set()
    for p in paragraphs:
        if p.pid in seen:
            raise ValueError(f"Duplicate paragraph id: {p.pid}")
        seen.add(p.pid)
    return paragraphs


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a .jsonl file, a JSON list, or a JSON object with a "nodes" list/dict."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in fh if line.strip()]
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "nodes" in data:
        nodes = data["nodes"]
        return list(nodes.values() if isinstance(nodes, dict) else nodes)
    raise ValueError(f"Unrecognised corpus format in {path}")


def read_paragraphs(path: str | Path) -> list[Paragraph]:
    return to_paragraphs(read_records(path))
