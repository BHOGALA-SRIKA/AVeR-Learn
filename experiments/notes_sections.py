"""Stand-in PDF reader for the MCQ demo: notes PDF -> sections (heading + lines).

Bhavya's document tree replaces this in the pipeline. It only exists so the MCQ
pieces can be run on the real notes today. It relies on how the notes are typed:
headings are short bold lines, "•" starts a bullet and "o" a sub-bullet, and each
page has a "name (email) page-number" footer that is dropped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402

from src.indexing.records import Paragraph  # noqa: E402

BULLET = "•"


def _lines(pdf_path):
    doc = pymupdf.open(pdf_path)
    for pno, page in enumerate(doc, 1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = re.sub(r"\s+", " ", "".join(s["text"] for s in spans)).strip()
                text = re.sub(r"^o(?=[A-Z0-9(])", "o ", text)  # sub-bullet glued to its word
                bold = all(s["flags"] & 16 for s in spans)
                font = spans[0]["font"]
                yield pno, text, bold, font


def read_sections(pdf_path, doc_title: str | None = None) -> list[Paragraph]:
    """One Paragraph per section. Lines keep their bullet markers ("• ", "o ")."""
    name = Path(pdf_path).name
    sections, current, after_bullet = [], None, False
    title_lines = []
    for pno, text, bold, font in _lines(pdf_path):
        if font.startswith("Calibri"):  # footer: author line and page number
            continue
        if text == BULLET:
            after_bullet = True
            continue
        is_heading = bold and not after_bullet and len(text.split()) <= 12 and not text.endswith(":")
        if is_heading and current is None and len(title_lines) < 2:
            title_lines.append(text)  # course title, unit title
            continue
        if is_heading:
            current = {"heading": text, "lines": [], "page_start": pno, "page_end": pno}
            sections.append(current)
        elif current is not None:
            prefix = "• " if after_bullet else ""
            current["lines"].append(prefix + text)
            current["page_end"] = pno
        after_bullet = False

    chapter = doc_title or (title_lines[-1] if title_lines else name)
    out = []
    for i, s in enumerate(sections):
        body = "\n".join(s["lines"]).strip()
        if not body:
            continue
        out.append(Paragraph(
            pid=f"{Path(name).stem}-s{i:02d}", body=body, heading=s["heading"], chapter=chapter,
            meta={"source_pdf": name, "page_start": s["page_start"], "page_end": s["page_end"]},
        ))
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        for p in read_sections(path):
            print(f"== {p.pid} [{p.chapter} > {p.heading}] p{p.meta['page_start']}")
            print(p.body)
            print()
