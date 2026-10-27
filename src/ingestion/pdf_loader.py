import re
from pathlib import Path

import pymupdf


def detect_subject(pdf_path: Path) -> str:
    """Infer the subject from a parent directory."""
    for parent in pdf_path.parents:
        if parent.name.upper() in {"OS", "DBMS"}:
            return parent.name.upper()

    return "UNKNOWN"


def clean_page_text(text: str) -> str:
    """Normalize whitespace while preserving line boundaries."""
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n")]

    return "\n".join(lines).strip()


def is_heading(line: str) -> bool:
    """
    Detect likely headings using conservative heuristics.

    This is a starting point; PDF-specific heading detection
    may require additional rules.
    """
    line = line.strip()

    if not line or len(line) > 90:
        return False

    # Ignore list items.
    if re.match(r"^(?:[-•*]|\d+[.)])\s", line):
        return False

    # Ignore ordinary sentence endings.
    if line.endswith((".", ",", ";")):
        return False

    # Known headings from the example OS material.
    known_headings = {
        "introduction",
        "need for an operating system",
        "general definition of an operating system",
        "genesis of operating systems",
        "computer system structure",
        "computer system operation",
        "a modern computer system",
        "booting the system",
        "interrupts",
        "interrupt handling",
        "interrupt timeline",
        "storage structure",
        "storage structure – main memory",
        "storage hierarchy",
        "direct memory access (dma)",
        "i/o operation flow",
    }

    if line.lower() in known_headings:
        return True

    # General heading heuristic.
    words = line.split()

    if len(words) <= 7 and line.isupper():
        return True

    return False


def split_large_text(text: str, max_words: int = 100) -> list[str]:
    """Split oversized content into bounded chunks."""
    words = text.split()
    chunks = []

    for start in range(0, len(words), max_words):
        chunk = " ".join(words[start:start + max_words]).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def extract_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract heading-aware chunks from a PDF."""
    pdf_path = Path(pdf_path)

    records = []
    chunk_number = 0

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = clean_page_text(page.get_text("text"))

            current_heading = ""
            current_lines = []

            def save_current_section():
                nonlocal chunk_number

                content = " ".join(current_lines).strip()

                if not content:
                    return

                chunks = split_large_text(content)

                for chunk in chunks:
                    chunk_number += 1

                    records.append({
                        "paragraph_id": (
                            f"{pdf_path.stem}_p{chunk_number:05d}"
                        ),
                        "document_id": pdf_path.stem,
                        "subject": detect_subject(pdf_path),
                        "page_number": page_number,
                        "heading_path": current_heading,
                        "text": chunk,
                    })

            for line in text.splitlines():
                line = line.strip()

                if not line:
                    continue

                if is_heading(line):
                    save_current_section()
                    current_lines = []
                    current_heading = line
                else:
                    current_lines.append(line)

            save_current_section()

    return records


if __name__ == "__main__":
    pdf_folder = Path("data/raw")

    for pdf_path in pdf_folder.rglob("*.pdf"):
        records = extract_pdf(pdf_path)

        print(f"\nFile: {pdf_path}")
        print(f"Extracted chunks: {len(records)}")

        for record in records[:5]:
            print("\n", record)