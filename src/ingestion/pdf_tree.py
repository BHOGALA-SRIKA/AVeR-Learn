import pymupdf
import re
import math

from collections import defaultdict
from pathlib import Path


# ============================================================
# 1. TEXT NORMALIZATION
# ============================================================

def clean_text(text: str) -> str:
    """Normalize extracted text."""

    text = text.replace("\u00a0", " ")
    text = text.replace("\r", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Normalize text for repeated-header comparison."""

    return re.sub(
        r"\s+",
        " ",
        text.casefold()
    ).strip()


# ============================================================
# 2. FONT AND FORMATTING INFORMATION
# ============================================================

def is_bold(span: dict) -> bool:
    """Check whether a text span is bold."""

    return bool(span["flags"] & 16)


# ============================================================
# 3. PDF PAGE LAYOUT EXTRACTION
# ============================================================

def extract_page_layout(page, page_number: int) -> list[dict]:
    """
    Extract text lines with formatting and position information.
    """

    page_data = page.get_text("dict")

    records = []

    for block in page_data["blocks"]:

        if "lines" not in block:
            continue

        for line in block["lines"]:

            spans = line["spans"]

            text = "".join(
                span["text"] for span in spans
            )

            text = clean_text(text)

            if not text:
                continue

            bbox = line["bbox"]

            font_sizes = [
                span["size"]
                for span in spans
                if span["text"].strip()
            ]

            if not font_sizes:
                continue

            records.append({

                "text": text,

                "page_number": page_number,

                "bbox": {
                    "x0": round(bbox[0], 2),
                    "y0": round(bbox[1], 2),
                    "x1": round(bbox[2], 2),
                    "y1": round(bbox[3], 2)
                },

                "font_size": round(
                    max(font_sizes), 2
                ),

                "bold": any(
                    is_bold(span) for span in spans
                ),

                "fonts": list({
                    span["font"] for span in spans
                }),

                "span_count": len(spans)

            })

    # Preserve approximate reading order.
    records.sort(
        key=lambda item: (
            item["bbox"]["y0"],
            item["bbox"]["x0"]
        )
    )

    return records


# ============================================================
# 4. REPEATED HEADER AND FOOTER DETECTION
# ============================================================

def detect_running_headers(pages: list[dict]) -> set[str]:
    """
    Identify repeated text near page boundaries.

    A candidate must appear near the top or bottom
    of multiple pages.
    """

    occurrences = defaultdict(set)

    for page in pages:

        page_number = page["page_number"]
        page_height = page["height"]

        for line in page["lines"]:

            text = normalize_for_matching(
                line["text"]
            )

            if len(text) < 5:
                continue

            y0 = line["bbox"]["y0"]

            near_top = y0 <= page_height * 0.15

            near_bottom = y0 >= page_height * 0.85

            if near_top or near_bottom:

                occurrences[text].add(page_number)

    minimum_pages = max(
        2,
        math.ceil(len(pages) * 0.30)
    )

    return {
        text
        for text, page_numbers in occurrences.items()
        if len(page_numbers) >= minimum_pages
    }


def mark_running_headers(pages: list[dict]) -> set[str]:
    """
    Mark repeated headers without deleting source records.
    """

    headers = detect_running_headers(pages)

    for page in pages:

        for line in page["lines"]:

            normalized = normalize_for_matching(
                line["text"]
            )

            line["is_running_header"] = (
                normalized in headers
            )

    return headers


# ============================================================
# 5. HEADING PATTERN DETECTION
# ============================================================

def is_numbered_heading(text: str) -> bool:
    """
    Check whether text follows common heading-numbering patterns.
    """

    patterns = [

        r"^(chapter|unit|module)\s+[\w.-]+",

        r"^\d+(\.\d+)*\s+[A-Z]",

        r"^unit\s*[-:]?\s*\d+"

    ]

    return any(
        re.match(
            pattern,
            text.strip(),
            re.IGNORECASE
        )
        for pattern in patterns
    )


def looks_like_sentence(text: str) -> bool:
    """
    Identify lines that look like complete sentences.
    """

    if re.search(r"[.!?]$", text.strip()):
        return True

    sentence_starters = [

        "the ",
        "this ",
        "these ",
        "those ",
        "it ",
        "they ",
        "we ",
        "a ",
        "an "

    ]

    return text.lower().startswith(
        tuple(sentence_starters)
    )


# ============================================================
# 6. DOCUMENT METADATA DETECTION
# ============================================================

def is_metadata(text: str) -> bool:
    """
    Identify common document metadata.

    Examples:
    - Email addresses
    - Department information
    - Textbook labels
    - Reference book labels
    - Closing slide text
    """

    metadata_patterns = [

        r"^[\w.-]+@[\w.-]+\.\w+$",

        r"^department of ",

        r"^text book\s*:",

        r"^reference book\s*:",

        r"^evaluation policy$",

        r"^thank you$"

    ]

    return any(
        re.match(
            pattern,
            text.strip(),
            re.IGNORECASE
        )
        for pattern in metadata_patterns
    )


# ============================================================
# 7. HEADING CANDIDATE DETECTION
# ============================================================

def detect_heading_candidates(pages: list[dict]) -> None:
    """
    Identify possible headings using:

    1. Bold formatting
    2. Relative font size
    3. Numbering patterns
    4. Text length
    5. Sentence structure
    6. Bullet/list detection
    7. Metadata detection

    Classifications:

    strong_candidate
    review_candidate
    not_heading

    This function does not assign hierarchy levels.
    """

    # --------------------------------------------------------
    # Calculate median font size separately for each page.
    # --------------------------------------------------------

    page_medians = {}

    for page in pages:

        sizes = [

            line["font_size"]

            for line in page["lines"]

            if not line.get(
                "is_running_header",
                False
            )

        ]

        if sizes:

            sizes.sort()

            page_medians[page["page_number"]] = (
                sizes[len(sizes) // 2]
            )

    # --------------------------------------------------------
    # Evaluate every extracted line.
    # --------------------------------------------------------

    for page in pages:

        median_font_size = page_medians.get(
            page["page_number"],
            0
        )

        for line in page["lines"]:

            text = line["text"].strip()

            score = 0

            reasons = []

            # ------------------------------------------------
            # Ignore repeated headers.
            # ------------------------------------------------

            if line.get(
                "is_running_header",
                False
            ):

                line["heading_status"] = "not_heading"
                line["heading_score"] = 0
                line["heading_reasons"] = [
                    "running_header"
                ]

                line["is_heading_candidate"] = False

                continue

            # ------------------------------------------------
            # Ignore document metadata.
            # ------------------------------------------------

            if is_metadata(text):

                line["heading_status"] = "not_heading"
                line["heading_score"] = 0
                line["heading_reasons"] = [
                    "document_metadata"
                ]

                line["is_heading_candidate"] = False

                continue

            # ------------------------------------------------
            # Ignore extremely short text.
            # ------------------------------------------------

            if len(text) < 3:

                line["heading_status"] = "not_heading"
                line["heading_score"] = 0
                line["heading_reasons"] = [
                    "too_short"
                ]

                line["is_heading_candidate"] = False

                continue

            is_bullet_item = bool(
                re.match(
                    r"^\s*[➢▪•●◦▫‣⁃\-–—*]\s*",
                    text
                )
            )

            is_indented_list_item = (
                line["bbox"]["x0"] >= 85
                and len(text.split()) <= 12
            )

            if is_bullet_item or is_indented_list_item:
                line["heading_status"] = "not_heading"
                line["heading_score"] = 0
                line["heading_reasons"] = ["list_item"]
                line["is_heading_candidate"] = False
                continue

            # ------------------------------------------------
            # Formatting evidence: bold.
            # ------------------------------------------------

            if line["bold"]:

                score += 2

                reasons.append("bold")

            # ------------------------------------------------
            # Relative font size.
            # ------------------------------------------------

            if (
                median_font_size > 0
                and line["font_size"] > median_font_size * 1.05
            ):

                score += 2

                reasons.append("larger_font")

            # ------------------------------------------------
            # Structural numbering.
            # ------------------------------------------------

            if is_numbered_heading(text):

                score += 2

                reasons.append("numbered_heading")

            # ------------------------------------------------
            # Short text.
            # ------------------------------------------------

            if len(text) <= 100:

                score += 1

                reasons.append("short_text")

            # ------------------------------------------------
            # Heading-like colon.
            # ------------------------------------------------

            if (
                text.endswith(":")
                and len(text) <= 60
            ):

                score += 1

                reasons.append("heading_like_colon")

            # ------------------------------------------------
            # Sentence-like text penalty.
            # ------------------------------------------------

            if looks_like_sentence(text):

                score -= 2

                reasons.append("sentence_like")

            # ------------------------------------------------
            # Long text penalty.
            # ------------------------------------------------

            if len(text) > 150:

                score -= 2

                reasons.append("long_text")

            # ------------------------------------------------
            # Bullet/list item penalty.
            # ------------------------------------------------

            if re.match(
                r"^[➢▪•●◦\-]\s*",
                text
            ):

                score -= 3

                reasons.append("bullet_item")

            # ------------------------------------------------
            # Ordinary numbered list penalty.
            # ------------------------------------------------

            if re.match(
                r"^\d+\.\s+",
                text
            ):

                if not is_numbered_heading(text):

                    score -= 2

                    reasons.append(
                        "possible_numbered_list"
                    )

            # ------------------------------------------------
            # Explanatory text after a colon.
            # ------------------------------------------------

            if (
                ":" in text
                and len(text) > 60
            ):

                score -= 2

                reasons.append(
                    "explanatory_colon"
                )

            # ------------------------------------------------
            # Classification.
            # ------------------------------------------------

            if score >= 5:

                status = "strong_candidate"

            elif score >= 3:

                status = "review_candidate"

            else:

                status = "not_heading"

            # ------------------------------------------------
            # Save classification in original line record.
            # ------------------------------------------------

            line["heading_score"] = score

            line["heading_reasons"] = reasons

            line["heading_status"] = status

            line["is_heading_candidate"] = (
                status != "not_heading"
            )

# ============================================================
# 8. HEADING GROUPING
# ============================================================

def is_incomplete_heading(text: str) -> bool:
    """
    Return True when a short text fragment plausibly needs continuation.

    This is intentionally conservative. It is not a linguistic parser;
    it only supplies evidence to the layout-based grouping heuristic.
    """

    text = clean_text(text)

    if not text:
        return False

    # These endings usually indicate a completed heading or label.
    if text.endswith((".", "?", "!", ":", ")", "]", ";")):
        return False

    # Very long lines are unlikely to be incomplete heading fragments.
    return len(text.split()) <= 4 and len(text) <= 45


def starts_with_lowercase(text: str) -> bool:
    """Check whether a continuation fragment begins with lowercase text."""

    text = clean_text(text)
    return bool(text) and text[0].islower()


def can_merge_heading_lines(
    first: dict,
    second: dict,
    page_height: float
) -> bool:
    """
    Conservatively determine whether two adjacent extracted lines belong
    to one heading.

    Two patterns are supported:
      1. Vertically stacked continuation lines.
      2. Horizontally adjacent fragments on approximately the same baseline.

    Both patterns require compatible formatting and a plausible incomplete
    first fragment. Lines are never merged across pages.
    """

    if not first.get("is_heading_candidate", False):
        return False

    if not second.get("is_heading_candidate", False):
        return False

    if first.get("is_running_header", False):
        return False

    if second.get("is_running_header", False):
        return False

    if first.get("page_number") != second.get("page_number"):
        return False

    first_box = first["bbox"]
    second_box = second["bbox"]

    # Formatting must be reasonably consistent.
    size1 = float(first.get("font_size", 0))
    size2 = float(second.get("font_size", 0))

    if size1 <= 0 or size2 <= 0:
        return False

    if abs(size1 - size2) > max(size1, size2) * 0.10:
        return False

    if first.get("bold", False) != second.get("bold", False):
        return False

    # Do not join a fragment that already looks complete.
    if not is_incomplete_heading(first.get("text", "")):
        return False

    # Case 1: vertically stacked continuation.
    vertical_gap = second_box["y0"] - first_box["y1"]
    same_left_alignment = abs(first_box["x0"] - second_box["x0"]) <= 35

    vertical_continuation = (
        same_left_alignment
        and -3 <= vertical_gap <= page_height * 0.035
        and starts_with_lowercase(second.get("text", ""))
    )

    if vertical_continuation:
        return True

    # Case 2: horizontally adjacent fragments on the same baseline.
    # A wider gap is allowed because PDF text spans can have separate
    # bounding boxes even when they form one visual phrase.
    same_baseline = abs(first_box["y0"] - second_box["y0"]) <= 3
    horizontal_gap = second_box["x0"] - first_box["x1"]

    horizontal_continuation = (
        same_baseline
        and -2 <= horizontal_gap <= 45
        and starts_with_lowercase(second.get("text", ""))
    )

    return horizontal_continuation


def group_heading_lines(pages: list[dict]) -> list[dict]:
    """
    Combine consecutive heading lines into grouped headings.

    Original page lines remain unchanged.

    Returns a separate list of heading records.
    """

    grouped_headings = []

    for page in pages:

        lines = page["lines"]

        page_height = page["height"]

        i = 0

        while i < len(lines):

            current = lines[i]

            if not current.get(
                "is_heading_candidate",
                False
            ):
                i += 1
                continue

            group = [current]

            j = i + 1

            while j < len(lines):

                next_line = lines[j]

                if can_merge_heading_lines(
                    group[-1],
                    next_line,
                    page_height
                ):

                    group.append(next_line)

                    j += 1

                else:
                    break

            # Combine the text of grouped lines.
            combined_text = clean_text(
                " ".join(
                    item["text"]
                    for item in group
                )
            )

            # Calculate combined bounding box.
            bbox = {

                "x0": min(
                    item["bbox"]["x0"]
                    for item in group
                ),

                "y0": min(
                    item["bbox"]["y0"]
                    for item in group
                ),

                "x1": max(
                    item["bbox"]["x1"]
                    for item in group
                ),

                "y1": max(
                    item["bbox"]["y1"]
                    for item in group
                )

            }

            grouped_headings.append({

                "text": combined_text,

                "page_number": page["page_number"],

                "bbox": bbox,

                "font_size": max(
                    item["font_size"]
                    for item in group
                ),

                "bold": all(
                    item["bold"]
                    for item in group
                ),

                "heading_score": min(
                    item["heading_score"]
                    for item in group
                ),

                "heading_status": (
                    "strong_candidate"
                    if all(
                        item["heading_status"] == "strong_candidate"
                        for item in group
                    )
                    else "review_candidate"
                ),

                "source_lines": [
                    item["text"]
                    for item in group
                ],

                "source_line_count": len(group)

            })

            i = j

    return grouped_headings

# ============================================================
# 8. COMPLETE PDF EXTRACTION PIPELINE
# ============================================================

def extract_pdf_layout(pdf_path: str | Path) -> dict:
    """
    Extract layout information from an entire PDF.

    Pipeline:

    1. Extract page text and layout.
    2. Detect repeated headers.
    3. Mark repeated headers.
    4. Detect heading candidates.
    5. Return structured extraction results.
    """

    pdf_path = Path(pdf_path)

    pages = []

    with pymupdf.open(pdf_path) as document:

        for page_number, page in enumerate(
            document,
            start=1
        ):

            records = extract_page_layout(
                page,
                page_number
            )

            pages.append({

                "page_number": page_number,

                "width": round(
                    page.rect.width,
                    2
                ),

                "height": round(
                    page.rect.height,
                    2
                ),

                "image_count": len(
                    page.get_images(full=True)
                ),

                "lines": records

            })

    # Detect repeated headers.
    running_headers = mark_running_headers(pages)

    # Detect heading candidates.
    detect_heading_candidates(pages)

    # Group heading lines.
    grouped_headings = group_heading_lines(pages)

    return {
        "document_id": pdf_path.stem,
        "source_file": pdf_path.name,
        "total_pages": len(pages),
        "running_headers": sorted(running_headers),
        "pages": pages,
        "grouped_headings": grouped_headings
    }


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    pdf_folder = Path("data/raw")

    for pdf_path in pdf_folder.rglob("*.pdf"):

        print("\n" + "=" * 60)

        print(f"Processing: {pdf_path.name}")

        print("=" * 60)

        result = extract_pdf_layout(pdf_path)

        total_lines = sum(
            len(page["lines"])
            for page in result["pages"]
        )

        print("Pages:", result["total_pages"])

        print("Extracted lines:", total_lines)

        print(
            "Detected running headers:",
            result["running_headers"]
        )

        print("\nHeading candidates:")

        for page in result["pages"]:

            for line in page["lines"]:

                if line["is_heading_candidate"]:

                    print(
                        "Page:",
                        page["page_number"],
                        "| Status:",
                        line["heading_status"],
                        "| Score:",
                        line["heading_score"],
                        "| Text:",
                        line["text"]
                    )