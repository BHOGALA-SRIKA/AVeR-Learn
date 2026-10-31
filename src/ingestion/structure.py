import re
from collections import Counter, defaultdict


# ============================================================
# 1. DOCUMENT TYPE DETECTION
# ============================================================

def detect_document_type(document: dict) -> dict:
    """
    Determine whether the input is lecture slides or textbook-style notes.

    This is a heuristic classification based on page density and
    repeated headers. The result should be treated as provisional.
    """

    pages = document.get("pages", [])

    if not pages:
        return {
            "document_type": "unknown",
            "confidence": "low",
            "reasons": ["No pages found"]
        }

    total_pages = len(pages)

    average_lines = sum(
        len(page.get("lines", []))
        for page in pages
    ) / total_pages

    short_pages = sum(
        len(page.get("lines", [])) <= 8
        for page in pages
    )

    short_page_ratio = short_pages / total_pages

    score = 0
    reasons = []

    if short_page_ratio >= 0.40:
        score += 1
        reasons.append("Many pages contain little text")

    if average_lines <= 12:
        score += 1
        reasons.append("Low average lines per page")

    if document.get("running_headers"):
        score += 1
        reasons.append("Repeated page headers detected")

    if score >= 2:
        return {
            "document_type": "lecture_slides",
            "confidence": "medium",
            "reasons": reasons
        }

    return {
        "document_type": "textbook_notes",
        "confidence": "medium",
        "reasons": reasons
    }


# ============================================================
# 2. TEXT HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""

    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def is_metadata_text(text: str) -> bool:
    """Identify common document metadata."""

    text = text.strip()

    patterns = [
        r"^[\w.+-]+@[\w.-]+\.\w+$",
        r"^department of ",
        r"^reference book\s*:?",
        r"^text book\s*:?",
        r"^evaluation policy$",
        r"^thank you$",
        r"^references$",
        r"^operating systems\s*\([^)]+\)$",
    ]

    return any(
        re.match(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def is_chapter_heading(text: str) -> bool:
    """Identify explicitly numbered chapters, units and modules."""

    patterns = [
        r"^\s*chapter\s+[\w.-]+",
        r"^\s*unit\s*[-:]?\s*\d+",
        r"^\s*module\s*[-:]?\s*\d+",
    ]

    return any(
        re.match(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def is_subsection_heading(text: str) -> bool:
    """Identify headings that commonly introduce subsection content."""

    patterns = [
        r"^\s*from the .+ perspective\s*:?\s*$",
        r"^\s*examples?\s*:?\s*$",
        r"^\s*applications?\s*:?\s*$",
    ]

    return any(
        re.match(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def is_course_outline(text: str) -> bool:
    """Identify numbered unit entries."""

    return bool(
        re.match(
            r"^\s*unit\s*[-:]?\s*\d+",
            text,
            re.IGNORECASE
        )
    )


# ============================================================
# 3. PAGE CONTEXT
# ============================================================

def get_page_texts(document: dict) -> dict:
    """Return extracted text indexed by page number."""

    page_texts = {}

    for page in document.get("pages", []):

        page_number = page["page_number"]

        page_texts[page_number] = [
            line["text"]
            for line in page.get("lines", [])
        ]

    return page_texts


def get_page_records(document: dict) -> dict:
    """Return complete extracted line records indexed by page."""

    return {
        page["page_number"]: page.get("lines", [])
        for page in document.get("pages", [])
    }


def is_course_outline_page(texts: list[str]) -> bool:
    """Check whether a page belongs to a course-outline section."""

    combined = " ".join(texts).casefold()

    return (
        "overview" in combined
        and "course contents" in combined
    )


def get_bbox_center_y(bbox: dict) -> float | None:
    """Calculate the vertical center of a bounding box."""

    if not isinstance(bbox, dict):
        return None

    if "y0" not in bbox or "y1" not in bbox:
        return None

    return (bbox["y0"] + bbox["y1"]) / 2


def is_author_candidate(
    heading: dict,
    page_lines: list[dict],
    tolerance: float = 100.0
) -> bool:
    """
    Identify likely author information using its proximity to
    department or email information on the same page.

    This avoids classifying every heading on an author-information
    page as metadata.
    """

    if is_metadata_text(heading.get("text", "")):
        return False

    heading_bbox = heading.get("bbox")

    heading_center = get_bbox_center_y(heading_bbox)

    if heading_center is None:
        return False

    anchor_centers = []

    for line in page_lines:

        line_text = line.get("text", "").strip()

        is_department = (
            "department of" in line_text.casefold()
        )

        is_email = (
            re.search(
                r"[\w.+-]+@[\w.-]+\.\w+",
                line_text
            ) is not None
        )

        if is_department or is_email:

            line_center = get_bbox_center_y(
                line.get("bbox")
            )

            if line_center is not None:
                anchor_centers.append(line_center)

    if not anchor_centers:
        return False

    return any(
        abs(heading_center - anchor_center) <= tolerance
        for anchor_center in anchor_centers
    )


def is_repeated_at_same_position(
    heading: dict,
    document: dict,
    tolerance: float = 5.0
) -> bool:
    """
    Detect whether the same text appears at approximately the same
    position on multiple pages.

    This helps distinguish repeated slide-template labels from
    genuine headings that happen to share the same wording.
    """

    text = normalize_text(heading.get("text", ""))

    target_bbox = heading.get("bbox")

    if not isinstance(target_bbox, dict):
        return False

    if "x0" not in target_bbox or "y0" not in target_bbox:
        return False

    target_page = heading.get("page_number")

    matching_pages = set()

    for page in document.get("pages", []):

        page_number = page.get("page_number")

        # Do not count the heading's own page as a repeated occurrence.
        if page_number == target_page:
            continue

        for line in page.get("lines", []):

            if normalize_text(line.get("text", "")) != text:
                continue

            bbox = line.get("bbox", {})

            if not isinstance(bbox, dict):
                continue

            if "x0" not in bbox or "y0" not in bbox:
                continue

            same_position = (
                abs(bbox["x0"] - target_bbox["x0"]) <= tolerance
                and abs(bbox["y0"] - target_bbox["y0"]) <= tolerance
            )

            if same_position:
                matching_pages.add(page_number)
                break

    return len(matching_pages) >= 1


def find_repeated_text(document: dict) -> set[str]:
    """
    Find text that appears on multiple pages.

    This function identifies repeated text candidates.
    Position-based verification is performed separately.
    """

    occurrences = defaultdict(set)

    for page in document.get("pages", []):

        page_number = page["page_number"]

        for line in page.get("lines", []):

            normalized = normalize_text(line.get("text", ""))

            if len(normalized) >= 5:
                occurrences[normalized].add(page_number)

    minimum_pages = max(
        2,
        round(len(document.get("pages", [])) * 0.20)
    )

    return {
        text
        for text, page_numbers in occurrences.items()
        if len(page_numbers) >= minimum_pages
    }


# ============================================================
# 4. HEADING CLASSIFICATION
# ============================================================

def classify_heading(
    heading: dict,
    document_type: str,
    page_texts: dict,
    repeated_text: set[str],
    document: dict
) -> dict:
    """
    Classify an extracted heading.

    Possible classifications:
    metadata
    course_outline
    repeated_document_text
    chapter
    subsection
    slide_title
    section
    review_required
    """

    text = heading.get("text", "").strip()

    page_number = heading.get("page_number")

    normalized = normalize_text(text)

    result = {
        "kind": "review_required",
        "confidence": "low",
        "classification_reasons": []
    }

    if not text:
        return result

    page_lines = page_texts.get(page_number, [])

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if is_metadata_text(text):

        result.update({
            "kind": "metadata",
            "confidence": "high"
        })

        result["classification_reasons"].append(
            "Recognized document metadata pattern"
        )

        return result

    # --------------------------------------------------------
    # Course outline
    # Check before chapter classification.
    # --------------------------------------------------------

    if (
        is_course_outline(text)
        and is_course_outline_page(page_lines)
    ):

        result.update({
            "kind": "course_outline",
            "confidence": "high"
        })

        result["classification_reasons"].append(
            "Unit entry appears on a course-outline page"
        )

        return result

    # --------------------------------------------------------
    # Repeated document text
    # --------------------------------------------------------

    if (
        normalized in repeated_text
        and is_repeated_at_same_position(heading, document)
    ):

        result.update({
            "kind": "repeated_document_text",
            "confidence": "medium"
        })

        result["classification_reasons"].append(
            "Text appears repeatedly at approximately the same "
            "position across multiple pages"
        )

        return result

    # --------------------------------------------------------
    # Chapter
    # --------------------------------------------------------

    if is_chapter_heading(text):

        result.update({
            "kind": "chapter",
            "confidence": "high"
        })

        result["classification_reasons"].append(
            "Explicit chapter, unit or module numbering"
        )

        return result

    # --------------------------------------------------------
    # Subsection
    # --------------------------------------------------------

    if is_subsection_heading(text):

        result.update({
            "kind": "subsection",
            "confidence": "medium"
        })

        result["classification_reasons"].append(
            "Matches a subsection text pattern"
        )

        return result

    # --------------------------------------------------------
    # Slide-based documents
    # --------------------------------------------------------

    if document_type == "lecture_slides":

        result.update({
            "kind": "slide_title",
            "confidence": "medium"
        })

        result["classification_reasons"].append(
            "Heading occurs in a slide-based document"
        )

        return result

    # --------------------------------------------------------
    # Textbook-style documents
    # --------------------------------------------------------

    if heading.get("heading_status") == "strong_candidate":

        result.update({
            "kind": "section",
            "confidence": "medium"
        })

        result["classification_reasons"].append(
            "Strong heading candidate"
        )

        return result

    if heading.get("heading_status") == "review_candidate":

        result.update({
            "kind": "section",
            "confidence": "low"
        })

        result["classification_reasons"].append(
            "Possible section requiring verification"
        )

    return result


# ============================================================
# 5. COMPLETE STRUCTURAL CLASSIFICATION
# ============================================================

def classify_document_structure(document: dict) -> dict:
    """
    Classify all extracted heading candidates in a document.
    """

    document_type = detect_document_type(document)

    page_texts = get_page_texts(document)

    page_records = get_page_records(document)

    repeated_text = find_repeated_text(document)

    classified_headings = []

    for heading in document.get("grouped_headings", []):

        page_number = heading.get("page_number")

        current_page_records = page_records.get(
            page_number,
            []
        )

        # First perform normal heading classification.
        classification = classify_heading(
            heading,
            document_type["document_type"],
            page_texts,
            repeated_text,
            document
        )

        # ----------------------------------------------------
        # Author metadata detection
        # ----------------------------------------------------

        if (
            document_type["document_type"] == "lecture_slides"
            and is_author_candidate(
                heading,
                current_page_records
            )
        ):

            classification = {
                "kind": "metadata",
                "confidence": "medium",
                "classification_reasons": [
                    "Heading is positioned close to department "
                    "or email information on the same page"
                ]
            }

        # ----------------------------------------------------
        # Slide-template label detection
        # ----------------------------------------------------

        elif (
            classification["kind"] == "repeated_document_text"
            and document_type["document_type"] == "lecture_slides"
        ):

            classification["kind"] = "slide_template_label"

            classification["classification_reasons"].append(
                "Repeated text appears at approximately the same "
                "position across slides"
            )

        classified_headings.append({
            **heading,
            **classification
        })

    # --------------------------------------------------------
    # Count classification results
    # --------------------------------------------------------

    counts = Counter(
        heading["kind"]
        for heading in classified_headings
    )

    return {
        "document_id": document.get("document_id"),
        "source_file": document.get("source_file"),
        "document_type": document_type,
        "heading_counts": dict(counts),
        "classified_headings": classified_headings
    }