import json
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def normalize(text: str) -> str:
    """Normalize text for matching."""

    return re.sub(r"\s+", " ", text.casefold()).strip()


def make_prefix(document_id: str) -> str:
    """Create a stable ID prefix from the document name."""

    prefix = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        document_id
    ).strip("_").upper()

    return prefix or "DOC"


def count_tokens(text: str) -> int:
    """
    Approximate token count using whitespace tokenization.

    This is not a model-specific tokenizer.
    """

    return len(text.split())


def make_node(
    node_id,
    level,
    kind,
    title,
    text,
    parent_id,
    page_start,
    page_end,
    doc_order
):
    """Create a hierarchy node."""

    return {
        "id": node_id,
        "level": level,
        "kind": kind,
        "title": title,
        "text": text,
        "parent_id": parent_id,
        "children": [],
        "breadcrumb": "",
        "doc_order": doc_order,
        "page_start": page_start,
        "page_end": page_end,
        "n_tokens": count_tokens(text),
        "n_tokens_breadcrumb": 0
    }


# ============================================================
# 2. PARAGRAPH EXTRACTION
# ============================================================

def get_heading_texts(classified_headings: list[dict]) -> set[str]:
    """
    Collect heading text and original source lines.

    These lines should not be extracted again as paragraphs.
    """

    heading_texts = set()

    for heading in classified_headings:

        heading_texts.add(
            normalize(heading.get("text", ""))
        )

        for source_line in heading.get("source_lines", []):

            heading_texts.add(
                normalize(source_line)
            )

    return heading_texts


def should_skip_line(
    line: dict,
    heading_texts: set[str]
) -> bool:
    """Exclude headers, metadata and heading lines from paragraphs."""

    text = normalize(line.get("text", ""))

    if not text:
        return True

    if line.get("is_running_header", False):
        return True

    if text in heading_texts:
        return True

    if line.get("heading_status") in {
        "strong_candidate",
        "review_candidate"
    }:
        return True

    metadata_patterns = [
        r"^[\w.+-]+@[\w.-]+\.\w+$",
        r"^department of ",
        r"^reference book\s*:?",
        r"^text book\s*:?",
        r"^evaluation policy$",
        r"^thank you$",
    ]

    if any(
        re.match(pattern, text, re.IGNORECASE)
        for pattern in metadata_patterns
    ):
        return True

    return False


def extract_paragraph_blocks(
    page: dict,
    heading_texts: set[str]
) -> list[dict]:
    """
    Group extracted body lines into paragraph-like blocks.

    Lines are grouped using vertical distance and horizontal alignment.

    This is a layout-based approximation, not semantic paragraph detection.
    """

    valid_lines = []

    for line in page.get("lines", []):

        if should_skip_line(line, heading_texts):
            continue

        valid_lines.append(line)

    valid_lines.sort(
        key=lambda line: (
            line["bbox"]["y0"],
            line["bbox"]["x0"]
        )
    )

    paragraphs = []

    current_group = []

    def save_group(group):

        if not group:
            return

        text = " ".join(
            line["text"].strip()
            for line in group
        )

        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return

        paragraphs.append({
            "text": text,
            "page_start": page["page_number"],
            "page_end": page["page_number"],
            "bbox": {
                "x0": min(line["bbox"]["x0"] for line in group),
                "y0": min(line["bbox"]["y0"] for line in group),
                "x1": max(line["bbox"]["x1"] for line in group),
                "y1": max(line["bbox"]["y1"] for line in group)
            }
        })

    for line in valid_lines:

        if not current_group:
            current_group.append(line)
            continue

        previous = current_group[-1]

        vertical_gap = (
            line["bbox"]["y0"] - previous["bbox"]["y1"]
        )

        font_size = float(
            previous.get("font_size", 10)
        )

        same_alignment = (
            abs(
                line["bbox"]["x0"]
                - previous["bbox"]["x0"]
            ) <= 35
        )

        same_column = (
            line["bbox"]["x0"]
            >= previous["bbox"]["x0"] - 10
        )

        should_join = (
            -2 <= vertical_gap <= max(4, font_size * 0.75)
            and same_alignment
            and same_column
        )

        if should_join:
            current_group.append(line)

        else:
            save_group(current_group)
            current_group = [line]

    save_group(current_group)

    return paragraphs


# ============================================================
# 3. BREADCRUMB GENERATION
# ============================================================

def get_breadcrumb(
    nodes: dict,
    node_id: str
) -> str:
    """Build breadcrumbs using structural ancestors only."""

    parts = []

    current_id = node_id

    while current_id:

        node = nodes[current_id]

        # Paragraph content must never appear in breadcrumbs.
        if node["kind"] == "document":
            pass

        elif node["kind"] == "paragraph":
            pass

        else:
            label = node.get("title", "").strip()

            if label:
                parts.append(label)

        current_id = node["parent_id"]

    parts.reverse()

    return " > ".join(parts)


def finalize_nodes(nodes: dict):
    """Calculate breadcrumbs and breadcrumb token counts."""

    for node_id, node in nodes.items():

        breadcrumb = get_breadcrumb(
            nodes,
            node_id
        )

        node["breadcrumb"] = breadcrumb

        node["n_tokens_breadcrumb"] = count_tokens(
            breadcrumb
        )


# ============================================================
# 4. NODE MANAGEMENT
# ============================================================

def add_node(
    nodes: dict,
    node: dict
):
    """Add a node and update its parent's children."""

    nodes[node["id"]] = node

    parent_id = node["parent_id"]

    if parent_id:

        if parent_id not in nodes:
            raise ValueError(
                f"Parent node does not exist: {parent_id}"
            )

        nodes[parent_id]["children"].append(
            node["id"]
        )


# ============================================================
# 5. BUILD TEXTBOOK-STYLE HIERARCHY
# ============================================================

def build_textbook_hierarchy(
    extraction: dict,
    classification: dict,
    nodes: dict,
    root_id: str,
    prefix: str
) -> list[dict]:
    """
    Build chapter -> section -> subsection -> paragraph hierarchy.
    """

    headings_by_page = defaultdict(list)

    for heading in classification["classified_headings"]:

        kind = heading["kind"]

        if kind in {
            "chapter",
            "section",
            "subsection"
        }:

            headings_by_page[
                heading["page_number"]
            ].append(heading)

    heading_texts = get_heading_texts(
        classification["classified_headings"]
    )
    current_chapter = root_id
    current_section = None
    current_subsection = None

    chapter_count = 0
    section_count = 0
    subsection_count = 0
    paragraph_count = 0
    doc_order = 0

    paragraph_records = []

    for page in extraction["pages"]:

        page_number = page["page_number"]

        page_headings = headings_by_page[page_number]

        paragraph_blocks = extract_paragraph_blocks(
            page,
            heading_texts
        )

        events = []

        for heading in page_headings:

            events.append({
                "type": "heading",
                "y": heading["bbox"]["y0"],
                "data": heading
            })

        for paragraph in paragraph_blocks:

            events.append({
                "type": "paragraph",
                "y": paragraph["bbox"]["y0"],
                "data": paragraph
            })

        events.sort(
            key=lambda event: event["y"]
        )

        for event in events:

            doc_order += 1

            if event["type"] == "heading":

                heading = event["data"]

                kind = heading["kind"]

                text = heading["text"]

                if kind == "chapter":

                    chapter_count += 1

                    node_id = (
                        f"{prefix}_CH{chapter_count:03d}"
                    )

                    node = make_node(
                        node_id,
                        1,
                        "chapter",
                        text,
                        text,
                        root_id,
                        page_number,
                        page_number,
                        doc_order
                    )

                    add_node(nodes, node)

                    current_chapter = node_id
                    current_section = None
                    current_subsection = None

                elif kind == "section":

                    section_count += 1

                    node_id = (
                        f"{prefix}_S{section_count:03d}"
                    )

                    node = make_node(
                        node_id,
                        2,
                        "section",
                        text,
                        text,
                        current_chapter,
                        page_number,
                        page_number,
                        doc_order
                    )

                    add_node(nodes, node)

                    current_section = node_id
                    current_subsection = None

                elif kind == "subsection":

                    subsection_count += 1

                    parent_id = (
                        current_section
                        or current_chapter
                    )

                    node_id = (
                        f"{prefix}_SS{subsection_count:03d}"
                    )

                    node = make_node(
                        node_id,
                        3,
                        "subsection",
                        text,
                        text,
                        parent_id,
                        page_number,
                        page_number,
                        doc_order
                    )

                    add_node(nodes, node)

                    current_subsection = node_id

            else:

                paragraph = event["data"]

                paragraph_count += 1

                parent_id = (
                    current_subsection
                    or current_section
                    or current_chapter
                )

                node_id = (
                    f"{prefix}_P{paragraph_count:04d}"
                )

                node = make_node(
                    node_id,
                    4,
                    "paragraph",
                    "",
                    paragraph["text"],
                    parent_id,
                    paragraph["page_start"],
                    paragraph["page_end"],
                    doc_order
                )

                node["bbox"] = paragraph["bbox"]

                add_node(nodes, node)

                paragraph_records.append(node.copy())

    return paragraph_records


# ============================================================
# 6. BUILD LECTURE-SLIDE HIERARCHY
# ============================================================

def build_slide_hierarchy(
    extraction: dict,
    classification: dict,
    nodes: dict,
    root_id: str,
    prefix: str
) -> list[dict]:
    """
    Build document -> slide -> heading/paragraph hierarchy.

    Every original PDF page is represented as a slide node.
    """

    headings_by_page = defaultdict(list)

    for heading in classification["classified_headings"]:

        if heading["kind"] in {
            "slide_title",
            "course_outline"
        }:

            headings_by_page[
                heading["page_number"]
            ].append(heading)

    heading_texts = get_heading_texts(
        classification["classified_headings"]
    )

    # Fallback: detect short, bold question-style slide titles
    # that the main heading classifier may have missed.
    for page in extraction["pages"]:

        page_number = page["page_number"]

        existing_page_texts = {
            normalize(heading["text"])
            for heading in headings_by_page[page_number]
        }

        for line in page.get("lines", []):

            text = line.get("text", "").strip()
            normalized_text = normalize(text)

            if not text:
                continue

            if normalized_text in heading_texts:
                continue

            if normalized_text in existing_page_texts:
                continue

            if line.get("is_running_header", False):
                continue

            if not line.get("bold", False):
                continue

            if not text.endswith("?"):
                continue

            if len(text) > 80:
                continue

            if line.get("font_size", 0) < 18:
                continue

            inferred_heading = {
                "kind": "slide_title",
                "text": text,
                "page_number": page_number,
                "bbox": line["bbox"],
                "source_lines": [text]
            }

            headings_by_page[page_number].append(
                inferred_heading
            )

            heading_texts.add(normalized_text)
            existing_page_texts.add(normalized_text)

    slide_count = 0
    heading_count = 0
    paragraph_count = 0
    doc_order = 0

    paragraph_records = []

    for page in extraction["pages"]:

        page_number = page["page_number"]

        slide_count += 1
        doc_order += 1

        slide_id = (
            f"{prefix}_SLIDE_P{page_number:03d}"
        )

        slide_node = make_node(
            slide_id,
            1,
            "slide",
            f"Slide {page_number}",
            "",
            root_id,
            page_number,
            page_number,
            doc_order
        )

        add_node(nodes, slide_node)

        page_headings = headings_by_page[page_number]

        paragraph_blocks = extract_paragraph_blocks(
            page,
            heading_texts
        )

        events = []

        for heading in page_headings:

            events.append({
                "type": "heading",
                "y": heading["bbox"]["y0"],
                "data": heading
            })

        for paragraph in paragraph_blocks:

            events.append({
                "type": "paragraph",
                "y": paragraph["bbox"]["y0"],
                "data": paragraph
            })

        events.sort(
            key=lambda event: event["y"]
        )

        current_heading = slide_id

        for event in events:

            doc_order += 1

            if event["type"] == "heading":

                heading = event["data"]

                heading_count += 1

                heading_id = (
                    f"{prefix}_SL{page_number:03d}"
                    f"_H{heading_count:03d}"
                )

                heading_node = make_node(
                    heading_id,
                    2,
                    heading["kind"],
                    heading["text"],
                    heading["text"],
                    slide_id,
                    page_number,
                    page_number,
                    doc_order
                )

                add_node(nodes, heading_node)

                current_heading = heading_id

            else:

                paragraph = event["data"]

                paragraph_count += 1

                paragraph_id = (
                    f"{prefix}_P{paragraph_count:04d}"
                )

                paragraph_node = make_node(
                    paragraph_id,
                    4,
                    "paragraph",
                    "",
                    paragraph["text"],
                    current_heading,
                    page_number,
                    page_number,
                    doc_order
                )

                paragraph_node["bbox"] = paragraph["bbox"]

                add_node(nodes, paragraph_node)

                paragraph_records.append(
                    paragraph_node.copy()
                )

    return paragraph_records


# ============================================================
# 7. MAIN HIERARCHY BUILDER
# ============================================================

def build_hierarchy(
    extraction: dict,
    classification: dict
) -> dict:
    """
    Build the complete document hierarchy.

    Input:
        PDF extraction output
        Structural classification output

    Output:
        Document tree and paragraph records.
    """

    document_id = extraction["document_id"]

    prefix = make_prefix(document_id)

    root_id = f"{prefix}_DOCUMENT"

    nodes = {}

    root_node = make_node(
        root_id,
        0,
        "document",
        document_id,
        "",
        None,
        1,
        extraction["total_pages"],
        0
    )

    add_node(nodes, root_node)

    document_type = classification[
        "document_type"
    ]["document_type"]

    if document_type == "lecture_slides":

        paragraphs = build_slide_hierarchy(
            extraction,
            classification,
            nodes,
            root_id,
            prefix
        )

    else:

        paragraphs = build_textbook_hierarchy(
            extraction,
            classification,
            nodes,
            root_id,
            prefix
        )

    finalize_nodes(nodes)

    # Refresh paragraph records after breadcrumb calculation.
    paragraphs = [
        nodes[paragraph["id"]]
        for paragraph in paragraphs
    ]

    return {
        "schema_version": "1.0",
        "document_id": document_id,
        "source_file": extraction["source_file"],
        "document_type": document_type,
        "root_ids": [root_id],
        "nodes": nodes,
        "paragraphs": paragraphs
    }


# ============================================================
# 8. SAVE OUTPUT FILES
# ============================================================

def save_hierarchy(
    result: dict,
    output_folder: str | Path
):
    """Save tree.json and paragraphs.jsonl."""

    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    tree_path = output_folder / "tree.json"

    paragraphs_path = (
        output_folder / "paragraphs.jsonl"
    )

    tree_data = {
        "schema_version": result["schema_version"],
        "document_id": result["document_id"],
        "source_file": result["source_file"],
        "document_type": result["document_type"],
        "root_ids": result["root_ids"],
        "nodes": result["nodes"]
    }

    with open(
        tree_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tree_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    with open(
        paragraphs_path,
        "w",
        encoding="utf-8"
    ) as file:

        for paragraph in result["paragraphs"]:

            file.write(
                json.dumps(
                    paragraph,
                    ensure_ascii=False
                ) + "\n"
            )

    print("Generated:", tree_path)
    print("Generated:", paragraphs_path)

    print(
        "Total nodes:",
        len(result["nodes"])
    )

    print(
        "Total paragraphs:",
        len(result["paragraphs"])
    )


# ============================================================
# 9. VALIDATION
# ============================================================

def validate_hierarchy(result: dict) -> list[str]:
    """Perform basic structural validation."""

    errors = []

    nodes = result["nodes"]

    # Check root.
    for root_id in result["root_ids"]:

        if root_id not in nodes:
            errors.append(
                f"Missing root node: {root_id}"
            )

    # Check IDs and parent-child consistency.
    for node_id, node in nodes.items():

        if node_id != node["id"]:
            errors.append(
                f"Node key does not match ID: {node_id}"
            )

        parent_id = node["parent_id"]

        if parent_id is not None:

            if parent_id not in nodes:
                errors.append(
                    f"Invalid parent for {node_id}"
                )

            elif node_id not in nodes[parent_id]["children"]:
                errors.append(
                    f"Parent-child mismatch: {node_id}"
                )

        for child_id in node["children"]:

            if child_id not in nodes:
                errors.append(
                    f"Missing child node: {child_id}"
                )

            elif nodes[child_id]["parent_id"] != node_id:
                errors.append(
                    f"Invalid child relationship: {child_id}"
                )

    # Check unique paragraph IDs.
    paragraph_ids = [
        paragraph["id"]
        for paragraph in result["paragraphs"]
    ]

    if len(paragraph_ids) != len(set(paragraph_ids)):
        errors.append(
            "Duplicate paragraph IDs detected"
        )

    # Check paragraph references.
    for paragraph in result["paragraphs"]:

        if paragraph["id"] not in nodes:
            errors.append(
                f"Paragraph missing from tree: {paragraph['id']}"
            )

    return errors