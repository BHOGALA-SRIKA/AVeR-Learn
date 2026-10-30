import json
from pathlib import Path

import pytest

from src.ingestion.pdf_tree import extract_pdf_layout
from src.ingestion.structure import classify_document_structure
from src.ingestion.hierarchy import build_hierarchy, validate_hierarchy


PDF_FILES = [
    Path("data/raw/dsa.pdf"),
    Path("data/raw/unit1_topic1.pdf"),
]


@pytest.mark.parametrize("pdf_path", PDF_FILES)
def test_pdf_extraction(pdf_path):
    assert pdf_path.exists()

    extraction = extract_pdf_layout(pdf_path)

    assert extraction["total_pages"] > 0
    assert len(extraction["pages"]) > 0


@pytest.mark.parametrize("pdf_path", PDF_FILES)
def test_hierarchy_generation(pdf_path):
    extraction = extract_pdf_layout(pdf_path)
    classification = classify_document_structure(extraction)

    result = build_hierarchy(extraction, classification)

    assert result["nodes"]
    assert result["paragraphs"]

    errors = validate_hierarchy(result)

    assert errors == []


@pytest.mark.parametrize("pdf_path", PDF_FILES)
def test_node_relationships(pdf_path):
    extraction = extract_pdf_layout(pdf_path)
    classification = classify_document_structure(extraction)

    result = build_hierarchy(extraction, classification)
    nodes = result["nodes"]

    for node_id, node in nodes.items():
        assert node_id == node["id"]

        parent_id = node["parent_id"]

        if parent_id is not None:
            assert parent_id in nodes
            assert node_id in nodes[parent_id]["children"]


@pytest.mark.parametrize("pdf_path", PDF_FILES)
def test_paragraph_records(pdf_path):
    extraction = extract_pdf_layout(pdf_path)
    classification = classify_document_structure(extraction)

    result = build_hierarchy(extraction, classification)

    paragraph_ids = [
        paragraph["id"]
        for paragraph in result["paragraphs"]
    ]

    assert len(paragraph_ids) == len(set(paragraph_ids))

    for paragraph in result["paragraphs"]:
        assert paragraph["text"].strip()
        assert paragraph["parent_id"] in result["nodes"]
        assert paragraph["breadcrumb"]


def test_generated_json_files_exist():
    for folder in ["dsa", "unit1_topic1"]:
        output_folder = Path("data/processed") / folder

        assert (output_folder / "tree.json").exists()
        assert (output_folder / "paragraphs.jsonl").exists()

        with open(output_folder / "tree.json", encoding="utf-8") as file:
            tree = json.load(file)

        assert tree["nodes"]

