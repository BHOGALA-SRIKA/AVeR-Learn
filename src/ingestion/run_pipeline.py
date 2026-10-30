from pathlib import Path

from src.ingestion.pdf_tree import extract_pdf_layout
from src.ingestion.structure import classify_document_structure
from src.ingestion.hierarchy import (
    build_hierarchy,
    save_hierarchy,
    validate_hierarchy
)


RAW_FOLDER = Path("data/raw")
OUTPUT_FOLDER = Path("data/processed")


def process_pdf(pdf_path: Path):

    print("\n" + "=" * 60)
    print("Processing:", pdf_path.name)
    print("=" * 60)

    # 1. Extract PDF layout.
    extraction = extract_pdf_layout(pdf_path)

    # 2. Classify document structure.
    classification = classify_document_structure(
        extraction
    )

    # 3. Build hierarchy.
    result = build_hierarchy(
        extraction,
        classification
    )

    # 4. Validate.
    errors = validate_hierarchy(result)

    if errors:

        print("\nVALIDATION ERRORS:")

        for error in errors:
            print("-", error)

        raise ValueError(
            "Hierarchy validation failed."
        )

    print("\nValidation passed.")

    # 5. Save output in a separate folder per document.
    document_output = (
        OUTPUT_FOLDER / extraction["document_id"]
    )

    save_hierarchy(
        result,
        document_output
    )


def main():

    pdf_files = list(
        RAW_FOLDER.rglob("*.pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_path in pdf_files:

        process_pdf(pdf_path)


if __name__ == "__main__":
    main()