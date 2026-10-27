import json
from pathlib import Path

from src.ingestion.pdf_loader import extract_pdf
from src.indexing.bm25_index import BM25Index


def load_records() -> list[dict]:
    """Load extracted records or create them from local PDFs."""
    processed_file = Path("data/processed/paragraphs.json")

    if processed_file.exists():
        with processed_file.open("r", encoding="utf-8") as file:
            return json.load(file)

    pdf_folder = Path("data/raw")
    records = []

    for pdf_path in pdf_folder.rglob("*.pdf"):
        records.extend(extract_pdf(pdf_path))

    if not records:
        raise RuntimeError(
            "No paragraphs found. Add PDFs to data/raw first."
        )

    processed_file.parent.mkdir(parents=True, exist_ok=True)

    with processed_file.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    return records


if __name__ == "__main__":
    records = load_records()
    retriever = BM25Index(records)

    print(f"Indexed {len(records)} paragraphs.")

    while True:
        query = input("\nEnter a question (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        results = retriever.search(
    query,
    top_k=3,
    heading_boost=4.0,
)

        if not results:
            print("No results found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print("Document:", result["document_id"])
            print("Page:", result["page_number"])
            print("Paragraph ID:", result["paragraph_id"])
            print("BM25 score:", round(result["retrieval_score"], 4))
            print("Text:", result["text"][:1000])