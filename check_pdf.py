import pymupdf
from pathlib import Path

pdf_folder = Path("data/raw")

for pdf_path in pdf_folder.rglob("*.pdf"):
    print(f"\nChecking: {pdf_path}")

    try:
        with pymupdf.open(pdf_path) as document:
            print("Pages:", len(document))

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text")

                if text.strip():
                    print(f"\n--- Page {page_number} ---")
                    print(text[:500])
                    break
            else:
                print("No extractable text found.")
    except Exception as error:
        print("Error:", error)