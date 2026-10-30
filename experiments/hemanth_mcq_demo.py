"""MCQ demo on the notes PDFs: offline, CPU only, no embeddings.

    python experiments/hemanth_mcq_demo.py                       # all data/raw/unit1_topic*.pdf
    python experiments/hemanth_mcq_demo.py path/a.pdf path/b.pdf
    python experiments/hemanth_mcq_demo.py --topic "process states"

Writes every template MCQ to data/mcq/unit1_template_mcqs.json and, for the
topic, the passages and the prompt an SLM would get (data/mcq/example_slm_prompt.txt).
The PDF reading here is a stand-in until Bhavya's document tree is ready.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.notes_sections import read_sections  # noqa: E402
from src.mcq import generate_for_topic, generate_mcqs  # noqa: E402
from src.mcq.context import ContextRetriever, build_prompt  # noqa: E402


def show(m) -> None:
    print(f"\n{m.mcq_id} [{m.kind}, {m.difficulty}] {m.question}")
    for letter, option in zip("ABCD", m.options):
        print(f"   {'*' if letter == m.answer else ' '} {letter}. {option}")
    ev = m.evidence
    print(f"   source: {ev['source_pdf']} p{ev['page_start']} - {ev['heading']}")


def main() -> None:
    # Windows terminals may not print "→" or "•" otherwise
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*")
    ap.add_argument("--topic", default="process states")
    ap.add_argument("--out", default=str(ROOT / "data" / "mcq"))
    args = ap.parse_args()
    pdfs = args.pdfs or sorted(glob.glob(str(ROOT / "data" / "raw" / "unit1_topic*.pdf")))
    if not pdfs:
        sys.exit("No PDFs given and none found in data/raw/unit1_topic*.pdf")

    t0 = time.perf_counter()
    paragraphs = [p for f in pdfs for p in read_sections(f)]
    t1 = time.perf_counter()
    mcqs = generate_mcqs(paragraphs)
    t2 = time.perf_counter()

    print(f"PDFs: {', '.join(Path(f).name for f in pdfs)}")
    print(f"Sections: {len(paragraphs)}   MCQs: {len(mcqs)}   "
          f"by difficulty: {dict(Counter(m.difficulty for m in mcqs))}   by kind: {dict(Counter(m.kind for m in mcqs))}")
    print(f"Time: read {t1 - t0:.2f}s, generate {t2 - t1:.2f}s (CPU, no model)")
    print(f"All answers found in the notes: {all(m.checks['answer_in_notes'] for m in mcqs)}")

    for m in mcqs[:: max(1, len(mcqs) // 4)][:4]:
        show(m)

    print(f'\n=== Topic mode: "{args.topic}"')
    for m in generate_for_topic(paragraphs, args.topic, n=3):
        show(m)

    retriever = ContextRetriever(paragraphs)
    passages = retriever.retrieve(args.topic)
    print(f'\n=== Notes for the SLM on "{args.topic}": {[p.heading for p in passages]}')
    for t in ["semaphores", "virtual memory"]:
        print(f'    "{t}" -> {"not in the notes, no questions" if not retriever.retrieve(t) else "found"}')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "unit1_template_mcqs.json").write_text(
        json.dumps([m.to_dict() for m in mcqs], indent=2, ensure_ascii=False), encoding="utf-8")
    if passages:
        prompt = build_prompt(args.topic, passages, n=3, difficulty="hard", hints=retriever.option_hints(passages))
        (out / "example_slm_prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"\nWrote {out / 'unit1_template_mcqs.json'} and {out / 'example_slm_prompt.txt'}")


if __name__ == "__main__":
    main()
