"""Quick side-by-side check of every lexical method (Hemanth, Member 2) on the pilot OS PDF.

Run from the repository root:

    python experiments/member2_lexical_check.py
    python experiments/member2_lexical_check.py --show "what is DMA"

IMPORTANT: this is a smoke check, NOT the project benchmark. The 20 questions
below were written while looking at the 28 paragraphs, so they share more
words with the text than real student questions will, and they favour
lexical methods. Real results must come from the team benchmark
(data/benchmark/questions.json + qrels.json) and Sreeram's evaluation
harness. Use this only to confirm the methods work and to see how they differ.

For each method it prints how many questions have an essential paragraph at
rank 1 ("top1") and in the top 3 ("top3"), plus build and query time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.indexing.lexical import METHODS, LexicalIndex  # noqa: E402

CORPUS = REPO / "data" / "processed" / "paragraphs.json"
P = "unit1_topic1_p000"

# (question, essential paragraph ids, student_wording)
QUESTIONS = [
    ("What is an interrupt service routine?", [P + "19"], False),
    ("What steps does the CPU follow to handle an interrupt?", [P + "20"], False),
    ("What is the bootstrap program and where is it stored?", [P + "17"], False),
    ("Which process is created first after booting?", [P + "18"], False),
    ("What is direct memory access used for?", [P + "28"], False),
    ("How does a device controller tell the CPU an operation is finished?", [P + "16"], False),
    ("What is the kernel?", [P + "13"], False),
    ("What are the two roles of the OS from the system view?", [P + "12"], False),
    ("How do the user view and system view of an OS differ?", [P + "10"], False),
    ("Why do we need an operating system?", [P + "04", P + "03"], False),
    ("What is caching?", [P + "25"], False),
    ("Explain the fetch decode execute cycle", [P + "23"], False),
    ("Is main memory volatile?", [P + "22"], False),
    ("What is a trap?", [P + "19"], False),
    ("What are the four components of a computer system?", [P + "08", P + "09"], False),
    ("What are the goals of an operating system?", [P + "07"], False),
    # Student-style wording: fewer words shared with the book.
    ("How is data copied from disk to memory without the processor?", [P + "28"], True),
    ("What happens when the computer is switched on?", [P + "17"], True),
    ("What is an ISR?", [P + "19"], True),
    ("How did OSes start, were there always operating systems?", [P + "06"], True),
]


def run(method: str, expand: bool) -> dict:
    t0 = time.perf_counter()
    index = LexicalIndex(method, expand=expand).build(CORPUS)
    build_ms = (time.perf_counter() - t0) * 1000
    top1 = top3 = top1_student = 0
    t0 = time.perf_counter()
    for question, gold, student in QUESTIONS:
        ranked = [h.pid for h in index.search(question, k=3)]
        hit1 = ranked[0] in gold
        top1 += hit1
        top3 += any(pid in gold for pid in ranked)
        top1_student += hit1 and student
    query_ms = (time.perf_counter() - t0) * 1000 / len(QUESTIONS)
    return {"top1": top1, "top3": top3, "top1_student": top1_student, "build_ms": build_ms, "query_ms": query_ms}


def show(question: str) -> None:
    for method in METHODS:
        for expand in (False, True):
            index = LexicalIndex(method, expand=expand).build(CORPUS)
            hits = ", ".join(f"{h.pid[-3:]} ({h.score:.3f})" for h in index.search(question, k=3))
            print(f"{method + ('+exp' if expand else ''):18s} {hits}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", metavar="QUESTION", help="print the top 3 paragraphs of every method for one question")
    args = parser.parse_args()
    if args.show:
        show(args.show)
        return

    n, n_student = len(QUESTIONS), sum(s for *_, s in QUESTIONS)
    print(f"Smoke check: {n} questions ({n_student} in student wording), 28 paragraphs\n")
    print(f"{'method':18s} {'top1':>6s} {'top3':>6s} {'student top1':>13s} {'build ms':>9s} {'query ms':>9s}")
    for method in METHODS:
        for expand in (False, True):
            r = run(method, expand)
            label = method + ("+exp" if expand else "")
            print(
                f"{label:18s} {r['top1']:>3d}/{n:<2d} {r['top3']:>3d}/{n:<2d} "
                f"{r['top1_student']:>9d}/{n_student:<3d} {r['build_ms']:>9.1f} {r['query_ms']:>9.2f}"
            )


if __name__ == "__main__":
    main()
