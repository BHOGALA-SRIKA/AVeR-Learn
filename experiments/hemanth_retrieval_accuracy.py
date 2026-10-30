"""Retrieval accuracy for MCQ generation - no language model needed.

    python experiments/hemanth_retrieval_accuracy.py

Three checks on the Unit 1 notes (data/raw/unit1_topic*.pdf):

A. Student questions (data/benchmark/drafts/hemanth_pilot_questions.json): is the
   section holding the gold quote ranked first / in the top 3? Out-of-scope
   questions must get no passages.
B. Topic requests: every concept / term the notes define, typed as a topic
   ("context switch", "throughput"). Is its own section first / in the top 3?
C. Topics from later units ("deadlock", "paging" ...): must get no passages, so
   the SLM is never asked to make up questions.

Note: the questions in A were also used while tuning, so A is a development
score. Use a held-out question set for the final number.
"""

from __future__ import annotations

import glob
import json
import sys
import os
from pathlib import Path

# NumPy's maths library starts one thread (and memory buffer) per CPU core; on a laptop
# that is also running the SLM this can fail with "OpenBLAS error: Memory allocation".
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.notes_sections import load_sections  # noqa: E402
from src.indexing.concepts import content_tokens, extract_concepts  # noqa: E402
from src.mcq.context import ContextRetriever  # noqa: E402

OUT_OF_SCOPE_TOPICS = [
    "semaphores", "deadlock", "banker's algorithm", "page replacement", "paging", "segmentation",
    "thrashing", "virtual memory", "round robin scheduling", "shortest job first", "multithreading",
    "threads", "critical section problem", "file allocation methods", "disk scheduling", "RAID",
    "virtualization", "cloud computing", "translation lookaside buffer", "dining philosophers problem",
    "inter process communication", "message passing", "priority inversion", "memory fragmentation",
    "demand paging",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pdfs = args or sorted(glob.glob(str(ROOT / "data" / "raw" / "unit1_topic*.pdf")))
    paragraphs, source = load_sections(pdfs, ROOT, prefer_trees="--stand-in" not in sys.argv)
    print(f"Sections: {len(paragraphs)} ({source})")
    cr = ContextRetriever(paragraphs)

    def gold_pids(quotes):
        out = set()
        for q in quotes:
            qt = content_tokens(q)
            for p in paragraphs:
                if qt and len(qt & content_tokens(f"{p.heading} {p.body}")) / len(qt) >= 0.9:
                    out.add(p.pid)
        return out

    # A
    bench = ROOT / "data" / "benchmark" / "drafts" / "hemanth_pilot_questions.json"
    if bench.exists():
        qs = json.loads(bench.read_text(encoding="utf-8"))
        top1 = top3 = n = oos_ok = oos = 0
        for q in qs:
            got = [p.pid for p in cr.retrieve(q["question"])]
            if q.get("out_of_scope"):
                oos += 1
                oos_ok += not got
                continue
            gold = gold_pids([g["quote"] for g in q["gold_quotes"] if g["grade"] == 2])
            n += 1
            top1 += bool(got[:1] and got[0] in gold)
            top3 += bool(set(got[:3]) & gold)
            if not set(got[:3]) & gold:
                print(f"   A miss: {q['qid']} {q['question']}")
        print(f"A) student questions: right section first {top1}/{n}, in top 3 {top3}/{n}; "
              f"out-of-scope refused {oos_ok}/{oos}")

    # B
    cs = extract_concepts(paragraphs)
    topics: dict[str, set[str]] = {}
    for s in cs.sections:
        topics.setdefault(s.name.lower(), set(s.pids))
    for d in cs.definitions:
        topics.setdefault(d.term.lower(), {d.pid})
    b1 = b3 = 0
    for t, gold in topics.items():
        got = [p.pid for p in cr.retrieve(t)]
        b1 += bool(got[:1] and got[0] in gold)
        b3 += bool(set(got[:3]) & gold)
    print(f"B) topic requests: right section first {b1}/{len(topics)}, in top 3 {b3}/{len(topics)}")

    # C
    leaked = [t for t in OUT_OF_SCOPE_TOPICS if cr.retrieve(t)]
    print(f"C) topics not in the notes refused: {len(OUT_OF_SCOPE_TOPICS) - len(leaked)}/{len(OUT_OF_SCOPE_TOPICS)}"
          + (f"  (not refused: {', '.join(leaked)})" if leaked else ""))


if __name__ == "__main__":
    main()
