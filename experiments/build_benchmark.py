"""Merge the question drafts into the benchmark and check every gold quote (Member 4).

    python experiments/build_benchmark.py
    python experiments/build_benchmark.py --corpus data/processed/corpus_unit1.json   # old stand-in corpus

Reads every file in data/benchmark/drafts/, fills default fields, assigns a
dev/test split, checks that each gold quote is found in the corpus, and writes:

- data/benchmark/questions.json   the merged benchmark
- data/benchmark/qrels.trec       paragraph judgements for this corpus (TREC format)

Split: about one third "dev" (tune parameters, write glossary entries, look at
failures) and two thirds "test" (touch only for the final numbers). The split
is a hash of the question ID, so it never changes when questions are added,
and a split already set in a draft is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.benchmark import build_qrels  # noqa: E402
from src.indexing.records import read_paragraphs  # noqa: E402

DRAFTS = REPO / "data" / "benchmark" / "drafts"
OUT = REPO / "data" / "benchmark"
DEFAULTS = {"out_of_scope": False, "source_visible": True, "student_wording": False,
            "annotator": "Claude (AI-drafted, needs human review)"}


def split_for(qid: str) -> str:
    return "dev" if int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 3 == 0 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(REPO / "data" / "processed" / "corpus_unit1_sections.jsonl"))
    args = parser.parse_args()

    questions = []
    for path in sorted(DRAFTS.glob("*.json")):
        for q in json.loads(path.read_text(encoding="utf-8")):
            q = {**DEFAULTS, **q, "draft_file": path.name}
            if q.get("split") in (None, "", "unassigned"):
                q["split"] = split_for(q["qid"])
            questions.append(q)
    if len({q["qid"] for q in questions}) != len(questions):
        raise SystemExit("Duplicate qid across draft files")

    judgements = build_qrels(questions, read_paragraphs(args.corpus))
    for qid, quote in judgements.unmatched:
        print(f"NOT FOUND in corpus  {qid}: {quote[:80]}")
    if judgements.unmatched:
        raise SystemExit("Fix the quotes above (copy them exactly from the PDF text) and run again.")

    (OUT / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / "qrels.trec").open("w", encoding="utf-8") as fh:
        for q in questions:
            for pid, grade in sorted(judgements.qrels[q["qid"]].items()):
                fh.write(f"{q['qid']} 0 {pid} {grade}\n")

    in_scope = [q for q in questions if not q["out_of_scope"]]
    print(f"{len(questions)} questions: {len(in_scope)} in scope, {len(questions) - len(in_scope)} out of scope")
    for split in ("dev", "test"):
        part = [q for q in in_scope if q["split"] == split]
        print(f"  {split:4s} {len(part):3d} in scope, {sum(q['student_wording'] for q in part)} student wording")
    print(f"Wrote {OUT / 'questions.json'} and {OUT / 'qrels.trec'}")


if __name__ == "__main__":
    main()
