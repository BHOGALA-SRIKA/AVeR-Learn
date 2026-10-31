"""Live demo of the evaluation part (Member 4). Run it in front of the class:

    python experiments/demo_evaluation.py
    python experiments/demo_evaluation.py --question "why does the cpu sit idle without multiprogramming"
    python experiments/demo_evaluation.py --no-dense      # skip the embedding model (no model file needed)

It walks through what the harness does for ONE question, step by step:

1. the benchmark (questions with gold evidence copied from the PDFs)
2. how gold quotes become relevant paragraphs
3. each method's top 3 paragraphs, with the gold ones marked
4. the metrics for this one question
5. how much of the evidence fits in a 512-token budget
6. where the full results and the HTML report are
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")

from src.evaluation.benchmark import build_qrels, deliver, evidence_coverage, full_text, load_questions  # noqa: E402
from src.evaluation.dense_baseline import make_index  # noqa: E402
from src.evaluation.metrics import evaluate_query  # noqa: E402
from src.indexing.records import read_paragraphs  # noqa: E402
from src.utils.token_counter import count_tokens  # noqa: E402

CORPUS = REPO / "data" / "processed" / "corpus_unit1_sections.jsonl"
QUESTIONS = REPO / "data" / "benchmark" / "questions.json"
RESULTS = REPO / "results" / "unit1_all_methods" / "dev"
DEFAULT_QID = "CL-022"  # student wording: "What should a good scheduler try to keep as small as possible?"


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\nSTEP {n}. {title}\n{'=' * 78}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qid", default=DEFAULT_QID, help="a benchmark question id, e.g. HEM-007")
    parser.add_argument("--question", help="your own question (no gold answer, so no metrics)")
    parser.add_argument("--no-dense", action="store_true", help="skip the embedding model")
    args = parser.parse_args()

    paragraphs = read_paragraphs(CORPUS)
    texts = {p.pid: full_text(p) for p in paragraphs}
    questions = load_questions(QUESTIONS)
    in_scope = [q for q in questions if not q["out_of_scope"]]

    step(1, "The benchmark")
    print(f"Corpus: {len(paragraphs)} sections (heading + text) from 3 PDFs (Unit 1: intro, CPU scheduling, processes),")
    print("        built from Member 1's tree.json with Member 2's sections_from_tree")
    print(f"Questions: {len(questions)} ({len(in_scope)} answerable, {len(questions) - len(in_scope)} out of scope)")
    print(f"  dev  {sum(q['split'] == 'dev' for q in in_scope):3d}   used while developing")
    print(f"  test {sum(q['split'] == 'test' for q in in_scope):3d}   kept hidden for the final numbers")
    print(f"  student wording: {sum(q['student_wording'] for q in in_scope)}   book wording: "
          f"{sum(not q['student_wording'] for q in in_scope)}")

    if args.question:
        q = {"qid": "YOURS", "question": args.question, "gold_quotes": [], "out_of_scope": True}
        qrels = {}
    else:
        q = next(x for x in questions if x["qid"] == args.qid)
        qrels = build_qrels([q], paragraphs).qrels[q["qid"]]

    step(2, "One question and its gold evidence")
    print(f"[{q['qid']}] {q['question']}")
    if q.get("student_wording"):
        print("(written the way a student would ask, not with the book's words)")
    for g in q.get("gold_quotes", []):
        print(f"  gold quote (grade {g['grade']}, {g['source_pdf']} p.{g.get('page', '?')}): \"{g['quote']}\"")
    if qrels:
        print("Matched to paragraphs: " + ", ".join(f"{pid} (grade {g})" for pid, g in qrels.items()))
    else:
        print("No gold evidence: we only show what each method retrieves.")

    step(3, "What each method retrieves (top 3)    [GOLD] = contains the evidence")
    methods = [("bm25", False), ("bm25f", False), ("char_tfidf", False)]  # bm25f = what the MCQ pipeline uses
    if not args.no_dense:
        methods.append(("dense", False))
    rankings = {}
    for method, expand in methods:
        label = method + ("+exp" if expand else "")
        t0 = time.perf_counter()
        index = make_index(method, expand).build(paragraphs)
        build_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        hits = index.search(q["question"], k=None)
        query_ms = (time.perf_counter() - t0) * 1000
        rankings[label] = [h.pid for h in hits]
        kind = "embedding model" if method == "dense" else "vectorless"
        print(f"\n{label}  ({kind}; index built in {build_ms:,.0f} ms, answered in {query_ms:.2f} ms)")
        for h in hits[:3]:
            mark = f"[GOLD {qrels[h.pid]}]" if h.pid in qrels else "       "
            print(f"  {h.rank}. {mark} {h.pid}  score {h.score:7.3f}  {texts[h.pid][:70]}...")

    if qrels:
        step(4, "Metrics for this question")
        print(f"{'method':14s} {'R@1':>5s} {'R@3':>5s} {'P@3':>5s} {'MRR':>5s} {'nDCG@10':>8s}")
        for label, ranked in rankings.items():
            m = evaluate_query(ranked, qrels)
            print(f"{label:14s} {m['R@1']:5.2f} {m['R@3']:5.2f} {m['P@3']:5.2f} {m['MRR@10']:5.2f} {m['nDCG@10']:8.3f}")
        print("R@k = share of gold paragraphs in the top k; P@3 = share of the top 3 that are gold;")
        print("MRR = 1 / rank of the first gold paragraph; nDCG = rewards essential evidence ranked high.")

        step(5, "Does the evidence fit in the generator's 512-token budget?")
        for label, ranked in rankings.items():
            delivered = deliver(ranked, texts, 512, count_tokens)
            cov = evidence_coverage(q, delivered, texts)
            used = sum(count_tokens(texts[p]) for p in delivered)
            print(f"{label:14s} {len(delivered):2d} paragraphs, {used:3d} tokens -> {cov['coverage']:.0%} of the "
                  f"essential evidence delivered{'  (all of it)' if cov['all_essential'] else ''}")

    step(6, "Full results over all questions")
    summary = RESULTS / "summary.json"
    if summary.exists():
        res = json.loads(summary.read_text(encoding="utf-8"))["results"]
        best = sorted(res, key=lambda l: -res[l]["mean"]["nDCG@10"])[:5]
        print("Top 5 by nDCG@10 on the dev split: " + ", ".join(f"{l} {res[l]['mean']['nDCG@10']:.3f}" for l in best))
    print(f"Tables:      {RESULTS / 'summary.md'}")
    print(f"HTML report: {RESULTS / 'report.html'}")
    print("Re-run everything: python -m src.evaluation.evaluate_retrieval experiments/configs/unit1_all_methods.yaml")


if __name__ == "__main__":
    main()
