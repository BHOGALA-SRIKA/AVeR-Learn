"""Merge checks for PR #1 (Hemanth's lexical indexing), on Bhavya's real ingestion output.

    python experiments/member2_pr_checks.py                       # data/processed/*/
    python experiments/member2_pr_checks.py --ingestion-dir PATH  # folder with <doc>/paragraphs.jsonl + tree.json

Checks:
  1. the test suite passes (run separately: python -m pytest -q)
  2. BM25 and BM25F scores equal an independent plain-Python implementation of the
     formulas (Lucene BM25; Robertson-Zaragoza-Taylor BM25F) for every paragraph and query
  3. the shared token counter gives the same n_tokens / n_tokens_breadcrumb as the
     ingestion output
  4. LexicalIndex builds from paragraphs.jsonl, scores every paragraph (zeros included)
     and scores tree nodes (sections, chapters) with score_nodes
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexing.lexical import LexicalIndex  # noqa: E402
from src.indexing.records import read_paragraphs  # noqa: E402
from src.text import tokenize  # noqa: E402
from src.utils.token_counter import count_breadcrumb, count_tokens, counter_name  # noqa: E402

QUERIES = [
    "interrupt service routine", "bootstrap program ROM", "direct memory access", "caching",
    "CPU scheduler ready queue", "dispatcher context switch", "dispatch latency", "turnaround time",
    "preemptive scheduling race condition", "process control block", "context switch overhead",
    "long-term scheduler degree of multiprogramming", "zombie orphan process", "fork exec",
    "What is the difference between a program and a process?", "semaphore",  # last: no match
]


# ----------------------------------------------------------------- reference formulas

def ref_bm25(docs: list[list[str]], query: list[str], k1=1.2, b=0.75) -> list[float]:
    n = len(docs)
    avg = sum(map(len, docs)) / n
    df = Counter(t for d in docs for t in set(d))
    out = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in query:
            if df[t] == 0 or tf[t] == 0:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return out


def ref_bm25f(fields: dict[str, list[list[str]]], query: list[str], w: dict, b: dict, k1=1.2) -> list[float]:
    names = list(w)
    n = len(next(iter(fields.values())))
    avg = {f: (sum(map(len, fields[f])) / n) or 1.0 for f in names}
    docs = [[t for f in fields for t in fields[f][i]] for i in range(n)]  # df over the whole paragraph
    df = Counter(t for d in docs for t in set(d))
    out = []
    for i in range(n):
        s = 0.0
        for t in query:
            if df[t] == 0:
                continue
            tf_tilde = 0.0
            for f in names:
                toks = fields[f][i]
                tf = toks.count(t)
                if tf:
                    norm = 1 - b[f] + b[f] * len(toks) / avg[f]
                    tf_tilde += w[f] * tf / norm
            if tf_tilde:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf_tilde / (k1 + tf_tilde)
        out.append(s)
    return out


# ----------------------------------------------------------------- checks

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingestion-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args()
    docs = sorted(Path(p).parent for p in glob.glob(os.path.join(args.ingestion_dir, "*", "paragraphs.jsonl")))
    if not docs:
        print(f"No <doc>/paragraphs.jsonl under {args.ingestion_dir}")
        return 1
    ok_all = True

    def report(name, ok, detail):
        nonlocal ok_all
        ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(f"Ingestion output: {', '.join(d.name for d in docs)}\n")
    for d in docs:
        print(f"--- {d.name}")
        raw = [json.loads(line) for line in open(d / "paragraphs.jsonl", encoding="utf-8")]
        paras = read_paragraphs(d / "paragraphs.jsonl")
        tree = json.loads((d / "tree.json").read_text(encoding="utf-8")) if (d / "tree.json").exists() else None

        # 2. BM25 / BM25F against the reference formulas
        body = [tokenize(p.body) for p in paras]
        fields = {"body": body, "heading": [tokenize(p.heading) for p in paras],
                  "chapter": [tokenize(p.chapter) for p in paras]}
        bm25 = LexicalIndex(method="bm25").build(paras)
        bm25f = LexicalIndex(method="bm25f").build(paras)
        w, b = {"body": 1.0, "heading": 2.0, "chapter": 1.0}, {"body": 0.75, "heading": 0.5, "chapter": 0.5}
        worst_bm25 = worst_bm25f = 0.0
        for q in QUERIES:
            qt = tokenize(q)
            ref = ref_bm25(body, qt)
            got = bm25.score(q)
            worst_bm25 = max(worst_bm25, max(abs(got[p.pid] - r) for p, r in zip(paras, ref)))
            ref_f = ref_bm25f(fields, qt, w, b)
            got_f = bm25f.score(q)
            worst_bm25f = max(worst_bm25f, max(abs(got_f[p.pid] - r) for p, r in zip(paras, ref_f)))
        report("BM25 = reference formula", worst_bm25 < 1e-4,
               f"{len(QUERIES)} queries x {len(paras)} paragraphs, largest difference {worst_bm25:.2e}")
        report("BM25F = reference formula", worst_bm25f < 1e-4,
               f"{len(QUERIES)} queries x {len(paras)} paragraphs, largest difference {worst_bm25f:.2e}")

        # 3. token counter vs ingestion
        n_tok = [(r, count_tokens(r["text"])) for r in raw if "n_tokens" in r]
        diff = [(r["id"], r["n_tokens"], c) for r, c in n_tok if r["n_tokens"] != c]
        report(f"n_tokens = count_tokens ({counter_name()})", not diff,
               f"{len(n_tok) - len(diff)}/{len(n_tok)} records match"
               + (f"; e.g. {diff[0][0]}: ingestion {diff[0][1]} vs counter {diff[0][2]}" if diff else ""))
        n_bc = [(r, count_breadcrumb(r.get("breadcrumb"))) for r in raw if "n_tokens_breadcrumb" in r]
        diff_bc = [(r["id"], r["n_tokens_breadcrumb"], c) for r, c in n_bc if r["n_tokens_breadcrumb"] != c]
        report("n_tokens_breadcrumb = count_breadcrumb", not diff_bc,
               f"{len(n_bc) - len(diff_bc)}/{len(n_bc)} records match"
               + (f"; e.g. {diff_bc[0][0]}: ingestion {diff_bc[0][1]} vs counter {diff_bc[0][2]}" if diff_bc else ""))

        # 4. LexicalIndex consumes the ingestion output
        scores = bm25f.score("context switch")
        ids = [r["id"] for r in raw if r.get("kind", "paragraph") == "paragraph" and str(r.get("text", "")).strip()]
        report("score() covers every paragraph, zeros included", set(scores) == set(ids),
               f"{len(scores)} scores for {len(ids)} paragraphs, {sum(v == 0 for v in scores.values())} are 0.0")
        if tree is not None:
            nodes = tree["nodes"]
            nodes = list(nodes.values()) if isinstance(nodes, dict) else nodes
            idx = LexicalIndex(method="bm25f").build(paras, tree=nodes)
            sec = [n["id"] for n in nodes if n.get("kind") in ("section", "subsection", "chapter")]
            ns = idx.score_nodes("context switch", sec)
            best = max(sec, key=lambda i: ns[i]) if ns else None
            best_title = next((n.get("title", "") for n in nodes if n["id"] == best), "") if best and ns[best] > 0 \
                else "(no node matches 'context switch')"
            report("score_nodes() scores tree sections and chapters", len(ns) == len(sec),
                   f"{len(ns)} section/chapter nodes scored; best for 'context switch': {best_title[:50]}")
        print()

    print("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
