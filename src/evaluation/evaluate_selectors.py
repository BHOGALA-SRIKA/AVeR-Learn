"""Evaluate Member 3's evidence selectors (EMTE and its baselines) on the benchmark (owner: Member 4).

    python -m src.evaluation.evaluate_selectors experiments/configs/member3_selectors.yaml
    python -m src.evaluation.evaluate_selectors experiments/configs/member3_selectors.yaml --split test

evaluate_retrieval.py scores *rankings* (Recall, MRR, nDCG). A selector does
something else: it decides WHICH text, and how much of it, goes to the
generator within a token budget. Its output is a set of tree regions, not a
ranking, so it is scored on what it delivers:

- ``cov@B``      mean fraction of each essential gold quote's words that are
                 in the delivered text (same definition as evaluate_retrieval)
- ``allEss@B``   1 if every essential quote is at least 80% delivered
- ``gold_share`` fraction of delivered words that belong to a gold quote
                 (how much of the context is evidence rather than noise)
- ``tokens``     delivered tokens, as counted by the selector (text + breadcrumbs)
- ``abstained``  the selector returned nothing

Setting: the app answers questions about ONE uploaded PDF, so every question is
run on the document its gold quotes come from (Member 1's tree.json and
paragraphs.jsonl for that document, Member 2's BM25F index, Member 3's
``retrieve_from_processed_corpus``). The reference row ``flat_sections`` is the
current MCQ pipeline on the same document: BM25F over Member 1's sections,
delivered top-down until the budget is full. A question whose quotes come from
several PDFs is run on the PDF with most of its essential quotes and scored on
that PDF's quotes only.

Out-of-scope questions have no gold quotes; each is run on every document and
only abstention is reported, together with how well a threshold on the top raw
BM25F score (EMTE's ``min_raw_score``) would separate them from in-scope ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.benchmark import covered_positions, deliver, evidence_coverage, full_text, load_questions, words
from src.evaluation.dense_baseline import make_index
from src.evaluation.evaluate_retrieval import REPO, sha256
from src.evaluation.significance import holm, randomization_test
from src.indexing.records import read_paragraphs
from src.retrieval.processed_pipeline import load_processed_corpus, retrieve_from_processed_corpus
from src.utils.token_counter import count_tokens, counter_name


THRESHOLDS = [0, 1.0, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]  # min_raw_score values shown in the summary


def machine() -> str:
    return f"{platform.processor() or platform.machine()} | {os.cpu_count()} CPUs | Python {platform.python_version()}"


def label_of(run: dict[str, Any]) -> str:
    return run.get("label") or run["method"]


def document_of(question: dict[str, Any]) -> str:
    """The PDF a question is run on: the one holding most of its essential quotes (ties: alphabetical)."""
    quotes = question.get("gold_quotes") or []
    if not quotes:
        raise ValueError(f"{question['qid']}: in-scope question without gold quotes")
    essential = [g for g in quotes if g["grade"] >= 2] or quotes
    counts: dict[str, int] = {}
    for g in essential:
        doc = Path(g["source_pdf"]).stem
        counts[doc] = counts.get(doc, 0) + 1
    return min(counts, key=lambda d: (-counts[d], d))


def on_document(question: dict[str, Any], document: str) -> dict[str, Any]:
    """The question with only the gold quotes from ``document`` (the app sees one uploaded PDF)."""
    return {**question, "gold_quotes": [g for g in question["gold_quotes"]
                                        if Path(g["source_pdf"]).stem == document]}


def region_text(nodes: dict[str, dict], node_id: str) -> str:
    """What the generator reads for a selected region: breadcrumb, title, then its paragraphs in order."""
    node = nodes[node_id]
    parts = [str(node.get("breadcrumb") or "")]
    if node.get("kind") == "paragraph":
        parts.append(str(node.get("text", "")))
    else:
        parts.append(str(node.get("title") or ""))
        stack, leaves = [node_id], []
        while stack:
            current = nodes[stack.pop()]
            if current.get("kind") == "paragraph":
                leaves.append(current)
            else:
                stack.extend(str(c) for c in current.get("children", []) if str(c) in nodes)
        leaves.sort(key=lambda n: int(n.get("doc_order", 0)))
        parts.extend(str(n.get("text", "")) for n in leaves)
    return "\n".join(p for p in parts if p)


def gold_share(question: dict[str, Any], delivered_texts: list[str]) -> float:
    """Fraction of delivered words that are part of any gold quote (grade 1 or 2)."""
    quotes = [words(g["quote"]) for g in question.get("gold_quotes") or []]
    total = hit = 0
    for text in delivered_texts:
        dw = words(text)
        covered: set[int] = set()
        for qw in quotes:
            covered |= covered_positions(dw, qw)
        total += len(dw)
        hit += len(covered)
    return hit / total if total else 0.0


class FlatSections:
    """The current MCQ pipeline on one document: BM25F sections, top-down until the budget is full."""

    def __init__(self, corpus_path: Path):
        self.paragraphs = read_paragraphs(corpus_path)
        self.indexes: dict[str, Any] = {}

    def select(self, document: str, query: str, budget: int) -> tuple[list[str], int, bool]:
        if document not in self.indexes:
            paras = [p for p in self.paragraphs if p.meta.get("document_id") == document]
            self.indexes[document] = (make_index("bm25f").build(paras), {p.pid: full_text(p) for p in paras})
        index, texts = self.indexes[document]
        hits = index.search(query, k=None)
        chosen = deliver([h.pid for h in hits if h.score > 0], texts, budget, count_tokens)
        return [texts[pid] for pid in chosen], sum(count_tokens(texts[pid]) for pid in chosen), not chosen


def evaluate(config: dict[str, Any], split: str, out_dir: Path) -> dict[str, Any]:
    qpath = REPO / config["questions"]
    budgets = config.get("budgets", [512])
    primary_budget = config.get("primary_budget", 512)
    questions = [q for q in load_questions(qpath) if split == "all" or q["split"] == split]
    in_scope = [q for q in questions if not q.get("out_of_scope")]
    flat = FlatSections(REPO / config["sections_corpus"])

    corpora: dict[str, Any] = {}
    build_ms: dict[str, float] = {}
    for doc in sorted({document_of(q) for q in in_scope} | set(config.get("oos_documents", []))):
        t0 = time.perf_counter()
        corpora[doc] = load_processed_corpus(doc, REPO / "data/processed")
        build_ms[doc] = (time.perf_counter() - t0) * 1000
    print(f"{split}: {len(in_scope)} in-scope, {len(questions) - len(in_scope)} out-of-scope questions; "
          f"documents {', '.join(corpora)}")

    # The evidence an abstention threshold (EMTE's min_raw_score) would look at
    top_scores: dict[str, dict[str, float]] = {"in_scope": {}, "out_of_scope": {}}
    for q in questions:
        if q.get("out_of_scope"):
            for doc in config.get("oos_documents", []):
                top_scores["out_of_scope"][f"{q['qid']}|{doc}"] = max(corpora[doc].index.score(q["question"]).values())
        else:
            top_scores["in_scope"][q["qid"]] = max(corpora[document_of(q)].index.score(q["question"]).values())

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for run in config["runs"]:
        label = label_of(run)
        per_query: dict[str, dict[str, float]] = {}
        latencies: list[float] = []
        oos: dict[str, dict[str, Any]] = {}
        tracemalloc.start()
        for q in questions:
            docs = [document_of(q)] if not q.get("out_of_scope") else config.get("oos_documents", [])
            for doc in docs:
                m: dict[str, float] = {}
                for b in budgets:
                    t0 = time.perf_counter()
                    if run["method"] == "flat_sections":
                        texts, tokens, abstained = flat.select(doc, q["question"], b)
                    else:
                        corpus = corpora[doc]
                        sel = retrieve_from_processed_corpus(
                            q["question"], corpus, method=run["method"], budget=b, params=run.get("params", {}))
                        nodes = corpus.tree["nodes"]
                        texts = [region_text(nodes, nid) for nid in sel.nodes]
                        tokens, abstained = sel.delivered_tokens, sel.abstained or not sel.nodes
                    latencies.append((time.perf_counter() - t0) * 1000)
                    m[f"tokens@{b}"] = tokens
                    m[f"regions@{b}"] = len(texts)
                    m[f"abstained@{b}"] = float(abstained)
                    if not q.get("out_of_scope"):
                        gq = on_document(q, doc)
                        cov = evidence_coverage(gq, list(range(len(texts))), dict(enumerate(texts)))
                        m[f"cov@{b}"] = cov["coverage"]
                        m[f"allEss@{b}"] = cov["all_essential"]
                        m[f"gold_share@{b}"] = gold_share(gq, texts)
                if q.get("out_of_scope"):
                    oos[f"{q['qid']}|{doc}"] = m
                else:
                    per_query[q["qid"]] = m
                    rows.append({"label": label, "qid": q["qid"], "document": doc,
                                 "student_wording": int(bool(q.get("student_wording"))), **m})
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        keys = list(next(iter(per_query.values())).keys())
        mean = {k: statistics.fmean(m[k] for m in per_query.values()) for k in keys}
        groups = {
            "student wording": [q["qid"] for q in in_scope if q.get("student_wording")],
            "book wording": [q["qid"] for q in in_scope if not q.get("student_wording")],
        }
        results[label] = {
            "run": run,
            "mean": mean,
            "groups": {g: {"n": len(ids), f"cov@{primary_budget}":
                           statistics.fmean(per_query[i][f"cov@{primary_budget}"] for i in ids)}
                       for g, ids in groups.items() if ids},
            "per_query": per_query,
            "out_of_scope": oos,
            "efficiency": {"select_ms_median": statistics.median(latencies),
                           "select_ms_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
                           "select_alloc_peak_mb": peak / 2**20},
        }
        print(f"  {label:26s} cov@{primary_budget} = {mean[f'cov@{primary_budget}']:.3f}  "
              f"tokens = {mean[f'tokens@{primary_budget}']:.0f}")

    with (out_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {"results": results, "build_ms": build_ms, "top_scores": top_scores}


def compare(results: dict[str, Any], baseline: str, metric: str) -> dict[str, Any]:
    """Paired randomization test of every selector against ``baseline`` on ``metric``, Holm-adjusted."""
    base = results[baseline]["per_query"]
    qids = sorted(base)
    pvalues, diffs = {}, {}
    for label, r in results.items():
        if label == baseline:
            continue
        a = [r["per_query"][q][metric] for q in qids]
        b = [base[q][metric] for q in qids]
        pvalues[label] = randomization_test(a, b)
        diffs[label] = {"diff": statistics.fmean(a) - statistics.fmean(b),
                        "wins": sum(x > y for x, y in zip(a, b)), "losses": sum(x < y for x, y in zip(a, b))}
    adjusted = holm(pvalues)
    return {label: {**diffs[label], "p": pvalues[label], "p_holm": adjusted[label]} for label in pvalues}


def summary_md(name: str, split: str, out: dict[str, Any], tests: dict[str, Any], baseline: str,
               budgets: list[int], primary_budget: int) -> str:
    results = out["results"]
    lines = [f"# {name} ({split})", "",
             f"Evidence selectors on one document per question. Token counter: `{counter_name()}`. "
             f"Significance: paired randomization test vs `{baseline}` on cov@{primary_budget}, Holm-adjusted.", "",
             "## Evidence delivered", "",
             "| Selector | " + " | ".join(f"cov@{b}" for b in budgets) + f" | allEss@{primary_budget} | "
             f"gold share@{primary_budget} | tokens@{primary_budget} | regions@{primary_budget} | "
             f"diff vs {baseline} | p (Holm) |",
             "|---|" + "---:|" * (len(budgets) + 6)]
    for label, r in sorted(results.items(), key=lambda kv: -kv[1]["mean"][f"cov@{primary_budget}"]):
        m, t = r["mean"], tests.get(label)
        sig = f"{t['diff']:+.3f} ({t['wins']}W/{t['losses']}L) | {t['p_holm']:.3f}" if t else "baseline | "
        lines.append(f"| {label} | " + " | ".join(f"{m[f'cov@{b}']:.3f}" for b in budgets)
                     + f" | {m[f'allEss@{primary_budget}']:.3f} | {m[f'gold_share@{primary_budget}']:.3f}"
                     f" | {m[f'tokens@{primary_budget}']:.0f} | {m[f'regions@{primary_budget}']:.1f} | {sig} |")
    lines += ["", f"## Student vs book wording (cov@{primary_budget})", "", "| Selector | student | book |", "|---|---:|---:|"]
    for label, r in results.items():
        g = r["groups"]
        lines.append(f"| {label} | {g['student wording'][f'cov@{primary_budget}']:.3f} (n={g['student wording']['n']}) | "
                     f"{g['book wording'][f'cov@{primary_budget}']:.3f} (n={g['book wording']['n']}) |")
    lines += ["", "## Speed and memory (selection only; index already built)", "",
              "| Selector | median ms | p95 ms | peak alloc MB |", "|---|---:|---:|---:|"]
    for label, r in results.items():
        e = r["efficiency"]
        lines.append(f"| {label} | {e['select_ms_median']:.2f} | {e['select_ms_p95']:.2f} | {e['select_alloc_peak_mb']:.1f} |")
    lines += ["", "Index build per document (Member 1 files + Member 2 BM25F): "
              + ", ".join(f"{d} {ms:.0f} ms" for d, ms in out["build_ms"].items())]
    oos_any = any(r["out_of_scope"] for r in results.values())
    lines += ["", "## Out-of-scope questions (should abstain)", ""]
    if oos_any:
        lines += ["Each out-of-scope question is run on every document. A good selector abstains on these "
                  "and not on in-scope questions.", "",
                  f"| Selector | out-of-scope abstained (right) | in-scope abstained (wrong) | "
                  f"tokens@{primary_budget} given for out-of-scope |", "|---|---:|---:|---:|"]
        for label, r in results.items():
            o, i = r["out_of_scope"].values(), r["per_query"].values()
            lines.append(f"| {label} | {sum(m[f'abstained@{primary_budget}'] for m in o):.0f}/{len(o)} | "
                         f"{sum(m[f'abstained@{primary_budget}'] for m in i):.0f}/{len(i)} | "
                         f"{statistics.fmean(m[f'tokens@{primary_budget}'] for m in o):.0f} |")
        top = out["top_scores"]
        lines += ["", "### Can a score threshold separate them? (top raw BM25F paragraph score)", "",
                  "| min_raw_score | out-of-scope abstained (right) | in-scope abstained (wrong) |", "|---:|---:|---:|"]
        for t in THRESHOLDS:
            lines.append(f"| {t} | {sum(s < t for s in top['out_of_scope'].values())}/{len(top['out_of_scope'])} | "
                         f"{sum(s < t for s in top['in_scope'].values())}/{len(top['in_scope'])} |")
        lines += ["", "In-scope top scores: " + ", ".join(f"{s:.2f}" for s in sorted(top["in_scope"].values())),
                  "", "Out-of-scope top scores: " + ", ".join(f"{s:.2f}" for s in sorted(top["out_of_scope"].values()))]
    else:
        lines.append(f"No out-of-scope questions in the {split} split.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--split", choices=("dev", "test", "all"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split = args.split or config.get("split", "dev")
    out_dir = REPO / "results" / config["name"] / split
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets, primary_budget = config.get("budgets", [512]), config.get("primary_budget", 512)

    out = evaluate(config, split, out_dir)
    baseline = config.get("baseline", "flat_sections")
    tests = compare(out["results"], baseline, f"cov@{primary_budget}")
    (out_dir / "summary.md").write_text(
        summary_md(config["name"], split, out, tests, baseline, budgets, primary_budget), encoding="utf-8")
    meta = {"config": config, "split": split, "token_counter": counter_name(), "machine": machine(),
            "questions_sha256": sha256(REPO / config["questions"]),
            "trees_sha256": {d: sha256(REPO / "data/processed" / d / "tree.json") for d in out["build_ms"]}}
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, **out, "significance": tests}, indent=1), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
