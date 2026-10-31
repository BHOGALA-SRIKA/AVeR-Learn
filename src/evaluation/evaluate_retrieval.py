"""Evaluation harness: run every method on the benchmark and write the results (owner: Member 4).

    python -m src.evaluation.evaluate_retrieval experiments/configs/unit1_all_methods.yaml
    python -m src.evaluation.evaluate_retrieval experiments/configs/unit1_all_methods.yaml --split test

One YAML config = one experiment. Output in results/<name>/<split>/:

- ``runs/<label>.trec``  ranked paragraphs per question (TREC run format:
                          qid Q0 pid rank score label), usable with trec_eval
- ``qrels.trec``         the judgements used
- ``per_query.csv``      every metric for every question and method
- ``summary.md``         the tables for the report
- ``summary.json``       the same numbers plus everything needed to reproduce
                          them (file hashes, tokenizer, token counter, machine)

Rules this harness enforces:

- Methods are compared on the SAME questions, the same corpus and the same
  judgements, so differences come from the methods only.
- The test split is for final numbers only. Tune on dev.
- Out-of-scope questions have no relevant paragraph, so they are left out of
  Recall/MRR/nDCG and listed separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.benchmark import build_qrels, deliver, evidence_coverage, load_questions
from src.evaluation.dense_baseline import make_index
from src.evaluation.metrics import evaluate_query, mean_metrics
from src.evaluation.significance import bootstrap_ci, holm, randomization_test
from src.indexing.expansion import DEFAULT_GLOSSARY
from src.indexing.records import read_paragraphs
from src.text.tokenize import TOKENIZER_VERSION
from src.utils.token_counter import count_tokens, counter_name

REPO = Path(__file__).resolve().parents[2]


def label_of(run: dict[str, Any]) -> str:
    extra = "".join(f",{k}={v}" for k, v in sorted(run.get("params", {}).items()))
    return run["method"] + ("+exp" if run.get("expand") else "") + (f"({extra[1:]})" if extra else "")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def evaluate(config: dict[str, Any], split: str, out_dir: Path) -> dict[str, Any]:
    corpus, qpath = REPO / config["corpus"], REPO / config["questions"]
    ks = tuple(config.get("ks", (1, 3, 5, 10)))
    budgets = config.get("budgets", [1024])
    primary = config.get("primary_metric", f"nDCG@{max(ks)}")

    paragraphs = read_paragraphs(corpus)
    texts = {p.pid: p.body for p in paragraphs}
    questions = [q for q in load_questions(qpath) if split == "all" or q["split"] == split]
    judgements = build_qrels(questions, paragraphs)
    if judgements.unmatched:
        raise SystemExit(f"{len(judgements.unmatched)} gold quotes not in the corpus; run experiments/build_benchmark.py")
    in_scope = [q for q in questions if not q.get("out_of_scope")]
    print(f"{split}: {len(in_scope)} in-scope and {len(questions) - len(in_scope)} out-of-scope questions, "
          f"{len(paragraphs)} paragraphs")

    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    with (out_dir / "qrels.trec").open("w", encoding="utf-8") as fh:
        for q in in_scope:
            for pid, grade in sorted(judgements.qrels[q["qid"]].items()):
                fh.write(f"{q['qid']} 0 {pid} {grade}\n")

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for run in config["runs"]:
        label = label_of(run)
        index = make_index(run["method"], run.get("expand", False), **run.get("params", {})).build(paragraphs)
        per_query, oos_top = [], []
        with (out_dir / "runs" / f"{label}.trec").open("w", encoding="utf-8") as fh:
            for q in questions:
                hits = index.search(q["question"], k=None)
                ranked = [h.pid for h in hits]
                for h in hits[:100]:
                    fh.write(f"{q['qid']} Q0 {h.pid} {h.rank} {h.score:.6f} {label}\n")
                if q.get("out_of_scope"):
                    oos_top.append(hits[0].score)
                    continue
                m = evaluate_query(ranked, judgements.qrels[q["qid"]], ks)
                for b in budgets:
                    delivered = deliver(ranked, texts, b, count_tokens)
                    cov = evidence_coverage(q, delivered, texts)
                    m[f"cov@{b}"] = cov["coverage"]
                    m[f"allEss@{b}"] = cov["all_essential"]
                    m[f"paras@{b}"] = len(delivered)
                per_query.append(m)
                rows.append({"label": label, "qid": q["qid"], "split": q["split"], "type": q.get("type", ""),
                             "student_wording": int(bool(q.get("student_wording"))), "top1": ranked[0], **m})
        groups = {
            "all": per_query,
            "student wording": [m for m, q in zip(per_query, in_scope) if q.get("student_wording")],
            "book wording": [m for m, q in zip(per_query, in_scope) if not q.get("student_wording")],
        }
        for t in sorted({q.get("type", "") for q in in_scope}):
            groups[f"type: {t}"] = [m for m, q in zip(per_query, in_scope) if q.get("type") == t]
        results[label] = {
            "run": run,
            "mean": mean_metrics(per_query),
            "groups": {g: {"n": len(ms), primary: mean_metrics(ms).get(primary)} for g, ms in groups.items() if ms},
            "per_query": {q["qid"]: m for q, m in zip(in_scope, per_query)},
            "oos_top1_score": oos_top,
            "inscope_top1_score_median": None,
        }
        print(f"  {label:22s} {primary} = {results[label]['mean'][primary]:.3f}")

    if config.get("efficiency"):
        for run in config["runs"]:
            label = label_of(run)
            cmd = [sys.executable, "-m", "src.evaluation.efficiency", "--method", run["method"],
                   "--corpus", str(corpus), "--questions", str(qpath),
                   "--repeats", str(config.get("efficiency_repeats", 3)),
                   "--params", json.dumps(run.get("params", {}))]
            if run.get("expand"):
                cmd.append("--expand")
            out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True)
            results[label]["efficiency"] = json.loads(out.stdout.strip().splitlines()[-1])
            print(f"  efficiency {label}")

    with (out_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    significance = compare(results, config.get("baseline", "bm25"), config.get("significance_metrics", [primary]))
    meta = {
        "experiment": config["name"],
        "split": split,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(questions) - len(in_scope),
        "n_paragraphs": len(paragraphs),
        "corpus": {"path": config["corpus"], "sha256": sha256(corpus)},
        "questions": {"path": config["questions"], "sha256": sha256(qpath)},
        "glossary_sha256": sha256(DEFAULT_GLOSSARY),
        "tokenizer_version": TOKENIZER_VERSION,
        "token_counter": counter_name(),
        "python": platform.python_version(),
        "config": config,
    }
    summary = {"meta": meta, "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_query"}
                                         for k, v in results.items()}, "significance": significance}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "summary.md").write_text(render(meta, results, significance, ks, budgets, primary), encoding="utf-8")
    return summary


def compare(results: dict[str, Any], baseline: str, metrics: list[str]) -> dict[str, Any]:
    """Each method vs the baseline: mean difference, 95% CI, randomization p, Holm-adjusted p."""
    if baseline not in results:
        return {}
    out: dict[str, Any] = {}
    base = results[baseline]["per_query"]
    qids = list(base)
    for metric in metrics:
        rows, pvals = {}, {}
        for label, r in results.items():
            if label == baseline:
                continue
            a = [r["per_query"][q][metric] for q in qids]
            b = [base[q][metric] for q in qids]
            lo, hi = bootstrap_ci(a, b)
            pvals[label] = randomization_test(a, b)
            rows[label] = {"diff": statistics.fmean(a) - statistics.fmean(b), "ci95": [lo, hi],
                           "p": pvals[label], "wins": sum(x > y for x, y in zip(a, b)),
                           "losses": sum(x < y for x, y in zip(a, b))}
        for label, p in holm(pvals).items():
            rows[label]["p_holm"] = p
        out[metric] = {"baseline": baseline, "n": len(qids), "methods": rows}
    return out


def _table(header: list[str], rows: list[list[Any]]) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def render(meta, results, significance, ks, budgets, primary) -> str:
    order = sorted(results, key=lambda l: -results[l]["mean"][primary])
    k_max = max(ks)
    cols = [f"R@{k}" for k in ks] + ["P@3", f"MRR@{k_max}", f"nDCG@{k_max}"]
    parts = [
        f"# {meta['experiment']} ({meta['split']} split)\n",
        f"{meta['n_in_scope']} in-scope questions, {meta['n_out_of_scope']} out-of-scope, "
        f"{meta['n_paragraphs']} paragraphs. Tokenizer {meta['tokenizer_version']}, token counter "
        f"{meta['token_counter']}, run {meta['date_utc']}. Sorted by {primary}.\n",
        "## Ranking quality\n",
        _table(["method"] + cols, [[l] + [results[l]["mean"][c] for c in cols] for l in order]),
        "\n## Evidence coverage within a token budget\n",
        "cov = share of the essential gold-quote words delivered; allEss = share of questions where "
        "every essential quote is at least 80% delivered; paras = paragraphs delivered.\n",
        _table(["method"] + [f"{m}@{b}" for b in budgets for m in ("cov", "allEss", "paras")],
               [[l] + [results[l]["mean"][f"{m}@{b}"] for b in budgets for m in ("cov", "allEss", "paras")]
                for l in order]),
        f"\n## {primary} by question group\n",
    ]
    groups = list(results[order[0]]["groups"])
    parts.append(_table(["method"] + [f"{g} (n={results[order[0]]['groups'][g]['n']})" for g in groups],
                        [[l] + [results[l]["groups"][g][primary] for g in groups] for l in order]))
    if all("efficiency" in results[l] for l in order):
        parts += [
            "\n## Speed and memory\n",
            f"Machine: {results[order[0]]['efficiency']['machine']}\n",
            "index KB = what is stored per PDF; model MB = model file shared by all PDFs (0 for lexical). "
            "Process peak MB includes Python itself (baseline about "
            f"{results[order[0]]['efficiency']['baseline_mb']:.0f} MB).\n",
            _table(["method", "build ms", "query ms (median)", "query ms (p95)", "index KB", "model MB",
                    "build alloc MB", "process peak MB"],
                   [[l, e["build_ms"], e["query_ms_median"], e["query_ms_p95"], e["index_bytes"] / 1024,
                     e["model_bytes"] / 2**20, e["build_alloc_mb"], e["process_peak_mb"]]
                    for l in order for e in [results[l]["efficiency"]]]),
        ]
    for metric, s in significance.items():
        parts += [
            f"\n## Is it better than {s['baseline']}? ({metric}, n = {s['n']} questions)\n",
            "Paired randomization test (10,000 sign flips), Holm-corrected over all methods. "
            "Significant = Holm p < 0.05.\n",
            _table(["method", "diff", "95% CI", "wins/losses", "p", "Holm p", "significant"],
                   [[l, r["diff"], f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]", f"{r['wins']}/{r['losses']}",
                     r["p"], r["p_holm"], "yes" if r["p_holm"] < 0.05 else "no"]
                    for l in order if l in s["methods"] for r in [s["methods"][l]]]),
        ]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--split", help="dev, test or all (overrides the config)")
    parser.add_argument("--no-efficiency", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.no_efficiency:
        config["efficiency"] = False
    split = args.split or config.get("split", "dev")
    out_dir = REPO / "results" / config["name"] / split
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")
    t0 = time.perf_counter()
    evaluate(config, split, out_dir)
    print(f"Done in {time.perf_counter() - t0:.0f} s. Results: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
