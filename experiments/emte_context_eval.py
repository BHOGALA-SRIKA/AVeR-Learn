"""Does EMTE give the SLM the evidence it needs, in fewer tokens, without a model?

Compares, for the 18 SLM topics (Unit 1 notes):

- ``bm25f``:   ``ContextRetriever.retrieve`` - whole BM25F-ranked sections packed into the budget,
  at most 4 (what ``slm_eval.py`` uses today)
- ``bm25f_8``: the same with at most 8 sections, so the budget, not the cap, is the limit
- ``emte``:  ``EmteContext.retrieve``      - Srika's EMTE over Bhavya's paragraph tree

Measured per topic, at the same token budgets:

- evidence found: share of the topic's template-MCQ evidence quotes that are inside the
  context, by the same rule the MCQ checker uses (``validate._find_evidence``)
- prompt tokens: tokens of the notes block the SLM would read (shared token counter)
- on-topic share: share of those tokens in passages that contain the topic's words
- time: milliseconds to select the context (CPU, no model)
- refusal: topics not in the notes must get no context

    python experiments/emte_context_eval.py
    python experiments/emte_context_eval.py --budgets 256 512

Writes results/mcq_context/summary.md and results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from notes_sections import load_sections  # noqa: E402
from slm_eval import NOT_IN_NOTES, TOPICS  # noqa: E402
from src.indexing.concepts import content_tokens  # noqa: E402
from src.mcq.context import QUESTION_WORDS, ContextRetriever  # noqa: E402
from src.mcq.emte_context import EmteContext, notes_block, prompt_tokens  # noqa: E402
from src.mcq.generate import generate_for_topic  # noqa: E402
from src.mcq.validate import _find_evidence  # noqa: E402
from src.utils.token_counter import count_tokens  # noqa: E402

DOCS = ["unit1_topic1", "unit1_topic2", "unit1_topic3"]


def on_topic_share(topic: str, passages) -> float:
    """Share of the context tokens in passages that are about the topic (same rule as
    ``ContextRetriever.about``: they contain the topic's words, not just one of them)."""
    core = content_tokens(topic) - QUESTION_WORDS
    total = sum(count_tokens(p.text) for p in passages)
    good = sum(count_tokens(p.text) for p in passages
               if core and ContextRetriever._covers(core, f"{p.heading} {p.text}"))
    return good / total if total else 0.0


class Bm25fSections:
    """``ContextRetriever.retrieve`` with up to ``max_passages`` sections (default there: 4)."""

    def __init__(self, gate: ContextRetriever, max_passages: int) -> None:
        self.gate, self.max_passages = gate, max_passages

    def retrieve(self, topic, budget_tokens):
        return self.gate.retrieve(topic, budget_tokens=budget_tokens, max_passages=self.max_passages)


def build(kind, sections):
    tracemalloc.start()
    t0 = time.perf_counter()
    gate = ContextRetriever(sections)
    if kind == "emte":
        obj = EmteContext(DOCS, gate, ROOT / "data" / "processed")
    else:
        obj = Bm25fSections(gate, 4 if kind == "bm25f" else 8)
    ms = (time.perf_counter() - t0) * 1000
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return obj, ms, peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--n", type=int, default=5, help="template MCQs per topic whose evidence must be found")
    ap.add_argument("--out", default=str(ROOT / "results" / "mcq_context"))
    args = ap.parse_args()

    sections, source = load_sections()
    quotes = {t: [m.evidence["quote"] for m in generate_for_topic(sections, t, n=args.n)] for t in TOPICS}
    built = {k: build(k, sections) for k in ("bm25f", "bm25f_8", "emte")}

    rows, per_topic = [], []
    for budget in args.budgets:
        for kind, (obj, _, _) in built.items():
            found = total = 0
            toks, share, ms, n_pass = [], [], [], []
            for topic in TOPICS:
                t0 = time.perf_counter()
                passages = obj.retrieve(topic, budget_tokens=budget)
                ms.append((time.perf_counter() - t0) * 1000)
                hit = sum(_find_evidence(q, passages) is not None for q in quotes[topic]) if passages else 0
                found += hit
                total += len(quotes[topic])
                toks.append(prompt_tokens(passages))
                share.append(on_topic_share(topic, passages))
                n_pass.append(len(passages))
                per_topic.append({"budget": budget, "method": kind, "topic": topic, "evidence_found": hit,
                                  "evidence_total": len(quotes[topic]), "prompt_tokens": toks[-1],
                                  "on_topic_share": round(share[-1], 3), "passages": len(passages),
                                  "ms": round(ms[-1], 2), "context": [p.cite() for p in passages]})
            refused = sum(not obj.retrieve(t, budget_tokens=budget) for t in NOT_IN_NOTES)
            rows.append({"budget": budget, "method": kind, "evidence_found": found, "evidence_total": total,
                         "evidence_rate": round(found / total, 3) if total else 0.0,
                         "prompt_tokens_mean": round(statistics.mean(toks), 1),
                         "on_topic_share": round(statistics.mean(share), 3),
                         "passages_mean": round(statistics.mean(n_pass), 1),
                         "ms_median": round(statistics.median(ms), 2),
                         "refused": refused, "refused_total": len(NOT_IN_NOTES)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup = {k: {"build_ms": round(v[1], 1), "build_peak_alloc_mb": round(v[2], 2)} for k, v in built.items() if k != "bm25f_8"}
    (out / "results.json").write_text(json.dumps({"sections_from": source, "setup": setup, "summary": rows,
                                                  "per_topic": per_topic}, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    lines = [
        "# SLM context: BM25F sections vs EMTE (Unit 1 notes)",
        "",
        f"Topics: {len(TOPICS)} (same as `slm_eval.py`). Sections from: {source}.",
        f"Evidence = the template-MCQ evidence quotes for each topic ({args.n} per topic at most), found with the "
        "MCQ checker's own rule. Prompt tokens = the notes block the SLM reads, shared token counter. "
        "No model is used; CPU only.",
        "",
        "| Budget | Method | Evidence found | Prompt tokens (mean) | On-topic share | Passages | ms / topic | "
        "Off-syllabus refused |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['budget']} | {r['method']} | {r['evidence_found']}/{r['evidence_total']} "
                     f"({r['evidence_rate']:.0%}) | {r['prompt_tokens_mean']:.0f} | {r['on_topic_share']:.0%} | "
                     f"{r['passages_mean']} | {r['ms_median']} | {r['refused']}/{r['refused_total']} |")
    lines += ["", "Setup cost (build once per set of PDFs):", ""]
    for k, v in setup.items():
        lines.append(f"- {k}: {v['build_ms']} ms, peak Python allocation {v['build_peak_alloc_mb']} MB")
    lines += ["", "Per-topic details, including the selected sections, are in `results.json`."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
