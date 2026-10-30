"""Speed and memory of one retrieval method (owner: Member 4).

    python -m src.evaluation.efficiency --method bm25f --expand --corpus data/processed/corpus_unit1.json \
        --questions data/benchmark/questions.json

Prints one JSON line. The harness runs this in a NEW Python process for every
method, so memory left over from one method never counts against the next.

What is measured:

- ``build_ms``:        median of ``repeats`` index builds (wall clock)
- ``query_ms_median`` / ``query_ms_p95``: per question, after one warm-up pass
- ``index_bytes``:     size of the saved index file (``LexicalIndex.save``)
- ``build_alloc_mb``:  peak Python + NumPy memory allocated while building
  (tracemalloc; deterministic, but misses memory used by C libraries)
- ``process_peak_mb``: peak memory of the whole process (psutil), and
  ``baseline_mb``, the memory after imports and corpus loading, before any
  index exists. The difference is the method's footprint.

Laptop timings vary by a few percent between runs; close other programs and
report the machine (``machine``) with the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import psutil

from src.evaluation.benchmark import load_questions
from src.indexing.lexical import LexicalIndex
from src.indexing.records import read_paragraphs


def _peak_mb(proc: psutil.Process) -> float:
    info = proc.memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset / 2**20
    import resource  # Linux / macOS

    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (2**20 if sys.platform == "darwin" else 2**10)


def measure(method: str, expand: bool, corpus: str, questions: list[str], repeats: int = 3,
            params: dict | None = None) -> dict:
    proc = psutil.Process()
    paragraphs = read_paragraphs(corpus)
    baseline = proc.memory_info().rss / 2**20
    params = params or {}

    build_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        index = LexicalIndex(method, expand=expand, **params).build(paragraphs)
        build_times.append((time.perf_counter() - t0) * 1000)

    for q in questions:  # warm-up
        index.search(q, k=10)
    latencies = []
    for _ in range(repeats):
        for q in questions:
            t0 = time.perf_counter()
            index.search(q, k=10)
            latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.pkl"
        index.save(path)
        index_bytes = path.stat().st_size

    del index
    tracemalloc.start()
    LexicalIndex(method, expand=expand, **params).build(paragraphs)
    _, alloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": method,
        "expand": expand,
        "n_paragraphs": len(paragraphs),
        "n_queries": len(questions),
        "build_ms": statistics.median(build_times),
        "query_ms_median": statistics.median(latencies),
        "query_ms_p95": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        "index_bytes": index_bytes,
        "build_alloc_mb": alloc_peak / 2**20,
        "baseline_mb": baseline,
        "process_peak_mb": _peak_mb(proc),
        "machine": f"{platform.processor() or platform.machine()} | {os.cpu_count()} CPUs | "
                   f"{psutil.virtual_memory().total / 2**30:.0f} GB RAM | {platform.platform()}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True)
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--params", default="{}", help="JSON dict of method parameters")
    args = parser.parse_args()
    texts = [q["question"] for q in load_questions(args.questions)]
    print(json.dumps(measure(args.method, args.expand, args.corpus, texts, args.repeats, json.loads(args.params))))


if __name__ == "__main__":
    main()
