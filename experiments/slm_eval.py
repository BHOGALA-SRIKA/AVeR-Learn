"""How accurate are small language models at making MCQs from our retrieved notes?

Runs on your own computer through Ollama (offline after the one-time model
download). Two ways of using the SLM are compared:

- free    - the SLM writes whole MCQs from the retrieved notes
            topic -> ContextRetriever (vectorless RAG) -> prompt -> SLM -> validate_mcq
- locked  - evidence-locked: the engine makes the MCQ (answer, options, evidence are
            fixed and verified), the SLM only rewrites the question and writes the
            explanation; check_rewrite keeps the template question if the rewrite is bad

Setup (once): install Ollama (https://ollama.com/download), then e.g.
    ollama pull qwen2.5:1.5b
Run:
    python experiments/slm_eval.py --quick                  # both modes, 6 topics
    python experiments/slm_eval.py --mode locked --models qwen2.5:1.5b
    python experiments/slm_eval.py --backend mock --quick   # test the script, no model
or simply: powershell -ExecutionPolicy Bypass -File experiments\\run_slm_eval.ps1

Output in data/mcq/slm_eval/<date-time>/ (saved after every model, and on Ctrl+C):
    report.md     - one row per model and mode
    results.json  - every MCQ with its checks
    review.csv    - for a human check (is it correct? is the wording good?)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
import os
from pathlib import Path

# NumPy's maths library starts one thread (and memory buffer) per CPU core; on a laptop
# that is also running the SLM this can fail with "OpenBLAS error: Memory allocation".
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.notes_sections import read_sections  # noqa: E402
from src.mcq import generate_for_topic  # noqa: E402
from src.mcq.context import MCQ_SCHEMA, ContextRetriever, build_prompt  # noqa: E402
from src.mcq.rewrite import REWRITE_SCHEMA, apply_rewrite, build_rewrite_prompt  # noqa: E402
from src.mcq.validate import validate_mcq  # noqa: E402

DEFAULT_MODELS = ["qwen2.5:1.5b", "gemma3:1b", "llama3.2:1b", "qwen3:1.7b"]
TOPICS = [
    "interrupts", "bootstrap program", "storage hierarchy", "caching", "direct memory access",
    "device controller",
    "CPU scheduler", "dispatcher", "scheduling criteria", "preemptive scheduling", "CPU and I/O bursts",
    "process states", "process control block", "context switch", "long-term scheduler",
    "fork and exec", "zombie and orphan processes", "cascading termination",
]
QUICK_TOPICS = ["interrupts", "dispatcher", "scheduling criteria", "process states", "context switch",
                "zombie and orphan processes"]
NOT_IN_NOTES = ["semaphores", "deadlock", "virtual memory", "threads"]


# --------------------------------------------------------------------- backends

class ModelError(Exception):
    pass


class Ollama:
    def __init__(self, host: str, timeout: int, cpu_only: bool = True) -> None:
        self.host, self.timeout, self.cpu_only = host.rstrip("/"), timeout, cpu_only

    def _call(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.host + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise ModelError(f"HTTP {e.code}: {body}") from None

    def available(self) -> list[str]:
        return [m["name"] for m in self._call("/api/tags").get("models", [])]

    def generate(self, model: str, prompt: str, schema: dict) -> dict:
        payload = {
            "model": model, "prompt": prompt, "stream": False, "format": schema,
            "options": {"temperature": 0.2, "seed": 42, "num_ctx": 4096},
            "keep_alive": "10m",
        }
        if self.cpu_only:
            payload["options"]["num_gpu"] = 0  # the project targets CPU-only laptops; also avoids GPU memory errors
        if any(t in model for t in ("qwen3", "deepseek-r1")):
            payload["think"] = False  # reasoning models: answer directly
        try:
            r = self._call("/api/generate", payload)
        except ModelError as first:
            # older Ollama versions or some models reject "think" or a JSON schema: retry simpler
            payload.pop("think", None)
            payload["format"] = "json"
            try:
                r = self._call("/api/generate", payload)
            except ModelError:
                raise first from None
        return {"text": r.get("response", ""), "tokens": r.get("eval_count", 0),
                "gen_seconds": r.get("eval_duration", 0) / 1e9}

    def loaded(self) -> list[str]:
        try:
            return [m.get("name") or m.get("model") for m in self._call("/api/ps").get("models", [])]
        except (ModelError, urllib.error.URLError, OSError):
            return []

    def unload(self, model: str) -> None:
        """Free the model's memory now instead of after keep_alive (10 min)."""
        try:
            self._call("/api/generate", {"model": model, "keep_alive": 0})
        except (ModelError, urllib.error.URLError, OSError):
            pass

    def memory_gb(self, model: str) -> float | None:
        try:
            for m in self._call("/api/ps").get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    return round(m.get("size", 0) / 1e9, 2)
        except (ModelError, urllib.error.URLError, OSError):
            pass
        return None


class Mock:
    """No model: realistic good and bad replies, to test the script end to end."""

    def __init__(self, paragraphs) -> None:
        self.paragraphs = paragraphs
        self.calls = 0

    def available(self) -> list[str]:
        return ["mock"]

    def generate(self, model: str, prompt: str, schema: dict) -> dict:
        self.calls += 1
        if schema is REWRITE_SCHEMA:
            draft = re.search(r"Question draft: (.+)", prompt).group(1).strip()
            answer = re.search(r"Correct answer: [A-D]\. (.+)", prompt).group(1).strip()
            if self.calls % 3 == 0:  # a bad rewrite that gives the answer away
                reply = {"question": f"Is it {answer}?", "explanation": "Because."}
            else:
                reply = {"question": draft.replace("which term matches this description", "which term means"),
                         "explanation": f"The notes describe {answer} in this way, so {answer} is correct."}
            text = json.dumps(reply)
        else:
            topic = re.search(r"Topic: (.+)", prompt).group(1).strip()
            given = set(re.findall(r"^\[([^\]]+)\]", prompt, re.M))
            good = [{"question": m.question, "options": m.options, "answer": m.answer,
                     "explanation": "From the notes.", "evidence_quote": m.evidence["quote"]}
                    for m in generate_for_topic(self.paragraphs, topic, n=10) if m.evidence["pid"] in given][:2]
            bad = {"question": f"Which is true about {topic}?", "options": ["It was invented in 1999",
                   "It is stored in the cloud", "It is stored in the cloud", "None"], "answer": "A",
                   "explanation": "", "evidence_quote": "It was invented in 1999 by engineers"}
            text = json.dumps({"mcqs": good + [bad]})
        return {"text": text, "tokens": len(text.split()), "gen_seconds": 0.01}

    def memory_gb(self, model: str) -> float | None:
        return None

    def loaded(self) -> list[str]:
        return []

    def unload(self, model: str) -> None:
        pass


# --------------------------------------------------------------------- parsing

def parse_json(text: str):
    text = (text or "").strip()
    candidates = [text]
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)  # JSON inside extra words or ``` fences
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def parse_mcqs(text: str) -> list[dict] | None:
    data = parse_json(text)
    if isinstance(data, dict):
        data = data.get("mcqs", data.get("questions", [data]))
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return None


def problem_kind(problem: str) -> str:
    if "not in the notes" in problem:
        return "evidence made up"
    if "answer key" in problem:
        return "wrong answer key"
    if "also correct" in problem or "also stated" in problem or "true fact" in problem:
        return "two correct options"
    if "longer" in problem:
        return "answer given away"
    return "format"


# --------------------------------------------------------------------- runs

def run_free(model, topics, backend, retriever, args, rows) -> dict:
    stats, kinds = Counter(), Counter()
    seconds = gen = 0.0
    tokens = 0
    for i, topic in enumerate(topics, 1):
        passages = retriever.retrieve(topic)
        if not passages:
            print(f"  [free {i}/{len(topics)}] {topic}: not in the notes, skipped")
            continue
        prompt = build_prompt(topic, passages, n=args.n, difficulty=args.difficulty,
                              hints=retriever.option_hints(passages))
        t0 = time.perf_counter()
        try:
            r = backend.generate(model, prompt, MCQ_SCHEMA)
        except (ModelError, urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"  [free {i}/{len(topics)}] {topic}: model call failed ({e})")
            stats["calls_failed"] += 1
            continue
        seconds += time.perf_counter() - t0
        tokens += r["tokens"]
        gen += r["gen_seconds"]
        stats["calls"] += 1
        mcqs = parse_mcqs(r["text"])
        if mcqs is None:
            stats["bad_json"] += 1
            print(f"  [free {i}/{len(topics)}] {topic}: reply was not valid JSON")
            rows.append({"model": model, "mode": "free", "topic": topic, "ok": False,
                         "problems": ["reply was not valid JSON"], "raw": r["text"][:500]})
            continue
        ok_here = 0
        for mcq in mcqs:
            v = validate_mcq(mcq, passages, retriever.concepts)
            stats["mcqs"] += 1
            stats["passed"] += v["ok"]
            ok_here += v["ok"]
            for p in v["problems"]:
                kinds[problem_kind(p)] += 1
            rows.append({"model": model, "mode": "free", "topic": topic, **mcq, "ok": v["ok"],
                         "problems": v["problems"], "evidence_pid": v["evidence_pid"]})
        print(f"  [free {i}/{len(topics)}] {topic}: {ok_here}/{len(mcqs)} passed  ({time.perf_counter() - t0:.1f}s)")
    n = stats["mcqs"]
    return {
        "model": model, "mode": "free", "mcqs": n, "passed": stats["passed"],
        "pass_rate": round(stats["passed"] / n, 3) if n else 0.0,
        "calls_failed": stats["calls_failed"],
        "valid_json_rate": round(1 - stats["bad_json"] / stats["calls"], 3) if stats["calls"] else 0.0,
        "problems": dict(kinds),
        "seconds_per_call": round(seconds / stats["calls"], 1) if stats["calls"] else None,
        "tokens_per_second": round(tokens / gen, 1) if gen else None,
    }


def run_locked(model, topics, backend, retriever, paragraphs, args, rows) -> dict:
    by_pid = {p.pid: p for p in paragraphs}
    stats, why = Counter(), Counter()
    seconds = gen = 0.0
    tokens = 0
    for i, topic in enumerate(topics, 1):
        mcqs = generate_for_topic(paragraphs, topic, n=args.n)
        if not mcqs:
            print(f"  [locked {i}/{len(topics)}] {topic}: no template MCQs, skipped")
            continue
        accepted_here = 0
        t_topic = time.perf_counter()
        for m in mcqs:
            sec = by_pid[m.evidence["pid"]]
            notes = f"{sec.heading}\n{sec.body}"
            t0 = time.perf_counter()
            try:
                r = backend.generate(model, build_rewrite_prompt(m, notes), REWRITE_SCHEMA)
            except (ModelError, urllib.error.URLError, OSError, TimeoutError) as e:
                print(f"  [locked {i}/{len(topics)}] {topic}: model call failed ({e})")
                stats["calls_failed"] += 1
                continue
            seconds += time.perf_counter() - t0
            tokens += r["tokens"]
            gen += r["gen_seconds"]
            stats["calls"] += 1
            data = parse_json(r["text"])
            if not isinstance(data, dict):
                data = {}
            final, problems = apply_rewrite(m, str(data.get("question", "")), str(data.get("explanation", "")), notes)
            # options, answer and evidence are unchanged; re-check the final MCQ anyway
            v = validate_mcq({"question": final["question"], "options": final["options"], "answer": final["answer"],
                              "evidence_quote": m.evidence["quote"]},
                             retriever.retrieve(topic) + [_as_passage(sec)], retriever.concepts)
            stats["mcqs"] += 1
            stats["accepted"] += final["rewrite_accepted"]
            stats["valid"] += v["ok"]
            accepted_here += final["rewrite_accepted"]
            for p in problems:
                why[p] += 1
            rows.append({"model": model, "mode": "locked", "topic": topic, "question": final["question"],
                         "template_question": m.question, "options": final["options"], "answer": final["answer"],
                         "explanation": final["explanation"], "evidence_quote": m.evidence["quote"],
                         "difficulty": m.difficulty, "rewrite_accepted": final["rewrite_accepted"],
                         "proposed_question": final["proposed_question"],
                         "proposed_explanation": final["proposed_explanation"],
                         "ok": v["ok"], "problems": problems + v["problems"]})
        print(f"  [locked {i}/{len(topics)}] {topic}: {accepted_here}/{len(mcqs)} rewrites accepted"
              f"  ({time.perf_counter() - t_topic:.1f}s)")
    n = stats["mcqs"]
    return {
        "model": model, "mode": "locked", "mcqs": n, "passed": stats["valid"],
        "pass_rate": round(stats["valid"] / n, 3) if n else 0.0,
        "rewrites_accepted": stats["accepted"],
        "rewrite_rate": round(stats["accepted"] / n, 3) if n else 0.0,
        "calls_failed": stats["calls_failed"],
        "problems": dict(why),
        "seconds_per_call": round(seconds / stats["calls"], 1) if stats["calls"] else None,
        "tokens_per_second": round(tokens / gen, 1) if gen else None,
    }


def _as_passage(p):
    from src.mcq.context import Passage
    return Passage(p.pid, p.heading, p.chapter, p.body, 1.0)


# --------------------------------------------------------------------- output

def save(out_dir, summary, rows, pdfs, topics, args, refused) -> None:
    """Write report.md, results.json and review.csv (called after every model and on Ctrl+C)."""
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "mcqs": rows}, indent=2,
                                                     ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "topic", "question", "A", "B", "C", "D", "answer", "explanation",
                    "evidence_quote", "passed_checker", "problems",
                    "human_correct (y/n)", "human_good_wording (y/n)", "human_good_wrong_options (y/n)"])
        for r in rows:
            opts = (r.get("options") or []) + [""] * 4
            w.writerow([r["model"], r["mode"], r["topic"], r.get("question", ""), *opts[:4], r.get("answer", ""),
                        r.get("explanation", ""), r.get("evidence_quote", ""), "yes" if r["ok"] else "no",
                        "; ".join(r["problems"]), "", "", ""])
    lines = [
        f"# SLM MCQ accuracy - {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Notes: {', '.join(Path(p).name for p in pdfs)}. Topics: {len(topics)}, {args.n} MCQs per topic. "
        f"Device: {'GPU allowed' if getattr(args, 'gpu', False) else 'CPU only'}. "
        f"Topics not in the notes refused before the SLM: {len(refused)}/{len(NOT_IN_NOTES)}.",
        "",
        "| Model | Mode | MCQs | Passed checker | Rewrites accepted | Wrong answer key | Two correct options | "
        "Evidence made up | Failed calls | s / call | tokens/s | Memory GB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        p = s.get("problems", {})
        rew = f"{s['rewrites_accepted']} ({s['rewrite_rate']:.0%})" if s["mode"] == "locked" else "-"
        lines.append(
            f"| {s['model']} | {s['mode']} | {s['mcqs']} | {s['passed']} ({s['pass_rate']:.0%}) | {rew} | "
            f"{p.get('wrong answer key', 0)} | {p.get('two correct options', 0)} | {p.get('evidence made up', 0)} | "
            f"{s['calls_failed']} | {s['seconds_per_call']} | {s['tokens_per_second']} | {s.get('memory_gb')} |")
    lines += ["",
              "free = the SLM writes whole MCQs; locked = the engine fixes answer, options and evidence and the SLM "
              "only rewrites the question and explanation (a rejected rewrite keeps the template question).",
              "Passed checker = grounded in the notes and well-formed. The checker agreed with a hand check on 17 of "
              "18 qwen2.5:1.5b MCQs; fill in review.csv for the real accuracy."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--mode", choices=["free", "locked", "both"], default="both")
    ap.add_argument("--backend", choices=["ollama", "mock"], default="ollama")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--quick", action="store_true", help="6 topics instead of 18")
    ap.add_argument("--n", type=int, default=3, help="MCQs per topic")
    ap.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--timeout", type=int, default=600, help="seconds allowed per model call")
    ap.add_argument("--gpu", action="store_true", help="let Ollama use the graphics card (default: CPU only)")
    ap.add_argument("--pdfs", nargs="*")
    args = ap.parse_args()

    pdfs = args.pdfs or sorted(glob.glob(str(ROOT / "data" / "raw" / "unit1_topic*.pdf")))
    paragraphs = [p for f in pdfs for p in read_sections(f)]
    retriever = ContextRetriever(paragraphs)
    topics = QUICK_TOPICS if args.quick else TOPICS
    modes = ["free", "locked"] if args.mode == "both" else [args.mode]

    if args.backend == "mock":
        backend, models = Mock(paragraphs), ["mock"]
    else:
        backend = Ollama(args.host, args.timeout, cpu_only=not args.gpu)
        try:
            have = backend.available()
        except (ModelError, urllib.error.URLError, OSError):
            sys.exit("Cannot reach Ollama at " + args.host + ". Is the Ollama app running?")
        models = [m for m in args.models if m in have or f"{m}:latest" in have]
        for m in args.models:
            if m not in models:
                print(f"skip {m}: not downloaded (run: ollama pull {m})")
        if not models:
            sys.exit("No models to test. Download one first, e.g.: ollama pull qwen2.5:1.5b")

    refused = [t for t in NOT_IN_NOTES if not retriever.retrieve(t)]
    print(f"Topics not in the notes refused before reaching the SLM: {len(refused)}/{len(NOT_IN_NOTES)}")
    out_dir = ROOT / "data" / "mcq" / "slm_eval" / datetime.now().strftime("%Y%m%d-%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    # models left in memory by an earlier run can use up the laptop's RAM
    for m in backend.loaded():
        print(f"Unloading {m} from memory")
        backend.unload(m)
    try:
        for model in models:
            if model != models[0]:
                backend.unload(models[models.index(model) - 1])  # one model in memory at a time
            for mode in modes:
                print(f"\n=== {model} ({mode})")
                if mode == "free":
                    s = run_free(model, topics, backend, retriever, args, rows)
                else:
                    s = run_locked(model, topics, backend, retriever, paragraphs, args, rows)
                s["memory_gb"] = backend.memory_gb(model)
                summary.append(s)
                save(out_dir, summary, rows, pdfs, topics, args, refused)
    except KeyboardInterrupt:
        print("\nStopped. Saving what was finished so far...")
    save(out_dir, summary, rows, pdfs, topics, args, refused)
    print("\n" + "\n".join((out_dir / "report.md").read_text(encoding="utf-8").splitlines()[4:]))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
