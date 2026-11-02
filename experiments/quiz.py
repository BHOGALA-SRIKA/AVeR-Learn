"""Take a quiz in the terminal: answer the MCQs yourself and get them marked.

The MCQs come from the notes (template MCQs, no model needed). For each question you
type A, B, C or D; the quiz says right or wrong, shows the line from the notes and the
page, and gives a score at the end. With the pass mark reached you are told to move up a
level, otherwise to stay. Questions you have already seen are not asked again.

    python experiments/quiz.py                                  # 5 medium questions, any topic
    python experiments/quiz.py --topic "scheduling criteria"     # only this topic
    python experiments/quiz.py --level hard --n 10 --pass-mark 0.6
    python experiments/quiz.py --reset                           # forget the questions already seen
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from notes_sections import load_sections  # noqa: E402
from src.mcq.context import ContextRetriever  # noqa: E402
from src.mcq.generate import generate_for_topic, generate_mcqs  # noqa: E402

LEVELS = ["easy", "medium", "hard"]
PROGRESS = Path.home() / ".aver_learn_quiz.json"  # outside the repo, so it is never committed


def load_seen() -> set[str]:
    try:
        return set(json.loads(PROGRESS.read_text(encoding="utf-8")).get("seen", []))
    except (OSError, ValueError):
        return set()


def save_seen(seen: set[str]) -> None:
    try:
        PROGRESS.write_text(json.dumps({"seen": sorted(seen)}, indent=1), encoding="utf-8")
    except OSError:
        pass  # the quiz still works, it just cannot remember


def key_of(m) -> str:
    """Same question = same concept, kind and evidence, even if option order changes."""
    return f"{m.kind}|{m.concept}|{m.evidence.get('pid')}|{m.answer_text}"


def ask(m, i: int, n: int) -> bool | None:
    print(f"\nQuestion {i}/{n}  [{m.difficulty}]")
    print(m.question)
    for letter, opt in zip("ABCD", m.options):
        print(f"   {letter}. {opt}")
    while True:
        a = input("Your answer (A/B/C/D, Q to stop): ").strip().upper()
        if a in {"A", "B", "C", "D", "Q"}:
            break
        print("Please type A, B, C or D.")
    if a == "Q":
        return None
    ev = m.evidence
    if a == m.answer:
        print("Correct!")
    else:
        print(f"Wrong. The answer is {m.answer}. {m.answer_text}")
    print(f'   From the notes: "{ev.get("quote", "")}"  ({ev.get("source_pdf")}, page {ev.get("page_start")})')
    return a == m.answer


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Answer MCQs from the notes in the terminal.")
    ap.add_argument("--topic", help="only questions about this topic")
    ap.add_argument("--level", choices=LEVELS, default="medium")
    ap.add_argument("--n", type=int, default=5, help="questions in this round")
    ap.add_argument("--pass-mark", type=float, default=0.6, help="share correct needed to move up (0.6 = 60%%)")
    ap.add_argument("--reset", action="store_true", help="forget which questions were already asked")
    ap.add_argument("--pdfs", nargs="*")
    args = ap.parse_args()

    pdfs = args.pdfs or sorted(glob.glob(str(ROOT / "data" / "raw" / "unit1_topic*.pdf")))
    sections, _ = load_sections(pdfs, ROOT)
    if args.topic:
        if not ContextRetriever(sections).rank(args.topic):
            sys.exit(f'"{args.topic}" is not in the notes, so there are no questions on it.')
        pool = generate_for_topic(sections, args.topic, n=1000)
    else:
        pool = generate_mcqs(sections)

    seen = set() if args.reset else load_seen()
    by_level = {lv: [m for m in pool if m.difficulty == lv] for lv in LEVELS}
    fresh = [m for m in by_level[args.level] if key_of(m) not in seen]
    print(f"Questions in the notes{f' on {args.topic!r}' if args.topic else ''}: "
          + ", ".join(f"{lv} {len(by_level[lv])}" for lv in LEVELS)
          + f". Not seen yet at {args.level}: {len(fresh)}.")
    if not by_level[args.level]:
        others = [lv for lv in LEVELS if by_level[lv]]
        sys.exit(f"The notes give no {args.level} questions yet. Try --level {others[0]}." if others
                 else "No questions could be made from these notes.")
    if not fresh:
        others = [lv for lv in LEVELS if lv != args.level and by_level[lv]]
        hint = f" Try --level {others[0]}" if others else ""
        sys.exit(f"No new {args.level} questions left.{hint} or --reset to start again.")

    random.shuffle(fresh)
    round_ = fresh[: args.n]
    right = asked = 0
    for i, m in enumerate(round_, 1):
        result = ask(m, i, len(round_))
        if result is None:
            break
        asked += 1
        right += result
        seen.add(key_of(m))
        save_seen(seen)

    if not asked:
        return
    score = right / asked
    print(f"\nScore: {right}/{asked} ({score:.0%}). Pass mark {args.pass_mark:.0%}.")
    i = LEVELS.index(args.level)
    if score >= args.pass_mark:
        nxt = LEVELS[i + 1] if i + 1 < len(LEVELS) else None
        print(f"Passed. Next: --level {nxt}" if nxt else "Passed the hardest level. Well done!")
    else:
        print(f"Not yet. Try --level {args.level} again: you will get new questions.")


if __name__ == "__main__":
    main()
