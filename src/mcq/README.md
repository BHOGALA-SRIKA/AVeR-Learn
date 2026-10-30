# MCQ generation (owner: Hemanth)

Offline, CPU only, no embeddings. Turns uploaded notes into checked MCQs.

```
PDF -> sections (Bhavya's tree)
    -> ContextRetriever.retrieve(topic)      vectorless RAG: BM25F + glossary, token budget,
                                             returns [] if the topic is not in the notes
    -> build_prompt(...)                     prompt for the small language model (SLM),
                                             with related terms as wrong-option hints
    -> SLM writes MCQs as JSON
    -> validate_mcq(mcq, passages)           rejects made-up answers, two correct options,
                                             broken format, answers given away by length
    -> student
```

`generate_mcqs(paragraphs)` makes MCQs without any model, straight from the notes.
Use them as the fallback when the SLM output fails validation, as reference
questions for evaluating the SLM, and to test the pipeline today.

## Files

| File | What it does |
|---|---|
| `src/indexing/concepts.py` | Finds MCQ-worthy concepts: term/definition lists ("Throughput - Number of processes completed per unit time"), concept sections (Dispatcher, Long-Term Scheduler) and aliases (CPU scheduler = short-term scheduler). |
| `src/indexing/distractors.py` | Chooses wrong options: siblings from the same list first, then lexical neighbours (BM25). `is_safe` makes sure no wrong option is also correct. |
| `src/mcq/generate.py` | Template MCQs (`identify`, `define`, `fact`) with a difficulty label and the source page. `generate_for_topic` for a topic the student types. |
| `src/mcq/context.py` | Retrieval and prompt for the SLM. |
| `src/mcq/validate.py` | Checks any MCQ (SLM or human) against the notes: evidence really in the notes, answer key backed by it, no wrong option that is also true. |
| `src/mcq/rewrite.py` | Evidence-locked mode: the SLM only rewrites the question and writes the explanation; `check_rewrite` guards it. |
| `experiments/hemanth_mcq_demo.py` | Runs everything on the notes PDFs. |
| `src/indexing/tree_sections.py` | Reads Bhavya's `tree.json` (feature/member1-ingestion) as sections, repairing bullet marks, bold bullets stored as headings, wrapped lines and page numbers. |
| `experiments/notes_sections.py` | `load_sections`: uses `data/processed/<doc>/tree.json` when present, else a stand-in PDF reader. |

## Difficulty (for adaptive learning)

Difficulty comes from how close the wrong options are to the answer:

- **hard**: all wrong options are siblings (Ready vs Waiting vs Running; short- vs long-term scheduler)
- **medium**: some wrong options come from the same chapter
- **easy**: some wrong options come from other chapters

`generate_mcqs(paragraphs, levels=("hard",))` returns only hard questions, and so on.
`build_prompt(..., difficulty="hard")` asks the SLM for the same.

## Accuracy so far (Unit 1 notes, topics 1-3)

Retrieval (`python experiments/hemanth_retrieval_accuracy.py`, no model needed):

| Check | Result |
|---|---|
| Student questions: right section first / in top 3 | 10/13 / 12/13 |
| Topic requests ("context switch", "throughput" ...): right section first / in top 3 | 90/105 / 99/105 |
| Topics not in the notes refused (deadlock, paging, threads ...) | 24/25 + 2/2 questions |

The student questions were also used for tuning, so these are development
numbers; a held-out question set gives the final score.

Template MCQs: 56 (24 hard, 32 medium), every answer found in the notes, every
one passes the validator, about 0.5 s on CPU with no model.

SLM accuracy, first run (2026-10-30, 6 topics, laptop CPU, models from Ollama):

| Model | Size | MCQs written | Truly correct (hand check) | Checker passed |
|---|---|---|---|---|
| qwen2.5:1.5b | ~1.0 GB | 18 | 6 (33%) | 7 (agrees with the hand check on 17 of 18) |
| gemma3:1b | ~0.8 GB | 15 | - | 0 |
| llama3.2:1b | ~1.3 GB | 15 | - | 1 |
| qwen3:1.7b | ~1.4 GB | 0 (every call failed: HTTP 500) | - | - |

Small models writing whole MCQs often get the answer key wrong ("waiting for an
event" -> Running) or offer true statements as wrong options. That is why
`src/mcq/rewrite.py` adds the **evidence-locked** mode: the engine fixes the
answer, options and evidence, and the SLM only rewrites the question and writes
the explanation. A rewrite that changes the meaning, gives the answer away or
adds facts is rejected and the template question is kept, so the answer key
stays correct whatever the model does. `slm_eval.py` compares both modes.

## With Bhavya's document tree (checked 2026-10-30)

On the Unit 1 notes, her tree.json read through `tree_sections.py` gives the same
retrieval accuracy as the stand-in reader (student questions 10/13 first, 12/13 in
top 3; off-syllabus topics refused 24/25) and 53 template MCQs (26 hard), none
failing the checker. The scripts use her trees automatically once
`data/processed/unit1_topic*/tree.json` exist; add `--stand-in` to compare.

## Run

```
python experiments/hemanth_mcq_demo.py
python experiments/hemanth_mcq_demo.py --topic "scheduling criteria"
python experiments/hemanth_retrieval_accuracy.py
python -m pytest tests/test_mcq.py
```

SLM accuracy (offline after a one-time model download):

```
# install Ollama from https://ollama.com/download, then:
ollama pull qwen2.5:1.5b
ollama pull gemma3:1b
ollama pull llama3.2:1b
ollama pull qwen3:1.7b
python experiments/slm_eval.py --quick                 # both modes, 6 topics
python experiments/slm_eval.py --mode locked --models qwen2.5:1.5b
python experiments/slm_eval.py                         # all 18 topics
```

Results go to `data/mcq/slm_eval/<date-time>/` (`report.md`, `results.json`, `review.csv`).
`python experiments/slm_eval.py --backend mock` tests the script without any model.

## Not done yet

- Human check of SLM questions (`review.csv`) for real correctness, not just grounding.
- Student-performance tracking for adaptive difficulty (this code only labels difficulty).
