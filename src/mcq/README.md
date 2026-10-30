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
| `src/mcq/validate.py` | Checks any MCQ (SLM or human) against the notes. |
| `experiments/hemanth_mcq_demo.py` | Runs everything on the notes PDFs. |
| `experiments/notes_sections.py` | Stand-in PDF reader until Bhavya's tree is ready. |

## Difficulty (for adaptive learning)

Difficulty comes from how close the wrong options are to the answer:

- **hard**: all wrong options are siblings (Ready vs Waiting vs Running; short- vs long-term scheduler)
- **medium**: some wrong options come from the same chapter
- **easy**: some wrong options come from other chapters

`generate_mcqs(paragraphs, levels=("hard",))` returns only hard questions, and so on.
`build_prompt(..., difficulty="hard")` asks the SLM for the same.

## Unit 1 notes (topics 1-3) result

79 sections, 56 MCQs (24 hard, 32 medium), every answer found in the notes,
every template MCQ passes the validator, about 0.5 s on CPU with no model. Topics
not in the notes ("semaphores", "virtual memory", "deadlock", "threads") return
no passages, so no questions are made up. All MCQs still need human review.

## Run

```
python experiments/hemanth_mcq_demo.py data/raw/unit1_topic1.pdf data/raw/unit1_topic2.pdf data/raw/unit1_topic3.pdf
python experiments/hemanth_mcq_demo.py --topic "scheduling criteria"
python -m pytest tests/test_mcq.py
```

## Not done yet

- Running an actual SLM (e.g. a small instruct model through llama.cpp) on the
  prompt and measuring how many of its MCQs pass `validate_mcq`.
- Using Bhavya's document tree instead of `notes_sections.py`.
- Student-performance tracking for adaptive difficulty (this code only labels difficulty).
