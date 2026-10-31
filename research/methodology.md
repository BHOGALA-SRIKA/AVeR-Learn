# Methodology

Each member describes how their component is built and evaluated, so results can be
reproduced and checked.

---

## Lexical retrieval and MCQ support - Hemanth

### Data

- Corpus: the Unit 1 notes PDFs (`data/raw/unit1_topic1-3.pdf`: introduction and computer
  system organization; CPU scheduling; process concept).
- Structure: Bhavya's ingestion output (`data/processed/<doc>/tree.json`, `paragraphs.jsonl`).
  `src/indexing/tree_sections.py` rebuilds sections (heading + body) from it and repairs known
  ingestion patterns (bullet marks stored as separate records, bold bullet items stored as
  headings, wrapped sentences stored as headings, page numbers). On the Unit 1 notes this
  gives the same retrieval accuracy as a direct PDF reader.

### Text processing

One shared tokenizer (`src/text/tokenize.py`, versioned by `TOKENIZER_VERSION`): Unicode
NFKC, lower case, hyphenated words joined, "i/o" kept whole, Lucene stop words plus
question words, Snowball English stemmer. The same tokenizer is used for indexing,
queries and every check. Token budgets use one shared counter (`src/utils/token_counter.py`),
which ingestion should also use (open point, 2026-10-31).

### Retrieval methods and parameters

| Method | Defaults | Dev grid |
|---|---|---|
| BM25 (Lucene variant) | k1 = 1.2, b = 0.75 | k1 in {0.9, 1.2, 1.5}, b in {0.5, 0.75, 0.9} |
| **BM25F** (default) | w = body 1, heading 2, chapter 1; b = 0.75 / 0.5 / 0.5; k1 = 1.2 | w_heading, w_chapter in {1, 2, 3, 5}; b_f in {0.3, 0.5, 0.75} |
| QL Dirichlet / JM | mu = 250 / lambda = 0.7 | mu in {50, 100, 250, 500, 1000, 2000}; lambda in {0.1, 0.3, 0.5, 0.7} |
| RM3 | 10 feedback docs, 10 terms, original-query weight 0.5 | fb_docs in {3, 5, 10}, fb_terms in {5, 10, 20}, weight in {0.3, 0.5, 0.7} |
| RRF | k = 60 over BM25F, QL, char TF-IDF | - |
| Glossary expansion | **off** | on/off decided on dev |

### Fairness rules

1. Tune parameters on the **dev split only**; report the test split once, at the end.
2. The glossary is **frozen** (`data/glossary/os_glossary.json`, version
   `1.0-frozen-2026-10-31`): textbook terms only, never added to fix a question. A test
   (`test_glossary_is_frozen_and_does_not_leak_smoke_questions`) fails if a glossary phrase
   appears inside a smoke-test question. This rule was added after review found two
   entries that copied smoke-test questions.
3. Benchmark questions: about half written without looking at the source; gold quotes are
   checked programmatically to occur in the notes.
4. Numbers measured on questions that were also used while developing are labelled
   "development", never "results".

### Retrieval evaluation

- Section hit rate: is the section holding the gold quote ranked first / in the top 3?
- Ranking metrics (MRR, nDCG) come from Sreeram's evaluation harness.
- Out-of-scope refusal: topics from later units ("deadlock", "paging") must return no
  passages. Rule: a 1-3 word topic must not contain words the notes never use, and the best
  section must contain the topic's words (all of a 1-2 word topic).
- Correctness of the scoring code: BM25 and BM25F are compared with plain-Python versions of
  the formulas on every paragraph (`experiments/member2_pr_checks.py`; max difference 3e-7).

### MCQ generation and checking

1. **Template MCQs** (`src/mcq/generate.py`): concepts and term/definition lists are found in
   the notes (`src/indexing/concepts.py`); wrong options are siblings from the same list
   first, then lexical neighbours (`src/indexing/distractors.py`); `is_safe` rejects options
   that are aliases of the answer, mean the same, or are true of the asked concept.
   Difficulty: hard (all wrong options are siblings), medium (same chapter), easy (other
   chapters).
2. **Checker** (`src/mcq/validate.py`) for any MCQ: 4 distinct options; quoted evidence
   found in the notes (word for word, or 90% of word pairs in order); answer key backed by
   the evidence; no wrong option that is part of the answer, stated in the evidence, or a
   stated fact about the asked concept; answer not much longer than the others.
   Agreement with a hand check: 17 of 18 qwen2.5:1.5b MCQs.
3. **Evidence-locked SLM mode** (`src/mcq/rewrite.py`): the engine fixes answer, options
   and evidence; the SLM only rewrites the question and writes an explanation;
   `check_rewrite` rejects rewrites that change the meaning, give the answer away, or have
   explanations longer than 3 sentences / 55 words or with words not in the notes. A
   rejected rewrite keeps the template question.

### SLM evaluation (`experiments/slm_eval.py`)

- Runtime: Ollama on the laptop, **CPU only** (`num_gpu = 0`), temperature 0.2, seed 42,
  context 4096 tokens, one model in memory at a time.
- Models: qwen2.5:1.5b, gemma3:1b, llama3.2:1b, qwen3:1.7b (thinking off).
- Modes: free (SLM writes whole MCQs) vs locked (SLM rewrites only).
- Measures: MCQs passing the checker, wrong answer keys, two correct options, made-up
  evidence, rewrites accepted, seconds per call, tokens/s, memory. Every run is saved in
  `data/mcq/slm_eval/<date-time>/` with `review.csv` for a human check.

### Threats to validity

- Small question sets (13 notes questions, 20 smoke questions, 6 SLM topics): treat all
  numbers here as development numbers.
- The hand check of SLM MCQs was done by one person on 18 MCQs.
- Section and concept extraction rely on how these notes are typed (bold headings, bullet
  lists); other courses' notes may need adjustments.
- SLM speed depends on the laptop; GPU runs (2026-10-30 19:17) are not comparable with
  CPU runs.
