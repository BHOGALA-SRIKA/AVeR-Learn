# Experiment log

Dated entries, newest last. Each entry: what was run, the result, and what it means.

---

## Hemanth - lexical retrieval and MCQ support

**2026-10-29 - Pilot smoke check** (`experiments/member2_lexical_check.py`, 20 questions,
28 paragraphs of unit1_topic1). BM25 15/20 at rank 1, BM25F 16/20. Headings help.
The "+exp" rows of that run are invalid (see 2026-10-31).

**2026-10-30 - Benchmark questions rebuilt on the notes.** 13 questions + 2 out of scope on
unit1_topic1-3; every gold quote found once in the notes by script.

**2026-10-30 - Retrieval for MCQ topics** (`experiments/hemanth_retrieval_accuracy.py`).
First run: student questions 3/13 right section first, because the "not in the notes"
rule refused long questions. Fixes: refuse only when key words are absent from the notes;
rank sections whose heading or exact phrase matches the topic first. After: 10/13 first,
12/13 in top 3; topic requests 90/105 first; out-of-scope topics refused 24/25
("segmentation" leaks because the notes say "segments of a program"). Development numbers:
the same questions were used while fixing.

**2026-10-30 - SLM writes whole MCQs** (run 20261030-1854, GPU allowed, 6 topics).
qwen2.5:1.5b: 18 MCQs, hand check 6 correct (33%); old checker passed 9, of which 5 were
wrong. gemma3:1b and llama3.2:1b: almost none passed. qwen3:1.7b: every call failed
(HTTP 500). Typical errors: wrong answer key, three true "wrong" options. Checker fixed;
new checker agrees with the hand check on 17 of 18.

**2026-10-30 - Evidence-locked mode, CPU only** (run 20261030-2147, 6 topics, 14 MCQs per
model).

| Model | Correct answer keys | Rewrites accepted | s / call | Memory |
|---|---|---|---|---|
| qwen3:1.7b | 14/14 | 12/14 (86%) | 8.4 | 1.89 GB |
| qwen2.5:1.5b | 14/14 | 8/14 (57%) | 8.2 | 1.17 GB |

All 28 rewrites read by hand: qwen3:1.7b wording and explanations best; one wrong qwen2.5
explanation (orphan/zombie mixed up) was correctly rejected. Chosen SLM: qwen3:1.7b.

**2026-10-30 - Bhavya's tree.json as input.** Same accuracy as the stand-in PDF reader
(10/13 first, 12/13 top 3; 24/25 refused); 53 template MCQs, none failing the checker.
With her improved output (commit e7b597d): 11/13 first, 13/13 top 3.

**2026-10-31 - Merge checks** (`experiments/member2_pr_checks.py`). BM25 and BM25F equal
plain-Python versions of the formulas to within 3e-7 on all 4 documents. Ingestion's
`n_tokens` uses a whitespace word count, not the shared counter (e.g. 3 vs 13 tokens);
to be fixed in ingestion.

**2026-10-31 - Glossary leak (Sreeram's review).** Two glossary entries ("switched on",
"without the processor") copied smoke-test questions word for word, so "+exp" scored 4/4 on
student-wording questions. 14 student-phrasing entries removed; glossary frozen to
textbook terms; leak-guard test added; expansion off by default. Re-run: bm25f+exp 17/20
(2/4 student wording) vs bm25f 16/20 (2/4). Notes questions: 10/13 first without
expansion, 11/13 with; 13/13 in top 3 either way. Sreeram's benchmark (old glossary):
bm25f 0.821, bm25f+exp 0.813. Decision on expansion: on Sreeram's dev split.
