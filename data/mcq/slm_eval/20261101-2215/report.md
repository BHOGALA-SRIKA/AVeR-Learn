# SLM MCQ accuracy - 2026-11-01 22:30

Notes: unit1_topic1.pdf, unit1_topic2.pdf, unit1_topic3.pdf. Topics: 18, 3 MCQs per topic. Device: CPU only. Free-mode notes: bm25f, budget 600 tokens. Topics not in the notes refused before the SLM: 4/4.

| Model | Mode | MCQs | Passed checker | Rewrites accepted | Wrong answer key | Two correct options | Evidence made up | Failed calls | s / call | tokens/s | Memory GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3:1.7b | free | 39 | 15 (38%) | - | 8 | 20 | 9 | 0 | 50.1 | 15.3 | 1.65 |

qwen3:1.7b free mode: 15 of 54 requested MCQs passed (28%); replies that were not valid JSON count as not delivered.

free = the SLM writes whole MCQs; locked = the engine fixes answer, options and evidence and the SLM only rewrites the question and explanation (a rejected rewrite keeps the template question).
Passed checker = grounded in the notes and well-formed. The checker agreed with a hand check on 17 of 18 qwen2.5:1.5b MCQs; fill in review.csv for the real accuracy.
