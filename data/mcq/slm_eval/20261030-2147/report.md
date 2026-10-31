# SLM MCQ accuracy - 2026-10-30 21:52

Notes: unit1_topic1.pdf, unit1_topic2.pdf, unit1_topic3.pdf. Topics: 6, 3 MCQs per topic. Device: CPU only. Topics not in the notes refused before the SLM: 4/4.

| Model | Mode | MCQs | Passed checker | Rewrites accepted | Wrong answer key | Two correct options | Evidence made up | Failed calls | s / call | tokens/s | Memory GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:1.5b | locked | 14 | 14 (100%) | 8 (57%) | 0 | 0 | 0 | 0 | 8.2 | 18.6 | 1.17 |
| qwen3:1.7b | locked | 14 | 14 (100%) | 12 (86%) | 0 | 0 | 0 | 0 | 8.4 | 15.8 | 1.89 |

free = the SLM writes whole MCQs; locked = the engine fixes answer, options and evidence and the SLM only rewrites the question and explanation (a rejected rewrite keeps the template question).
Passed checker = grounded in the notes and well-formed. The checker agreed with a hand check on 17 of 18 qwen2.5:1.5b MCQs; fill in review.csv for the real accuracy.
