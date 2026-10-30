# SLM MCQ accuracy - 2026-10-30 19:33

Notes: unit1_topic1.pdf, unit1_topic2.pdf, unit1_topic3.pdf. Topics: 6, 3 MCQs per topic. Topics not in the notes refused before the SLM: 4/4.

| Model | Mode | MCQs | Passed checker | Rewrites accepted | Wrong answer key | Two correct options | Evidence made up | Failed calls | s / call | tokens/s | Memory GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:1.5b | free | 18 | 7 (39%) | - | 5 | 13 | 3 | 0 | 6.6 | 121.6 | 1.17 |
| qwen2.5:1.5b | locked | 14 | 14 (100%) | 11 (79%) | 0 | 0 | 0 | 0 | 2.9 | 103.2 | 1.17 |
| gemma3:1b | free | 16 | 1 (6%) | - | 11 | 8 | 4 | 0 | 12.1 | 125.8 | 0.88 |
| gemma3:1b | locked | 14 | 14 (100%) | 3 (21%) | 0 | 0 | 0 | 0 | 2.9 | 80.5 | 0.88 |
| llama3.2:1b | free | 3 | 0 (0%) | - | 0 | 3 | 3 | 4 | 184.6 | 110.1 | 1.52 |
| llama3.2:1b | locked | 14 | 14 (100%) | 4 (29%) | 0 | 0 | 0 | 0 | 2.8 | 94.5 | 1.52 |
| qwen3:1.7b | free | 0 | 0 (0%) | - | 0 | 0 | 0 | 6 | None | None | 1.71 |
| qwen3:1.7b | locked | 12 | 12 (100%) | 9 (75%) | 0 | 0 | 0 | 2 | 5.3 | 42.9 | 1.71 |

free = the SLM writes whole MCQs; locked = the engine fixes answer, options and evidence and the SLM only rewrites the question and explanation (a rejected rewrite keeps the template question).
Passed checker = grounded in the notes and well-formed. The checker agreed with a hand check on 17 of 18 qwen2.5:1.5b MCQs; fill in review.csv for the real accuracy.
