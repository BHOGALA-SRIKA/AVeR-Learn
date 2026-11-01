# SLM context: BM25F sections vs EMTE (Unit 1 notes)

Topics: 18 (same as `slm_eval.py`). Sections from: unit1_topic1: tree.json; unit1_topic2: tree.json; unit1_topic3: tree.json.
Evidence = the template-MCQ evidence quotes for each topic (5 per topic at most), found with the MCQ checker's own rule. Prompt tokens = the notes block the SLM reads, shared token counter. No model is used; CPU only.

| Budget | Method | Evidence found | Prompt tokens (mean) | On-topic share | Passages | ms / topic | Off-syllabus refused |
|---|---|---|---|---|---|---|---|
| 256 | bm25f | 43/48 (90%) | 259 | 76% | 3.6 | 0.31 | 4/4 |
| 256 | bm25f_8 | 43/48 (90%) | 343 | 65% | 4.9 | 0.65 | 4/4 |
| 256 | emte | 44/48 (92%) | 335 | 71% | 4.8 | 6.47 | 4/4 |
| 384 | bm25f | 43/48 (90%) | 271 | 76% | 3.6 | 0.36 | 4/4 |
| 384 | bm25f_8 | 43/48 (90%) | 454 | 56% | 6.3 | 0.52 | 4/4 |
| 384 | emte | 46/48 (96%) | 458 | 60% | 6.3 | 9.16 | 4/4 |
| 512 | bm25f | 43/48 (90%) | 271 | 76% | 3.6 | 0.37 | 4/4 |
| 512 | bm25f_8 | 48/48 (100%) | 497 | 55% | 6.6 | 0.43 | 4/4 |
| 512 | emte | 47/48 (98%) | 541 | 56% | 6.7 | 7.57 | 4/4 |

Setup cost (build once per set of PDFs):

- bm25f: 117.0 ms, peak Python allocation 0.31 MB
- emte: 352.1 ms, peak Python allocation 4.36 MB

Per-topic details, including the selected sections, are in `results.json`.
