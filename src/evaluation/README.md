# Evaluation and system measurement (Member 4)

Measures how well each retrieval method finds the evidence for a question, how much of that evidence fits in the generator's token budget, and how fast and small each method is.

## Run it

```powershell
python experiments/build_corpus.py                                           # sections from Member 1's tree.json
python experiments/build_benchmark.py                                        # merge drafts, check quotes, assign splits
python -m src.evaluation.evaluate_retrieval experiments/configs/unit1_all_methods.yaml            # dev split
python -m src.evaluation.evaluate_retrieval experiments/configs/unit1_all_methods.yaml --no-efficiency   # fast (~5 s)
python -m pytest tests/test_evaluation.py
```

Results go to `results/<experiment>/<split>/summary.md` (tables for the report), `summary.json`, `per_query.csv` and `runs/*.trec`.

## Files

| File | What it holds |
| --- | --- |
| `metrics.py` | Recall@k, Precision@k, MRR, nDCG@k (graded) |
| `benchmark.py` | Loading questions, matching gold quotes to paragraphs (qrels), evidence coverage within a token budget |
| `significance.py` | Paired randomization test, bootstrap confidence interval, Holm correction |
| `efficiency.py` | Build time, query latency, index size, peak memory (one fresh process per method) |
| `evaluate_retrieval.py` | The harness: config → runs → tables |
| `experiments/build_benchmark.py` | Builds `data/benchmark/questions.json` and `qrels.trec` from `data/benchmark/drafts/` |
| `experiments/configs/*.yaml` | One file per experiment |

## Benchmark rules

1. **Gold evidence is quoted text, not paragraph IDs.** A quote is copied exactly from the PDF with a grade (2 = essential, 1 = helpful). The harness finds the paragraphs that contain it, so the benchmark survives changes to the PDF loader and chunk size.
2. **Dev and test.** About 1/3 of questions are `dev` (look at failures, tune parameters, write glossary entries), 2/3 are `test` (final numbers only, run once). The split is a hash of the qid and never changes.
3. **The glossary must not see test questions.** Glossary entries are added only from the textbook or from dev failures. `summary.json` records the glossary hash, so a changed glossary is visible.
4. **Human review.** Every draft question is marked with its annotator. A teammate must check the question, the reference answer and every gold quote before the final run, then change `annotator` to their name.
5. **Out-of-scope questions** have no gold quotes and are left out of Recall/MRR/nDCG; they test whether the system can say "not in this PDF" (EMTE).

## Metric choices

- Primary metric: **nDCG@10**. It uses the grades, so ranking essential evidence above helpful evidence counts.
- **Evidence coverage** (`cov@B`, `allEss@B`) answers the question the generator cares about: with B tokens of context, is the evidence actually inside? Paragraphs are delivered in rank order until the next one does not fit. Tokens are counted with `src/utils/token_counter.py`, and the counter name is recorded.
- **Significance:** paired randomization test against the baseline, Holm-corrected over all methods (Smucker, Allan & Carterette 2007). With 15 dev questions, only large gaps can be significant.
