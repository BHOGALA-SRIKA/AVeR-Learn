# Evaluation (draft, Member 4)

> Draft written from the dev-split results of 31 October 2026. Numbers marked *dev* will be replaced by the
> test-split numbers after the benchmark has been reviewed by a second person and the glossary is frozen.

## 1. What we evaluate and why

AVeR-Learn generates MCQs from a page range of an uploaded PDF. The small language model (SLM) can only write a
correct question if the retrieval step hands it the right evidence inside a small context budget. We therefore
evaluate retrieval on its own, before generation, and ask three questions:

1. **Effectiveness.** Does the method put the paragraphs that contain the answer at the top?
2. **Evidence delivery.** With a fixed token budget for the SLM (256, 512 or 1,024 tokens), is the evidence
   actually inside the context?
3. **Efficiency.** How long do indexing and a query take, how much memory is needed, and how large is the stored
   index on a student laptop, fully offline?

The project's claim is that **vectorless (lexical) retrieval is good enough while being much lighter** than
embedding-based retrieval. To test it fairly, we compare the lexical methods against a small embedding model and a
hybrid of both, on the same questions, corpus and judgements.

## 2. Benchmark

**Corpus.** Three Unit 1 Operating Systems PDFs (introduction and computer-system organisation; CPU scheduling;
process concept), split by the project's PDF loader into 55 paragraphs.

**Questions.** 49 questions: 45 answerable from the PDFs and 4 out-of-scope (e.g. "What is the difference between
SaaS, PaaS and IaaS?"), used later to test whether the system can say "not in this PDF". Each question records
its type (factual, conceptual, comparative, application, multi-step) and whether it uses *student wording*
(the student's own words, few words shared with the book) or *book wording*.

**Gold evidence as quotes.** Instead of labelling paragraph IDs, annotators copy the exact sentences that answer
the question from the PDF and grade them: 2 = essential, 1 = helpful. The harness matches each quote to the
paragraphs that contain it (runs of at least 3 consecutive words covering half the quote or 8 words). This keeps
the benchmark valid when the PDF loader or chunk size changes, which paragraph IDs would not. A paragraph's
relevance grade is the highest grade of the quotes it contains.

**Splits.** About one third of the questions are *dev* (used to inspect failures, tune parameters and write
glossary entries), two thirds are *test* (used once, for the final numbers). The split is a hash of the question
ID, so it is fixed. In total: 15 dev and 30 test answerable questions.

**Avoiding leakage.** The pilot smoke test of the indexing module contained glossary entries ("switched on",
"without the processor") that matched its own test questions word for word, which inflated the benefit of
query expansion. The benchmark therefore follows two rules: glossary entries may come only from the textbook or
from dev failures, and the glossary is frozen (its hash is recorded in every result file) before the test split
is run.

## 3. Metrics

For a query with ranked paragraphs and graded judgements (Manning et al., 2008; Järvelin & Kekäläinen, 2002):

| Metric | Definition | Why |
|---|---|---|
| Recall@k | relevant paragraphs in the top k / all relevant paragraphs | Is the evidence retrieved at all? |
| Precision@k | relevant paragraphs in the top k / k | How much of the context is useful? |
| MRR@10 | 1 / rank of the first relevant paragraph | How soon is the first useful paragraph found? |
| **nDCG@10** (primary) | DCG with gain 2^grade − 1 and discount log2(rank + 1), divided by the ideal DCG | Uses the grades: essential evidence ranked high counts most |
| Coverage@B | share of the essential gold-quote words inside the paragraphs delivered within B tokens | What the SLM actually receives |
| AllEssential@B | share of questions where every essential quote is at least 80% delivered | Can the SLM answer at all? |

Paragraphs are delivered in rank order until the next one no longer fits the budget. Tokens are counted with the
project's shared token counter (`approx-v1`), which is recorded with every result. Out-of-scope questions have no
relevant paragraph and are excluded from these metrics. All means are macro averages (every question counts equally).

## 4. Methods compared

| Group | Methods |
|---|---|
| Vectorless (lexical) | BM25, BM25 with headings, BM25F, TF-IDF, character n-gram TF-IDF, Query Likelihood (Dirichlet, Jelinek-Mercer), RRF fusion, RM3 feedback; each with and without glossary expansion |
| Embedding | `dense`: BAAI/bge-small-en-v1.5 (33M parameters, 384 dimensions, 64 MB ONNX model on the CPU) |
| Hybrid | RRF of BM25 and dense |

Default parameters were used for every method (no tuning yet), so no method had an advantage from tuning.

## 5. Statistical testing

All methods answer the same questions, so differences are tested per question with a **paired randomization
test** (10,000 random sign flips of the per-question differences; Smucker, Allan & Carterette, 2007), with a
95% bootstrap confidence interval for the mean difference. Because 21 methods are compared with the same
baseline (BM25), p-values are corrected with the **Holm–Bonferroni** method. A difference is called significant
only when the Holm-adjusted p < 0.05.

## 6. Efficiency measurement

Each method is measured in a fresh Python process so that memory from one method does not count against another.
Reported: median index build time over 5 builds; median and 95th-percentile query time after a warm-up pass; size
of the saved index per PDF collection; size of the model file (embedding methods only); peak process memory.
Machine: Intel Core (12 logical CPUs), 16 GB RAM, Windows 11, CPU only.

## 7. Results (dev split, 15 questions)

### 7.1 Effectiveness and evidence delivery

| Method | R@1 | R@5 | MRR@10 | nDCG@10 | AllEssential@512 |
|---|---|---|---|---|---|
| BM25 (baseline) | 0.622 | 0.833 | 0.840 | 0.821 | 0.800 |
| BM25F | 0.622 | 0.833 | 0.840 | 0.821 | 0.800 |
| BM25F + glossary expansion | 0.622 | 0.811 | 0.848 | 0.813 | 0.667 |
| Char n-gram TF-IDF | 0.689 | 0.900 | 0.880 | 0.863 | 0.867 |
| RRF (lexical fusion) | 0.622 | 0.833 | 0.847 | 0.828 | 0.800 |
| BM25F + RM3 | 0.589 | 0.833 | 0.810 | 0.815 | 0.800 |
| Query Likelihood (Dirichlet) | 0.589 | 0.833 | 0.800 | 0.794 | 0.800 |
| Dense (bge-small) | 0.589 | 0.978 | 0.856 | **0.895** | **0.933** |
| Hybrid (BM25 + dense) | 0.689 | 0.833 | 0.886 | 0.875 | 0.800 |

No method is significantly better than BM25 after Holm correction. The largest gap, dense vs BM25 (+0.075
nDCG@10), has a 95% CI of [−0.046, +0.205] and p = 0.30.

### 7.2 Student wording vs book wording (nDCG@10)

| Method | Book wording (n = 10) | Student wording (n = 5) |
|---|---|---|
| BM25 | 0.913 | 0.635 |
| Char n-gram TF-IDF | **0.984** | 0.621 |
| Dense | 0.960 | **0.766** |
| Hybrid | 0.961 | 0.702 |

### 7.3 Efficiency

| Method | Index build (55 paragraphs) | Query (median) | Peak memory | Extra disk |
|---|---|---|---|---|
| BM25 | 7 ms | 0.07 ms | 109 MB | none |
| BM25F + expansion | 12 ms | 0.46 ms | 109 MB | none |
| Char n-gram TF-IDF | 88 ms | 2.9 ms | 114 MB | none |
| Dense | 2,790 ms | 6.4 ms | 815 MB | 64 MB model |
| Hybrid | 2,565 ms | 7.6 ms | 818 MB | 64 MB model |

Peak memory includes the Python interpreter (about 105 MB) for all methods.

## 8. Discussion

1. **Vectorless retrieval is competitive on book-worded questions.** Character n-gram TF-IDF (0.984) and BM25
   (0.913) are as good as or better than the embedding model (0.960) when the question uses the book's terms.
2. **The gap is vocabulary mismatch.** On student-worded questions the embedding model leads (0.766 vs 0.635).
   For example, "What should a good scheduler try to keep as small as possible?" has no word in common with the
   book's "Minimize: turnaround time, waiting time, response time"; every lexical method misses it.
3. **Glossary expansion did not help on fair questions.** It lowered nDCG@10 slightly (0.821 → 0.813), because
   the common expansions ("OS" → "operating system", "CPU" → "central processing unit") add frequent words while
   the student phrases in the benchmark were not in the glossary. Its earlier gains came from entries matching the
   smoke-test questions.
4. **BM25F equals BM25 on this corpus**, because the PDF loader leaves the chapter field empty and many headings
   missing. Structure-aware ranking can only be judged once the document tree provides those fields.
5. **The cost of embeddings is large on a laptop:** about 400× slower indexing, about 100× slower queries,
   about 7.5× the peak memory and a 64 MB model file, for a gain that is not statistically significant on the
   current benchmark.

## 9. Limitations and threats to validity

- **Small benchmark.** With 15 dev questions only very large differences can be significant; the test split
  (30 questions) and more units are needed.
- **AI-drafted questions.** Most questions and gold quotes were drafted with AI assistance and are pending human
  review; annotator names are recorded per question.
- **One subject, one lecturer's notes.** Results may differ for textbooks with longer paragraphs.
- **Default parameters.** No method was tuned; tuning on dev may change the ranking.
- **Timings** are from one laptop and vary by a few percent between runs.

## 10. Reproducing the results

```
python experiments/build_benchmark.py
python -m src.evaluation.evaluate_retrieval experiments/configs/unit1_all_methods.yaml --split dev
python -m src.evaluation.report_html results/unit1_all_methods/dev
```

Every result file records the corpus, question and glossary hashes, the tokenizer version, the token counter,
the configuration and the machine.

## References

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR*.
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM TOIS*, 20(4).
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*, ch. 8. Cambridge University Press.
- Robertson, S., Zaragoza, H., & Taylor, M. (2004). Simple BM25 extension to multiple weighted fields. *CIKM*.
- Smucker, M. D., Allan, J., & Carterette, B. (2007). A comparison of statistical significance tests for information retrieval evaluation. *CIKM*.
- Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged resources to advance general Chinese embedding (BGE models). arXiv:2309.07597.
