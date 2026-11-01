# Evaluation (draft, Member 4)

> Draft written from the dev-split results of 31 October 2026, on the section corpus built from Member 1's
> document trees. The numbers will be replaced by the test-split numbers after the benchmark has been reviewed by
> a second person. The glossary was frozen on 31 October 2026.

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
process concept). Member 1's ingestion turns each PDF into a document tree (`tree.json`) using font size and bold
flags; Member 2's `sections_from_tree` joins its line-level nodes into **78 sections** (heading + body, median 32
words, no page footers). Sections are the unit the MCQ pipeline retrieves, so they are the unit we evaluate.
An earlier pilot used a simpler stand-in reader (55 chunks, 33 without a heading, page footers inside the text);
on that corpus BM25F could not use headings and scored the same as BM25.

**Questions.** 49 questions: 45 answerable from the PDFs and 4 out-of-scope (e.g. "What is the difference between
SaaS, PaaS and IaaS?"), used later to test whether the system can say "not in this PDF". Each question records
its type (factual, conceptual, comparative, application, multi-step) and whether it uses *student wording*
(the student's own words, few words shared with the book) or *book wording*.

**Gold evidence as quotes.** Instead of labelling paragraph IDs, annotators copy the exact sentences that answer
the question from the PDF and grade them: 2 = essential, 1 = helpful. The harness matches each quote to the
sections that contain it, heading included (runs of at least 3 consecutive words covering half the quote or 8
words). This keeps
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
| Hybrid | RRF of BM25F and dense |

Default parameters were used for every method (no tuning yet), so no method had an advantage from tuning. The
embedding model reads heading + text, and the hybrid uses BM25F, so every heading-aware method sees the same
information.

## 5. Statistical testing

All methods answer the same questions, so differences are tested per question with a **paired randomization
test** (10,000 random sign flips of the per-question differences; Smucker, Allan & Carterette, 2007), with a
95% bootstrap confidence interval for the mean difference. Because 21 methods are compared with the same
baseline (BM25F, the method the MCQ pipeline uses), p-values are corrected with the **Holm–Bonferroni** method.
A difference is called significant only when the Holm-adjusted p < 0.05. A few planned pairwise comparisons
(headings: BM25F vs BM25; expansion: BM25F+exp vs BM25F) are also reported with their unadjusted p.

## 6. Efficiency measurement

Each method is measured in a fresh Python process so that memory from one method does not count against another.
Reported: median index build time over 5 builds; median and 95th-percentile query time after a warm-up pass; size
of the saved index per PDF collection; size of the model file (embedding methods only); peak process memory.
Machine: Intel Core (12 logical CPUs), 16 GB RAM, Windows 11, CPU only.

## 7. Results (dev split, 15 questions)

### 7.1 Effectiveness and evidence delivery

| Method | R@1 | R@5 | MRR@10 | nDCG@10 | AllEssential@512 |
|---|---|---|---|---|---|
| BM25 (text only) | 0.456 | 0.667 | 0.656 | 0.598 | 0.600 |
| BM25 with headings pasted into the text | 0.589 | 0.789 | 0.757 | 0.765 | 0.800 |
| **BM25F (baseline, used by the MCQ pipeline)** | 0.589 | 0.789 | 0.763 | 0.770 | 0.800 |
| BM25F + frozen glossary expansion | 0.589 | 0.767 | 0.763 | 0.773 | 0.800 |
| BM25F + RM3 | 0.456 | 0.856 | 0.712 | 0.739 | 0.867 |
| RRF (lexical fusion) | 0.589 | 0.789 | 0.745 | 0.748 | 0.800 |
| Char n-gram TF-IDF (text only) | 0.389 | 0.700 | 0.625 | 0.600 | 0.667 |
| Query Likelihood, Dirichlet (text only) | 0.456 | 0.667 | 0.646 | 0.590 | 0.600 |
| Dense (bge-small) | 0.544 | **1.000** | 0.817 | **0.854** | **1.000** |
| Hybrid (BM25F + dense) | **0.611** | 0.922 | **0.844** | 0.836 | 0.933 |

No method is significantly better than BM25F after Holm correction. Dense vs BM25F: +0.084 nDCG@10, 95% CI
[−0.020, +0.198], p = 0.15 (6 questions better, 1 worse). Hybrid vs BM25F: +0.066, CI [+0.027, +0.111],
p = 0.03 unadjusted, not significant after correction.

### 7.2 Student wording vs book wording (nDCG@10)

| Method | Book wording (n = 10) | Student wording (n = 5) |
|---|---|---|
| BM25 (text only) | 0.711 | 0.373 |
| BM25F | 0.939 | 0.433 |
| Dense | **0.963** | **0.636** |
| Hybrid | **0.963** | 0.582 |

### 7.3 Efficiency

| Method | Index build (78 sections) | Query (median) | Peak memory | Extra disk |
|---|---|---|---|---|
| BM25 | 9 ms | 0.06 ms | 110 MB | none |
| BM25F | 12 ms | 0.16 ms | 110 MB | none |
| Char n-gram TF-IDF | 55 ms | 1.7 ms | 114 MB | none |
| Dense | 3,695 ms | 7.7 ms | 1,294 MB | 64 MB model |
| Hybrid | 3,375 ms | 7.0 ms | 1,294 MB | 64 MB model |

Peak memory includes the Python interpreter (about 105 MB) for all methods. Timings vary between runs on a
laptop (an earlier run measured 7 ms / 0.13 ms for BM25F and 2,982 ms / 5.5 ms for dense); the ratios stay in
the same range.

## 8. Discussion

1. **Document structure matters most.** Using the section heading lifts BM25 from 0.598 to 0.770 nDCG@10
   (BM25F; +0.17, better on 6 questions and worse on none, p = 0.03 unadjusted). Methods that ignore headings
   (BM25 on text only, TF-IDF, Query Likelihood) lose most, because a section's topic words are often only in its
   heading. Pasting the heading into the text gives almost the same result as formal BM25F (0.765 vs 0.770).
2. **On book-worded questions, vectorless retrieval matches embeddings:** BM25F 0.939 vs dense 0.963.
3. **The gap is vocabulary mismatch.** On student-worded questions the embedding model leads (0.636 vs 0.433).
   For example, "What should a good scheduler try to keep as small as possible?" has no word in common with the
   book's "Minimize: turnaround time, waiting time, response time"; every lexical method misses it.
4. **Glossary expansion has no effect once the glossary is fair** (0.770 → 0.773, 1 question better, 1 worse).
   Its earlier gains came from entries matching the smoke-test questions. Expansion therefore stays off.
5. **The cost of embeddings is large on a laptop:** about 300–430× slower indexing, about 40–50× slower queries,
   about 12× the peak memory and a 64 MB model file, for a gain that is not statistically significant on the
   current benchmark. The embedding model does deliver all essential evidence within 512 tokens on every dev
   question (BM25F: 80%), which is where its advantage would matter for the SLM.

## 9. Limitations and threats to validity

- **Small benchmark.** With 15 dev questions only very large differences can be significant; the test split
  (30 questions) and more units are needed.
- **AI-drafted questions.** Most questions and gold quotes were drafted with AI assistance. All 55 gold quotes
  were checked automatically to appear word for word on the stated PDF page, and every question's judged sections
  were compared with the top sections retrieved by BM25F and the embedding model; seven questions gained or lost
  a judged section as a result (recorded in `check_notes`). A human review by a team member who did not develop
  any compared method is still pending; the review status is recorded per question.
- **One subject, one lecturer's notes.** Results may differ for textbooks with longer paragraphs.
- **Default parameters.** No method was tuned; tuning on dev may change the ranking.
- **Timings** are from one laptop and vary by a few percent between runs.

## 10. Reproducing the results

```
python experiments/build_corpus.py
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
