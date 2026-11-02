# Evaluation (Member 4)

> Final numbers: test split (30 answerable + 4 out-of-scope questions), run once on 1 November 2026 after the
> benchmark was human-reviewed and every parameter was frozen on dev. The glossary was frozen on 31 October 2026;
> EMTE's abstention threshold (min_raw_score = 0) was fixed on dev before the test run.

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

**Questions.** 53 questions: 45 answerable from the PDFs and 8 out-of-scope (e.g. "What is the difference between
SaaS, PaaS and IaaS?"; their topics belong to later units of the course), used to test whether the system can say
"not in this PDF". Each question records
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
ID, so it is fixed. In total: 15 dev and 30 test answerable questions, and 4 out-of-scope questions in each split
(the 4 dev ones were added on purpose so that abstention thresholds are chosen on dev, never on test).

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
| Evidence selectors (Member 3) | Fixed-K, score-ratio, score-gap, Adaptive-k, section-then-paragraph, fixed granularity (paragraph / subsection / section), Auto-Merge, HiChunk-style Auto-Merge (with and without stop rule), INEX-style element selection, and the team's proposed EMTE, all on Member 1's tree with Member 2's BM25F index; reference: the current pipeline (BM25F sections delivered top-down) |

Default parameters were used for every method (no tuning), so no method had an advantage from tuning. The
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

## 7. Results (test split, 30 questions)

### 7.1 Effectiveness and evidence delivery

| Method | R@1 | R@5 | MRR@10 | nDCG@10 | AllEssential@512 |
|---|---|---|---|---|---|
| BM25 (text only) | 0.489 | 0.728 | 0.698 | 0.670 | 0.767 |
| BM25 with headings pasted into the text | 0.539 | 0.806 | 0.768 | 0.762 | 0.867 |
| **BM25F (baseline, used by the MCQ pipeline)** | 0.539 | 0.817 | 0.770 | 0.766 | 0.867 |
| BM25F + frozen glossary expansion | 0.539 | 0.833 | 0.791 | 0.795 | 0.900 |
| BM25F + RM3 | 0.489 | 0.817 | 0.725 | 0.732 | 0.833 |
| RRF (lexical fusion) | 0.572 | 0.800 | 0.787 | 0.790 | 0.833 |
| Char n-gram TF-IDF (text only) | 0.478 | 0.711 | 0.679 | 0.673 | 0.767 |
| Query Likelihood, Dirichlet (text only) | 0.478 | 0.717 | 0.676 | 0.678 | 0.800 |
| Dense (bge-small) | 0.522 | 0.817 | 0.769 | 0.766 | 0.867 |
| Hybrid (BM25F + dense) | **0.639** | **0.917** | **0.873** | **0.850** | **0.933** |

**Dense and BM25F are tied** on the test split: 0.766 nDCG@10 each, difference 0.000, 95% CI [−0.090, +0.094],
p = 0.996 (dense better on 8 questions, worse on 9). No method is significantly better than BM25F after Holm
correction. Hybrid vs BM25F: +0.084, CI [+0.002, +0.176], p = 0.074 unadjusted. Headings (BM25F vs BM25): +0.096,
BM25F better on 8 questions and worse on 3, p = 0.065 unadjusted (dev: +0.172, p = 0.028). Glossary expansion:
+0.029, p = 0.12; it stays off, as decided on dev.

### 7.2 Student wording vs book wording (nDCG@10)

| Method | Book wording (n = 21) | Student wording (n = 9) |
|---|---|---|
| BM25 (text only) | 0.716 | 0.564 |
| BM25F | 0.868 | 0.528 |
| Dense | 0.898 | 0.458 |
| Hybrid | **0.945** | **0.628** |

### 7.3 Development split (15 questions), for comparison

| Method | nDCG@10 | Book (n = 10) | Student (n = 5) |
|---|---|---|---|
| BM25 | 0.598 | 0.711 | 0.373 |
| BM25F | 0.770 | 0.939 | 0.433 |
| Dense | 0.854 | 0.963 | 0.636 |
| Hybrid | 0.836 | 0.963 | 0.582 |

On dev the embedding model led by +0.084 (p = 0.15) through the student-worded questions; on the larger test split
that lead disappears. Neither split shows a significant difference between dense and BM25F.

### 7.4 Efficiency

| Method | Index build (78 sections) | Query (median) | Peak memory | Extra disk |
|---|---|---|---|---|
| BM25 | 9 ms | 0.06 ms | 110 MB | none |
| BM25F | 12 ms | 0.16 ms | 110 MB | none |
| Char n-gram TF-IDF | 55 ms | 1.7 ms | 114 MB | none |
| Dense | 3,695 ms | 7.7 ms | 1,294 MB | 64 MB model |
| Hybrid | 3,375 ms | 7.0 ms | 1,294 MB | 64 MB model |

Measured in the dev run. Peak memory includes the Python interpreter (about 105 MB). Timings vary between runs on a
laptop: in the test run the machine was busier (BM25F 17 ms build / 0.78 ms query; dense 6,731 ms / 96 ms), but the
ratios stay in the same range (dense about 300–400× slower to index, 50–120× slower per query, 12× the memory).

### 7.5 Evidence selectors (Member 3) and EMTE

`src/evaluation/evaluate_selectors.py` runs each selector on the PDF that holds the question's essential quotes (the
app answers questions about one uploaded PDF) and scores the text it delivers: coverage of the essential quotes,
tokens used, and abstention. Selectors return sets of tree regions, not rankings, so Recall/MRR/nDCG do not apply.
Questions whose quotes come from two PDFs (CL-016, CL-026, test only) are run on the PDF with most essential quotes;
this rule was fixed before the selector test run.

| Selector (test, 30 questions) | Coverage@256 | Coverage@512 | AllEssential@512 | Tokens@512 | vs EMTE (Holm p) |
|---|---|---|---|---|---|
| Current pipeline: BM25F sections top-down | 0.869 | **0.983** | **0.967** | 413 | +0.042 (1.000) |
| **EMTE (proposed)** | **0.895** | 0.941 | 0.900 | 430 | — |
| Fixed granularity: section | 0.802 | 0.956 | 0.933 | 505 | +0.015 (1.000) |
| HiChunk-style Auto-Merge | 0.714 | 0.793 | 0.667 | 409 | −0.148 (0.058) |
| Section-then-paragraph | 0.609 | 0.703 | 0.567 | 428 | −0.238 (0.002) |
| Fixed granularity: paragraph | 0.593 | 0.692 | 0.533 | 509 | −0.249 (0.002) |
| Auto-Merge (plain) | 0.619 | 0.619 | 0.467 | 131 | −0.322 (0.002) |
| Adaptive-k | 0.558 | 0.574 | 0.367 | 242 | −0.367 (0.002) |
| INEX-style element selection | 0.487 | 0.546 | 0.300 | 508 | −0.395 (0.002) |
| Score-ratio | 0.521 | 0.544 | 0.333 | 203 | −0.397 (0.002) |
| Fixed-K (K = 5) | 0.462 | 0.462 | 0.200 | 114 | −0.479 (0.002) |
| Score-gap | 0.448 | 0.461 | 0.300 | 149 | −0.480 (0.002) |

EMTE is significantly better than eight of the nine other Member 3 selectors (Holm p ≤ 0.002; HiChunk-style:
p = 0.058) and has the **highest coverage at the tight 256-token budget** (0.895). At 512 tokens it is not
significantly different from the current pipeline (0.941 vs 0.983; 2 questions worse, none better) or from
section-level selection; on dev the three were tied at 0.948. With the default tau_up = 0, EMTE expands nearly every
seed to its whole section (100 of 117 regions on dev), so it behaves like evidence-ordered section selection; the
dev ablations (aggregator, tau_up, max_regions) did not improve on the default. All selectors take tens of
milliseconds per question (EMTE median 49 ms).

### 7.6 Saying "not in this PDF" (abstention)

Each out-of-scope question is run on every Unit 1 PDF (12 cases per split). With the threshold frozen on dev
(min_raw_score = 0), EMTE abstained on 1 of 12 test cases and never on an answerable question; the other selectors
likewise deliver unrelated text for out-of-scope questions. On dev no threshold separated the two groups:
out-of-scope top BM25F scores (1.51–5.15) overlapped the answerable ones (1.65–4.85), and the questions a threshold
wrongly refused were mostly student-worded. On test the out-of-scope questions happened to score lower (0–2.01 vs
1.74–8.21), so a threshold of 1.7 would have caught 7 of 12 without refusing a real question; because this was seen
only on test, it is reported as an observation, not used as a result.

## 8. Discussion

1. **Vectorless retrieval matches the embedding model on unseen questions.** On the 30 test questions BM25F and
   bge-small have the same nDCG@10 (0.766), while BM25F indexes hundreds of times faster, answers a query tens of
   times faster, needs about 12× less memory and no model file. The dev lead of the embedding model (+0.084, not
   significant) did not carry over to test. For an offline MCQ app on a student laptop, BM25F is the better trade-off.
2. **Document structure helps.** Using the section heading lifts BM25 by +0.17 (dev) and +0.10 (test) nDCG@10.
   Methods that ignore headings (BM25 on text only, TF-IDF, Query Likelihood) are the weakest on both splits.
3. **Combining both is best but costs the most.** The hybrid (BM25F + dense) has the highest scores on test
   (0.850 nDCG@10, 93% of questions with all essential evidence in 512 tokens), but needs the embedding model, and
   its gain over BM25F is not significant after correction.
4. **Student wording is the weak spot for every method** (BM25F, dense and hybrid: 0.46–0.63 on test vs 0.87–0.95 for book wording).
   Glossary expansion did not fix it (+0.003 dev, +0.029 test, both not significant), and the embedding model was
   not consistently better on it either (dev 0.636 vs 0.433; test 0.458 vs 0.528).
5. **EMTE beats the other proposed selectors but not the simple section pipeline.** On these short documents
   (862–1,646 tokens per PDF) a 512-token budget is a large share of a document, so delivering whole sections in
   BM25F order is hard to beat. EMTE's advantage appears at the tight 256-token budget on test (0.895 vs 0.869),
   which is where an evidence-aware selector should help; longer documents are needed to confirm it.
6. **Lexical scores cannot reliably detect out-of-scope questions** on this corpus, because an out-of-scope
   question can share common words (disk, data, algorithm) with the notes while a student-worded in-scope question
   may share almost none.

## 9. Limitations and threats to validity

- **Small benchmark.** 30 test questions; only large differences can be significant, and dev and test disagreed
  on dense vs BM25F. More units and subjects are needed.
- **Human review.** Questions and gold quotes were drafted with AI assistance (Hemanth's 15 with AI help). All gold
  quotes were checked automatically to appear word for word on the stated PDF page; the answerable questions were
  then reviewed by Bhavya (and the 34 CL questions also by Srika), and the out-of-scope questions by Sreeram and
  Srika. Reviewers are recorded per question. Bhavya did not build any compared method; Srika built EMTE, which
  is why Bhavya's review of every answerable question matters.
- **One subject, one lecturer's notes, short PDFs.** Results may differ for textbooks with longer sections.
- **Default parameters.** No retrieval method was tuned; EMTE's ablations were run on dev only.
- **Single-document setting for selectors.** Two test questions draw evidence from two PDFs; they were scored on the
  PDF with most essential quotes.
- **Timings** are from one laptop and varied between runs (see 7.4); the ratios are stable.

## 10. Reproducing the results

```
python experiments/build_corpus.py
python experiments/build_benchmark.py
python -m src.evaluation.evaluate_retrieval experiments/configs/unit1_all_methods.yaml --split test
python -m src.evaluation.report_html results/unit1_all_methods/test
python -m src.evaluation.evaluate_selectors experiments/configs/member3_selectors.yaml --split test
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
