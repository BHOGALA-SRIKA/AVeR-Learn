# Literature review

Each member adds the part of the literature behind their component. Keep claims tied to a
reference, and keep "what we observed" separate from "what the paper says".

---

## Lexical (vectorless) retrieval - Hemanth

AVeR-Learn must run offline on a CPU laptop without embeddings or a vector database, so
retrieval is lexical: a paragraph is scored from the words it shares with the query. This
section compares the lexical models implemented in `src/indexing/` and explains why BM25F
is the default.

### Why lexical retrieval is a fair choice

On the BEIR benchmark, BM25 was a strong zero-shot baseline across 18 datasets, and many
dense (embedding) retrievers fell below it outside their training domain (Thakur et al.,
2021). Lexical models need no model weights, no GPU and no training data, index in
milliseconds, and their scores can be explained word by word. Their known weakness is
**vocabulary mismatch**: a student who writes "software interrupt" gets no credit for
a paragraph that says "trap". Glossary expansion, pseudo-relevance feedback and character
n-grams (below) are the classical answers.

### The models

**TF-IDF with cosine similarity** (Salton & Buckley, 1988). A paragraph and the query are
vectors of term weights `tf(t,d) * log(N / df(t))`, compared by cosine. Simple and fast, but
term frequency grows without limit (ten repeats count ten times) and length is handled only
by vector normalisation.

**BM25** (Robertson et al., 1995; Robertson & Zaragoza, 2009). From the probabilistic
relevance framework:

```
score(q, d) = sum over t in q of  idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

- `k1` controls **saturation**: the 1st occurrence of a word counts much more than the 10th.
- `b` controls **length normalisation**: long paragraphs are not favoured just for being long.
- Many variants exist that differ in the idf and the `(k1+1)` factor. On TREC collections
  the differences in effectiveness were small (Kamphuis et al., 2020). We use the **Lucene
  variant**, `idf = log(1 + (N - df + 0.5) / (df + 0.5))`, which is never negative, through
  the BM25S library, which precomputes scores into a sparse matrix (Lù, 2024).

**BM25F** (Robertson, Zaragoza & Taylor, 2004). For documents with fields. Each field `f`
(here: body, nearest heading, chapter title) has a weight `w_f` and its own length
normalisation `b_f`. Counts are **combined across fields first** and saturated **once**:

```
tf~(t,d)   = sum over f of  w_f * tf_f(t,d) / (1 - b_f + b_f * len_f(d) / avglen_f)
score(q,d) = sum over t of  idf(t) * tf~(t,d) / (k1 + tf~(t,d))
```

Adding separate per-field BM25 scores instead would count a word that appears in both the
heading and the body as two independent pieces of evidence, which never saturates; the
paper shows combining first is the principled form. Course notes are strongly structured
(a paragraph under "Context Switch" is about context switches even when the body says "the
CPU switches ..."), so heading evidence matters: this is why **BM25F is our default**.

**Query likelihood language models** (Ponte & Croft, 1998; Zhai & Lafferty, 2004). Rank a
paragraph by the probability that its language model generates the query. Unseen words need
**smoothing** with the collection model:

- Dirichlet: `p(t|d) = (tf(t,d) + mu * p(t|C)) / (|d| + mu)`, good for short queries.
- Jelinek-Mercer: `p(t|d) = (1 - lambda) * tf(t,d)/|d| + lambda * p(t|C)`, better for long queries.

Smoothing plays the role of idf and length normalisation (Zhai & Lafferty, 2004). Scores are
log-probabilities, never 0, so "does the paragraph match at all" must be checked separately
(`LexicalIndex.matches()`).

**Pseudo-relevance feedback, RM3** (Lavrenko & Croft, 2001; Abdul-Jaleel et al., 2004).
Search once, build a relevance model from the top `k` paragraphs, add its most likely terms
to the query and search again. It helps when the first ranking is good and **drifts** when
it is not; on very small collections (a few hundred paragraphs) the feedback set is thin.

**Reciprocal Rank Fusion** (Cormack, Clarke & Büttcher, 2009). Combines rankings, not
scores: `RRF(d) = sum over runs r of 1 / (60 + rank_r(d))`. Robust because it needs no score
calibration across different models.

**Character n-gram TF-IDF** (McNamee & Mayfield, 2004). Indexes overlapping character
n-grams, so it tolerates spelling variants and word-form changes ("preempt" /
"pre-emption") at the cost of a larger index and noisier matches.

### Comparison

| Model | Length normalisation | Term saturation | Uses headings | Parameters | Strength for course notes | Weakness |
|---|---|---|---|---|---|---|
| TF-IDF (cosine) | vector norm | no | no | none | simplest baseline | repeated words dominate |
| BM25 | `b` | yes, `k1` | no (or one merged field) | k1, b | strong, well understood | ignores structure |
| **BM25F** | per field `b_f` | yes, once | **yes, weighted** | k1, w_f, b_f | uses headings and chapters | more parameters to tune |
| QL Dirichlet | via smoothing | implicit | no | mu | strong on short queries | scores never 0 |
| QL Jelinek-Mercer | via smoothing | implicit | no | lambda | long queries | needs tuning |
| RM3 (on BM25/BM25F) | as base model | as base model | as base model | fb_docs, fb_terms, weight | bridges vocabulary | drifts on small corpora |
| RRF | none (ranks) | none | as inputs | k = 60 | robust fusion | slower, needs several runs |
| Char n-gram TF-IDF | vector norm | no | no | n-gram range | spelling variants | noisy, bigger index |

### What we observed (not yet benchmark results)

- Our BM25 and BM25F give the same scores as plain-Python versions of the formulas above,
  to within 3e-7 (`experiments/member2_pr_checks.py`).
- On the 20-question pilot smoke check, BM25F ranked the right paragraph first for 16/20
  questions vs 15/20 for BM25: headings help. This is a smoke check, not a result.
- Glossary expansion: an earlier glossary contained student phrasings, two of which copied
  smoke-test questions word for word, so its 4/4 score on student-wording questions was
  not valid (found in review, 2026-10-31). With the glossary frozen to textbook terms,
  expansion gives 2/4, the same as no expansion, and on Sreeram's fair benchmark it slightly
  lowered BM25F (0.821 -> 0.813, old glossary). Expansion is therefore off by default.
- Final comparisons use Sreeram's benchmark with a held-out test split.

### References

- Abdul-Jaleel, N., Allan, J., Croft, W. B., Diaz, F., Larkey, L., Li, X., Smucker, M. D., & Wade, C. (2004). UMass at TREC 2004: Novelty and HARD. *TREC 2004*.
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR 2009*.
- Kamphuis, C., de Vries, A. P., Boytsov, L., & Lin, J. (2020). Which BM25 do you mean? A large-scale reproducibility study of scoring variants. *ECIR 2020*.
- Lavrenko, V., & Croft, W. B. (2001). Relevance-based language models. *SIGIR 2001*.
- Lù, X. H. (2024). BM25S: Orders of magnitude faster lexical search via eager sparse scoring. *arXiv:2407.03618*.
- McNamee, P., & Mayfield, J. (2004). Character n-gram tokenization for European language text retrieval. *Information Retrieval*, 7(1-2).
- Ponte, J. M., & Croft, W. B. (1998). A language modeling approach to information retrieval. *SIGIR 1998*.
- Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M., & Gatford, M. (1995). Okapi at TREC-3. *TREC-3*.
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4).
- Robertson, S., Zaragoza, H., & Taylor, M. (2004). Simple BM25 extension to multiple weighted fields. *CIKM 2004*.
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5).
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. *NeurIPS 2021 Datasets and Benchmarks*.
- Zhai, C., & Lafferty, J. (2004). A study of smoothing methods for language models applied to information retrieval. *ACM Transactions on Information Systems*, 22(2).
