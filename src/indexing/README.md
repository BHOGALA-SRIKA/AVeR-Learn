# Lexical indexing (Hemanth, Member 2)

Vectorless, offline, CPU-only retrieval. Every method scores every paragraph from word statistics alone: no embeddings, no neural model, no network.

## Quick start

```python
from src.indexing.lexical import LexicalIndex

index = LexicalIndex("bm25f", expand=True).build("data/processed/paragraphs.json")
for hit in index.search("what is an ISR", k=3):
    print(hit.rank, hit.pid, round(hit.score, 3))

scores = index.score("what is an ISR")   # {"unit1_topic1_p00019": 5.42, ...} for every paragraph
index.save("indexes/bm25f_exp.pkl")
index = LexicalIndex().load("indexes/bm25f_exp.pkl")
```

Compare all methods on the pilot PDF:

```powershell
pip install -r requirements.txt
python experiments/member2_lexical_check.py
python experiments/member2_lexical_check.py --show "what is DMA"
python -m pytest tests/test_tokenizer.py tests/test_indexing.py
```

## Files

| File | What it holds |
| --- | --- |
| `src/text/tokenize.py` | The shared tokenizer every module must use |
| `src/indexing/lexical.py` | `LexicalIndex`, the interface Srika (Member 3) calls; the method list |
| `src/indexing/baselines.py` | BM25 (bm25s, Lucene), TF-IDF, character TF-IDF, Query Likelihood |
| `src/indexing/bm25f.py` | Formal BM25F |
| `src/indexing/expansion.py` | OS glossary query expansion |
| `src/indexing/fusion.py` | Reciprocal Rank Fusion |
| `src/indexing/rm3.py` | RM3 pseudo-relevance feedback |
| `src/utils/token_counter.py` | The shared token counter for context budgets |
| `src/indexing/records.py` | Turns paragraph records into body / heading / chapter fields |
| `src/indexing/bm25_index.py` | The original rank_bm25 code, kept unchanged as the reference baseline |
| `data/glossary/os_glossary.json` | Abbreviations and synonyms (research pair: extend this) |

## The tokenizer

Queries and paragraphs go through the same five steps, so "Interrupts" in a question matches "interrupt" in the book:

1. Unicode normalisation (fixes PDF ligatures such as "ﬁ") and lowercase.
2. Rejoin words split across PDF lines ("inter-↵rupt" → "interrupt").
3. Split into words, keeping "i/o" whole.
4. Remove stopwords: Lucene's English list plus question words ("what", "how", "explain") that never help find the answer.
5. Snowball stemming: "scheduling" → "schedul", "interrupts" → "interrupt".

Its rules are versioned (`TOKENIZER_VERSION`). A saved index built with a different tokenizer version refuses to load, so nobody compares scores from two tokenizers by accident. Freeze it before the final benchmark.

## The methods, in plain words

| Method | Idea | Why it is here |
| --- | --- | --- |
| `bm25` | Counts query words in the paragraph. Rare words count more (IDF). Repeats help less and less (k1). Long paragraphs are penalised (b). | The standard lexical baseline |
| `bm25_headings` | Same, with heading and chapter words pasted into the text | The simplest way to use headings, to compare against BM25F |
| `bm25f` | Heading, chapter and body are separate fields with their own weights. A word's counts are combined across fields first, then BM25's diminishing returns are applied once. | The principled way to use document structure; the main method |
| `tfidf` | Word weights, cosine similarity | Classic baseline |
| `char_tfidf` | Matches 3–5 letter pieces of words | Catches variants stemming misses ("multiprogrammed" vs "multiprogramming") |
| `ql_dirichlet` | Which paragraph's word distribution is most likely to produce the question? | A different theory (language models); a fair second baseline |
| `ql_jm` | Same, simpler smoothing | Optional comparison |
| `rrf` | Combines the rankings of bm25f, ql_dirichlet and char_tfidf using only rank positions | Different methods fail on different questions |
| `bm25_rm3`, `bm25f_rm3` | Searches once, takes the words most typical of the top paragraphs, and searches again with them added | Expansion from the corpus itself, no glossary needed. Can drift if the first search is wrong, and is known to be weak on small collections, so test it on dev questions |
| `expand=True` | Adds glossary terms to the question: "DMA" → "direct memory access", "switched on" → "boot" | Bridges student words and book words, BM25's main weakness |

### BM25F formula (Robertson, Zaragoza & Taylor 2004)

```
B_f(d)     = 1 − b_f + b_f · len_f(d) / avg_len_f        length normalisation per field
tf~(t, d)  = Σ_f  w_f · tf_f(t, d) / B_f(d)              combine the fields FIRST
score(q,d) = Σ_t∈q  idf(t) · tf~ / (k1 + tf~)             then diminishing returns, once
idf(t)     = log(1 + (N − df(t) + 0.5) / (df(t) + 0.5))   df over the whole paragraph
```

Why combine first: if heading and body were scored separately and added, a word in both would count as two independent pieces of evidence that never saturate. The original repo code adds a bonus per heading word; that is a different, simpler model, not BM25F. A test checks that BM25F with only the body field gives exactly the same scores as BM25.

## Token counter (context budgets)

`src/utils/token_counter.py` counts tokens for budgets such as "1,024 evidence tokens". Everyone uses it: Bhavya for `n_tokens` in the tree, Srika's selectors and EMTE, Sreeram's delivered-token metrics. It is separate from the retrieval tokenizer, which drops stopwords and stems words; a budget must count every token the answer model will read.

```python
from src.utils.token_counter import count_tokens, counter_name
count_tokens("An interrupt transfers control to the ISR.")   # 10
counter_name()                                               # "approx-v1": record it with every result
```

- `approx-v1` (default): a library-free approximation of a subword tokenizer. Words are one token, plus one per 8 extra letters; each digit and each symbol is one token.
- `HFTokenizerCounter("…/tokenizer.json")`: exact counts with the generator's own tokenizer file (needs `pip install tokenizers`; offline; no neural network). Switch to it once the team picks the Phase 1 generator, then recompute all token counts.

## Score contract (Interface Agreement, section 6)

- `score(query)` returns a score for every paragraph. None is left uncomputed. Higher is better.
- Scores are raw, and their ranges differ by method: BM25/BM25F from 0 up to `upper_bound(query)`, TF-IDF cosine 0–1, Query Likelihood negative log-probabilities. Never add scores from two methods; use `rrf`.
- A paragraph with no query word scores 0 for BM25, BM25F and TF-IDF. Query Likelihood still gives it a low score; `matches(query)` says which paragraphs really match.
- `upper_bound(query)` (BM25, BM25F): `score / upper_bound` lies in [0, 1), the absolute scale EMTE can use.
- `score_nodes(query, node_ids)`: paragraph IDs get paragraph scores; sections and chapters are scored as their title plus all text below them, against other nodes of the same level. This needs the tree: `build(paragraphs, tree=tree)`.
- `info()` lists the method, parameters, tokenizer version and paragraph count, for experiment records.

## Tuning on dev questions only

| Method | Parameters | Grid |
| --- | --- | --- |
| bm25, bm25_headings | k1, b | k1 ∈ {0.9, 1.2, 1.5}; b ∈ {0.5, 0.75, 0.9} |
| bm25f | weights, b per field, k1 | w_heading, w_chapter ∈ {1, 2, 3, 5} (w_body = 1); b_f ∈ {0.3, 0.5, 0.75}; k1 ∈ {0.9, 1.2, 1.5} |
| ql_dirichlet | mu | {50, 100, 250, 500, 1000, 2000} |
| ql_jm | lam | {0.1, 0.3, 0.5, 0.7} |
| rrf | k | 60 (paper default) |
| bm25_rm3, bm25f_rm3 | fb_docs, fb_terms, orig_weight | fb_docs ∈ {3, 5, 10}; fb_terms ∈ {5, 10, 20}; orig_weight ∈ {0.3, 0.5, 0.7} |

Example: `LexicalIndex("bm25f", weights={"body": 1, "heading": 3, "chapter": 1}, k1=0.9)`.

## Pilot smoke check (not the benchmark)

`experiments/member2_lexical_check.py`, 20 questions on the 28-paragraph pilot PDF, written while looking at the text:

| Method | Correct at rank 1 | In top 3 | Student-wording questions at rank 1 |
| --- | --- | --- | --- |
| bm25 | 15/20 | 18/20 | 1/4 |
| bm25f | 16/20 | 18/20 | 2/4 |
| bm25f + expansion | 19/20 | 20/20 | 4/4 |
| rrf + expansion | 19/20 | 20/20 | 4/4 |
| bm25f_rm3 | 16/20 | 18/20 | 2/4 |

This only shows the methods work and differ. The questions share many words with the text and there are only 20, so these numbers are not results. Real comparisons need the team benchmark and Sreeram's evaluation harness.

## Open points for the team

1. **Breadcrumb root:** if Bhavya's breadcrumbs start with the document title, set `BREADCRUMB_SKIP_ROOT = True` in `records.py`.
2. **Tree format for `score_nodes`:** accepts a list of node dicts, a dict id → node, or an object with `.nodes`. Confirm with Bhavya's `load_tree`.
3. **Glossary:** the seed list is small. The research pair should extend it from the textbook's index and headings, and expansion must be checked on dev questions, since it can also add noise.
4. **Saved indexes** are pickle files: load only files the team made.
