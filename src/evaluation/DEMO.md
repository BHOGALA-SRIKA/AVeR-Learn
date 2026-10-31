# Demo guide: evaluation part (Member 4)

About 10 minutes. Open PowerShell in the project folder first:

```powershell
cd D:\AVeR-Learn-feature-member2-indexing
```

## 0. Before mam arrives (once)

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m src.evaluation.evaluate_retrieval experiments/configs/unit1_all_methods.yaml --split dev
.\.venv\Scripts\python -m src.evaluation.report_html results/unit1_all_methods/dev
```

The second command takes about 3 minutes (it measures speed and memory). Then open
`results\unit1_all_methods\dev\report.html` in the browser and keep it open.

## 1. Tests (30 s)

```powershell
.\.venv\Scripts\python -m pytest -q
```

Say: "All tests pass, including 14 for my evaluation code. The metric tests use values worked out by hand, so they
check the formulas, not just that the code runs."

## 2. A student-worded question (3 min)

```powershell
.\.venv\Scripts\python experiments/demo_evaluation.py
```

Walk through the 6 steps. Points to make:
- Gold evidence is **quoted text from the PDF**, not paragraph numbers, so the benchmark still works if the PDF
  loader changes.
- "What should a good scheduler try to keep as small as possible?" shares no words with the book's "Minimize",
  so every vectorless method misses it and the embedding model finds it.
- Step 5 shows what the SLM actually gets in 512 tokens: 0% of the evidence for BM25, 100% for dense.

## 3. A book-worded question (1 min)

```powershell
.\.venv\Scripts\python experiments/demo_evaluation.py --qid HEM-007 --no-dense
```

Say: "When the question uses the book's words, BM25 finds the right paragraph first, in under a millisecond."

## 4. Mam's own question (1 min, optional)

```powershell
.\.venv\Scripts\python experiments/demo_evaluation.py --question "type mam's question here"
```

No gold answer exists for it, so only the retrieved paragraphs are shown, not metrics.

## 5. The report (4 min)

Show `report.html`, top to bottom:
1. **The four tiles:** embeddings are +0.075 nDCG@10 better, but **not significantly** (p = 0.30), while being about
   100× slower per question, 400× slower to index and 7.5× more memory, plus a 64 MB model.
2. **Ranking quality chart:** blue = vectorless, orange = embeddings.
3. **Student vs book wording:** on book-worded questions, char n-gram TF-IDF (0.984) beats the embedding model
   (0.960); the gap is only on student wording.
4. **Significance table:** no method beats BM25 after Holm correction; with 15 dev questions only large gaps can.

## Likely questions and answers

| Question | Answer |
|---|---|
| Why nDCG as the main metric? | It uses graded relevance: essential evidence ranked high counts more than helpful evidence. |
| Why not just accuracy? | Retrieval returns a ranking; Recall/MRR/nDCG measure the ranking, and coverage measures what the SLM receives within its token budget. |
| Why a randomization test? | Both methods answer the same questions, so the test is paired; it makes no normality assumption and is recommended for IR (Smucker et al., 2007). |
| Why Holm correction? | We compare 21 methods with one baseline; without correction about one would look significant by chance. |
| Why dev and test splits? | Anything we tune or look at (parameters, glossary) uses dev; test is run once, so final numbers are not tuned to it. |
| Are the questions reliable? | They are AI-drafted and are being reviewed by a teammate; annotator names are stored per question. |
| Is the embedding model offline? | Yes. Downloaded once (64 MB), then loaded with the Hugging Face hub set to offline. |
| What is left? | Human review of questions, the final test-split run, and evaluating Srika's EMTE with the same harness. |
