# Demo guide: evaluation part (Member 4)

About 10 minutes. Open PowerShell in the project folder first:

```powershell
cd D:\AVeR-Learn-feature-member2-indexing
```

## 0. Before mam arrives (once)

```powershell
.\.venv\Scripts\python -m pytest -q
```

Then open `results\unit1_all_methods\test\report.html` (final results) in the browser and keep it open.
Do **not** rerun the test split for the demo: it was run once, on 1 November 2026, and that run is the result.
To show the harness running live, run the dev split instead (about 3 minutes, it measures speed and memory):
`.\.venv\Scripts\python -m src.evaluation.evaluate_retrieval experiments/configs/unit1_all_methods.yaml --split dev`

## 1. Tests (30 s)

```powershell
.\.venv\Scripts\python -m pytest -q
```

Say: "All tests pass, including 19 for my evaluation code. The metric tests use values worked out by hand, so they
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

Say: "When the question uses the book's words, BM25F finds the right section first, in under a millisecond."

## 4. Mam's own question (1 min, optional)

```powershell
.\.venv\Scripts\python experiments/demo_evaluation.py --question "type mam's question here"
```

No gold answer exists for it, so only the retrieved paragraphs are shown, not metrics.

## 5. The report (4 min)

Show the test `report.html` (30 unseen questions), top to bottom:
1. **The four tiles:** on unseen questions the embedding model and BM25F score **exactly the same** (0.766
   nDCG@10, p = 0.996), while the embedding model is tens of times slower per question, hundreds of times slower
   to index, uses about 12× more memory and needs a 64 MB model. (On the 15 dev questions it was +0.084, not
   significant; that lead did not hold up.)
2. **Ranking quality chart:** blue = vectorless, orange = embeddings. Methods that use section headings (BM25F,
   BM25 with headings, RRF) are well above those that ignore them: headings from Member 1's tree matter.
3. **Student vs book wording:** every method is much weaker on student wording (BM25F 0.528, dense 0.458,
   hybrid 0.628) than on book wording (0.87–0.95).
4. **Significance table:** no method beats BM25F after Holm correction.
5. **EMTE (Member 3):** open `results\member3_selectors\test\summary.md`. EMTE is significantly better than 8 of
   the 9 other selectors and best at a 256-token budget (0.895), and ties the simple section pipeline at 512.

## Likely questions and answers

| Question | Answer |
|---|---|
| Why nDCG as the main metric? | It uses graded relevance: essential evidence ranked high counts more than helpful evidence. |
| Why not just accuracy? | Retrieval returns a ranking; Recall/MRR/nDCG measure the ranking, and coverage measures what the SLM receives within its token budget. |
| Why a randomization test? | Both methods answer the same questions, so the test is paired; it makes no normality assumption and is recommended for IR (Smucker et al., 2007). |
| Why Holm correction? | We compare 21 methods with one baseline; without correction about one would look significant by chance. |
| Why dev and test splits? | Anything we tune or look at (parameters, glossary) uses dev; test is run once, so final numbers are not tuned to it. |
| Are the questions reliable? | They were drafted with AI assistance, every quote was checked automatically against the PDF, and every question was reviewed by a teammate (Bhavya, who built none of the compared methods); reviewer names are stored per question. |
| Is the embedding model offline? | Yes. Downloaded once (64 MB), then loaded with the Hugging Face hub set to offline. |
| Why is the test split not rerun? | It is run once, after everything is frozen on dev, so the final numbers are not tuned to it. |
| Can it say "not in this PDF"? | Not reliably: on dev no score threshold separated out-of-scope questions from student-worded real ones, so the threshold stays off. This is a measured limitation of lexical retrieval. |
| What is next? | More units and longer PDFs (where EMTE's budgeted selection should matter more) and generating MCQs from the delivered evidence. |
