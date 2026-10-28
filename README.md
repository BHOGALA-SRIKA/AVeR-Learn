# AVeR-Learn
### Adaptive Vectorless Retrieval, Explanation-Quality Detection and Confidence Recalibration for Resource-Constrained Offline Learning

AVeR-Learn is a research project investigating how educational learning systems can retrieve reliable, curriculum-grounded evidence efficiently under limited computational and memory resources.

The project begins by developing an offline, vectorless retrieval engine that uses lexical matching and document structure to select relevant evidence from course materials. Later stages aim to investigate adaptive evidence selection, explanation-quality assessment, confidence recalibration, and evidence-grounded assessment generation.

**Current development focus:** Phase 1 — Vectorless Retrieval Foundation.

## Problem Statement

Educational assistants need relevant and sufficient course evidence to answer students' questions accurately. However, processing too much material increases computational cost, while retrieving too little may omit important information.

Embedding-based retrieval can introduce additional model, storage, and computational requirements. AVeR-Learn investigates whether lexical and structural retrieval methods can provide adequate evidence for a bounded educational corpus while reducing these overheads.

## Research Objectives

1. Build a structured corpus from educational materials while preserving document hierarchy and source provenance.
2. Implement and evaluate vectorless lexical retrieval using BM25 and TF-IDF.
3. Investigate heading-aware and hierarchical retrieval methods.
4. Develop a reproducible benchmark containing questions and manually verified supporting passages.
5. Compare retrieval relevance, evidence coverage, latency, memory consumption, index size, and preprocessing cost.
6. Investigate adaptive evidence selection using the proposed Evidence-Mass Tree Expansion (EMTE) approach.
7. Establish a validated retrieval foundation for subsequent offline learning, explanation evaluation, and confidence-recalibration research.

## Research Hypotheses

**H1 — Efficient vectorless retrieval:** Hierarchical lexical retrieval can provide adequate curriculum-grounded evidence for a bounded educational corpus while reducing resource overhead compared with an embedding-based retrieval baseline.

**H2 — Evidence-adaptive retrieval:** Adaptive evidence selection can reduce unnecessary context while maintaining answer quality compared with suitable fixed retrieval and hierarchical-merging baselines.

These hypotheses will be evaluated experimentally; neither is assumed to be true in advance.

## Phase 1 Scope

The initial phase focuses on the following components:

- PDF ingestion and text extraction.
- Paragraph-level processing and document hierarchy.
- BM25 baseline retrieval.
- TF-IDF baseline retrieval.
- Heading-aware and hierarchical lexical retrieval.
- A manually labelled retrieval benchmark.
- Retrieval evaluation using Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and nDCG.
- Resource measurements, including retrieval latency, index size, memory use, and preprocessing cost.
- A later comparison with an embedding-based retrieval baseline.
- Development and evaluation of EMTE after establishing reliable baselines.

## Proposed Architecture

```text
Educational PDFs and Slides
           |
           v
Text Extraction and Cleaning
           |
           v
Structured Document Corpus
(chapters, sections, paragraphs)
           |
           v
Lexical Indexing
(BM25 and TF-IDF)
           |
           v
Heading-Aware and Hierarchical Retrieval
           |
           v
Evidence Selection and Context Assembly
           |
           v
Evaluation and Resource Measurement
           |
           v
Adaptive Retrieval Research (EMTE)
```

## Current Implementation Status

### Implemented or tested

- Python development environment.
- PDF text extraction using PyMuPDF.
- Paragraph extraction and processed JSON records.
- BM25-based retrieval using `rank-bm25`.
- Initial heading-aware ranking and token-normalization experiments.
- Preliminary manual tests using Operating Systems material.

The current pilot PDF has nine pages, and the latest reported processed corpus contains 28 paragraphs.

### In progress

- Improving paragraph segmentation and heading hierarchy.
- Stabilizing retrieval scoring and result reporting.
- Building a reproducible benchmark with manually verified relevant passages.
- Implementing evaluation metrics and additional retrieval baselines.

### Planned

- TF-IDF and hierarchical retrieval comparisons.
- Embedding-based comparison baseline.
- Formal evaluation of retrieval quality and resource cost.
- EMTE implementation and comparison with suitable adaptive-merging baselines.
- Subsequent phases covering offline answering, explanation quality, confidence recalibration, and evidence-grounded assessment generation.

## Evaluation Plan

The retrieval methods will be evaluated using the same corpus, query set, relevance labels, and comparable resource budgets.

| Category | Metrics |
|---|---|
| Retrieval relevance | Recall@K, Precision@K, MRR, nDCG |
| Evidence coverage | Fraction of labelled supporting passages retrieved |
| Efficiency | Query latency, preprocessing time, peak memory, index size |
| Context efficiency | Number of tokens selected and answer quality at comparable budgets |
| Robustness | Performance across question types and degraded document structure |

An embedding-based retriever will be used as a comparison baseline for the vectorless retrieval hypothesis, not as part of the vectorless retrieval path.

## Technology Stack

- Python
- PyMuPDF
- rank-bm25
- scikit-learn
- NumPy and pandas, where required
- pytest
- Git and GitHub

Additional dependencies will be introduced when their corresponding components are implemented.

## Repository Structure

```text
AVeR-Learn/
├── data/
│   ├── raw/
│   │   ├── OS/
│   │   └── DBMS/
│   ├── processed/
│   └── benchmark/
├── src/
│   ├── ingestion/
│   ├── indexing/
│   ├── retrieval/
│   └── evaluation/
├── tests/
├── experiments/
├── research/
├── requirements.txt
├── README.md
└── .gitignore
```

The structure represents the planned organization; some directories and modules may still be under development.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/BHOGALA-SRIKA/AVeR-Learn
cd AVeR-Learn
```


### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the retrieval system

Use the project's current retrieval entry point after placing the relevant PDF in the configured data directory. The exact execution command will be documented once the entry point and configuration are finalized.

## Experimental Reproducibility

Each experiment should document:

- The method and configuration tested.
- The dataset and question set.
- The relevance labels and evaluation procedure.
- The results, limitations, and observed failure cases.
- The code version and changes relative to the baseline.

Development data will be used for tuning. The held-out test set will remain separate from parameter selection.

## Research Contribution

The project does not claim BM25, TF-IDF, hierarchical retrieval, or score aggregation as individually novel techniques. The intended contribution is an experimentally validated combination of vectorless retrieval, document hierarchy, adaptive evidence selection, and resource-aware evaluation in a bounded educational setting.

The final contribution claims will depend on the measured results and comparisons with suitable baselines.

## Team

AVeR-Learn is developed collaboratively by a team of four members. Research, implementation, benchmarking, and evaluation responsibilities are being organized across the team.

## Project Status

**Phase 1: In progress.** The initial PDF ingestion and BM25 retrieval pipeline has been tested. Formal benchmark construction, comparative evaluation, and adaptive evidence-selection research remain to be completed.
