# AVeR-Learn Experiments

This directory contains the experimental records for AVeR-Learn, an offline learning-material retrieval project.

## Purpose

The experiments document:

- Retrieval methods implemented and evaluated.
- Dataset characteristics and test queries.
- Observed retrieval results.
- Limitations and failure cases.
- Planned improvements and evaluation metrics.

## Experiment Log

| Experiment | Description | Status |
|---|---|---|
| Experiment 01 | Baseline BM25 retrieval | Completed |
| Experiment 02 | Heading-aware ranking and token normalization | Preliminary evaluation completed |
| Experiment 03 | Controlled retrieval benchmark and metric evaluation | Planned |

## Evaluation Principles

Experiments should use documented test queries and manually verified relevant paragraphs. Improvements should be supported by repeatable results rather than a few selected examples.

Future evaluations will investigate Recall@K and Mean Reciprocal Rank (MRR), alongside retrieval latency where appropriate.