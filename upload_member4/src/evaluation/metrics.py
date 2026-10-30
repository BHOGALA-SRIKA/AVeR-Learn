"""Ranking metrics for one query and their means (owner: Member 4, evaluation).

Inputs for one query:

- ``ranked``: paragraph IDs, best first (a method's output).
- ``qrels``:  paragraph ID -> relevance grade. 2 = essential evidence,
  1 = helpful, 0 or missing = not relevant (TREC graded judgements).

A paragraph counts as relevant for Recall, Precision and MRR when its grade
is 1 or more. nDCG uses the grades themselves.

Formulas (Manning, Raghavan & Schuetze 2008, ch. 8; Jarvelin & Kekalainen 2002):

    Recall@k    = |relevant in top k| / |relevant|
    Precision@k = |relevant in top k| / k
    RR          = 1 / rank of the first relevant paragraph (0 if none in the cut-off)
    DCG@k       = sum over ranks i <= k of (2^grade_i - 1) / log2(i + 1)
    nDCG@k      = DCG@k / DCG@k of the ideal ranking

Queries with no relevant paragraph (out-of-scope questions) have no defined
Recall or nDCG. ``evaluate_query`` returns None for them; the harness reports
them separately instead of counting them as 0 or 1.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _relevant(qrels: Mapping[str, int]) -> set[str]:
    return {pid for pid, grade in qrels.items() if grade >= 1}


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = _relevant(qrels)
    if not relevant:
        raise ValueError("Recall is undefined for a query with no relevant paragraph.")
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")
    relevant = _relevant(qrels)
    return sum(pid in relevant for pid in ranked[:k]) / k


def reciprocal_rank(ranked: Sequence[str], qrels: Mapping[str, int], k: int | None = None) -> float:
    relevant = _relevant(qrels)
    for rank, pid in enumerate(ranked[:k] if k else ranked, start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(grades: Sequence[int], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(grades[:k], start=1))


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    ideal = dcg_at_k(sorted(qrels.values(), reverse=True), k)
    if ideal == 0:
        raise ValueError("nDCG is undefined for a query with no relevant paragraph.")
    return dcg_at_k([qrels.get(pid, 0) for pid in ranked], k) / ideal


def evaluate_query(
    ranked: Sequence[str], qrels: Mapping[str, int], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float] | None:
    """All metrics for one query, e.g. {"R@3": 1.0, "P@3": 0.33, "MRR@10": 0.5, "nDCG@10": 0.7}.

    MRR is cut at the largest k, so a method is not rewarded for a hit at rank 50.
    Returns None when the query has no relevant paragraph.
    """
    if not _relevant(qrels):
        return None
    out: dict[str, float] = {}
    for k in ks:
        out[f"R@{k}"] = recall_at_k(ranked, qrels, k)
        out[f"P@{k}"] = precision_at_k(ranked, qrels, k)
        out[f"nDCG@{k}"] = ndcg_at_k(ranked, qrels, k)
    out[f"MRR@{max(ks)}"] = reciprocal_rank(ranked, qrels, max(ks))
    return out


def mean_metrics(per_query: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Macro average: every query counts equally."""
    if not per_query:
        return {}
    return {name: sum(q[name] for q in per_query) / len(per_query) for name in per_query[0]}
