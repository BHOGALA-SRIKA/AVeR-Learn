"""Paragraph-level retrieval baselines for AVeR-Learn Member 3."""
from __future__ import annotations
from typing import Any, Mapping
from ._common import Selection, _prepare, _counter, _fit_paragraphs


def select_fixed_k(query: str, tree: Any, index: Any, budget: int,
                   params: Mapping[str, Any] | None = None) -> Selection:
    params = params or {}
    nodes, ids, scores, ranked = _prepare(query, tree, index, params)
    if not ranked:
        return Selection([], [], 0, True)
    k = int(params.get("k", params.get("top_k", 5)))
    if k < 1:
        raise ValueError("k must be >= 1")
    include_breadcrumbs = bool(params.get("include_breadcrumbs", True))
    return _fit_paragraphs(ranked, nodes, budget, _counter(params), k, include_breadcrumbs)


def select_score_ratio(query: str, tree: Any, index: Any, budget: int,
                       params: Mapping[str, Any] | None = None) -> Selection:
    """Keep the leading prefix with score_i / best_score >= ratio_threshold."""
    params = params or {}
    nodes, ids, scores, ranked = _prepare(query, tree, index, params)
    if not ranked:
        return Selection([], [], 0, True)
    ratio = float(params.get("ratio_threshold", 0.5))
    if not 0 <= ratio <= 1:
        raise ValueError("ratio_threshold must be in [0, 1]")
    best = scores[ranked[0]]
    if best <= 0:
        return Selection([], ranked, 0, True)
    prefix = []
    for pid in ranked:
        if scores[pid] <= 0 or scores[pid] / best < ratio:
            break
        prefix.append(pid)
    include_breadcrumbs = bool(params.get("include_breadcrumbs", True))
    return _fit_paragraphs(prefix, nodes, budget, _counter(params), include_breadcrumbs=include_breadcrumbs)


def select_score_gap(query: str, tree: Any, index: Any, budget: int,
                     params: Mapping[str, Any] | None = None) -> Selection:
    """Keep the prefix ending immediately before the largest adjacent score gap."""
    params = params or {}
    nodes, ids, scores, ranked = _prepare(query, tree, index, params)
    if not ranked:
        return Selection([], [], 0, True)
    threshold = float(params.get("gap_threshold", 0.5))
    if threshold < 0:
        raise ValueError("gap_threshold must be non-negative")
    if len(ranked) == 1:
        prefix = ranked
    else:
        gaps = [scores[left] - scores[right] for left, right in zip(ranked, ranked[1:])]
        largest_index = max(range(len(gaps)), key=lambda i: gaps[i])
        prefix = ranked[:largest_index + 1] if gaps[largest_index] > threshold else ranked
    include_breadcrumbs = bool(params.get("include_breadcrumbs", True))
    return _fit_paragraphs(prefix, nodes, budget, _counter(params), include_breadcrumbs=include_breadcrumbs)


def select_adaptive_k(query: str, tree: Any, index: Any, budget: int,
                      params: Mapping[str, Any] | None = None) -> Selection:
    """Paper adaptation: find the largest adjacent gap in the top search_frac."""
    params = params or {}
    nodes, ids, scores, ranked = _prepare(query, tree, index, params)
    if not ranked:
        return Selection([], [], 0, True)

    # Legacy interface retained for existing project callers/tests.
    if any(key in params for key in ("min_k", "max_k", "relative_gap_threshold")):
        min_k = int(params.get("min_k", 1))
        max_k = min(int(params.get("max_k", 8)), len(ranked))
        if min_k < 1 or max_k < min_k:
            raise ValueError("Require 1 <= min_k <= max_k")
        relative_gap = float(params.get("relative_gap_threshold", 0.25))
        absolute_gap = params.get("gap_threshold")
        top = max(abs(scores[ranked[0]]), 1e-12)
        chosen_k = max_k
        for i in range(min(max_k - 1, len(ranked) - 1)):
            gap = scores[ranked[i]] - scores[ranked[i + 1]]
            significant = gap > float(absolute_gap) if absolute_gap is not None else gap / top > relative_gap
            if significant:
                chosen_k = max(min_k, i + 1)
                break
        include_breadcrumbs = bool(params.get("include_breadcrumbs", True))
        return _fit_paragraphs(ranked[:chosen_k], nodes, budget, _counter(params), chosen_k, include_breadcrumbs)

    search_frac = float(params.get("search_frac", 0.90))
    buffer = int(params.get("buffer", 5))
    if not 0 < search_frac <= 1:
        raise ValueError("search_frac must be in (0, 1]")
    if buffer < 0:
        raise ValueError("buffer must be non-negative")

    n = len(ranked)
    limit = max(2, int(n * search_frac))
    limit = min(n, limit)
    if limit < 2:
        chosen_k = n
    else:
        gaps = [
            scores[ranked[i]] - scores[ranked[i + 1]]
            for i in range(limit - 1)
        ]
        largest_gap_index = max(range(len(gaps)), key=lambda i: gaps[i])
        chosen_k = min(n, largest_gap_index + 1 + buffer)

    include_breadcrumbs = bool(params.get("include_breadcrumbs", True))
    return _fit_paragraphs(ranked[:chosen_k], nodes, budget, _counter(params), chosen_k, include_breadcrumbs)


def select(query: str, tree: Any, index: Any, budget: int,
           params: Mapping[str, Any] | None = None) -> Selection:
    params = params or {}
    method = str(params.get("method", "fixed_k")).lower()
    registry = {
        "fixed_k": select_fixed_k,
        "score_ratio": select_score_ratio,
        "score_gap": select_score_gap,
        "adaptive_k": select_adaptive_k,
    }
    if method not in registry:
        raise ValueError(f"Unknown selector method: {method}")
    return registry[method](query, tree, index, budget, params)
