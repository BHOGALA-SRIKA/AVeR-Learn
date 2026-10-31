"""Evidence-Mass Tree Expansion (EMTE).

Paragraph-level lexical implementation of the AVeR-Learn Phase 1 procedure.
Internal-region cost is descendant paragraph text plus one region breadcrumb.
Selected regions are non-overlapping and always budget checked.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

from ._common import Selection, _prepare, _counter


def _descendants(nodes: Mapping[str, dict], node_id: str) -> list[str]:
    if node_id not in nodes:
        raise KeyError(node_id)
    found: list[str] = []
    stack = list(reversed(nodes[node_id].get("children", [])))
    while stack:
        child = str(stack.pop())
        if child not in nodes:
            raise ValueError(f"Missing child node {child}")
        node = nodes[child]
        if node.get("kind") == "paragraph":
            found.append(child)
        else:
            stack.extend(reversed(node.get("children", [])))
    found.sort(key=lambda pid: (int(nodes[pid].get("doc_order", 0)), pid))
    return found


def _ancestors(nodes: Mapping[str, dict], node_id: str) -> list[str]:
    result: list[str] = []
    current = nodes[node_id].get("parent_id")
    seen: set[str] = set()
    while current is not None:
        current = str(current)
        if current in seen:
            raise ValueError("Cycle detected in parent links")
        seen.add(current)
        if current not in nodes:
            raise ValueError(f"Missing parent node {current}")
        result.append(current)
        current = nodes[current].get("parent_id")
    return result


def _paragraph_tokens(node: Mapping[str, Any], counter) -> int:
    value = node.get("n_tokens")
    count = int(value) if value is not None else int(counter(str(node.get("text", ""))))
    if count < 0:
        raise ValueError("Token count cannot be negative")
    return count


def _region_cost(nodes: Mapping[str, dict], node_id: str, counter, include_breadcrumbs: bool = True) -> int:
    """Cost of delivered region: paragraph text plus one region breadcrumb."""
    node = nodes[node_id]
    if node.get("kind") == "paragraph":
        content_cost = _paragraph_tokens(node, counter)
    else:
        leaves = _descendants(nodes, node_id)
        content_cost = (
            sum(_paragraph_tokens(nodes[pid], counter) for pid in leaves)
            if leaves else _paragraph_tokens(node, counter)
        )
    breadcrumb_cost = int(node.get("n_tokens_breadcrumb", 0) or 0) if include_breadcrumbs else 0
    if breadcrumb_cost < 0:
        raise ValueError("Breadcrumb token count cannot be negative")
    return content_cost + breadcrumb_cost


def _aggregate(values: list[float], method: str, saturation: float) -> float:
    if not values:
        return 0.0
    if method == "sum":
        return sum(values)
    if method == "max":
        return max(values)
    if method == "mean":
        return sum(values) / len(values)
    if method == "saturating":
        total = sum(max(0.0, value) for value in values)
        return 1.0 - math.exp(-total / max(saturation, 1e-12))
    raise ValueError("aggregator must be sum, max, mean, or saturating")


def _query_terms(query: str, index: Any, params: Mapping[str, Any]) -> list[str]:
    supplied = params.get("query_tokens")
    if supplied is not None:
        return list(dict.fromkeys(str(t) for t in supplied))
    tokenizer = getattr(index, "tokenize", None)
    if callable(tokenizer):
        return list(dict.fromkeys(str(t) for t in tokenizer(query)))
    tokenizer = getattr(index, "tokenize_query", None)
    if callable(tokenizer):
        return list(dict.fromkeys(str(t) for t in tokenizer(query)))
    # Fallback is intentionally simple; for BM25F absolute scaling, provide the
    # shared Member 2 tokenizer through index.tokenize or params.query_tokens.
    return list(dict.fromkeys(re.findall(r"\w+", query.lower())))


def _lookup_idf(mapping: Any, term: str, depth: int | None = None) -> float:
    if mapping is None:
        return 0.0
    if isinstance(mapping, (list, tuple)):
        if depth is None or depth >= len(mapping):
            return 0.0
        mapping = mapping[depth]
    if isinstance(mapping, Mapping):
        return float(mapping.get(term, 0.0))
    return 0.0


def _absolute_upper_bound(query: str, index: Any, params: Mapping[str, Any]) -> float:
    """Compute U(q) from the BM25F term-field IDF components in the spec.

    An explicit ``params['upper_bound']`` or index method can be used by tests
    and alternative index adapters. Otherwise the index must expose:
    ``idf_body``, ``idf_head`` (depth -> term -> IDF), ``k1``, ``w_h`` and
    ``gamma``. ``vocab`` is optional when the IDF maps already define terms.
    """
    explicit = params.get("upper_bound")
    if callable(explicit):
        return float(explicit(query))
    if explicit is not None:
        return float(explicit)

    for method_name in ("score_upper_bound", "absolute_score_upper_bound", "upper_bound"):
        method = getattr(index, method_name, None)
        if callable(method):
            return float(method(query))

    idf_body = getattr(index, "idf_body", None)
    idf_head = getattr(index, "idf_head", None)
    k1 = getattr(index, "k1", None)
    w_h = getattr(index, "w_h", None)
    gamma = getattr(index, "gamma", None)
    if any(value is None for value in (idf_body, idf_head, k1, w_h, gamma)):
        raise ValueError(
            "absolute score_scale requires index.score_upper_bound(query), "
            "params['upper_bound'], or index BM25F fields "
            "(idf_body, idf_head, k1, w_h, gamma)"
        )

    vocab = getattr(index, "vocab", None)
    terms = _query_terms(query, index, params)
    if vocab is not None:
        terms = [term for term in terms if term in vocab]
    total = 0.0
    for term in terms:
        total += _lookup_idf(idf_body, term)
        # Header depths are indexed from 0; the specification uses gamma**d.
        if isinstance(idf_head, Mapping):
            head_depths = sorted(idf_head, key=lambda value: int(value))
            for depth_key in head_depths:
                depth = int(depth_key)
                total += float(w_h) * (float(gamma) ** depth) * _lookup_idf(
                    idf_head[depth_key], term
                )
        else:
            for depth in range(len(idf_head)):
                total += float(w_h) * (float(gamma) ** depth) * _lookup_idf(
                    idf_head, term, depth
                )
    return (float(k1) + 1.0) * total


def _scale_scores(
    query: str,
    index: Any,
    raw_scores: Mapping[str, float],
    method: str,
    params: Mapping[str, Any],
) -> dict[str, float]:
    positives = {key: max(0.0, float(value)) for key, value in raw_scores.items()}
    if method in ("relative", "max"):
        maximum = max(positives.values(), default=0.0)
        return {key: value / maximum for key, value in positives.items()} if maximum else positives
    if method == "absolute":
        upper = _absolute_upper_bound(query, index, params)
        if upper <= 0:
            return {key: 0.0 for key in positives}
        return {key: value / upper for key, value in positives.items()}
    if method == "none":
        return positives
    if method == "sum":
        total = sum(positives.values())
        return {key: value / total for key, value in positives.items()} if total else positives
    raise ValueError("score_scale must be relative, absolute, max, sum, or none")


def _propagate(
    candidates: Mapping[str, float],
    nodes: Mapping[str, dict],
    aggregator: str,
    saturation: float,
) -> dict[str, float]:
    """Recompute evidence mass for every node from the remaining paragraphs."""
    mass: dict[str, float] = {}
    for node_id in nodes:
        leaves = _descendants(nodes, node_id) if nodes[node_id].get("kind") != "paragraph" else [node_id]
        mass[node_id] = _aggregate(
            [candidates[pid] for pid in leaves if pid in candidates],
            aggregator,
            saturation,
        )
    return mass


def select_emte(
    query: str,
    tree: Any,
    index: Any,
    budget: int,
    params: Mapping[str, Any] | None = None,
) -> Selection:
    """Select evidence regions using recomputed evidence mass and budgeted ascent."""
    params = dict(params or {})
    nodes, paragraph_ids, raw_scores, ranked = _prepare(query, tree, index, params)
    if not ranked or budget <= 0:
        return Selection([], ranked, 0, True)

    counter = _counter(params)
    section_level = int(params.get("section_level", 2))
    aggregator = str(params.get("aggregator", "saturating"))
    saturation = float(params.get("saturation", params.get("lambda", 1.0)))
    score_scale = str(params.get("score_scale", "relative"))
    tau_stop = float(params.get("tau_stop", 0.0))
    tau_up = float(params.get("tau_up", 0.0))
    epsilon = float(params.get("epsilon", 1e-9))
    max_candidates = max(0, int(params.get("max_candidates", 30)))
    max_regions = params.get("max_regions")
    if max_regions is not None:
        max_regions = int(max_regions)
        if max_regions < 1:
            raise ValueError("max_regions must be >= 1 or None")
    include_breadcrumbs = bool(params.get("include_breadcrumbs", True))

    scores = _scale_scores(query, index, raw_scores, score_scale, params)
    if score_scale == "absolute" and max(scores.values(), default=0.0) < epsilon:
        return Selection([], ranked, 0, True)

    candidates = {
        pid: scores[pid]
        for pid in ranked[:max_candidates]
        if scores.get(pid, 0.0) >= epsilon
    }
    if not candidates:
        return Selection([], ranked, 0, True)

    selected: list[str] = []
    used = 0

    while candidates and (max_regions is None or len(selected) < max_regions):
        # Mass must be recomputed after every region has been removed.
        mass = _propagate(candidates, nodes, aggregator, saturation)

        # Highest remaining evidence score first; ties follow document order.
        seed_order = sorted(
            candidates,
            key=lambda pid: (-candidates[pid], int(nodes[pid].get("doc_order", 0)), pid),
        )
        seed = None
        for candidate_id in seed_order:
            token_cost = _region_cost(nodes, candidate_id, counter, include_breadcrumbs)
            if token_cost > 0 and used + token_cost <= budget:
                seed = candidate_id
                break
        if seed is None:
            # No remaining candidate can fit; all are exhausted.
            break

        seed_cost = _region_cost(nodes, seed, counter, include_breadcrumbs)
        if selected and candidates[seed] / max(_paragraph_tokens(nodes[seed], counter), 1) < tau_stop:
            # Step 13: only subsequent regions are subject to tau_stop.
            break

        region = seed
        region_cost = seed_cost
        current = seed

        # Climb only to section level; never swallow a previously selected region.
        while True:
            parent_id = nodes[current].get("parent_id")
            if parent_id is None:
                break
            parent_id = str(parent_id)
            if parent_id not in nodes:
                raise ValueError(f"Missing parent node {parent_id}")
            parent = nodes[parent_id]
            if int(parent.get("level", 99)) < section_level:
                break

            parent_leaves = set(_descendants(nodes, parent_id))
            selected_leaves: set[str] = set()
            for selected_id in selected:
                selected_leaves.update(
                    [selected_id] if nodes[selected_id].get("kind") == "paragraph"
                    else _descendants(nodes, selected_id)
                )
            if parent_leaves.intersection(selected_leaves):
                break

            parent_cost = _region_cost(nodes, parent_id, counter, include_breadcrumbs)
            extra_tokens = parent_cost - region_cost
            parent_mass = mass.get(parent_id, 0.0)
            if extra_tokens <= 0:
                delta = float("inf")
            else:
                delta = (parent_mass - mass.get(region, candidates.get(seed, 0.0))) / extra_tokens

            if used + parent_cost > budget:
                break
            if delta + epsilon < tau_up:
                break

            region = parent_id
            region_cost = parent_cost
            current = parent_id

        if region_cost <= 0 or used + region_cost > budget:
            # Defensive fallback: remove this seed and continue with the next.
            candidates.pop(seed, None)
            continue

        selected.append(region)
        used += region_cost
        covered = (
            {region} if nodes[region].get("kind") == "paragraph"
            else set(_descendants(nodes, region))
        )
        for paragraph_id in covered:
            candidates.pop(paragraph_id, None)

    selected.sort(key=lambda nid: (int(nodes[nid].get("doc_order", 0)), nid))
    used = sum(_region_cost(nodes, node_id, counter, include_breadcrumbs) for node_id in selected)
    return Selection(selected, ranked, used, not bool(selected))


def emte_ablation(
    query: str,
    tree: Any,
    index: Any,
    budget: int,
    base_params: Mapping[str, Any] | None = None,
    aggregators: tuple[str, ...] = ("sum", "max", "mean", "saturating"),
) -> dict[str, Selection]:
    """Run controlled aggregator ablations with identical remaining settings."""
    base = dict(base_params or {})
    return {
        name: select_emte(query, tree, index, budget, {**base, "aggregator": name})
        for name in aggregators
    }


def emte_ablation_suite(
    query: str,
    tree: Any,
    index: Any,
    budget: int,
    base_params: Mapping[str, Any] | None = None,
    *,
    aggregators: tuple[str, ...] = ("sum", "max", "mean", "saturating"),
    score_scales: tuple[str, ...] = ("relative", "absolute"),
    tau_stop_values: tuple[float, ...] = (0.0005, 0.001, 0.002, 0.004),
    region_limits: tuple[int | None, ...] = (1, None),
    max_regions_values: tuple[int | None, ...] | None = None,
    heading_weights: tuple[float, ...] | None = None,
) -> dict[str, Selection]:
    """Run a deterministic EMTE ablation grid.

    ``region_limits`` is the public compatibility name;
    ``max_regions_values`` is accepted as an equivalent newer spelling.
    Heading-weight ablation requires an index adapter that can rebuild itself
    with the requested heading weight.
    """
    if heading_weights is not None and not callable(getattr(index, "with_heading_weight", None)):
        raise TypeError("heading ablation requires index.with_heading_weight(weight)")

    limits = region_limits if max_regions_values is None else max_regions_values
    base = dict(base_params or {})
    results: dict[str, Selection] = {}
    index_variants: list[tuple[str, Any]] = [("heading=default", index)]
    if heading_weights is not None:
        index_variants = [(f"heading={weight:g}", index.with_heading_weight(weight)) for weight in heading_weights]

    for heading_label, variant_index in index_variants:
        for aggregator in aggregators:
            for scale in score_scales:
                for tau_stop in tau_stop_values:
                    for max_regions in limits:
                        limit_label = "all" if max_regions is None else str(max_regions)
                        label = (f"{heading_label}|agg={aggregator}|scale={scale}|"
                                 f"tau_stop={tau_stop:g}|max_regions={limit_label}")
                        settings = {**base, "aggregator": aggregator, "score_scale": scale,
                                    "tau_stop": tau_stop, "max_regions": max_regions}
                        results[label] = select_emte(query, tree, variant_index, budget, settings)
    return results
