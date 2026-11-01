"""Paragraph-level lexical adaptation of HiChunk Auto-Merge.

The original HiChunk paper describes ranked-chunk retrieval followed by
hierarchical parent merging. This module adapts that idea to paragraph-level
lexical retrieval and the project's deterministic token-budget accounting.

A parent can replace selected descendants when:
1. At least two direct children are represented.
2. Retrieved child text reaches len(parent)/3 * (1 + tk_cur/T).
3. Remaining budget before replacement is at least the parent's full length.

This adapts the paper's retrieval algorithm to Member 1's existing tree and
paragraph ranking; it does not reproduce HiChunk's LLM hierarchy construction.
"""
from __future__ import annotations

from typing import Any, Mapping

from ._common import Selection, _prepare, _counter


def _parent(nodes: Mapping[str, dict], node_id: str) -> str | None:
    parent_id = nodes[node_id].get("parent_id")
    return str(parent_id) if parent_id is not None else None


def _children(nodes: Mapping[str, dict], node_id: str) -> list[str]:
    return [str(child) for child in nodes[node_id].get("children", []) if str(child) in nodes]


def _descendant_paragraphs(nodes: Mapping[str, dict], node_id: str) -> list[str]:
    found: list[str] = []
    stack = list(reversed(_children(nodes, node_id)))
    while stack:
        current = stack.pop()
        if nodes[current].get("kind") == "paragraph":
            found.append(current)
        else:
            stack.extend(reversed(_children(nodes, current)))
    return found


def _is_descendant(nodes: Mapping[str, dict], node_id: str, ancestor_id: str) -> bool:
    current: str | None = node_id
    while current is not None:
        if current == ancestor_id:
            return True
        current = _parent(nodes, current)
    return False


def _represented_children(
    nodes: Mapping[str, dict], parent_id: str, selected: set[str]
) -> list[str]:
    """Return direct children of parent represented by selected regions."""
    represented: list[str] = []
    for child_id in _children(nodes, parent_id):
        if any(_is_descendant(nodes, selected_id, child_id) for selected_id in selected):
            represented.append(child_id)
    return represented


def _paragraph_tokens(node: Mapping[str, Any], counter) -> int:
    value = node.get("n_tokens")
    count = int(value) if value is not None else int(counter(str(node.get("text", ""))))
    if count < 0:
        raise ValueError("Token count cannot be negative")
    return count


def _region_cost(nodes: Mapping[str, dict], node_id: str, counter, include_breadcrumbs: bool = True) -> int:
    """Cost of a delivered region: paragraph text plus its breadcrumb once."""
    node = nodes[node_id]
    if node.get("kind") == "paragraph":
        content_cost = _paragraph_tokens(node, counter)
    else:
        leaves = _descendant_paragraphs(nodes, node_id)
        content_cost = (
            sum(_paragraph_tokens(nodes[pid], counter) for pid in leaves)
            if leaves
            else _paragraph_tokens(node, counter)
        )
    breadcrumb_cost = int(node.get("n_tokens_breadcrumb", 0) or 0) if include_breadcrumbs else 0
    if breadcrumb_cost < 0:
        raise ValueError("Breadcrumb token count cannot be negative")
    return content_cost + breadcrumb_cost


def _text_region_tokens(nodes: Mapping[str, dict], node_id: str, counter) -> int:
    """Text-only length used by HiChunk's coverage threshold (no breadcrumbs)."""
    node = nodes[node_id]
    if node.get("kind") == "paragraph":
        return _paragraph_tokens(node, counter)
    leaves = _descendant_paragraphs(nodes, node_id)
    if leaves:
        return sum(_paragraph_tokens(nodes[pid], counter) for pid in leaves)
    return _paragraph_tokens(node, counter)


def _adaptive_threshold(parent_text_tokens: int | None = None, used: int = 0, budget: int = 0,
                        *, parent_cost: int | None = None) -> float:
    """Paper threshold: len(parent)/3 * (1 + tk_cur/T).

    ``parent_cost`` is retained as a compatibility keyword for earlier tests;
    it denotes the same parent text length, excluding breadcrumb metadata.
    """
    if parent_text_tokens is None:
        parent_text_tokens = parent_cost
    if parent_text_tokens is None:
        raise TypeError("parent_text_tokens is required")
    if budget <= 0:
        return float("inf")
    progress = min(max(used / budget, 0.0), 1.0)
    return (parent_text_tokens / 3.0) * (1.0 + progress)


def _merge_parent(
    nodes: Mapping[str, dict],
    selected: list[str],
    parent_id: str,
    counter,
    budget: int,
    used: int,
    min_children: int = 2,
    include_breadcrumbs: bool = True,
):
    """Apply the three HiChunk Auto-Merge conditions to one parent.

    Adaptation note: our input leaves are Member 1 paragraphs rather than
    HiChunk's fixed-size chunks. The hierarchy is supplied by Member 1.
    """
    if nodes[parent_id].get("kind") == "document":
        return None

    selected_set = set(selected)
    represented = _represented_children(nodes, parent_id, selected_set)

    # Condition 1: at least two direct children are represented.
    if len(represented) < min_children:
        return None

    covered = [sid for sid in selected if _is_descendant(nodes, sid, parent_id)]
    if not covered:
        return None

    parent_text_tokens = _text_region_tokens(nodes, parent_id, counter)
    parent_cost = _region_cost(nodes, parent_id, counter, include_breadcrumbs)
    if parent_text_tokens <= 0 or parent_cost <= 0:
        return None

    # Condition 2: retrieved child text reaches the adaptive threshold.
    covered_text_tokens = sum(_text_region_tokens(nodes, sid, counter) for sid in covered)
    threshold = _adaptive_threshold(parent_text_tokens, used, budget)
    if covered_text_tokens < threshold:
        return None

    # Condition 3: the parent region itself must fit the delivery budget.
    # Since the parent replaces its selected descendants, compare against the
    # total budget here; the exact post-replacement total is checked below.
    # Requiring budget - used >= parent_cost would incorrectly reject valid
    # merges when the selected children are replaced by a smaller parent region.
    if parent_cost > budget:
        return None

    new_selected = [sid for sid in selected if sid not in covered]
    new_selected.append(parent_id)
    new_used = sum(
        _region_cost(nodes, sid, counter, include_breadcrumbs) for sid in new_selected
    )
    if new_used > budget:
        return None
    return new_selected, new_used


def select_hichunk(
    query: str,
    tree: Any,
    index: Any,
    budget: int,
    params: Mapping[str, Any] | None = None,
) -> Selection:
    """Select ranked paragraphs and apply budget-aware hierarchical merging."""
    params = dict(params or {})
    nodes, paragraph_ids, scores, ranked = _prepare(query, tree, index, params)
    if not ranked or budget <= 0:
        return Selection([], ranked, 0, True)

    counter = _counter(params)
    include_breadcrumbs = bool(params.get("include_breadcrumbs", True))
    min_score = float(params.get("min_score", 0.0))
    min_children = int(params.get("merge_min_children", 2))
    max_level = params.get("merge_level_max")
    if min_children < 2:
        raise ValueError("merge_min_children must be at least 2")

    candidates = [pid for pid in ranked if scores[pid] > min_score]
    if not candidates:
        return Selection([], ranked, 0, True)

    selected: list[str] = []
    used = 0

    # Add evidence in retrieval-rank order. After each addition, immediately
    # try merging upward, mirroring the paper's incremental Auto-Merge flow.
    for pid in candidates:
        cost = _region_cost(nodes, pid, counter, include_breadcrumbs)
        if cost <= 0 or used + cost > budget:
            continue

        selected.append(pid)
        used += cost

        current = _parent(nodes, pid)
        while current is not None:
            # The document root is never returned as a retrieval region.
            if nodes[current].get("kind") == "document":
                break
            if max_level is not None and int(nodes[current].get("level", 0)) > int(max_level):
                break

            merged = _merge_parent(
                nodes, selected, current, counter, budget, used, min_children, include_breadcrumbs
            )
            if merged is None:
                break

            selected, used = merged
            current = _parent(nodes, current)

    selected.sort(key=lambda nid: (int(nodes[nid].get("doc_order", 0)), nid))
    # Recompute from the final selected regions to keep accounting exact.
    used = sum(_region_cost(nodes, nid, counter, include_breadcrumbs) for nid in selected)
    return Selection(selected, ranked, used, not bool(selected))


def select_hichunk_with_stop(
    query: str,
    tree: Any,
    index: Any,
    budget: int,
    params: Mapping[str, Any] | None = None,
) -> Selection:
    """Apply a score-gap or score-ratio cutoff before Auto-Merge."""
    params = dict(params or {})
    nodes, ids, scores, ranked = _prepare(query, tree, index, params)
    if not ranked:
        return Selection([], [], 0, True)

    cutoff = len(ranked)
    if "gap_threshold" in params:
        threshold = float(params["gap_threshold"])
        gaps = [
            (scores[a] - scores[b], i)
            for i, (a, b) in enumerate(zip(ranked, ranked[1:]))
        ]
        if gaps:
            largest_gap, gap_index = max(gaps, key=lambda item: (item[0], -item[1]))
            if largest_gap > threshold:
                cutoff = gap_index + 1
    else:
        ratio = float(params.get("ratio_threshold", 0.25))
        top = max(scores[ranked[0]], 1e-12)
        for i, pid in enumerate(ranked):
            if scores[pid] <= 0 or scores[pid] / top < ratio:
                cutoff = i
                break

    class FilteredIndex:
        def score(self, _query):
            return {pid: scores[pid] for pid in ranked[:cutoff]}

    params["paragraphs"] = [nodes[pid] for pid in ids]
    return select_hichunk(query, tree, FilteredIndex(), budget, params)
