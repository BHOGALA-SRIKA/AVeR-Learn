"""Tree-based lexical baselines over Member 1's real processed hierarchy.

INEX uses one BM25 collection over all candidate elements so paragraph,
subsection and section scores are on a common scale. Granularity is selected
on the development split, never independently per query.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from src.text.tokenize import tokenize

from ._common import Selection, _node_map, _counter, _ordered_paragraph_ids


def _descendant_paragraphs(nodes: Mapping[str, dict], node_id: str) -> list[str]:
    out: list[str] = []
    stack = list(reversed([str(x) for x in nodes[node_id].get("children", [])]))
    while stack:
        current = stack.pop()
        if current not in nodes:
            continue
        node = nodes[current]
        if node.get("kind") == "paragraph":
            out.append(current)
        else:
            stack.extend(reversed([str(x) for x in node.get("children", []) if str(x) in nodes]))
    return out


def _text_tokens(nodes: Mapping[str, dict], node_id: str, counter) -> int:
    """Original-text token count, excluding breadcrumb metadata."""
    node = nodes[node_id]
    if node.get("kind") == "paragraph":
        value = node.get("n_tokens")
        return int(value) if value is not None else int(counter(str(node.get("text", ""))))
    leaves = _descendant_paragraphs(nodes, node_id)
    if leaves:
        return sum(_text_tokens(nodes, pid, counter) for pid in leaves)
    return int(counter(str(node.get("text", ""))))


def _region_cost(nodes: Mapping[str, dict], node_id: str, counter, include_breadcrumbs=True) -> int:
    content = _text_tokens(nodes, node_id, counter)
    crumb = int(nodes[node_id].get("n_tokens_breadcrumb") or 0) if include_breadcrumbs else 0
    if content < 0 or crumb < 0:
        raise ValueError("Token counts cannot be negative")
    return content + crumb


def _rank_nodes(index, query: str, node_ids: list[str]) -> tuple[list[str], dict[str, float]]:
    if not node_ids:
        return [], {}
    scores = {str(k): float(v) for k, v in index.score_nodes(query, node_ids).items()}
    order = {nid: i for i, nid in enumerate(node_ids)}
    ranked = sorted(node_ids, key=lambda nid: (-scores[nid], order[nid]))
    return ranked, scores


def _fit_regions(ranked: list[str], nodes: Mapping[str, dict], budget: int, counter,
                 include_breadcrumbs=True, max_items: int | None = None) -> Selection:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    chosen: list[str] = []
    used = 0
    for nid in ranked:
        if max_items is not None and len(chosen) >= max_items:
            break
        if any(_is_ancestor(nodes, nid, old) or _is_ancestor(nodes, old, nid) for old in chosen):
            continue
        cost = _region_cost(nodes, nid, counter, include_breadcrumbs)
        if cost > 0 and used + cost <= budget:
            chosen.append(nid)
            used += cost
    return Selection(chosen, ranked, used, not bool(chosen))


def _is_ancestor(nodes: Mapping[str, dict], ancestor: str, node_id: str) -> bool:
    current = nodes.get(node_id, {}).get("parent_id")
    while current is not None:
        current = str(current)
        if current == ancestor:
            return True
        current = nodes.get(current, {}).get("parent_id")
    return False


def _element_text(nodes: Mapping[str, dict], node_id: str) -> str:
    node = nodes[node_id]
    if node.get("kind") == "paragraph":
        return str(node.get("text", ""))
    title = str(node.get("title") or "")
    body = " ".join(str(nodes[pid].get("text", "")) for pid in _descendant_paragraphs(nodes, node_id))
    return (title + " " + body).strip()


def _inex_bm25(query: str, nodes: Mapping[str, dict], candidate_ids: list[str],
               b: float, k1: float) -> dict[str, float]:
    """BM25 over one shared candidate-element collection (cross-level comparable)."""
    if not 0.0 <= b <= 1.0:
        raise ValueError("inex_b must be between 0 and 1")
    if k1 <= 0:
        raise ValueError("inex_k1 must be positive")
    q_terms = tokenize(query)
    docs = {nid: tokenize(_element_text(nodes, nid)) for nid in candidate_ids}
    n_docs = len(docs)
    if not n_docs or not q_terms:
        return {nid: 0.0 for nid in candidate_ids}
    lengths = {nid: len(tokens) for nid, tokens in docs.items()}
    avg_len = sum(lengths.values()) / n_docs or 1.0
    dfs: Counter[str] = Counter()
    tfs: dict[str, Counter[str]] = {}
    for nid, terms in docs.items():
        tf = Counter(terms)
        tfs[nid] = tf
        dfs.update(tf.keys())
    # Robertson/Sparck Jones IDF, non-negative and consistent across levels.
    idf = {
        term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for term, df in dfs.items()
    }
    scores: dict[str, float] = {}
    for nid in candidate_ids:
        tf = tfs[nid]
        dl = lengths[nid]
        norm = k1 * (1.0 - b + b * dl / avg_len)
        score = 0.0
        for term in q_terms:
            freq = tf.get(term, 0)
            if freq:
                score += idf.get(term, 0.0) * (freq * (k1 + 1.0)) / (freq + norm)
        scores[nid] = score
    return scores


def select_section_then_paragraph(query: str, tree: Any, index: Any, budget: int,
                                  params: Mapping[str, Any] | None = None) -> Selection:
    """Rank sections, take top S, then rank/fill their paragraphs."""
    p = dict(params or {})
    nodes = _node_map(tree, p)
    sections = [nid for nid, n in nodes.items() if n.get("kind") == "section" or int(n.get("level", -1)) == 2]
    ranked_sections, _ = _rank_nodes(index, query, sections)
    if not ranked_sections or budget <= 0:
        return Selection([], ranked_sections, 0, True)
    top_s = int(p.get("top_sections", p.get("S", 3)))
    if top_s < 1:
        raise ValueError("top_sections must be >= 1")
    eligible: list[str] = []
    for sid in ranked_sections[:top_s]:
        eligible.extend(_descendant_paragraphs(nodes, sid))
    eligible = list(dict.fromkeys(eligible))
    scores = index.score(query)
    ranked = sorted(eligible, key=lambda pid: (-float(scores.get(pid, 0.0)), int(nodes[pid].get("doc_order", 0)), pid))
    result = _fit_regions(ranked, nodes, budget, _counter(p), bool(p.get("include_breadcrumbs", True)))
    result.ranked = ranked
    return result


def select_best_granularity(query: str, tree: Any, index: Any, budget: int,
                            params: Mapping[str, Any] | None = None) -> Selection:
    """Run one fixed granularity; select its value on dev frontier, not per query."""
    p = dict(params or {})
    granularity = str(p.get("granularity", "paragraph")).lower()
    if granularity not in {"paragraph", "subsection", "section"}:
        raise ValueError("granularity must be paragraph, subsection, or section")
    nodes = _node_map(tree, p)
    candidates = [nid for nid, n in nodes.items() if n.get("kind") == granularity]
    ranked, _ = _rank_nodes(index, query, candidates)
    return _fit_regions(ranked, nodes, budget, _counter(p), bool(p.get("include_breadcrumbs", True)))


def select_plain_automerge(query: str, tree: Any, index: Any, budget: int,
                           params: Mapping[str, Any] | None = None) -> Selection:
    """Team's LlamaIndex-style baseline (not the HiChunk paper algorithm)."""
    p = dict(params or {})
    nodes = _node_map(tree, p)
    para_ids = _ordered_paragraph_ids(nodes)
    scores = index.score(query)
    ranked = sorted(para_ids, key=lambda pid: (-float(scores.get(pid, 0.0)), int(nodes[pid].get("doc_order", 0)), pid))
    top_n = int(p.get("top_n", p.get("N", 8)))
    threshold = float(p.get("merge_fraction", p.get("r", 0.4)))
    if top_n < 1 or not 0.0 < threshold <= 1.0:
        raise ValueError("top_n must be >=1 and merge_fraction must be in (0,1]")
    max_level = int(p.get("max_merge_level", 2))
    selected = ranked[:top_n]
    parents = sorted(
        (nid for nid, n in nodes.items() if n.get("kind") in {"subsection", "section"}),
        key=lambda nid: int(nodes[nid].get("level", 0)), reverse=True,
    )
    for parent in parents:
        if int(nodes[parent].get("level", 99)) > max_level:
            continue
        children = [str(c) for c in nodes[parent].get("children", []) if str(c) in nodes]
        if not children:
            continue
        represented = [
            child for child in children
            if any(x == child or _is_ancestor(nodes, child, x) for x in selected)
        ]
        if len(represented) / len(children) < threshold:
            continue
        covered = [x for x in selected if _is_ancestor(nodes, parent, x)]
        if not covered:
            continue
        # Replace descendants only when the merged region itself fits and the
        # complete non-overlapping selection remains within the delivery budget.
        proposed = [x for x in selected if x not in covered] + [parent]
        proposed_cost = 0
        valid = True
        for candidate in proposed:
            if any(_is_ancestor(nodes, candidate, old) or _is_ancestor(nodes, old, candidate)
                   for old in proposed if old != candidate):
                # A proposed ancestor/descendant overlap is not a valid merge.
                valid = False
                break
            proposed_cost += _region_cost(
                nodes, candidate, _counter(p), bool(p.get("include_breadcrumbs", True))
            )
        if valid and proposed_cost <= budget:
            selected = proposed
    best_score = {}
    for nid in selected:
        leaves = [nid] if nodes[nid].get("kind") == "paragraph" else _descendant_paragraphs(nodes, nid)
        best_score[nid] = max((float(scores.get(pid, 0.0)) for pid in leaves), default=0.0)
    ordered = sorted(selected, key=lambda nid: (-best_score[nid], int(nodes[nid].get("doc_order", 0)), nid))
    return _fit_regions(ordered, nodes, budget, _counter(p), bool(p.get("include_breadcrumbs", True)))


def select_inex(query: str, tree: Any, index: Any, budget: int,
                params: Mapping[str, Any] | None = None) -> Selection:
    """INEX-style element selection with globally comparable node BM25 scores.

    `inex_b` is the BM25 length-normalisation parameter tuned on dev; it is
    applied to one collection containing paragraphs, subsections and sections.
    """
    p = dict(params or {})
    nodes = _node_map(tree, p)
    candidates = [nid for nid, n in nodes.items() if n.get("kind") in {"paragraph", "subsection", "section"}]
    b = float(p.get("inex_b", p.get("b", 0.75)))
    k1 = float(p.get("inex_k1", 1.2))
    scores = _inex_bm25(query, nodes, candidates, b, k1)
    order = {nid: i for i, nid in enumerate(candidates)}
    ranked = sorted(candidates, key=lambda nid: (-scores[nid], order[nid]))
    return _fit_regions(ranked, nodes, budget, _counter(p), bool(p.get("include_breadcrumbs", True)))
