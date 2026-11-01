"""Shared helpers for AVeR-Learn Member 3 selectors.

Tree contract: dictionary with tree["nodes"] mapping node IDs to node dictionaries.
Index contract: index.score(query) -> {paragraph_id: score}; a compatibility
adapter for the repository's original BM25Index.search() is also provided.

All budget calculations use the injected original-text counter where supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass
class Selection:
    nodes: list[str]
    ranked: list[str]
    delivered_tokens: int
    abstained: bool = False


def _counter(params: Mapping[str, Any]) -> Callable[[str], int]:
    fn = params.get("token_counter")
    if fn is None:
        # Temporary test-only fallback. Production integration must inject
        # Member 2's shared original-text token counter.
        return lambda text: len(text.split())
    if not callable(fn):
        raise TypeError("params['token_counter'] must be callable")
    return fn


def _node_map(tree: Any, params: Mapping[str, Any]) -> dict[str, dict]:
    if isinstance(tree, Mapping) and isinstance(tree.get("nodes"), Mapping):
        nodes = tree["nodes"]
    elif isinstance(tree, Mapping) and all(
        isinstance(v, Mapping) and "id" in v for v in tree.values()
    ):
        # Convenience for tests and compatible node-map inputs.
        nodes = tree
    elif isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        nodes = {str(n["id"]): n for n in tree}
    else:
        raise TypeError(
            "tree must follow Member 1's contract: {'schema_version', "
            "'document_id', 'root_ids', 'nodes': {node_id: node}}"
        )
    result = {str(k): dict(v) for k, v in nodes.items()}
    for node_id, node in result.items():
        if str(node.get("id")) != node_id:
            raise ValueError(f"Node key/id mismatch for {node_id}")
    return result


def _paragraph_records(tree: Any, index: Any, params: Mapping[str, Any]) -> list[dict]:
    supplied = params.get("paragraphs")
    if supplied is not None:
        return [dict(x) for x in supplied]
    if hasattr(index, "records"):
        return [dict(x) for x in index.records]
    nodes = _node_map(tree, params)
    return [n for n in nodes.values() if n.get("kind") == "paragraph"]


def _pid(record: Mapping[str, Any]) -> str:
    value = record.get("id", record.get("paragraph_id"))
    if value is None:
        raise ValueError("Paragraph record requires id or paragraph_id")
    return str(value)


def _scores(query: str, index: Any, paragraph_ids: list[str]) -> dict[str, float]:
    if hasattr(index, "score") and callable(index.score):
        raw = index.score(query)
        if not isinstance(raw, Mapping):
            raise TypeError("index.score(query) must return a mapping of IDs to scores")
        values = {str(k): float(v) for k, v in raw.items()}
    elif hasattr(index, "search") and callable(index.search):
        # Adapter for the current repository's BM25Index.search().
        results = index.search(query, top_k=max(1, len(paragraph_ids)))
        values = {}
        for row in results:
            pid = row.get("id", row.get("paragraph_id"))
            if pid is None:
                continue
            score = row.get("retrieval_score", row.get("bm25_score"))
            if score is None:
                raise ValueError("BM25 result is missing retrieval_score/bm25_score")
            values[str(pid)] = float(score)
    else:
        raise TypeError("Index must implement score(query) or search(query, top_k=...)")

    unknown = set(values) - set(paragraph_ids)
    if unknown:
        raise ValueError(f"Index returned IDs absent from paragraph corpus: {sorted(unknown)[:5]}")
    # Agreed global rule: absent lexical match is score 0.0.
    return {pid: values.get(pid, 0.0) for pid in paragraph_ids}


def _ordered_paragraph_ids(nodes: Mapping[str, dict]) -> list[str]:
    return sorted(
        (nid for nid, n in nodes.items() if n.get("kind") == "paragraph"),
        key=lambda nid: (int(nodes[nid].get("doc_order", 0)), nid),
    )


def _rank(scores: Mapping[str, float], nodes: Mapping[str, dict]) -> list[str]:
    order = {pid: i for i, pid in enumerate(_ordered_paragraph_ids(nodes))}
    return sorted(scores, key=lambda pid: (-scores[pid], order[pid]))


def _cost(node: Mapping[str, Any], counter: Callable[[str], int], include_breadcrumbs: bool = True) -> int:
    """Delivered paragraph cost: original text plus its breadcrumb exactly once."""
    text = str(node.get("text", ""))
    stored_text = node.get("n_tokens")
    text_tokens = int(stored_text) if stored_text is not None else int(counter(text))

    breadcrumb = str(node.get("breadcrumb", ""))
    stored_breadcrumb = node.get("n_tokens_breadcrumb")
    breadcrumb_tokens = 0
    if include_breadcrumbs:
        breadcrumb_tokens = (
            int(stored_breadcrumb)
            if stored_breadcrumb is not None
            else int(counter(breadcrumb))
        )
    if text_tokens < 0 or breadcrumb_tokens < 0:
        raise ValueError("Token counts cannot be negative")
    return text_tokens + breadcrumb_tokens


def _fit_paragraphs(
    ranked: list[str],
    nodes: Mapping[str, dict],
    budget: int,
    counter: Callable[[str], int],
    max_items: int | None = None,
    include_breadcrumbs: bool = True,
) -> Selection:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    chosen, used = [], 0
    for pid in ranked:
        if max_items is not None and len(chosen) >= max_items:
            break
        node = nodes[pid]
        cost = _cost(node, counter, include_breadcrumbs)
        if cost == 0:
            continue
        if used + cost <= budget:
            chosen.append(pid)
            used += cost
    return Selection(chosen, ranked, used, not bool(chosen))


def _prepare(query: str, tree: Any, index: Any, params: Mapping[str, Any]):
    if not query or not query.strip():
        return {}, [], {}, []
    nodes = _node_map(tree, params)
    paragraphs = _paragraph_records(tree, index, params)
    ids = [_pid(r) for r in paragraphs]
    if len(ids) != len(set(ids)):
        raise ValueError("Paragraph IDs must be unique")
    # Prefer node map text/metadata, but accept Member 1's flat paragraph file.
    for record in paragraphs:
        pid = _pid(record)
        if pid not in nodes:
            nodes[pid] = record
        elif not nodes[pid].get("text") and record.get("text"):
            nodes[pid]["text"] = record["text"]
    paragraph_ids = [pid for pid in ids if pid in nodes]
    scores = _scores(query, index, paragraph_ids)
    ranked = _rank(scores, nodes)
    return nodes, paragraph_ids, scores, ranked
