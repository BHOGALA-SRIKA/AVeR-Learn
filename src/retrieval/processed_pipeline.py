"""Strict Member 3 integration with Member 1 processed files and Member 2 LexicalIndex.

This is the production entry point for retrieval over the real AVeR-Learn corpus.
It intentionally has no synthetic/demo fallback: missing or inconsistent processed
files are hard errors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.indexing.lexical import LexicalIndex
from src.utils.token_counter import count_tokens, count_breadcrumb, counter_name

from ._common import Selection
from .selectors import (
    select_fixed_k,
    select_score_ratio,
    select_score_gap,
    select_adaptive_k,
)
from .hichunk import select_hichunk, select_hichunk_with_stop
from .emte import select_emte


@dataclass
class ProcessedCorpus:
    document_id: str
    document_dir: Path
    tree_path: Path
    paragraphs_path: Path
    tree: dict[str, Any]
    paragraphs: list[dict[str, Any]]
    index: LexicalIndex
    token_counter_name: str


METHODS = {
    "fixed_k": select_fixed_k,
    "score_ratio": select_score_ratio,
    "score_gap": select_score_gap,
    "adaptive_k": select_adaptive_k,
    "hichunk": select_hichunk,
    "hichunk_with_stop": select_hichunk_with_stop,
    "emte": select_emte,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if record.get("kind") != "paragraph":
                raise ValueError(
                    f"Expected paragraph record at {path}:{line_number}; "
                    f"got kind={record.get('kind')!r}"
                )
            if not record.get("id"):
                raise ValueError(f"Paragraph record has no id at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No paragraph records found in actual processed file: {path}")
    return records


def _validate_and_apply_shared_token_counts(
    tree: dict[str, Any], paragraphs: list[dict[str, Any]], document_id: str
) -> None:
    """Validate Member 1 output and refresh in-memory counts with Member 2's shared counter.

    Member 1's currently generated n_tokens fields use a different counting rule.
    Do not trust those stored counts for Member 3 budgets. Recompute in memory
    with the project's shared token_counter.py, without modifying source files.
    """
    if tree.get("document_id") != document_id:
        raise ValueError(
            f"Document ID mismatch: directory={document_id!r}, "
            f"tree={tree.get('document_id')!r}"
        )
    nodes = tree.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("Member 1 tree.json must contain a non-empty 'nodes' object")

    tree_paragraph_ids = {
        str(node_id)
        for node_id, node in nodes.items()
        if isinstance(node, dict) and node.get("kind") == "paragraph"
    }
    record_ids = [str(record["id"]) for record in paragraphs]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Member 1 paragraphs.jsonl contains duplicate paragraph IDs")

    if set(record_ids) != tree_paragraph_ids:
        missing_records = sorted(tree_paragraph_ids - set(record_ids))
        missing_tree_nodes = sorted(set(record_ids) - tree_paragraph_ids)
        raise ValueError(
            "Member 1 tree.json and paragraphs.jsonl do not describe the same paragraphs. "
            f"Missing from JSONL: {missing_records[:5]}; "
            f"missing paragraph nodes from tree: {missing_tree_nodes[:5]}"
        )

    # Ensure JSONL records and tree nodes refer to the same real text.
    for record in paragraphs:
        pid = str(record["id"])
        tree_node = nodes.get(pid)
        if tree_node is None:
            raise ValueError(f"Paragraph {pid} is missing from tree.json")
        if str(record.get("text", "")) != str(tree_node.get("text", "")):
            raise ValueError(f"Text mismatch between tree.json and paragraphs.jsonl for {pid}")

    # Refresh counts using the shared project counter. The original processed
    # files remain unchanged; the in-memory tree is what Member 3 receives.
    for node in nodes.values():
        if not isinstance(node, dict):
            raise ValueError("tree.json contains a non-object node")
        node["n_tokens"] = count_tokens(str(node.get("text", "")))
        node["n_tokens_breadcrumb"] = count_breadcrumb(node.get("breadcrumb"))

    for record in paragraphs:
        record["n_tokens"] = count_tokens(str(record.get("text", "")))
        record["n_tokens_breadcrumb"] = count_breadcrumb(record.get("breadcrumb"))


def load_processed_corpus(
    document_id: str,
    processed_root: str | Path = "data/processed",
    *,
    method: str = "bm25f",
    expand: bool = False,
    index_params: Mapping[str, Any] | None = None,
) -> ProcessedCorpus:
    """Load only the actual Member 1 processed files and build Member 2's index.

    Required paths:
      data/processed/<document_id>/tree.json
      data/processed/<document_id>/paragraphs.jsonl

    There is deliberately no fallback to raw PDFs, old paragraphs.json,
    fixtures, demos, or generated sample data.
    """
    if not document_id or Path(document_id).name != document_id or document_id in {".", ".."}:
        raise ValueError("document_id must be a single processed-document folder name")

    root = Path(processed_root).resolve()
    document_dir = root / document_id
    tree_path = document_dir / "tree.json"
    paragraphs_path = document_dir / "paragraphs.jsonl"

    if not document_dir.is_dir():
        raise FileNotFoundError(f"Processed document directory does not exist: {document_dir}")
    if not tree_path.is_file():
        raise FileNotFoundError(f"Member 1 tree file does not exist: {tree_path}")
    if not paragraphs_path.is_file():
        raise FileNotFoundError(f"Member 1 paragraph file does not exist: {paragraphs_path}")

    with tree_path.open("r", encoding="utf-8") as handle:
        tree = json.load(handle)
    if not isinstance(tree, dict):
        raise ValueError(f"Expected a JSON object in {tree_path}")

    paragraphs = _read_jsonl(paragraphs_path)
    _validate_and_apply_shared_token_counts(tree, paragraphs, document_id)

    # Build Member 2's actual lexical index directly from Member 1's JSONL path.
    index = LexicalIndex(method=method, expand=expand, **dict(index_params or {})).build(
        paragraphs_path,
        tree=tree,
    )

    expected_ids = [str(record["id"]) for record in paragraphs]
    actual_ids = [str(pid) for pid in index.paragraph_ids]
    if actual_ids != expected_ids:
        raise ValueError(
            "Member 2 LexicalIndex paragraph order/IDs do not match Member 1 paragraphs.jsonl"
        )

    return ProcessedCorpus(
        document_id=document_id,
        document_dir=document_dir,
        tree_path=tree_path,
        paragraphs_path=paragraphs_path,
        tree=tree,
        paragraphs=paragraphs,
        index=index,
        token_counter_name=counter_name(),
    )


def retrieve_from_processed_corpus(
    query: str,
    corpus: ProcessedCorpus,
    *,
    method: str,
    budget: int,
    params: Mapping[str, Any] | None = None,
) -> Selection:
    """Run one Member 3 method against a validated real processed corpus."""
    if method not in METHODS:
        raise ValueError(f"Unknown retrieval method {method!r}. Choose from {', '.join(METHODS)}")
    if not query or not query.strip():
        raise ValueError("A non-empty real user query is required")
    if budget <= 0:
        raise ValueError("budget must be a positive integer")

    settings = dict(params or {})
    # These are mandatory production bindings. Callers cannot replace them
    # with a fake index, a synthetic paragraph list, or a local test counter.
    settings["paragraphs"] = corpus.paragraphs
    settings["token_counter"] = count_tokens
    settings["include_breadcrumbs"] = True
    return METHODS[method](query, corpus.tree, corpus.index, budget, settings)
