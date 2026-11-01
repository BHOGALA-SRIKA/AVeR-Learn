"""HiChunk behaviour checks using AVeR-Learn's actual processed corpus.

The hierarchy, paragraph text and BM25F scores are loaded from committed
processed data. Queries are in-scope development benchmark questions.
The adaptive-threshold helper is tested independently because it is a
deterministic scalar function and does not require a retrieval fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.hichunk import _adaptive_threshold
from src.retrieval.processed_pipeline import load_processed_corpus
from src.utils.token_counter import count_tokens

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
QUESTIONS = ROOT / "data" / "benchmark" / "questions.json"
DOCUMENTS = ("unit1_topic1", "unit1_topic2", "unit1_topic3")
BUDGET = 2048


def _load_dev_query(document_id: str) -> str:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    source_pdf = f"{document_id}.pdf"
    matches = [
        item for item in questions
        if item.get("split") == "dev"
        and not item.get("out_of_scope", False)
        and any(
            quote.get("source_pdf", "").lower() == source_pdf.lower()
            for quote in item.get("gold_quotes", [])
        )
        and str(item.get("question", "")).strip()
    ]
    assert matches, f"No in-scope development question for {source_pdf}"
    return str(matches[0]["question"]).strip()


@pytest.fixture(params=DOCUMENTS)
def real_hichunk_case(request):
    document_id = request.param
    corpus = load_processed_corpus(document_id, PROCESSED, method="bm25f")
    query = _load_dev_query(document_id)
    return document_id, corpus, query


def test_hichunk_adaptive_threshold_increases_with_budget_usage():
    early_threshold = _adaptive_threshold(parent_cost=30, used=0, budget=100)
    late_threshold = _adaptive_threshold(parent_cost=30, used=100, budget=100)
    assert early_threshold == 10
    assert late_threshold == 20
    assert late_threshold > early_threshold


def test_hichunk_runs_on_real_processed_hierarchy(real_hichunk_case):
    from src.retrieval.hichunk import select_hichunk

    document_id, corpus, query = real_hichunk_case
    result = select_hichunk(
        query=query,
        tree=corpus.tree,
        index=corpus.index,
        budget=BUDGET,
        params={
            "paragraphs": corpus.paragraphs,
            "token_counter": count_tokens,
            "include_breadcrumbs": True,
            "merge_min_children": 2,
            "merge_level_max": 3,
        },
    )

    node_ids = set(corpus.tree["nodes"])
    paragraph_ids = set(corpus.index.paragraph_ids)
    assert set(result.ranked) == paragraph_ids, document_id
    assert set(result.nodes) <= node_ids, document_id
    assert result.delivered_tokens <= BUDGET, document_id


def test_hichunk_never_selects_document_root(real_hichunk_case):
    from src.retrieval.hichunk import select_hichunk

    document_id, corpus, query = real_hichunk_case
    result = select_hichunk(
        query=query,
        tree=corpus.tree,
        index=corpus.index,
        budget=BUDGET,
        params={
            "paragraphs": corpus.paragraphs,
            "token_counter": count_tokens,
            "include_breadcrumbs": True,
            "merge_min_children": 2,
            "merge_level_max": 3,
        },
    )
    document_root = next(
        (node_id for node_id, node in corpus.tree["nodes"].items()
         if node.get("kind") == "document"),
        None,
    )
    if document_root is not None:
        assert document_root not in result.nodes, document_id
