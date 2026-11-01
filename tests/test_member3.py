"""Member 3 selector tests using AVeR-Learn's committed processed corpus.

All trees, paragraphs, BM25F scores and benchmark questions are loaded from
 data/processed/ and data/benchmark/questions.json. No synthetic nodes or fake
 indexes are used in this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.processed_pipeline import (
    load_processed_corpus,
    retrieve_from_processed_corpus,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
QUESTIONS = ROOT / "data" / "benchmark" / "questions.json"
DOCUMENTS = ("unit1_topic1", "unit1_topic2", "unit1_topic3")


def _real_question(document_id: str) -> str:
    source_pdf = f"{document_id}.pdf"
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    for item in questions:
        if (
            item.get("split") == "dev"
            and not item.get("out_of_scope", False)
            and any(
                q.get("source_pdf", "").lower() == source_pdf.lower()
                for q in item.get("gold_quotes", [])
            )
        ):
            query = str(item.get("question", "")).strip()
            if query:
                return query
    raise AssertionError(f"No source-grounded benchmark question for {source_pdf}")


@pytest.fixture(params=DOCUMENTS)
def real_case(request):
    document_id = request.param
    corpus = load_processed_corpus(document_id, PROCESSED)
    return corpus, _real_question(document_id)


def _run(real_case, method: str, params=None, budget: int = 2048):
    corpus, query = real_case
    return retrieve_from_processed_corpus(
        query, corpus, method=method, budget=budget, params=params or {}
    )


def _assert_real_selection(real_case, result, budget=2048):
    corpus, _ = real_case
    node_ids = set(corpus.tree["nodes"])
    paragraph_ids = set(corpus.index.paragraph_ids)
    assert result.delivered_tokens <= budget
    assert set(result.ranked) == paragraph_ids
    assert set(result.nodes) <= node_ids
    assert all(node_id in node_ids for node_id in result.nodes)


def test_fixed_k_on_real_processed_corpus(real_case):
    result = _run(real_case, "fixed_k", {"k": 5})
    _assert_real_selection(real_case, result)


def test_score_ratio_on_real_processed_corpus(real_case):
    result = _run(real_case, "score_ratio", {"ratio_threshold": 0.5})
    _assert_real_selection(real_case, result)


def test_score_gap_on_real_processed_corpus(real_case):
    result = _run(real_case, "score_gap", {"gap_threshold": 0.0})
    _assert_real_selection(real_case, result)


def test_adaptive_k_on_real_processed_corpus(real_case):
    result = _run(real_case, "adaptive_k", {"search_frac": 0.9, "buffer": 1})
    _assert_real_selection(real_case, result)


def test_hichunk_on_real_processed_corpus(real_case):
    result = _run(real_case, "hichunk", {"merge_min_children": 2, "merge_level_max": 3})
    _assert_real_selection(real_case, result)


def test_hichunk_with_stopping_on_real_processed_corpus(real_case):
    result = _run(
        real_case, "hichunk_with_stop",
        {"gap_threshold": 0.0, "merge_min_children": 2, "merge_level_max": 3},
    )
    _assert_real_selection(real_case, result)


def test_emte_on_real_processed_corpus(real_case):
    result = _run(real_case, "emte", {"aggregator": "sum", "score_scale": "relative"})
    _assert_real_selection(real_case, result)


def test_emte_all_aggregators_on_real_processed_corpus(real_case):
    corpus, query = real_case
    from src.retrieval.emte import emte_ablation

    results = emte_ablation(query, corpus.tree, corpus.index, 2048, {
        "paragraphs": corpus.paragraphs,
        "token_counter": __import__("src.utils.token_counter", fromlist=["count_tokens"]).count_tokens,
        "include_breadcrumbs": True,
        "section_level": 2,
    })
    assert set(results) == {"sum", "max", "mean", "saturating"}
    for result in results.values():
        _assert_real_selection(real_case, result)
