"""File-backed integration tests for Member 3.

These tests intentionally use the real files committed under data/processed/
and the real benchmark questions in data/benchmark/questions.json.
They do not construct synthetic trees or fake indexes.
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

DOCUMENTS = ("unit1_topic1", "unit1_topic2", "unit1_topic3", "dsa")
METHODS = (
    "fixed_k",
    "score_ratio",
    "score_gap",
    "adaptive_k",
    "hichunk",
    "hichunk_with_stop",
    "emte",
)


def _real_question_for(document_id: str) -> str:
    source_pdf = f"{document_id}.pdf"
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    for item in questions:
        if any(
            quote.get("source_pdf") == source_pdf
            for quote in item.get("gold_quotes", [])
        ):
            return item["question"]
    raise AssertionError(f"No real benchmark question found for {source_pdf}")


@pytest.mark.parametrize("document_id", DOCUMENTS)
def test_loads_actual_member1_files_and_builds_member2_index(document_id):
    corpus = load_processed_corpus(document_id, PROCESSED)

    assert corpus.tree_path == PROCESSED / document_id / "tree.json"
    assert corpus.paragraphs_path == PROCESSED / document_id / "paragraphs.jsonl"
    assert corpus.tree["document_id"] == document_id
    assert corpus.paragraphs
    assert corpus.index.info()["method"] == "bm25f"
    assert corpus.index.paragraph_ids == [record["id"] for record in corpus.paragraphs]
    assert corpus.token_counter_name


@pytest.mark.parametrize("document_id", ("unit1_topic1", "unit1_topic2", "unit1_topic3"))
@pytest.mark.parametrize("method", METHODS)
def test_member3_methods_run_on_actual_processed_corpus(document_id, method):
    corpus = load_processed_corpus(document_id, PROCESSED)
    query = _real_question_for(document_id)

    params = {"k": 5} if method == "fixed_k" else {}
    result = retrieve_from_processed_corpus(
        query,
        corpus,
        method=method,
        budget=2048,
        params=params,
    )

    actual_paragraph_ids = set(corpus.index.paragraph_ids)
    actual_node_ids = set(corpus.tree["nodes"])
    assert result.delivered_tokens <= 2048
    assert set(result.ranked) == actual_paragraph_ids
    assert set(result.nodes) <= actual_node_ids
    assert all(node_id in actual_node_ids for node_id in result.nodes)
