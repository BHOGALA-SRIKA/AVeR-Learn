"""Member 3 selector/ablation specification tests on real project data.

Uses the committed benchmark questions, processed hierarchy JSON and JSONL
paragraphs, and the production BM25F LexicalIndex. No mock index or toy tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.processed_pipeline import load_processed_corpus
from src.retrieval.selectors import select_score_gap, select_adaptive_k
from src.retrieval.emte import emte_ablation_suite
from src.utils.token_counter import count_tokens

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
QUESTIONS = ROOT / "data" / "benchmark" / "questions.json"
DOCUMENTS = ("unit1_topic1", "unit1_topic2", "unit1_topic3")


def _real_case(document_id: str):
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    source_pdf = f"{document_id}.pdf"
    query = next(
        (str(item["question"]).strip() for item in questions
         if item.get("split") == "dev"
         and not item.get("out_of_scope", False)
         and any(q.get("source_pdf", "").lower() == source_pdf.lower()
                 for q in item.get("gold_quotes", []))
         and str(item.get("question", "")).strip()),
        None,
    )
    if query is None:
        raise AssertionError(f"No source-grounded benchmark question for {source_pdf}")
    corpus = load_processed_corpus(document_id, PROCESSED)
    params = {
        "paragraphs": corpus.paragraphs,
        "token_counter": count_tokens,
        "include_breadcrumbs": True,
    }
    return query, corpus, params


@pytest.mark.parametrize("document_id", DOCUMENTS)
def test_score_gap_largest_adjacent_gap_on_real_scores(document_id):
    query, corpus, params = _real_case(document_id)
    result = select_score_gap(query, corpus.tree, corpus.index, 2048,
                              {**params, "gap_threshold": 0.0})
    assert result.ranked
    assert set(result.ranked) == set(corpus.index.paragraph_ids)
    assert set(result.nodes) <= set(corpus.tree["nodes"])
    assert result.delivered_tokens <= 2048


@pytest.mark.parametrize("document_id", DOCUMENTS)
def test_adaptive_k_uses_real_corpus_scores(document_id):
    query, corpus, params = _real_case(document_id)
    result = select_adaptive_k(query, corpus.tree, corpus.index, 2048,
                               {**params, "search_frac": 0.90, "buffer": 0})
    assert result.ranked
    assert set(result.ranked) == set(corpus.index.paragraph_ids)
    assert set(result.nodes) <= set(corpus.tree["nodes"])
    assert result.delivered_tokens <= 2048


@pytest.mark.parametrize("document_id", DOCUMENTS)
def test_emte_ablation_grid_on_real_index(document_id):
    query, corpus, params = _real_case(document_id)
    results = emte_ablation_suite(
        query, corpus.tree, corpus.index, 2048,
        {**params, "section_level": 2},
        aggregators=("sum", "max"),
        score_scales=("relative", "absolute"),
        tau_stop_values=(0.0, 0.01),
        region_limits=(1, None),
    )
    assert len(results) == 16
    assert any("agg=sum" in key for key in results)
    assert any("scale=absolute" in key for key in results)
    assert any("max_regions=all" in key for key in results)
    for result in results.values():
        assert result.delivered_tokens <= 2048
        assert set(result.nodes) <= set(corpus.tree["nodes"])
