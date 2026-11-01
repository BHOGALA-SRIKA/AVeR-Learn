"""Tests for the Member 3 selector evaluation (src/evaluation/evaluate_selectors.py)."""

import pytest

from src.evaluation.evaluate_selectors import document_of, gold_share, region_text
from src.evaluation.evaluate_retrieval import REPO


def test_gold_share_by_hand():
    q = {"gold_quotes": [{"quote": "the bootstrap program loads the kernel", "grade": 2}]}
    # 6 of the 10 delivered words are the quote
    assert gold_share(q, ["the bootstrap program loads the kernel", "unrelated words here too"]) == pytest.approx(0.6)
    assert gold_share(q, []) == 0.0


def test_document_of_needs_one_source():
    assert document_of({"qid": "a", "gold_quotes": [{"source_pdf": "unit1_topic2.pdf"}]}) == "unit1_topic2"
    with pytest.raises(ValueError):
        document_of({"qid": "b", "gold_quotes": [{"source_pdf": "x.pdf"}, {"source_pdf": "y.pdf"}]})


def test_region_text_on_real_tree():
    pytest.importorskip("src.retrieval.processed_pipeline")
    from src.retrieval.processed_pipeline import load_processed_corpus

    corpus = load_processed_corpus("unit1_topic1", REPO / "data/processed")
    nodes = corpus.tree["nodes"]
    section = next(n for n in nodes.values() if n["kind"] == "section" and len(n["children"]) > 1)
    text = region_text(nodes, section["id"])
    assert text.startswith(section["breadcrumb"])
    for child in section["children"]:
        if nodes[child]["kind"] == "paragraph":
            assert nodes[child]["text"] in text
