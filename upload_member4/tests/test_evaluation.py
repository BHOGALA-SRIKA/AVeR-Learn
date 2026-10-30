import math

import pytest

from src.evaluation.benchmark import build_qrels, deliver, evidence_coverage, words
from src.evaluation.metrics import (
    evaluate_query,
    mean_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.evaluation.significance import bootstrap_ci, holm, randomization_test
from src.indexing.records import Paragraph

RANKED = ["a", "b", "c", "d", "e"]
QRELS = {"b": 2, "d": 1, "z": 1}  # "z" is relevant but never retrieved


def test_recall_counts_unretrieved_relevant():
    assert recall_at_k(RANKED, QRELS, 1) == 0
    assert recall_at_k(RANKED, QRELS, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 3)


def test_precision_divides_by_k():
    assert precision_at_k(RANKED, QRELS, 2) == 0.5
    assert precision_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 5)
    assert precision_at_k(["b"], QRELS, 3) == pytest.approx(1 / 3)  # short list still divides by k


def test_reciprocal_rank_and_cutoff():
    assert reciprocal_rank(RANKED, QRELS) == 0.5
    assert reciprocal_rank(RANKED, QRELS, k=1) == 0
    assert reciprocal_rank(["x"], QRELS) == 0


def test_ndcg_by_hand():
    # DCG@3 of a, b, c = (2^2 - 1) / log2(3) = 3 / log2(3)
    # ideal grades 2, 1, 1 -> 3 + 1 / log2(3) + 1 / log2(4)
    ideal = 3 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(RANKED, QRELS, 3) == pytest.approx((3 / math.log2(3)) / ideal)
    assert ndcg_at_k(["b", "d", "z"], QRELS, 3) == pytest.approx(1.0)


def test_ndcg_rewards_essential_first():
    assert ndcg_at_k(["b", "d"], QRELS, 2) > ndcg_at_k(["d", "b"], QRELS, 2)


def test_out_of_scope_query_is_skipped():
    assert evaluate_query(RANKED, {}) is None
    assert evaluate_query(RANKED, {"a": 0}) is None


def test_evaluate_and_mean():
    one = evaluate_query(RANKED, QRELS, ks=(1, 3))
    assert set(one) == {"R@1", "P@1", "nDCG@1", "R@3", "P@3", "nDCG@3", "MRR@3"}
    perfect = evaluate_query(["b", "d", "z"], QRELS, ks=(1, 3))
    assert mean_metrics([one, perfect])["R@3"] == pytest.approx((1 / 3 + 1) / 2)


PARAS = [
    Paragraph("p1", "The kernel is the core part of the OS and the only program running at all times.",
              meta={"document_id": "t1"}),
    Paragraph("p2", "A trap is a software generated interrupt caused by an error or a user request.",
              meta={"document_id": "t1"}),
    Paragraph("p3", "The kernel is the core part of another thing entirely.", meta={"document_id": "t2"}),
]
QUESTION = {
    "qid": "Q1",
    "gold_quotes": [
        {"quote": "The kernel is: the core part of the OS, the only program running at all times",
         "grade": 2, "source_pdf": "t1.pdf"},
        {"quote": "A trap is a software generated interrupt", "grade": 1, "source_pdf": "t1.pdf"},
    ],
}


def test_words_normalises_ligatures_and_punctuation():
    assert words("Deﬁnition: I/O") == ["definition", "i", "o"]


def test_quotes_match_paragraphs_of_their_own_pdf_only():
    j = build_qrels([QUESTION, {"qid": "Q2", "gold_quotes": [{"quote": "not in any paragraph at all here",
                                                               "grade": 2}]}], PARAS)
    assert j.qrels["Q1"] == {"p1": 2, "p2": 1}  # p3 shares words but is from another PDF
    assert j.unmatched == [("Q2", "not in any paragraph at all here")]


def test_delivery_stops_at_budget():
    texts = {"a": "one two", "b": "three four five", "c": "six"}
    count = lambda t: len(t.split())  # noqa: E731
    assert deliver(["a", "b", "c"], texts, budget=5, count=count) == ["a", "b"]
    assert deliver(["b", "a", "c"], texts, budget=4, count=count) == ["b"]


def test_evidence_coverage_uses_essential_quotes():
    texts = {p.pid: p.body for p in PARAS}
    assert evidence_coverage(QUESTION, ["p1"], texts)["all_essential"] == 1.0
    miss = evidence_coverage(QUESTION, ["p2"], texts)
    assert miss["all_essential"] == 0.0 and miss["coverage"] < 0.2


def test_randomization_test():
    a = [1.0] * 30
    b = [0.0] * 30
    assert randomization_test(a, b, n=2000) < 0.01
    assert randomization_test(a, a) == 1.0
    noisy = [1, 0] * 15
    assert randomization_test(noisy, noisy[::-1], n=2000) > 0.5


def test_bootstrap_ci_contains_mean():
    lo, hi = bootstrap_ci([1, 1, 0, 1], [0, 0, 0, 1], n=2000)
    assert lo <= 0.5 <= hi


def test_holm_is_monotone_and_capped():
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.5})
    assert adj["x"] == pytest.approx(0.03)
    assert adj["y"] == pytest.approx(0.08)
    assert adj["z"] == 1.0 or adj["z"] == pytest.approx(0.5)
