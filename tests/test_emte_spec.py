"""EMTE specification tests against AVeR-Learn's actual processed corpus.

They load the committed tree.json and
paragraphs.jsonl files and build the BM25F LexicalIndex.

For the three Unit 1 documents, queries come from the real in-scope dev split
in data/benchmark/questions.json. The DSA processed source has no DSA-labelled
question in that benchmark, so its behavioural query is formed only from the
actual wording in DSA_P0029 ("Stacks", "Recursion", "Infix to postfix
conversion"). It is a source-grounded retrieval smoke/spec test, NOT a
benchmark question or an evaluation result.
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
PROCESSED_ROOT = ROOT / "data" / "processed"
QUESTIONS_FILE = ROOT / "data" / "benchmark" / "questions.json"
BUDGET = 512
AGGREGATORS = ("sum", "max", "mean", "saturating")

# Real dev benchmark question IDs and their matching processed document IDs.
BENCHMARK_CASES = (
    ("unit1_topic1", "CL-001"),
    ("unit1_topic2", "CL-015"),
    ("unit1_topic3", "CL-023"),
)

# This is an actual phrase/query drawn from DSA_P0029 in the real JSONL source.
# It is deliberately not labelled as a benchmark question.
DSA_SOURCE_QUERY = "Stacks Recursion Infix to postfix conversion"
DSA_SOURCE_PARAGRAPH_ID = "DSA_P0029"


def _load_real_dev_question(qid: str, document_id: str) -> str:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    matches = [
        item
        for item in questions
        if item.get("qid") == qid
        and item.get("split") == "dev"
        and not item.get("out_of_scope", False)
    ]
    assert len(matches) == 1, f"Expected exactly one in-scope dev question: {qid}"

    question = matches[0]
    expected_source = f"{document_id}.pdf"
    assert any(
        quote.get("source_pdf", "").lower() == expected_source.lower()
        for quote in question.get("gold_quotes", [])
    ), f"{qid} does not cite {expected_source}"
    return question["question"]


@pytest.fixture(scope="module")
def real_cases():
    """Load all four actual processed corpora and their grounded queries."""
    cases = []

    for document_id, qid in BENCHMARK_CASES:
        corpus = load_processed_corpus(
            document_id, PROCESSED_ROOT, method="bm25f"
        )
        query = _load_real_dev_question(qid, document_id)
        cases.append((document_id, qid, corpus, query, None))

    dsa_corpus = load_processed_corpus("dsa", PROCESSED_ROOT, method="bm25f")
    cases.append(
        (
            "dsa",
            "source:DSA_P0029",
            dsa_corpus,
            DSA_SOURCE_QUERY,
            DSA_SOURCE_PARAGRAPH_ID,
        )
    )
    return cases


@pytest.mark.parametrize(
    "document_id",
    ("unit1_topic1", "unit1_topic2", "unit1_topic3", "dsa"),
)
def test_emte_loads_actual_processed_tree_jsonl_and_bm25f_index(document_id):
    corpus = load_processed_corpus(
        document_id, PROCESSED_ROOT, method="bm25f"
    )

    assert corpus.tree_path == PROCESSED_ROOT / document_id / "tree.json"
    assert corpus.paragraphs_path == PROCESSED_ROOT / document_id / "paragraphs.jsonl"
    assert corpus.tree["document_id"] == document_id
    assert corpus.paragraphs
    assert corpus.index.info()["method"] == "bm25f"
    assert corpus.index.paragraph_ids == [p["id"] for p in corpus.paragraphs]
    assert corpus.token_counter_name


@pytest.mark.parametrize("case_index", range(4))
@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_emte_aggregators_run_on_real_source_queries(
    real_cases, case_index, aggregator
):
    document_id, case_name, corpus, query, expected_source_pid = real_cases[case_index]

    # DSA query must genuinely match the paragraph from which it was derived.
    if expected_source_pid is not None:
        matches = corpus.index.matches(query)
        assert matches.get(expected_source_pid) is True, (
            f"The DSA source query did not match {expected_source_pid}"
        )

    result = retrieve_from_processed_corpus(
        query,
        corpus,
        method="emte",
        budget=BUDGET,
        params={
            "aggregator": aggregator,
            "score_scale": "relative",
        },
    )

    paragraph_ids = set(corpus.index.paragraph_ids)
    tree_nodes = corpus.tree["nodes"]
    assert result.delivered_tokens <= BUDGET, (
        f"{case_name} ({document_id}) exceeded the evidence budget"
    )
    assert set(result.ranked) <= paragraph_ids
    assert set(result.nodes) <= set(tree_nodes)
    assert result.nodes == sorted(
        result.nodes,
        key=lambda node_id: (
            int(tree_nodes[node_id].get("doc_order", 0)), node_id
        ),
    )
    assert result.abstained is (len(result.nodes) == 0)


@pytest.mark.parametrize("case_index", range(4))
def test_emte_absolute_scaling_runs_with_real_bm25f_index(real_cases, case_index):
    document_id, case_name, corpus, query, expected_source_pid = real_cases[case_index]

    if expected_source_pid is not None:
        assert corpus.index.matches(query).get(expected_source_pid) is True

    result = retrieve_from_processed_corpus(
        query,
        corpus,
        method="emte",
        budget=BUDGET,
        params={
            "aggregator": "saturating",
            "score_scale": "absolute",
            "tau_up": 0.0,
        },
    )

    assert result.delivered_tokens <= BUDGET, f"{case_name} exceeded budget"
    assert set(result.ranked) <= set(corpus.index.paragraph_ids)
    assert set(result.nodes) <= set(corpus.tree["nodes"])


def _paragraph_descendants(nodes: dict, node_id: str) -> set[str]:
    node = nodes[node_id]
    if node.get("kind") == "paragraph":
        return {node_id}

    found: set[str] = set()
    stack = list(node.get("children") or [])
    while stack:
        child_id = stack.pop()
        child = nodes[child_id]
        if child.get("kind") == "paragraph":
            found.add(child_id)
        else:
            stack.extend(child.get("children") or [])
    return found


@pytest.mark.parametrize("case_index", range(4))
def test_emte_selected_regions_do_not_overlap_on_actual_tree(real_cases, case_index):
    document_id, case_name, corpus, query, expected_source_pid = real_cases[case_index]

    if expected_source_pid is not None:
        assert corpus.index.matches(query).get(expected_source_pid) is True

    result = retrieve_from_processed_corpus(
        query,
        corpus,
        method="emte",
        budget=1024,
        params={
            "aggregator": "sum",
            "score_scale": "relative",
            "tau_up": 0.0,
            "max_regions": 10,
        },
    )

    nodes = corpus.tree["nodes"]
    covered: set[str] = set()
    for selected_id in result.nodes:
        region = _paragraph_descendants(nodes, selected_id)
        overlap = covered.intersection(region)
        assert not overlap, (
            f"{case_name} ({document_id}) has overlapping selected regions: "
            f"{sorted(overlap)}"
        )
        covered.update(region)
