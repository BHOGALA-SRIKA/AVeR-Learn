import pytest

from src.retrieval.selectors import (
    select_fixed_k, select_score_ratio, select_score_gap, select_adaptive_k
)
from src.retrieval.hichunk import select_hichunk, select_hichunk_with_stop
from src.retrieval.emte import select_emte, emte_ablation


def node(i, level, kind, parent, children, text, order, tokens=None):
    return {
        "id": i, "level": level, "kind": kind, "title": "",
        "text": text, "parent_id": parent, "children": children,
        "breadcrumb": "OS > Unit 1", "doc_order": order,
        "page_start": 1, "page_end": 1,
        "n_tokens": tokens if tokens is not None else len(text.split()),
        "n_tokens_breadcrumb": 3,
    }


NODES = {
    "DOC": node("DOC", 0, "document", None, ["CH"], "", 0, 0),
    "CH": node("CH", 1, "chapter", "DOC", ["SEC_A", "SEC_B"], "", 1, 0),
    "SEC_A": node("SEC_A", 2, "section", "CH", ["P1", "P2"], "alpha beta gamma delta", 2, 4),
    "SEC_B": node("SEC_B", 2, "section", "CH", ["P3"], "epsilon zeta", 3, 2),
    "P1": node("P1", 4, "paragraph", "SEC_A", [], "alpha beta", 4, 2),
    "P2": node("P2", 4, "paragraph", "SEC_A", [], "gamma delta", 5, 2),
    "P3": node("P3", 4, "paragraph", "SEC_B", [], "epsilon zeta", 6, 2),
}
TREE = {
    "schema_version": "1.0",
    "document_id": "OS_UNIT1_TOPIC1",
    "root_ids": ["DOC"],
    "nodes": NODES,
}
PARAGRAPHS = [NODES[x] for x in ("P1", "P2", "P3")]


class FakeIndex:
    records = PARAGRAPHS

    def score(self, query):
        return {"P1": 10.0, "P2": 8.0, "P3": 1.0}


def count(text):
    return len(text.split())


def params(**kwargs):
    return {"token_counter": count, **kwargs}


def test_fixed_k():
    # Each paragraph costs 2 content tokens + 3 breadcrumb tokens.
    result = select_fixed_k("q", TREE, FakeIndex(), 10, params(k=2))
    assert result.nodes == ["P1", "P2"]
    assert result.delivered_tokens == 10


def test_ratio():
    result = select_score_ratio("q", TREE, FakeIndex(), 10, params(ratio_threshold=.5))
    assert result.nodes == ["P1", "P2"]


def test_gap():
    result = select_score_gap("q", TREE, FakeIndex(), 10, params(gap_threshold=3))
    assert result.nodes == ["P1", "P2"]


def test_adaptive():
    result = select_adaptive_k(
        "q", TREE, FakeIndex(), 10,
        params(min_k=1, max_k=3, relative_gap_threshold=.5)
    )
    assert result.nodes == ["P1", "P2"]


def test_missing_score_is_zero():
    class Missing(FakeIndex):
        def score(self, query):
            return {"P1": 10}
    result = select_fixed_k("q", TREE, Missing(), 10, params(k=3))
    assert result.ranked == ["P1", "P2", "P3"]


def test_hichunk_merges_selected_siblings():
    result = select_hichunk(
        "q", TREE, FakeIndex(), 10,
        params(merge_min_children=2, merge_level_max=2)
    )
    assert result.nodes == ["SEC_A"]
    # Merged region: P1 + P2 content (4) + one section breadcrumb (3).
    assert result.delivered_tokens == 7


def test_hichunk_stopping():
    result = select_hichunk_with_stop(
        "q", TREE, FakeIndex(), 10,
        params(gap_threshold=3, merge_min_children=2, merge_level_max=2)
    )
    assert result.nodes == ["SEC_A"]


def test_emte_returns_valid_budgeted_evidence():
    result = select_emte(
        "q", TREE, FakeIndex(), 10,
        params(section_level=2, aggregator="sum", score_scale="max",
               tau_stop=0, tau_up=0)
    )
    assert result.nodes
    assert result.delivered_tokens <= 10


def test_emte_ablation_runs_all_aggregators():
    results = emte_ablation("q", TREE, FakeIndex(), 10, params(section_level=2))
    assert set(results) == {"sum", "max", "mean", "saturating"}


def test_blank_query_abstains():
    result = select_fixed_k("", TREE, FakeIndex(), 10, params(k=2))
    assert result.abstained
    assert result.nodes == []


def test_tree_key_id_mismatch_rejected():
    bad = {"schema_version": "1.0", "document_id": "x", "root_ids": ["DOC"],
           "nodes": {"wrong": NODES["DOC"]}}
    with pytest.raises(ValueError):
        select_hichunk("q", bad, FakeIndex(), 10, params())


def test_emte_handles_missing_hierarchy_levels():
    # Direct document -> paragraph is valid: levels need not be contiguous.
    flat = {
        "schema_version": "1.0", "document_id": "flat", "root_ids": ["D"],
        "nodes": {
            "D": node("D", 0, "document", None, ["P"], "", 0, 0),
            "P": node("P", 4, "paragraph", "D", [], "alpha beta", 1, 2),
        },
    }

    class One:
        records = [flat["nodes"]["P"]]

        def score(self, query):
            return {"P": 1.0}

    result = select_emte("q", flat, One(), 5, params(section_level=2))
    assert result.nodes == ["P"]
