"""Specification-focused tests for Member 3 EMTE."""
from src.retrieval.emte import select_emte


class MockIndex:
    def __init__(self, scores, upper_bound=None):
        self.scores = scores
        self._upper_bound = upper_bound

    def score(self, query):
        return self.scores

    def score_upper_bound(self, query):
        if self._upper_bound is None:
            raise AssertionError("No upper bound configured")
        return self._upper_bound


def make_tree():
    nodes = {
        "DOC": {
            "id": "DOC", "kind": "document", "level": 0, "parent_id": None,
            "children": ["CH"], "n_tokens": 0, "n_tokens_breadcrumb": 0, "doc_order": 0,
        },
        "CH": {
            "id": "CH", "kind": "chapter", "level": 1, "parent_id": "DOC",
            "children": ["SEC"], "n_tokens": 0, "n_tokens_breadcrumb": 0, "doc_order": 1,
        },
        "SEC": {
            "id": "SEC", "kind": "section", "level": 2, "parent_id": "CH",
            "children": ["SUB_A", "SUB_B"], "n_tokens": 0, "n_tokens_breadcrumb": 0, "doc_order": 2,
        },
        "SUB_A": {
            "id": "SUB_A", "kind": "subsection", "level": 3, "parent_id": "SEC",
            "children": ["P1", "P2"], "n_tokens": 0, "n_tokens_breadcrumb": 1, "doc_order": 3,
        },
        "SUB_B": {
            "id": "SUB_B", "kind": "subsection", "level": 3, "parent_id": "SEC",
            "children": ["P3", "P4"], "n_tokens": 0, "n_tokens_breadcrumb": 1, "doc_order": 6,
        },
    }
    for i, pid in enumerate(("P1", "P2", "P3", "P4"), start=1):
        parent = "SUB_A" if i <= 2 else "SUB_B"
        nodes[pid] = {
            "id": pid, "kind": "paragraph", "level": 4, "parent_id": parent,
            "children": [], "text": f"paragraph {pid}", "n_tokens": 5,
            "n_tokens_breadcrumb": 1, "doc_order": 3 + i,
        }
    return {"nodes": nodes}


def test_emte_merges_both_high_paragraphs_to_subsection():
    result = select_emte(
        "q", make_tree(), MockIndex({"P1": 1.0, "P2": 0.9, "P3": 0.0, "P4": 0.0}),
        budget=30, params={"aggregator": "sum", "score_scale": "relative", "tau_up": 0.002},
    )
    assert "SUB_A" in result.nodes
    assert "P1" not in result.nodes and "P2" not in result.nodes


def test_emte_keeps_single_high_paragraph_without_climbing():
    result = select_emte(
        "q", make_tree(), MockIndex({"P1": 1.0, "P2": 0.0, "P3": 0.0, "P4": 0.0}),
        budget=30, params={"aggregator": "sum", "score_scale": "relative", "tau_up": 0.002},
    )
    assert result.nodes == ["P1"]


def test_emte_never_exceeds_budget():
    result = select_emte(
        "q", make_tree(), MockIndex({"P1": 1.0, "P2": 0.9, "P3": 0.8, "P4": 0.7}),
        budget=12, params={"aggregator": "sum", "score_scale": "relative", "tau_up": 0.0},
    )
    assert result.delivered_tokens <= 12


def test_emte_does_not_merge_across_already_selected_regions():
    result = select_emte(
        "q", make_tree(), MockIndex({"P1": 1.0, "P2": 0.0, "P3": 0.9, "P4": 0.0}),
        budget=30, params={"aggregator": "sum", "score_scale": "relative", "tau_up": 1000.0},
    )
    assert result.nodes == ["P1", "P3"]


def test_emte_absolute_scale_abstains_for_zero_upper_bound():
    result = select_emte(
        "unknown words", make_tree(),
        MockIndex({"P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0}, upper_bound=0.0),
        budget=30, params={"aggregator": "sum", "score_scale": "absolute", "epsilon": 1e-9},
    )
    # Selection's fourth field represents abstention; the public interface
    # exposes the selected nodes and delivered token count.
    assert result.nodes == []
    assert result.delivered_tokens == 0


def test_emte_absolute_scale_uses_upper_bound():
    result = select_emte(
        "q", make_tree(),
        MockIndex({"P1": 2.0, "P2": 0.0, "P3": 0.0, "P4": 0.0}, upper_bound=4.0),
        budget=30,
        params={
            "aggregator": "sum", "score_scale": "absolute",
            "tau_stop": 0.01, "tau_up": 0.01,
        },
    )
    # P1's normalized score is 2/4 = 0.5. Expanding to SEC adds
    # zero-scoring paragraphs, so its gain-per-extra-token is zero.
    assert result.nodes == ["P1"]


def test_emte_tries_next_seed_when_top_seed_does_not_fit():
    tree = make_tree()
    tree["nodes"]["P1"]["n_tokens"] = 50
    tree["nodes"]["P2"]["n_tokens"] = 5
    result = select_emte(
        "q", tree, MockIndex({"P1": 1.0, "P2": 0.8, "P3": 0.0, "P4": 0.0}),
        budget=6, params={"aggregator": "sum", "score_scale": "relative", "tau_up": 1000.0},
    )
    assert "P2" in result.nodes
    assert result.delivered_tokens <= 6
