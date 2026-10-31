from src.retrieval.hichunk import select_hichunk
from src.retrieval.hichunk import _adaptive_threshold


class MockIndex:
    """Controlled retrieval scores for paper-based tests."""

    def __init__(self, scores):
        self.scores = scores

    def score(self, query):
        return self.scores


def make_tree():
    """Section A has three paragraphs, but only two are retrieved."""
    nodes = {
        "DOC": {
            "id": "DOC", "kind": "document", "level": 0, "parent_id": None,
            "children": ["SEC_A"], "text": "", "n_tokens": 0, "doc_order": 0,
        },
        "SEC_A": {
            "id": "SEC_A", "kind": "section", "level": 2, "parent_id": "DOC",
            "children": ["A1", "A2", "A3"], "text": "Section A complete content",
            "n_tokens": 8, "n_tokens_breadcrumb": 0, "doc_order": 1,
        },
        "A1": {
            "id": "A1", "kind": "paragraph", "level": 4, "parent_id": "SEC_A",
            "children": [], "text": "First evidence paragraph", "n_tokens": 3,
            "n_tokens_breadcrumb": 0, "doc_order": 2,
        },
        "A2": {
            "id": "A2", "kind": "paragraph", "level": 4, "parent_id": "SEC_A",
            "children": [], "text": "Second evidence paragraph", "n_tokens": 3,
            "n_tokens_breadcrumb": 0, "doc_order": 3,
        },
        "A3": {
            "id": "A3", "kind": "paragraph", "level": 4, "parent_id": "SEC_A",
            "children": [], "text": "Third paragraph not retrieved", "n_tokens": 3,
            "n_tokens_breadcrumb": 0, "doc_order": 4,
        },
    }
    return {"nodes": nodes}


def test_hichunk_can_merge_partial_retrieved_siblings():
    """Two represented direct children may merge even when a third is absent."""
    result = select_hichunk(
        query="Explain Section A",
        tree=make_tree(),
        index=MockIndex({"A1": 0.95, "A2": 0.85, "A3": 0.0}),
        budget=20,
        params={"merge_min_children": 2, "merge_level_max": 3},
    )
    assert "SEC_A" in result.nodes
    assert "A1" not in result.nodes
    assert "A2" not in result.nodes
    assert result.delivered_tokens == 9
    assert result.delivered_tokens <= 20


def test_hichunk_adaptive_threshold_increases_with_budget_usage():
    early_threshold = _adaptive_threshold(parent_cost=30, used=0, budget=100)
    late_threshold = _adaptive_threshold(parent_cost=30, used=100, budget=100)
    assert early_threshold == 10
    assert late_threshold == 20
    assert late_threshold > early_threshold


def test_hichunk_does_not_merge_when_only_one_direct_child_is_represented():
    result = select_hichunk(
        "query", make_tree(),
        MockIndex({"A1": 0.9, "A2": 0.0, "A3": 0.0}),
        budget=20,
        params={"merge_min_children": 2},
    )
    assert result.nodes == ["A1"]
    assert result.delivered_tokens == 3


def test_hichunk_does_not_merge_when_complete_parent_exceeds_budget():
    # Two paragraphs fit (6 tokens), but the complete section costs 9.
    result = select_hichunk(
        "query", make_tree(),
        MockIndex({"A1": 0.9, "A2": 0.8, "A3": 0.0}),
        budget=8,
        params={"merge_min_children": 2},
    )
    assert result.nodes == ["A1", "A2"]
    assert result.delivered_tokens == 6


def test_hichunk_respects_merge_min_children_parameter():
    result = select_hichunk(
        "query", make_tree(),
        MockIndex({"A1": 0.9, "A2": 0.8, "A3": 0.0}),
        budget=20,
        params={"merge_min_children": 3},
    )
    assert result.nodes == ["A1", "A2"]


def test_hichunk_respects_merge_level_max():
    result = select_hichunk(
        "query", make_tree(),
        MockIndex({"A1": 0.9, "A2": 0.8, "A3": 0.0}),
        budget=20,
        params={"merge_level_max": 1},
    )
    assert result.nodes == ["A1", "A2"]


def test_hichunk_parent_merge_includes_unretrieved_sibling_content():
    """Auto-Merge returns the complete parent, not only retrieved paragraphs."""
    tree = make_tree()
    result = select_hichunk(
        "query", tree,
        MockIndex({"A1": 0.9, "A2": 0.8, "A3": 0.0}),
        budget=20,
    )
    assert "SEC_A" in result.nodes
    # Complete section content accounts for all three descendant paragraphs.
    assert result.delivered_tokens == 9


def test_hichunk_never_returns_document_root():
    result = select_hichunk(
        "query", make_tree(),
        MockIndex({"A1": 0.9, "A2": 0.8, "A3": 0.0}),
        budget=20,
    )
    assert "DOC" not in result.nodes
