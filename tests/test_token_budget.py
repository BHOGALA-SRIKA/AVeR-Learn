from src.retrieval.selectors import select_fixed_k

class Index:
    def score(self, query):
        return {"P1": 2.0, "P2": 1.0}

def test_fixed_k_counts_breadcrumb_tokens():
    tree = {"nodes": {
        "P1": {"id":"P1","kind":"paragraph","level":4,"parent_id":None,
               "children":[],"text":"a b c","n_tokens":3,
               "breadcrumb":"Unit > Section","n_tokens_breadcrumb":4,"doc_order":1},
        "P2": {"id":"P2","kind":"paragraph","level":4,"parent_id":None,
               "children":[],"text":"d e","n_tokens":2,
               "breadcrumb":"Unit > Section","n_tokens_breadcrumb":4,"doc_order":2},
    }}
    result = select_fixed_k("q", tree, Index(), 7, {"k":2})
    assert result.nodes == ["P1"]
    assert result.delivered_tokens == 7
