import pytest
from test_member3 import TREE, FakeIndex, params, PARAGRAPHS
from src.retrieval.selectors import select_score_gap, select_adaptive_k
from src.retrieval.emte import emte_ablation_suite


def test_score_gap_uses_largest_adjacent_gap():
    class Scores:
        records = PARAGRAPHS
        def score(self, query):
            return {"P1": 10.0, "P2": 9.0, "P3": 2.0}
    result = select_score_gap("q", TREE, Scores(), 20, params(gap_threshold=0.0))
    assert result.nodes == ["P1", "P2"]


def test_adaptive_k_uses_top_fraction_and_buffer():
    class Scores:
        records = PARAGRAPHS
        def score(self, query):
            return {"P1": 10.0, "P2": 9.0, "P3": 2.0}
    result = select_adaptive_k(
        "q", TREE, Scores(), 20,
        params(search_frac=0.90, buffer=0)
    )
    assert result.nodes == ["P1"]


def test_emte_ablation_suite_exposes_config_dimensions():
    results = emte_ablation_suite(
        "q", TREE, FakeIndex(), 20,
        params(section_level=2, upper_bound=20.0),
        aggregators=("sum", "max"),
        score_scales=("relative", "absolute"),
        tau_stop_values=(0.0, 0.01),
        region_limits=(1, None),
    )
    assert len(results) == 2 * 2 * 2 * 2
    assert any("agg=sum" in key for key in results)
    assert any("scale=absolute" in key for key in results)
    assert any("max_regions=all" in key for key in results)


def test_heading_ablation_requires_configurable_index():
    with pytest.raises(TypeError, match="with_heading_weight"):
        emte_ablation_suite(
            "q", TREE, FakeIndex(), 20, params(section_level=2, score_scale="relative"),
            score_scales=("relative",),
            heading_weights=(0.0, 1.0),
        )
