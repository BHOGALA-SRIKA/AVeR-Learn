# AVeR-Learn — Member 3 updated retrieval code

This bundle contains updated Member 3 retrieval modules and focused tests.

## Files to replace in your repository

Copy these files into the same paths in your AVeR-Learn repository:

- `src/retrieval/selectors.py`
- `src/retrieval/hichunk.py`
- `src/retrieval/emte.py`
- `src/retrieval/__init__.py`

Keep your existing `src/retrieval/_common.py` unless your local copy differs from the shared contract. This bundle includes the audited `_common.py` for reference.

Add the two test files:
- `tests/test_member3.py`
- `tests/test_member3_spec_updates.py`

## Changes

### Selectors
- Score-gap stopping now finds the largest adjacent score gap and optionally applies `gap_threshold` as a minimum gap.
- Adaptive-K now supports the specified `search_frac=0.90` and `buffer=5` method: find the largest adjacent gap in the top 90%, set `k` to the position before the gap, then return up to `k + buffer`.
- Compatibility: old `min_k`, `max_k`, and `relative_gap_threshold` parameters still invoke the earlier local Adaptive-K behaviour. New experiments should use `search_frac` and `buffer`.

### HiChunk
- Score-gap stopping now uses the largest gap.
- The merge budget check accounts for the cost of descendants being replaced: the remaining capacity after removing those descendants must fit the parent, and the final selected context must remain within budget.
- This is a paragraph-level lexical adaptation of HiChunk, not a reproduction of the paper's full RAG pipeline.

### EMTE
- Preserves the corrected first-region exemption for `tau_stop`.
- Adds `max_regions` to support single-region versus multi-region ablations.
- Adds `emte_ablation_suite()` for callable runs across aggregation, score scaling, early-stop thresholds, and region limits.
- Heading-weight ablations require an index adapter exposing `with_heading_weight(weight)`. The function raises a clear error if the adapter is unavailable; it does not silently claim to vary headings when it cannot.
- `emte_ablation()` remains available for backward-compatible aggregator-only runs.

## Example calls

```python
from src.retrieval import (
    select_adaptive_k,
    select_score_gap,
    select_hichunk,
    select_hichunk_with_stop,
    select_emte,
    emte_ablation_suite,
)

# Paper-style Adaptive-K
adaptive = select_adaptive_k(
    query, tree, index, budget,
    {"token_counter": token_counter, "search_frac": 0.90, "buffer": 5},
)

# Largest-gap stopping
gap = select_score_gap(
    query, tree, index, budget,
    {"token_counter": token_counter, "gap_threshold": 0.0},
)

# EMTE controlled ablations
variants = emte_ablation_suite(
    query, tree, index, budget,
    {
        "token_counter": token_counter,
        "section_level": 2,
        "tau_up": 0.001,
        "upper_bound": index.upper_bound(query),  # needed only if index lacks its own upper_bound method
    },
    aggregators=("sum", "max", "mean", "saturating"),
    score_scales=("relative", "absolute"),
    tau_stop_values=(0.0, 0.001),  # 0.0 disables early stopping
    region_limits=(1, None),       # one region vs multiple regions
)

# Heading-weight ablation, only if your index implements with_heading_weight()
heading_variants = emte_ablation_suite(
    query, tree, index, budget, base_params,
    aggregators=("saturating",),
    score_scales=("relative",),
    tau_stop_values=(0.0,),
    region_limits=(None,),
    heading_weights=(0.0, 0.5, 1.0),
)
```

## Run tests

From the repository root, with the project virtual environment activated:

```powershell
python -m pytest tests/test_member3.py tests/test_member3_spec_updates.py -v
```

The focused tests in this bundle were run successfully: **16 passed**.

Run the complete project suite after copying the files:

```powershell
python -m pytest -v
```

The full-project test suite has not been run against your local repository after these updated files. Do not commit until that command passes.
