"""Is method A really better than method B, or is the gap luck? (owner: Member 4)

Both methods answer the SAME questions, so the test is paired: it looks at the
per-question differences.

- ``randomization_test``: Fisher's paired randomization (permutation) test,
  recommended for IR by Smucker, Allan & Carterette (CIKM 2007). If the two
  methods were equally good, each question's difference would be equally
  likely to have either sign. We flip signs at random many times and count how
  often the mean difference is at least as large as the one we saw.
- ``bootstrap_ci``: 95% confidence interval of the mean difference.
- ``holm``: when comparing many methods against one baseline, some p-values
  fall below 0.05 by chance. Holm-Bonferroni correction controls that.

With few questions (e.g. 40) only large differences can be significant.
Report that honestly instead of claiming a winner.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def randomization_test(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a) != mean(b), paired by question."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.size == 0:
        raise ValueError("Need at least one paired question.")
    observed = abs(diff.mean())
    if observed == 0:
        return 1.0
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n, diff.size))
    extreme = np.count_nonzero(np.abs((signs * diff).mean(axis=1)) >= observed - 1e-12)
    return (extreme + 1) / (n + 1)


def bootstrap_ci(
    a: Sequence[float], b: Sequence[float], n: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for mean(a - b)."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (compare each to 0.05)."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out
