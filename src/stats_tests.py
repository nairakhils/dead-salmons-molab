"""Statistical tests, multiple-comparison corrections, and discovery
metrics.

Pure NumPy + SciPy. No statsmodels. No global state.

NaN policy for ``benjamini_hochberg``: NaN p-values are excluded from
the BH step-up procedure entirely. They are returned as ``reject =
False`` and ``p_adjusted = NaN``. This is the same convention used by
SciPy's ``false_discovery_control``: a NaN means "not tested", and we
should neither reject it nor let it advance the threshold.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


# --------------------------------------------------------- bonferroni


def bonferroni(p_values: ArrayLike, alpha: float = 0.05) -> dict[str, Any]:
    """Bonferroni correction.

    Multiplies each p-value by the number of tests m and clips at 1.
    Rejects where the adjusted p-value is below alpha (equivalently,
    where the raw p-value is below alpha / m). Preserves input shape.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = p_arr.size
    if m == 0:
        return {
            "reject": np.zeros(p_arr.shape, dtype=bool),
            "threshold": float(alpha),
            "p_adjusted": p_arr.copy(),
            "alpha": float(alpha),
            "method": "bonferroni",
        }
    threshold = alpha / m
    p_adjusted = np.minimum(p_arr * m, 1.0)
    reject = p_arr < threshold
    return {
        "reject": reject,
        "threshold": float(threshold),
        "p_adjusted": p_adjusted,
        "alpha": float(alpha),
        "method": "bonferroni",
    }


# --------------------------------------------------- benjamini_hochberg


def benjamini_hochberg(p_values: ArrayLike, alpha: float = 0.05) -> dict[str, Any]:
    """Benjamini-Hochberg step-up procedure controlling the FDR at level
    ``alpha``.

    Handles arbitrary input shape by flattening, running the procedure,
    and reshaping back. NaN p-values are excluded from the test family
    and returned as ``reject = False``, ``p_adjusted = NaN``.

    Adjusted p-values follow the standard convention:
    ``p_adj_(k) = min_{j>=k}( m * p_(j) / j )``, capped at 1.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    shape = p_arr.shape
    flat = p_arr.ravel()

    if flat.size == 0:
        return {
            "reject": np.zeros(shape, dtype=bool),
            "threshold": 0.0,
            "p_adjusted": flat.copy().reshape(shape),
            "alpha": float(alpha),
            "method": "benjamini-hochberg",
        }

    valid_mask = ~np.isnan(flat)
    valid = flat[valid_mask]
    m = valid.size

    reject_flat = np.zeros(flat.size, dtype=bool)
    p_adj_flat = np.full(flat.size, np.nan, dtype=np.float64)
    threshold: float = 0.0

    if m > 0:
        order = np.argsort(valid, kind="mergesort")
        p_sorted = valid[order]
        ranks = np.arange(1, m + 1, dtype=np.float64)

        # Threshold rank: largest k with p_(k) <= k/m * alpha.
        below = p_sorted <= ranks / m * alpha
        if below.any():
            k_max = int(np.max(np.where(below)[0])) + 1
            threshold = float(p_sorted[k_max - 1])
        else:
            k_max = 0
            threshold = 0.0

        reject_sorted = np.zeros(m, dtype=bool)
        reject_sorted[:k_max] = True

        # Adjusted p-values: cumulative min of m * p / k from the right,
        # then capped at 1.
        adj_sorted = np.minimum(p_sorted * m / ranks, 1.0)
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]

        # Scatter back into the valid positions of the flat array, in
        # original order.
        valid_reject = np.zeros(m, dtype=bool)
        valid_reject[order] = reject_sorted
        valid_adj = np.empty(m, dtype=np.float64)
        valid_adj[order] = adj_sorted

        valid_indices = np.where(valid_mask)[0]
        reject_flat[valid_indices] = valid_reject
        p_adj_flat[valid_indices] = valid_adj

    return {
        "reject": reject_flat.reshape(shape),
        "threshold": threshold,
        "p_adjusted": p_adj_flat.reshape(shape),
        "alpha": float(alpha),
        "method": "benjamini-hochberg",
    }


# --------------------------------------------------- two_sample_ttest_matrix


def two_sample_ttest_matrix(
    group_a: ArrayLike, group_b: ArrayLike, axis: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-feature Welch's t-test for many features at once.

    Wraps ``scipy.stats.ttest_ind`` with ``equal_var=False``. Returns
    ``(t_statistics, p_values)`` with the per-feature axis collapsed.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    result = stats.ttest_ind(a, b, axis=axis, equal_var=False)
    return np.asarray(result.statistic, dtype=np.float64), \
           np.asarray(result.pvalue, dtype=np.float64)


# --------------------------------------------------- one_sample_zscore_effect


def one_sample_zscore_effect(
    samples: ArrayLike, axis: int = 0
) -> NDArray[np.float64]:
    """Per-feature one-sample z-score against a zero null mean.

    ``z = mean(samples) / (std(samples, ddof=1) / sqrt(n))`` along
    ``axis``. Returns the collapsed array.
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.shape[axis]
    mean = x.mean(axis=axis)
    std = x.std(axis=axis, ddof=1)
    se = std / np.sqrt(n)
    return mean / se


# ----------------------------------------------------- permutation_pvalues


def permutation_pvalues(
    group_a: ArrayLike,
    group_b: ArrayLike,
    n_resamples: int = 1000,
    seed: int = 0,
) -> float:
    """Two-sided permutation p-value for a difference in means between
    two 1-D samples.

    Intended for small data; the implementation is the clear loop, not
    the fast vectorised version. Uses the +1/+1 Monte Carlo correction
    of North, Curtis & Sham (2002) / Phipson & Smyth (2010).
    """
    a = np.asarray(group_a, dtype=np.float64).ravel()
    b = np.asarray(group_b, dtype=np.float64).ravel()
    n_a = a.size
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)

    obs = abs(a.mean() - b.mean())

    count = 0
    for _ in range(n_resamples):
        permuted = rng.permutation(pooled)
        diff = abs(permuted[:n_a].mean() - permuted[n_a:].mean())
        if diff >= obs:
            count += 1
    return (1 + count) / (n_resamples + 1)


# ------------------------------------------------------- discovery metrics


def true_discovery_count(
    reject: ArrayLike, truth: ArrayLike
) -> int:
    """Number of rejections that are true signals."""
    r = np.asarray(reject, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    return int(np.sum(r & t))


def false_discovery_count(
    reject: ArrayLike, truth: ArrayLike
) -> int:
    """Number of rejections that are nulls (false positives)."""
    r = np.asarray(reject, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    return int(np.sum(r & ~t))


def false_discovery_proportion(
    reject: ArrayLike, truth: ArrayLike
) -> float:
    """FDP = (false discoveries) / (total discoveries). 0 if no
    discoveries.
    """
    r = np.asarray(reject, dtype=bool)
    n_rej = int(np.sum(r))
    if n_rej == 0:
        return 0.0
    return false_discovery_count(r, truth) / n_rej


def power(reject: ArrayLike, truth: ArrayLike) -> float:
    """Power = (true discoveries) / (number of true signals). 0 if
    there are no true signals.
    """
    t = np.asarray(truth, dtype=bool)
    n_signal = int(np.sum(t))
    if n_signal == 0:
        return 0.0
    return true_discovery_count(reject, t) / n_signal


def summarize_discoveries(
    p_values: ArrayLike, truth: ArrayLike, alpha: float = 0.05
) -> dict[str, dict[str, float]]:
    """Run Bonferroni and BH at the given alpha and report discovery
    counts, FDP, and power for each.
    """
    bonf = bonferroni(p_values, alpha=alpha)
    bh = benjamini_hochberg(p_values, alpha=alpha)
    truth_arr = np.asarray(truth, dtype=bool)
    out: dict[str, dict[str, float]] = {}
    for name, res in (("bonferroni", bonf), ("benjamini_hochberg", bh)):
        rej = res["reject"]
        out[name] = {
            "true_discoveries": float(true_discovery_count(rej, truth_arr)),
            "false_discoveries": float(false_discovery_count(rej, truth_arr)),
            "fdp": float(false_discovery_proportion(rej, truth_arr)),
            "power": float(power(rej, truth_arr)),
        }
    return out
