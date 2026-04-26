"""Tests for src/stats_tests.py.

Behaviour pinned against textbook examples and small hand-checked
arrays. Implementation is pure NumPy + SciPy (no statsmodels).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.stats_tests import (
    benjamini_hochberg,
    bonferroni,
    false_discovery_count,
    false_discovery_proportion,
    one_sample_zscore_effect,
    permutation_pvalues,
    power,
    summarize_discoveries,
    true_discovery_count,
    two_sample_ttest_matrix,
)


# ------------------------------------------------------------- bonferroni


def test_bonferroni_threshold_is_alpha_over_m() -> None:
    p = np.array([0.001, 0.02, 0.04, 0.5])
    out = bonferroni(p, alpha=0.05)
    assert out["alpha"] == 0.05
    np.testing.assert_allclose(out["threshold"], 0.05 / 4)
    np.testing.assert_array_equal(
        out["reject"], np.array([True, False, False, False])
    )
    np.testing.assert_allclose(
        out["p_adjusted"], np.minimum(p * 4, 1.0)
    )
    assert out["method"] == "bonferroni"


def test_bonferroni_preserves_shape_2d() -> None:
    p = np.array([[0.01, 0.04], [0.5, 0.001]])
    out = bonferroni(p, alpha=0.05)
    assert out["reject"].shape == p.shape
    assert out["p_adjusted"].shape == p.shape


def test_bonferroni_caps_adjusted_at_one() -> None:
    p = np.array([0.6, 0.7, 0.99])
    out = bonferroni(p, alpha=0.05)
    assert out["p_adjusted"].max() <= 1.0


def test_bonferroni_empty() -> None:
    out = bonferroni(np.array([]), alpha=0.05)
    assert out["reject"].shape == (0,)
    assert out["p_adjusted"].shape == (0,)


# --------------------------------------------------------- benjamini_hochberg


def test_bh_textbook_example() -> None:
    # Benjamini & Hochberg (1995) Table 1 example. At q = 0.05,
    # the first four sorted p-values should be rejected.
    p = np.array([
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278,
        0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719,
        0.6528, 0.7590, 1.0000,
    ])
    out = benjamini_hochberg(p, alpha=0.05)
    np.testing.assert_array_equal(
        out["reject"],
        np.array([True, True, True, True] + [False] * 11),
    )
    assert out["method"] == "benjamini-hochberg"


def test_bh_rejection_in_input_order_under_permutation() -> None:
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.001, 5), rng.uniform(0.5, 1.0, 20)])
    perm = rng.permutation(p.size)
    out_orig = benjamini_hochberg(p, alpha=0.10)
    out_perm = benjamini_hochberg(p[perm], alpha=0.10)
    np.testing.assert_array_equal(out_orig["reject"][perm], out_perm["reject"])


def test_bh_adjusted_pvalues_are_monotonic_in_sorted_order() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(size=200)
    out = benjamini_hochberg(p, alpha=0.05)
    order = np.argsort(p)
    p_adj_sorted = out["p_adjusted"][order]
    diffs = np.diff(p_adj_sorted)
    assert (diffs >= -1e-12).all(), "BH adjusted p-values must be non-decreasing in sorted order"


def test_bh_adjusted_pvalues_capped_at_one() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(0.5, 1.0, size=100)
    out = benjamini_hochberg(p, alpha=0.05)
    assert out["p_adjusted"].max() <= 1.0 + 1e-12


def test_bh_preserves_shape_2d() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(size=(4, 5))
    out = benjamini_hochberg(p, alpha=0.05)
    assert out["reject"].shape == p.shape
    assert out["p_adjusted"].shape == p.shape


def test_bh_global_null_low_discovery_rate() -> None:
    rng = np.random.default_rng(4)
    n = 1000
    counts = []
    for _ in range(40):
        p = rng.uniform(size=n)
        counts.append(int(benjamini_hochberg(p, alpha=0.05)["reject"].sum()))
    assert np.mean(counts) < 0.05 * n


def test_bh_nan_policy_rejects_false() -> None:
    # Documented policy: NaN inputs are treated as non-rejections.
    p = np.array([0.001, np.nan, 0.5])
    out = benjamini_hochberg(p, alpha=0.10)
    assert out["reject"][1] == False  # noqa: E712
    assert np.isnan(out["p_adjusted"][1])


def test_bh_empty() -> None:
    out = benjamini_hochberg(np.array([]), alpha=0.05)
    assert out["reject"].shape == (0,)
    assert out["p_adjusted"].shape == (0,)


# --------------------------------------------------- two_sample_ttest_matrix


def test_two_sample_ttest_matrix_shapes_and_finite() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((50, 8))
    b = rng.standard_normal((40, 8))
    t, p = two_sample_ttest_matrix(a, b, axis=0)
    assert t.shape == (8,)
    assert p.shape == (8,)
    assert np.isfinite(t).all()
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_two_sample_ttest_matrix_detects_signal() -> None:
    rng = np.random.default_rng(0)
    n = 400
    d = 4
    a = rng.standard_normal((n, d))
    b = rng.standard_normal((n, d))
    b[:, 0] += 0.5  # plant a moderate effect on feature 0
    _, p = two_sample_ttest_matrix(a, b, axis=0)
    assert p[0] < 0.001
    # The other features should look null-ish.
    assert (p[1:] > 0.01).all() or (p[1:].mean() > 0.05)


# --------------------------------------------------- one_sample_zscore_effect


def test_one_sample_zscore_effect_shape_and_value() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 3)) + np.array([0.0, 0.5, -0.5])
    z = one_sample_zscore_effect(x, axis=0)
    assert z.shape == (3,)
    assert abs(z[0]) < 2.0
    assert z[1] > 2.0
    assert z[2] < -2.0


# ----------------------------------------------------- permutation_pvalues


def test_permutation_pvalues_reproducible_with_seed() -> None:
    # Weak (null-ish) signal so the p-value is not pinned to the
    # minimum 1 / (B + 1); different seeds will then give different
    # counts of null permutations exceeding the observed difference.
    rng = np.random.default_rng(0)
    a = rng.standard_normal(40)
    b = rng.standard_normal(35) + 0.05
    p1 = permutation_pvalues(a, b, n_resamples=200, seed=42)
    p2 = permutation_pvalues(a, b, n_resamples=200, seed=42)
    p3 = permutation_pvalues(a, b, n_resamples=200, seed=43)
    assert p1 == p2
    assert p1 != p3


def test_permutation_pvalues_finds_signal() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(60)
    b = rng.standard_normal(60) + 0.8
    p = permutation_pvalues(a, b, n_resamples=400, seed=0)
    assert 0.0 < p < 0.05


def test_permutation_pvalues_null_calibration() -> None:
    rng = np.random.default_rng(0)
    rejects = 0
    trials = 50
    for t in range(trials):
        a = rng.standard_normal(40)
        b = rng.standard_normal(40)
        if permutation_pvalues(a, b, n_resamples=199, seed=t) < 0.10:
            rejects += 1
    assert rejects / trials < 0.30


# --------------------------------------------------------- discovery metrics


def test_discovery_counts_basic() -> None:
    truth = np.array([True, True, False, False, True, False])
    rej   = np.array([True, False, True, False, True, True])
    assert true_discovery_count(rej, truth) == 2  # positions 0, 4
    assert false_discovery_count(rej, truth) == 2  # positions 2, 5
    np.testing.assert_allclose(
        false_discovery_proportion(rej, truth), 2 / 4
    )
    np.testing.assert_allclose(power(rej, truth), 2 / 3)


def test_discovery_metrics_no_rejections() -> None:
    truth = np.array([True, False, True])
    rej = np.array([False, False, False])
    assert true_discovery_count(rej, truth) == 0
    assert false_discovery_count(rej, truth) == 0
    np.testing.assert_allclose(false_discovery_proportion(rej, truth), 0.0)
    np.testing.assert_allclose(power(rej, truth), 0.0)


def test_discovery_metrics_all_truth_zero() -> None:
    truth = np.array([False, False, False])
    rej = np.array([True, False, True])
    np.testing.assert_allclose(power(rej, truth), 0.0)
    np.testing.assert_allclose(false_discovery_proportion(rej, truth), 1.0)


def test_summarize_discoveries_keys() -> None:
    rng = np.random.default_rng(0)
    n = 200
    truth = rng.uniform(size=n) < 0.05
    p = np.where(truth, rng.uniform(0, 0.001, size=n), rng.uniform(size=n))
    s = summarize_discoveries(p, truth, alpha=0.05)
    for key in ("bonferroni", "benjamini_hochberg"):
        for sub in ("true_discoveries", "false_discoveries", "fdp", "power"):
            assert sub in s[key]
    assert s["bonferroni"]["true_discoveries"] >= 0
    assert s["benjamini_hochberg"]["true_discoveries"] >= 0
