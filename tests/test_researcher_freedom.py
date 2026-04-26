"""Tests for src/researcher_freedom.py."""
from __future__ import annotations

import numpy as np

from src.researcher_freedom import (
    run_single_analysis,
    simulate_p_hacking_curve,
    sweep_false_positives,
    sweep_power_fdr,
)


# ----------------------------------------------------- run_single_analysis


def test_single_analysis_keys_and_shapes() -> None:
    out = run_single_analysis(
        n_samples=40, n_features=30, n_signal=0,
        effect_size=0.0, alpha=0.05, correction="none", seed=0,
    )
    for key in ("p_values", "reject", "truth", "n_tested", "n_reject",
                "false_discoveries", "true_discoveries", "fdp", "power",
                "correction", "choose_best_subset", "metadata"):
        assert key in out
    assert out["p_values"].ndim == 1
    assert out["reject"].ndim == 1
    assert out["truth"].ndim == 1
    assert out["p_values"].size == out["n_tested"]
    assert out["reject"].size == out["n_tested"]
    assert out["truth"].size == out["n_tested"]


def test_single_analysis_subset_fraction_truncates() -> None:
    out = run_single_analysis(
        n_samples=40, n_features=20, n_signal=0,
        effect_size=0.0, alpha=0.05, correction="none", seed=0,
        subset_fraction=0.5,
    )
    assert out["n_tested"] == 10
    assert out["metadata"]["subset_indices"].shape == (10,)


def test_single_analysis_reproducible() -> None:
    kw = dict(n_samples=40, n_features=20, n_signal=2, effect_size=0.5,
              alpha=0.05, correction="bh_fdr", seed=7,
              subset_fraction=0.6, choose_best_subset=True,
              n_candidate_subsets=10)
    a = run_single_analysis(**kw)
    b = run_single_analysis(**kw)
    np.testing.assert_array_equal(a["p_values"], b["p_values"])
    np.testing.assert_array_equal(a["reject"], b["reject"])
    np.testing.assert_array_equal(a["truth"], b["truth"])
    np.testing.assert_array_equal(
        a["metadata"]["subset_indices"], b["metadata"]["subset_indices"]
    )


def test_single_analysis_no_correction_matches_threshold() -> None:
    out = run_single_analysis(
        n_samples=80, n_features=50, n_signal=0,
        effect_size=0.0, alpha=0.05, correction="none", seed=0,
    )
    np.testing.assert_array_equal(out["reject"], out["p_values"] < 0.05)


def test_single_analysis_bonferroni_kills_pure_null() -> None:
    out = run_single_analysis(
        n_samples=60, n_features=200, n_signal=0,
        effect_size=0.0, alpha=0.05, correction="bonferroni", seed=0,
    )
    # Bonferroni at alpha=0.05 over 200 nulls should rarely reject anything.
    assert out["false_discoveries"] <= 1


def test_single_analysis_signal_and_metrics() -> None:
    out = run_single_analysis(
        n_samples=200, n_features=20, n_signal=4, effect_size=1.5,
        alpha=0.05, correction="none", seed=0, signal_pattern="block",
    )
    # The first four features carry signal; truth is aligned with the
    # full feature index when no subset is taken.
    assert int(out["truth"][:4].sum()) == 4
    assert out["true_discoveries"] >= 3
    assert 0.0 <= out["fdp"] <= 1.0
    assert 0.0 <= out["power"] <= 1.0


def test_single_analysis_choose_best_subset_inflates_under_null() -> None:
    rng = np.random.default_rng(0)
    n_repeats = 20
    base = []
    best = []
    for _ in range(n_repeats):
        seed = int(rng.integers(1 << 30))
        base.append(run_single_analysis(
            n_samples=60, n_features=80, n_signal=0,
            effect_size=0.0, alpha=0.05, correction="none", seed=seed,
            subset_fraction=0.5, choose_best_subset=False,
        )["n_reject"])
        best.append(run_single_analysis(
            n_samples=60, n_features=80, n_signal=0,
            effect_size=0.0, alpha=0.05, correction="none", seed=seed,
            subset_fraction=0.5, choose_best_subset=True,
            n_candidate_subsets=15,
        )["n_reject"])
    assert np.mean(best) > np.mean(base)


# --------------------------------------------------- sweep_false_positives


def test_sweep_false_positives_shapes_and_keys() -> None:
    out = sweep_false_positives(
        feature_counts=[20, 50, 100], alpha=0.05,
        n_repeats=20, seed=0,
    )
    assert out["feature_counts"].shape == (3,)
    for method in ("none", "bonferroni", "bh_fdr"):
        sub = out["results"][method]
        for stat in ("mean", "median", "p05", "p95", "any_fp"):
            assert sub[stat].shape == (3,)


def test_sweep_false_positives_no_correction_calibrated() -> None:
    out = sweep_false_positives(
        feature_counts=[200], alpha=0.05,
        n_repeats=200, seed=0,
        correction_methods=("none",),
    )
    mean_fp = out["results"]["none"]["mean"][0]
    assert abs(mean_fp - 0.05 * 200) < 2.0


def test_sweep_false_positives_bonferroni_controls_fwer() -> None:
    out = sweep_false_positives(
        feature_counts=[100], alpha=0.05,
        n_repeats=300, seed=0,
        correction_methods=("bonferroni",),
    )
    any_fp = out["results"]["bonferroni"]["any_fp"][0]
    assert any_fp <= 0.07  # close to alpha=0.05


# ----------------------------------------------------- sweep_power_fdr


def test_sweep_power_fdr_shapes() -> None:
    out = sweep_power_fdr(
        effect_sizes=[0.0, 0.5, 1.0],
        n_signal_values=[2, 5],
        n_features=30, n_repeats=10, seed=0,
    )
    assert out["effect_sizes"].shape == (3,)
    assert out["n_signal_values"].shape == (2,)
    for method in ("none", "bonferroni", "bh_fdr"):
        sub = out["results"][method]
        assert sub["fdp_mean"].shape == (3, 2)
        assert sub["power_mean"].shape == (3, 2)


def test_sweep_power_fdr_power_grows_with_effect() -> None:
    out = sweep_power_fdr(
        effect_sizes=[0.0, 1.5],
        n_signal_values=[3],
        n_features=30, n_repeats=20, seed=0,
        correction_methods=("none",),
    )
    p = out["results"]["none"]["power_mean"][:, 0]
    assert p[1] > p[0]


# ------------------------------------------------ simulate_p_hacking_curve


def test_p_hacking_curve_shapes_and_keys() -> None:
    out = simulate_p_hacking_curve(
        n_choices=[1, 5, 20, 50],
        n_samples=40, n_features=40, alpha=0.05, seed=0,
    )
    assert out["n_choices"].shape == (4,)
    assert out["best_raw"].shape == (4,)
    assert out["best_corrected"].shape == (4,)


def test_p_hacking_curve_reproducible() -> None:
    kw = dict(n_choices=[1, 5, 10], n_samples=40, n_features=40,
              alpha=0.05, seed=11)
    a = simulate_p_hacking_curve(**kw)
    b = simulate_p_hacking_curve(**kw)
    np.testing.assert_array_equal(a["best_raw"], b["best_raw"])
    np.testing.assert_array_equal(a["best_corrected"], b["best_corrected"])


def test_p_hacking_curve_best_raw_non_decreasing_on_average() -> None:
    # Trying more configurations on the same data can only increase
    # the maximum count of naive rejections (it is a max over a growing
    # set of candidates).
    out = simulate_p_hacking_curve(
        n_choices=[1, 5, 25, 100],
        n_samples=40, n_features=80, alpha=0.05, seed=0,
    )
    raw = out["best_raw"]
    assert (np.diff(raw) >= 0).all()


def test_p_hacking_curve_corrected_does_not_explode() -> None:
    out = simulate_p_hacking_curve(
        n_choices=[1, 5, 25, 100],
        n_samples=60, n_features=80, alpha=0.05, seed=1,
    )
    # Bonferroni at choice level should keep corrected counts modest
    # under the null; certainly not exceed half the subset size.
    assert (out["best_corrected"] <= out["subset_size"] // 2).all()
