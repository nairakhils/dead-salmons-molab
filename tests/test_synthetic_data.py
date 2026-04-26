"""Tests for src/synthetic_data.py."""
from __future__ import annotations

import numpy as np

from src.synthetic_data import (
    feature_labels,
    make_interpretability_grid,
    make_null_feature_matrix,
    make_two_group_data,
)


# --------------------------------------------- make_null_feature_matrix


def test_null_matrix_shape() -> None:
    X = make_null_feature_matrix(n_samples=80, n_features=12, seed=0)
    assert X.shape == (80, 12)


def test_null_matrix_no_nans() -> None:
    X = make_null_feature_matrix(n_samples=200, n_features=20, seed=1)
    assert not np.isnan(X).any()


def test_null_matrix_seed_reproducible() -> None:
    X1 = make_null_feature_matrix(n_samples=50, n_features=10, seed=42)
    X2 = make_null_feature_matrix(n_samples=50, n_features=10, seed=42)
    np.testing.assert_array_equal(X1, X2)


def test_null_matrix_seed_changes_output() -> None:
    X1 = make_null_feature_matrix(n_samples=50, n_features=10, seed=1)
    X2 = make_null_feature_matrix(n_samples=50, n_features=10, seed=2)
    assert not np.allclose(X1, X2)


def test_null_matrix_features_approximately_standardised() -> None:
    X = make_null_feature_matrix(n_samples=4000, n_features=8, seed=0)
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(X.std(axis=0), 1.0, atol=0.05)


def test_null_matrix_correlation_increases_off_diagonal() -> None:
    X_indep = make_null_feature_matrix(
        n_samples=4000, n_features=8, seed=0, correlation=0.0
    )
    X_corr = make_null_feature_matrix(
        n_samples=4000, n_features=8, seed=0, correlation=0.7
    )

    def _avg_offdiag_abs(X: np.ndarray) -> float:
        C = np.corrcoef(X, rowvar=False)
        m = C.shape[0]
        mask = ~np.eye(m, dtype=bool)
        return float(np.abs(C[mask]).mean())

    assert _avg_offdiag_abs(X_corr) > _avg_offdiag_abs(X_indep) + 0.2


# ------------------------------------------------ make_two_group_data


def test_two_group_shapes() -> None:
    out = make_two_group_data(
        n_a=30, n_b=40, n_features=20, n_signal=5,
        effect_size=0.5, seed=0,
    )
    assert out["group_a"].shape == (30, 20)
    assert out["group_b"].shape == (40, 20)
    assert out["truth"].shape == (20,)
    assert out["effect"].shape == (20,)


def test_two_group_truth_count_block() -> None:
    out = make_two_group_data(
        n_a=20, n_b=20, n_features=30, n_signal=7,
        effect_size=0.5, seed=0, signal_pattern="block",
    )
    assert int(out["truth"].sum()) == 7
    np.testing.assert_array_equal(
        out["truth"][:7], np.ones(7, dtype=bool)
    )


def test_two_group_truth_count_random() -> None:
    out = make_two_group_data(
        n_a=20, n_b=20, n_features=50, n_signal=11,
        effect_size=0.5, seed=2, signal_pattern="random",
    )
    assert int(out["truth"].sum()) == 11


def test_two_group_truth_count_clustered() -> None:
    out = make_two_group_data(
        n_a=20, n_b=20, n_features=40, n_signal=8,
        effect_size=0.5, seed=3, signal_pattern="clustered",
    )
    assert int(out["truth"].sum()) == 8
    # Clustered means contiguous: True positions should be a single run.
    where = np.where(out["truth"])[0]
    assert (np.diff(where) == 1).all()


def test_two_group_effect_only_on_signal_features() -> None:
    out = make_two_group_data(
        n_a=20, n_b=20, n_features=30, n_signal=5,
        effect_size=0.4, seed=0, signal_pattern="block",
    )
    np.testing.assert_array_equal(out["effect"][:5] != 0, np.ones(5, dtype=bool))
    np.testing.assert_array_equal(out["effect"][5:], np.zeros(25))


def test_two_group_no_nans() -> None:
    out = make_two_group_data(
        n_a=40, n_b=40, n_features=20, n_signal=4,
        effect_size=0.5, seed=0,
    )
    assert not np.isnan(out["group_a"]).any()
    assert not np.isnan(out["group_b"]).any()


def test_two_group_seed_reproducible() -> None:
    kw = dict(n_a=20, n_b=20, n_features=30, n_signal=5,
              effect_size=0.4, seed=7, signal_pattern="random")
    o1 = make_two_group_data(**kw)
    o2 = make_two_group_data(**kw)
    np.testing.assert_array_equal(o1["group_a"], o2["group_a"])
    np.testing.assert_array_equal(o1["group_b"], o2["group_b"])
    np.testing.assert_array_equal(o1["truth"], o2["truth"])


def test_two_group_signal_actually_separates() -> None:
    # With a strong effect on the signal features, group_b means should
    # exceed group_a means there.
    out = make_two_group_data(
        n_a=200, n_b=200, n_features=10, n_signal=3,
        effect_size=2.0, seed=0, signal_pattern="block",
    )
    diff = out["group_b"].mean(axis=0) - out["group_a"].mean(axis=0)
    assert (diff[:3] > 0.5).all()
    assert (np.abs(diff[3:]) < 0.5).all()


# --------------------------------------------- make_interpretability_grid


def test_grid_shapes() -> None:
    g = make_interpretability_grid(
        height=12, width=14, n_signal_regions=2,
        effect_size=0.5, seed=0,
    )
    assert g["p_values"].shape == (12, 14)
    assert g["truth"].shape == (12, 14)
    assert g["effect"].shape == (12, 14)


def test_grid_null_only_truth_all_false() -> None:
    g = make_interpretability_grid(
        height=10, width=10, n_signal_regions=3,
        effect_size=0.5, seed=0, null_only=True,
    )
    assert not g["truth"].any()
    np.testing.assert_array_equal(g["effect"], np.zeros((10, 10)))


def test_grid_null_only_pvalues_uniform_on_average() -> None:
    g = make_interpretability_grid(
        height=80, width=80, n_signal_regions=0,
        effect_size=0.0, seed=0, null_only=True,
    )
    p = g["p_values"]
    assert (p >= 0).all() and (p <= 1).all()
    assert abs(p.mean() - 0.5) < 0.03
    assert abs(np.mean(p < 0.05) - 0.05) < 0.02


def test_grid_signal_regions_have_lower_pvalues() -> None:
    g = make_interpretability_grid(
        height=20, width=20, n_signal_regions=2,
        effect_size=2.0, seed=0, null_only=False,
    )
    p = g["p_values"]
    truth = g["truth"]
    if truth.any():
        assert p[truth].mean() < p[~truth].mean()


def test_grid_no_nans() -> None:
    g = make_interpretability_grid(
        height=15, width=15, n_signal_regions=1,
        effect_size=0.5, seed=0,
    )
    assert not np.isnan(g["p_values"]).any()
    assert not np.isnan(g["effect"]).any()


def test_grid_seed_reproducible() -> None:
    g1 = make_interpretability_grid(
        height=10, width=10, n_signal_regions=2, effect_size=1.0, seed=5
    )
    g2 = make_interpretability_grid(
        height=10, width=10, n_signal_regions=2, effect_size=1.0, seed=5
    )
    np.testing.assert_array_equal(g1["p_values"], g2["p_values"])
    np.testing.assert_array_equal(g1["truth"], g2["truth"])


# ------------------------------------------------------- feature_labels


def test_feature_labels_count_and_format() -> None:
    labs = feature_labels(5, kind="neuron")
    assert len(labs) == 5
    assert labs[0] == "neuron_0001"
    assert labs[-1] == "neuron_0005"


def test_feature_labels_kinds() -> None:
    for kind in ("neuron", "token", "head", "feature"):
        labs = feature_labels(3, kind=kind)
        assert all(lab.startswith(kind + "_") for lab in labs)


def test_feature_labels_zero_padding_grows() -> None:
    labs = feature_labels(15000, kind="feature")
    # Padding must accommodate the largest index without overflow.
    assert labs[-1] == "feature_15000"
    # All labels share width.
    assert len({len(lab) for lab in labs}) == 1
