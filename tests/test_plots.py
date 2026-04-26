"""Smoke tests for src/plots.py.

We only check that each function returns a ``matplotlib.figure.Figure``
on small dummy data &mdash; no snapshot or visual comparisons.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import plots


# ----------------------------------------------------- shared fixtures


@pytest.fixture
def lead_sweep() -> dict[str, np.ndarray]:
    methods = np.asarray(["none", "bonferroni", "bh_fdr"], dtype=object)
    feat = np.array([20, 50, 100, 200], dtype=np.int64)
    rng = np.random.default_rng(0)
    mean = rng.uniform(0, 5, size=(3, feat.size))
    p05 = mean - 0.5
    p95 = mean + 0.5
    return {
        "feature_counts": feat,
        "methods": methods,
        "mean_false_discoveries": mean,
        "median_false_discoveries": mean,
        "p05": p05,
        "p95": p95,
        "prob_any_false_discovery": np.clip(rng.uniform(size=(3, feat.size)), 0, 1),
        "expected_naive": 0.05 * feat.astype(float),
        "alpha": 0.05,
    }


@pytest.fixture
def heatmap_artifact() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    h, w = 8, 10
    p = rng.uniform(size=(h, w))
    truth = np.zeros((h, w), dtype=bool)
    truth[2:4, 3:6] = True
    p[truth] *= 0.01
    naive = p < 0.05
    bonf = p < 0.05 / (h * w)
    bh = p < 0.01
    return {
        "p_values": p,
        "truth": truth,
        "naive_reject": naive,
        "bonferroni_reject": bonf,
        "bh_fdr_reject": bh,
    }


@pytest.fixture
def sparse_sweep() -> dict[str, np.ndarray]:
    methods = np.asarray(["none", "bonferroni", "bh_fdr"], dtype=object)
    es = np.array([0.0, 0.5, 1.0])
    ns = np.array([5, 20], dtype=np.int64)
    rng = np.random.default_rng(0)
    fdp = rng.uniform(0, 0.5, size=(3, es.size, ns.size))
    pwr = rng.uniform(0, 1, size=(3, es.size, ns.size))
    return {
        "effect_sizes": es,
        "n_signal_values": ns,
        "methods": methods,
        "fdp_mean": fdp,
        "power_mean": pwr,
        "alpha": 0.05,
    }


@pytest.fixture
def p_hacking_sweep() -> dict[str, np.ndarray]:
    n_choices = np.array([1, 2, 5, 10, 20], dtype=np.int64)
    rng = np.random.default_rng(0)
    return {
        "n_choices": n_choices,
        "best_raw_mean": rng.uniform(0, 10, size=n_choices.size),
        "best_raw_median": rng.uniform(0, 10, size=n_choices.size),
        "best_raw_p05": rng.uniform(0, 5, size=n_choices.size),
        "best_raw_p95": rng.uniform(5, 15, size=n_choices.size),
        "prob_publishable": rng.uniform(0, 1, size=n_choices.size),
        "alpha": 0.05,
    }


# ------------------------------------------------------- style setup


def test_set_competition_style_runs() -> None:
    plots.set_competition_style()
    # rcParams should be modified to non-defaults; sanity-check one key.
    assert "figure.facecolor" in plt.rcParams


# ----------------------------------------------- plot_lead_false_positive


def test_plot_lead_false_positive_returns_figure(lead_sweep) -> None:
    fig = plots.plot_lead_false_positive(
        lead_sweep, selected_n_features=100,
        correction_method="bonferroni",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_lead_false_positive_no_expected_line(lead_sweep) -> None:
    fig = plots.plot_lead_false_positive(
        lead_sweep, selected_n_features=200,
        correction_method="none", show_expected=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# --------------------------------------------------- plot_discovery_heatmap


def test_plot_discovery_heatmap_basic(heatmap_artifact) -> None:
    fig = plots.plot_discovery_heatmap(
        p_values=heatmap_artifact["p_values"],
        reject_mask=heatmap_artifact["naive_reject"],
        title="naive",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_discovery_heatmap_with_truth(heatmap_artifact) -> None:
    fig = plots.plot_discovery_heatmap(
        p_values=heatmap_artifact["p_values"],
        reject_mask=heatmap_artifact["bh_fdr_reject"],
        truth=heatmap_artifact["truth"],
        title="BH-FDR",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ------------------------------------------------ plot_correction_comparison


def test_plot_correction_comparison(heatmap_artifact) -> None:
    fig = plots.plot_correction_comparison(heatmap_artifact)
    assert isinstance(fig, plt.Figure)
    # Three panels.
    assert len(fig.axes) >= 3
    plt.close(fig)


# ------------------------------------------------------ plot_power_fdr


def test_plot_power_fdr(sparse_sweep) -> None:
    fig = plots.plot_power_fdr(
        sparse_sweep, selected_effect_size=0.5, selected_n_signal=5
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# -------------------------------------------------- plot_p_hacking_sweep


def test_plot_p_hacking_sweep(p_hacking_sweep) -> None:
    fig = plots.plot_p_hacking_sweep(p_hacking_sweep, selected_n_choices=5)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ----------------------------------------------- plot_feature_table_summary


def test_plot_feature_table_summary_runs() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=8)
    labels = [f"feature_{i+1:04d}" for i in range(8)]
    rej = p < 0.5
    fig = plots.plot_feature_table_summary(
        p_values=p, labels=labels, reject_mask=rej, top_k=5,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
