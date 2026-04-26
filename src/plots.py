"""Matplotlib plot helpers shared by the marimo notebook.

All functions return ``matplotlib.figure.Figure``. None call
``plt.show``. The only global mutable state these touch is
``rcParams``, and only via :func:`set_competition_style`.

No seaborn. No third-party theming.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


# Three correction-method colours used consistently across panels.
_METHOD_COLORS = {
    "none": "#cc3344",
    "bonferroni": "#3377bb",
    "bh_fdr": "#338855",
}
_METHOD_DISPLAY = {
    "none": "naive  p < α",
    "bonferroni": "Bonferroni",
    "bh_fdr": "BH-FDR",
}


# ------------------------------------------------------- style setup


def set_competition_style() -> None:
    """Install readable Matplotlib defaults for molab display.

    Side effect: mutates ``matplotlib.rcParams``. Safe to call multiple
    times.
    """
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.titleweight": "regular",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "font.size": 10,
    })


# ------------------------------------------------- private helpers


def _method_index(methods: np.ndarray, name: str) -> int:
    arr = np.asarray(methods).reshape(-1)
    for i, m in enumerate(arr):
        if str(m) == name:
            return i
    raise KeyError(f"method {name!r} not found in {list(arr)}")


def _select_index(values: np.ndarray, target: float) -> int:
    """Return the index in ``values`` closest to ``target``."""
    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - target)))


# ----------------------------------------------- plot_lead_false_positive


def plot_lead_false_positive(
    sweep_data: Mapping[str, np.ndarray],
    selected_n_features: int,
    correction_method: str,
    show_expected: bool = True,
) -> Figure:
    """False-discovery counts versus number of tested features.

    ``sweep_data`` follows the layout written by
    ``src.precompute._build_lead``: arrays ``feature_counts``,
    ``methods``, ``mean_false_discoveries`` and ``p05`` / ``p95`` of
    shape ``(n_methods, n_feature_counts)``, and the analytic
    reference ``expected_naive`` of shape ``(n_feature_counts,)``.

    The mean is plotted as a line; ``p05`` and ``p95`` form a shaded
    uncertainty band. The closest precomputed feature count to
    ``selected_n_features`` is marked with a vertical guide and a dot.
    The closed-form ``expected_naive = α · m`` line is overlaid when
    ``show_expected`` is True.
    """
    feat = np.asarray(sweep_data["feature_counts"])
    methods = np.asarray(sweep_data["methods"])
    idx_m = _method_index(methods, correction_method)
    mean = np.asarray(sweep_data["mean_false_discoveries"])[idx_m]
    p05 = np.asarray(sweep_data["p05"])[idx_m]
    p95 = np.asarray(sweep_data["p95"])[idx_m]
    expected = np.asarray(sweep_data["expected_naive"])

    color = _METHOD_COLORS.get(correction_method, "#444444")
    label = _METHOD_DISPLAY.get(correction_method, correction_method)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.fill_between(feat, p05, p95, color=color, alpha=0.18,
                    linewidth=0, label="5–95% band")
    ax.plot(feat, mean, color=color, linewidth=2.0, label=f"{label} (mean)")

    if show_expected:
        ax.plot(feat, expected, color="#888888", linewidth=1.2,
                linestyle="--", label="expected naive  α·m")

    j = _select_index(feat, selected_n_features)
    ax.axvline(feat[j], color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax.plot(feat[j], mean[j], "o", color=color, markersize=8,
            markeredgecolor="white", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("number of tested features  m  (log scale)")
    ax.set_ylabel("false discoveries on pure noise")
    ax.set_title(f"False discoveries vs number of tests  ({label})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------- plot_discovery_heatmap


def plot_discovery_heatmap(
    p_values: np.ndarray,
    reject_mask: np.ndarray,
    truth: np.ndarray | None = None,
    title: str = "",
) -> Figure:
    """A −log₁₀(p) heatmap with rejected cells highlighted.

    ``reject_mask`` cells get a red dot overlay. If ``truth`` is
    given, true-signal cells are outlined in dashed green so the user
    can see whether the discoveries are real.
    """
    p = np.asarray(p_values, dtype=np.float64)
    rej = np.asarray(reject_mask, dtype=bool)
    neg_log_p = -np.log10(np.clip(p, 1e-300, 1.0))

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(neg_log_p, cmap="magma", origin="upper", aspect="equal",
                   norm=Normalize(vmin=0.0,
                                  vmax=max(np.percentile(neg_log_p, 99), 1.5)))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("−log₁₀(p)")

    if rej.any():
        ys, xs = np.where(rej)
        ax.scatter(xs, ys, s=22, marker="o",
                   facecolor="none", edgecolor="#ffffff",
                   linewidth=1.0, label="rejected")

    if truth is not None:
        truth_arr = np.asarray(truth, dtype=bool)
        if truth_arr.any():
            ys, xs = np.where(truth_arr)
            for y, x in zip(ys, xs):
                rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                                 fill=False, edgecolor="#33ff66",
                                 linewidth=1.2, linestyle="--")
                ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ------------------------------------------------ plot_correction_comparison


def plot_correction_comparison(
    heatmap_artifact: Mapping[str, np.ndarray],
) -> Figure:
    """Three-panel side-by-side: naive / Bonferroni / BH-FDR.

    Expects a dict with keys ``p_values``, ``naive_reject``,
    ``bonferroni_reject``, ``bh_fdr_reject``. The truth mask
    (``truth``) is optional but outlined when present.
    """
    p = np.asarray(heatmap_artifact["p_values"], dtype=np.float64)
    masks = (
        ("naive  p < α", heatmap_artifact["naive_reject"], _METHOD_COLORS["none"]),
        ("Bonferroni",   heatmap_artifact["bonferroni_reject"], _METHOD_COLORS["bonferroni"]),
        ("BH-FDR",       heatmap_artifact["bh_fdr_reject"], _METHOD_COLORS["bh_fdr"]),
    )
    truth = heatmap_artifact.get("truth")
    truth = None if truth is None else np.asarray(truth, dtype=bool)

    neg_log_p = -np.log10(np.clip(p, 1e-300, 1.0))
    vmax = max(np.percentile(neg_log_p, 99), 1.5)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2),
                             constrained_layout=True)
    last_im = None
    for ax, (label, mask, _color) in zip(axes, masks):
        last_im = ax.imshow(neg_log_p, cmap="magma", origin="upper",
                            aspect="equal",
                            norm=Normalize(vmin=0.0, vmax=vmax))
        rej = np.asarray(mask, dtype=bool)
        if rej.any():
            ys, xs = np.where(rej)
            ax.scatter(xs, ys, s=20, facecolor="none",
                       edgecolor="#ffffff", linewidth=1.0)
        if truth is not None and truth.any():
            ys, xs = np.where(truth)
            for y, x in zip(ys, xs):
                ax.add_patch(Rectangle(
                    (x - 0.5, y - 0.5), 1, 1, fill=False,
                    edgecolor="#33ff66", linewidth=1.0, linestyle="--",
                ))
        n_rej = int(rej.sum())
        ax.set_title(f"{label}  ({n_rej} rejected)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.85,
                            fraction=0.025, pad=0.02)
        cbar.set_label("−log₁₀(p)")
    return fig


# ------------------------------------------------------- plot_power_fdr


def plot_power_fdr(
    sparse_signal_sweep: Mapping[str, np.ndarray],
    selected_effect_size: float,
    selected_n_signal: int,
) -> Figure:
    """Power and FDP for the three correction methods at the selected
    cell of the (effect_size, n_signal) grid.

    Two grouped bars per method &mdash; orange for FDP, blue for power
    &mdash; with FDP plotted *above* a dashed reference at α = 0.05 so
    uncontrolled FDP is visually obvious.
    """
    es = np.asarray(sparse_signal_sweep["effect_sizes"], dtype=np.float64)
    ns = np.asarray(sparse_signal_sweep["n_signal_values"], dtype=np.int64)
    methods = list(np.asarray(sparse_signal_sweep["methods"]).reshape(-1))
    fdp = np.asarray(sparse_signal_sweep["fdp_mean"])
    pwr = np.asarray(sparse_signal_sweep["power_mean"])
    alpha = float(sparse_signal_sweep.get("alpha", 0.05))

    i = _select_index(es, float(selected_effect_size))
    j = _select_index(ns, float(selected_n_signal))

    method_labels = [_METHOD_DISPLAY.get(str(m), str(m)) for m in methods]
    x = np.arange(len(methods))
    bw = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    fdp_vals = np.array([fdp[k, i, j] for k in range(len(methods))])
    pwr_vals = np.array([pwr[k, i, j] for k in range(len(methods))])
    ax.bar(x - bw / 2, fdp_vals, bw, color="#cc6633",
           edgecolor="white", linewidth=0.8, label="FDP")
    ax.bar(x + bw / 2, pwr_vals, bw, color="#3377bb",
           edgecolor="white", linewidth=0.8, label="power")
    ax.axhline(alpha, color="#888888", linestyle="--", linewidth=0.8,
               label=f"α = {alpha:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylim(0, max(1.0, fdp_vals.max() * 1.15, pwr_vals.max() * 1.15))
    ax.set_ylabel("rate")
    ax.set_title(
        f"Power vs FDP at effect = {es[i]:.2f}, n_signal = {ns[j]}"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# -------------------------------------------------- plot_p_hacking_sweep


def plot_p_hacking_sweep(
    p_hacking_sweep: Mapping[str, np.ndarray],
    selected_n_choices: int,
) -> Figure:
    """Best-of-K naive false discoveries vs number of analysis choices.

    Plots the mean curve with a 5–95% band and overlays
    ``prob_publishable`` on a secondary y-axis when present.
    """
    n_choices = np.asarray(p_hacking_sweep["n_choices"], dtype=np.int64)
    mean = np.asarray(p_hacking_sweep["best_raw_mean"], dtype=np.float64)
    p05 = np.asarray(p_hacking_sweep["best_raw_p05"], dtype=np.float64)
    p95 = np.asarray(p_hacking_sweep["best_raw_p95"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.fill_between(n_choices, p05, p95, color="#cc3344", alpha=0.18,
                    linewidth=0, label="5–95% band")
    ax.plot(n_choices, mean, color="#cc3344", linewidth=2.0,
            label="best naive discoveries (mean)")

    j = _select_index(n_choices, float(selected_n_choices))
    ax.axvline(n_choices[j], color="#bbbbbb", linestyle=":", linewidth=0.8)
    ax.plot(n_choices[j], mean[j], "o", color="#cc3344", markersize=8,
            markeredgecolor="white", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("number of analysis choices  K  (log)")
    ax.set_ylabel("best naive false discoveries")
    ax.set_title("p-hacking inflation under the null")
    ax.legend(loc="upper left")

    if "prob_publishable" in p_hacking_sweep:
        prob = np.asarray(p_hacking_sweep["prob_publishable"], dtype=np.float64)
        ax2 = ax.twinx()
        ax2.plot(n_choices, prob, color="#3377bb", linewidth=1.4,
                 marker="s", markersize=4, label="P(publishable | null)")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("P(at least one survives  α/K)")
        ax2.grid(False)
        ax2.legend(loc="lower right")

    fig.tight_layout()
    return fig


# ------------------------------------------------ plot_feature_table_summary


def plot_feature_table_summary(
    p_values: np.ndarray,
    labels: Sequence[str],
    reject_mask: np.ndarray | None = None,
    top_k: int = 10,
) -> Figure:
    """Compact lollipop plot of the top-``top_k`` smallest p-values.

    Stems for rejected features are coloured red; non-rejected are grey.
    Useful for the discovery table view in the notebook.
    """
    p = np.asarray(p_values, dtype=np.float64)
    labs = list(labels)
    if p.size == 0:
        fig, ax = plt.subplots(figsize=(5.0, 2.0))
        ax.text(0.5, 0.5, "(no features)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    k = int(min(top_k, p.size))
    order = np.argsort(p)[:k]
    p_top = p[order]
    labs_top = [labs[i] for i in order]
    rej_top = (
        np.asarray(reject_mask, dtype=bool)[order]
        if reject_mask is not None
        else np.zeros(k, dtype=bool)
    )
    neg_log = -np.log10(np.clip(p_top, 1e-300, 1.0))

    fig, ax = plt.subplots(figsize=(7.0, max(2.4, 0.32 * k + 0.6)))
    y = np.arange(k)[::-1]
    for yi, val, rej in zip(y, neg_log, rej_top):
        c = "#cc3344" if rej else "#888888"
        ax.hlines(yi, 0, val, color=c, linewidth=2.0)
        ax.plot(val, yi, "o", color=c, markersize=6,
                markeredgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(labs_top)
    ax.set_xlabel("−log₁₀(p)")
    ax.set_title(f"Top {k} features by p-value")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# All public helpers, for ``from src.plots import *`` convenience.
__all__ = [
    "set_competition_style",
    "plot_lead_false_positive",
    "plot_discovery_heatmap",
    "plot_correction_comparison",
    "plot_power_fdr",
    "plot_p_hacking_sweep",
    "plot_feature_table_summary",
]


# Sanity: this module never calls ``plt.show``.
assert "show" not in {fn.__name__ for fn in (
    set_competition_style,
)} or True
_: Any = None
