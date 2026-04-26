"""Synthetic data generators for the multiple-testing demos.

Two regimes are supported:

- **Pure null:** every "feature" (token / neuron / head / channel) is
  drawn from N(0, 1) with no group difference. Naive significance
  testing should produce a known false-positive rate proportional to
  ``alpha``.
- **Sparse signal:** a small subset of features carries a real effect;
  the rest are null. This is the standard interpretability setup
  where most components have no causal role.

Pure NumPy. No global state. Seed-deterministic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


# ----------------------------------------------- make_null_feature_matrix


def _raw_correlated_normal(
    n_samples: int,
    n_features: int,
    correlation: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Raw (unstandardised) correlated normal samples.

    With ``correlation = 0``, each column is independent N(0, 1). With
    ``correlation = rho > 0``, each column is
    ``sqrt(1 - rho) * eps_j + sqrt(rho) * latent``, so every column
    has unit variance in expectation and any pair of columns has
    correlation rho in expectation. Sampling fluctuations are kept;
    callers that want exactly-standardised columns should standardise
    after the fact.
    """
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must be in [0, 1)")
    eps = rng.standard_normal((n_samples, n_features))
    if correlation == 0.0:
        return eps
    latent = rng.standard_normal((n_samples, 1))
    return np.sqrt(1.0 - correlation) * eps + np.sqrt(correlation) * latent


def make_null_feature_matrix(
    n_samples: int,
    n_features: int,
    seed: int,
    correlation: float = 0.0,
) -> NDArray[np.float64]:
    """A synthetic feature matrix with no group structure.

    With ``correlation = 0`` each column is independent N(0, 1). With
    ``correlation > 0`` each column is a mixture of an independent
    component and a single shared latent factor:

        ``X[:, j] = sqrt(1 - rho) * eps_j + sqrt(rho) * latent``

    so every column has unit variance and any pair of columns has
    correlation ``rho``. Each column is then re-standardised to zero
    mean and unit variance to absorb the finite-sample drift &mdash;
    use :func:`make_two_group_data` if you need group-to-group
    sampling differences preserved.
    """
    rng = np.random.default_rng(seed)
    X = _raw_correlated_normal(n_samples, n_features, correlation, rng)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    X = X / std
    return X


# --------------------------------------------------- make_two_group_data


def _signal_indices(
    n_features: int, n_signal: int, pattern: str, rng: np.random.Generator
) -> NDArray[np.int64]:
    if n_signal < 0 or n_signal > n_features:
        raise ValueError("n_signal must be in [0, n_features]")
    if pattern == "block":
        return np.arange(n_signal, dtype=np.int64)
    if pattern == "random":
        return np.sort(rng.choice(n_features, size=n_signal, replace=False))
    if pattern == "clustered":
        if n_signal == 0:
            return np.empty(0, dtype=np.int64)
        max_start = n_features - n_signal
        start = int(rng.integers(0, max_start + 1))
        return np.arange(start, start + n_signal, dtype=np.int64)
    raise ValueError(f"unknown signal_pattern {pattern!r}")


def make_two_group_data(
    n_a: int,
    n_b: int,
    n_features: int,
    n_signal: int,
    effect_size: float,
    seed: int,
    correlation: float = 0.0,
    signal_pattern: str = "block",
) -> dict[str, Any]:
    """Two synthetic feature matrices, signal added only to group_b.

    Both matrices are raw (un-standardised) correlated-normal samples,
    so the per-feature group means retain their natural sampling
    fluctuations. ``effect_size`` is added to group_b's columns at the
    chosen signal positions; the units are the underlying noise
    standard deviation (~1).
    """
    rng = np.random.default_rng(seed)
    rng_a = np.random.default_rng(int(rng.integers(1 << 30)))
    rng_b = np.random.default_rng(int(rng.integers(1 << 30)))
    pat_rng = np.random.default_rng(int(rng.integers(1 << 30)))

    group_a = _raw_correlated_normal(n_a, n_features, correlation, rng_a)
    group_b = _raw_correlated_normal(n_b, n_features, correlation, rng_b)

    idx = _signal_indices(n_features, n_signal, signal_pattern, pat_rng)

    truth = np.zeros(n_features, dtype=bool)
    effect = np.zeros(n_features, dtype=np.float64)
    if idx.size > 0:
        truth[idx] = True
        effect[idx] = float(effect_size)
        group_b[:, idx] = group_b[:, idx] + float(effect_size)

    return {
        "group_a": group_a,
        "group_b": group_b,
        "truth": truth,
        "effect": effect,
        "metadata": {
            "n_a": n_a,
            "n_b": n_b,
            "n_features": n_features,
            "n_signal": n_signal,
            "effect_size": float(effect_size),
            "correlation": float(correlation),
            "signal_pattern": signal_pattern,
            "seed": int(seed),
        },
    }


# ----------------------------------------------- make_interpretability_grid


def _place_blob(
    truth: NDArray[np.bool_],
    effect: NDArray[np.float64],
    rng: np.random.Generator,
    effect_size: float,
) -> None:
    """Stamp a small 2x2 blob at a random location into truth/effect."""
    h, w = truth.shape
    if h < 2 or w < 2:
        i = int(rng.integers(0, h))
        j = int(rng.integers(0, w))
        truth[i, j] = True
        effect[i, j] = effect_size
        return
    i = int(rng.integers(0, h - 1))
    j = int(rng.integers(0, w - 1))
    truth[i:i + 2, j:j + 2] = True
    effect[i:i + 2, j:j + 2] = effect_size


def make_interpretability_grid(
    height: int,
    width: int,
    n_signal_regions: int,
    effect_size: float,
    seed: int,
    null_only: bool = False,
) -> dict[str, NDArray]:
    """Synthetic 2-D heatmap data for visual demos.

    Under ``null_only=True``, ``truth`` is all-False and ``p_values`` is
    Uniform(0, 1) across the grid. Otherwise ``n_signal_regions`` small
    2x2 blobs are stamped at random positions; their pixels get
    ``effect = effect_size`` and lower (more significant) p-values
    drawn from ``Uniform(0, exp(-effect_size))`` clipped to [0, 1].
    """
    rng = np.random.default_rng(seed)
    truth = np.zeros((height, width), dtype=bool)
    effect = np.zeros((height, width), dtype=np.float64)
    p_values = rng.uniform(0.0, 1.0, size=(height, width))

    if not null_only and n_signal_regions > 0:
        for _ in range(n_signal_regions):
            _place_blob(truth, effect, rng, float(effect_size))
        # Lower p-values inside signal regions: smaller upper bound for
        # larger effect sizes.
        upper = float(np.clip(np.exp(-float(effect_size)), 0.0, 1.0))
        if upper <= 0.0:
            upper = 1e-6
        signal_count = int(truth.sum())
        if signal_count > 0:
            p_values[truth] = rng.uniform(0.0, upper, size=signal_count)

    return {
        "p_values": p_values,
        "truth": truth,
        "effect": effect,
    }


# ------------------------------------------------------- feature_labels


def feature_labels(n_features: int, kind: str = "neuron") -> list[str]:
    """Human-readable feature labels for tables and tooltips.

    ``kind`` becomes the prefix; the index is zero-padded to at least
    width 4 (e.g. ``neuron_0001``), or wider if ``n_features`` exceeds
    9999.
    """
    width = max(4, len(str(n_features)))
    return [f"{kind}_{i:0{width}d}" for i in range(1, n_features + 1)]
