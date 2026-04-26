"""Researcher-degrees-of-freedom simulation.

Engine for the notebook's interactive p-hacking sandbox. The functions
here let the user manufacture "discoveries" by changing analysis
choices (which subset of features, which correction method, how many
candidate analyses to try) while everything stays seed-deterministic
and fast enough to run live in marimo.

Pure NumPy + SciPy. No statsmodels. Reuses
:mod:`src.synthetic_data` for data generation and
:mod:`src.stats_tests` for tests, corrections, and discovery metrics.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from src.stats_tests import (
    benjamini_hochberg,
    bonferroni,
    false_discovery_count,
    false_discovery_proportion,
    power as discovery_power,
    true_discovery_count,
    two_sample_ttest_matrix,
)
from src.synthetic_data import make_two_group_data


CORRECTION_METHODS = ("none", "bonferroni", "bh_fdr")


# --------------------------------------------------------- helpers


def _apply_correction(
    p_values: NDArray[np.float64], correction: str, alpha: float
) -> NDArray[np.bool_]:
    """Return a boolean rejection mask for the requested correction."""
    if correction == "none":
        return p_values < alpha
    if correction == "bonferroni":
        return bonferroni(p_values, alpha=alpha)["reject"]
    if correction == "bh_fdr":
        return benjamini_hochberg(p_values, alpha=alpha)["reject"]
    raise ValueError(
        f"unknown correction {correction!r}; expected one of {CORRECTION_METHODS}"
    )


def _random_subset(
    n_features: int, subset_size: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Sorted random subset of feature indices."""
    if subset_size <= 0 or subset_size > n_features:
        raise ValueError(
            f"subset_size={subset_size} must be in (0, {n_features}]"
        )
    return np.sort(
        rng.choice(n_features, size=subset_size, replace=False)
    ).astype(np.int64)


# ----------------------------------------------------- run_single_analysis


def run_single_analysis(
    n_samples: int,
    n_features: int,
    n_signal: int,
    effect_size: float,
    alpha: float,
    correction: str,
    seed: int,
    correlation: float = 0.0,
    subset_fraction: float = 1.0,
    choose_best_subset: bool = False,
    n_candidate_subsets: int = 20,
    signal_pattern: str = "block",
) -> dict[str, Any]:
    """Run one analysis and report what the analyst would see.

    The two groups have ``n_samples`` rows each. ``effect_size`` is
    added to ``group_b`` at the ``n_signal`` signal positions.

    If ``subset_fraction < 1.0`` a random subset of features is drawn.
    If ``choose_best_subset=True`` the function draws
    ``n_candidate_subsets`` such subsets, runs the t-test on each, and
    keeps the subset with the most naive ``p < alpha`` rejections.
    The chosen subset's p-values are then run through ``correction``.
    """
    if not 0.0 < subset_fraction <= 1.0:
        raise ValueError("subset_fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    data_seed = int(rng.integers(1 << 30))

    data = make_two_group_data(
        n_a=n_samples, n_b=n_samples, n_features=n_features,
        n_signal=n_signal, effect_size=effect_size, seed=data_seed,
        correlation=correlation, signal_pattern=signal_pattern,
    )
    group_a = data["group_a"]
    group_b = data["group_b"]
    full_truth = data["truth"]

    subset_size = max(1, int(np.floor(subset_fraction * n_features)))

    candidate_counts: list[int] = []
    if choose_best_subset:
        best_idx: NDArray[np.int64] | None = None
        best_count = -1
        for _ in range(n_candidate_subsets):
            cand = _random_subset(n_features, subset_size, rng)
            _, p_cand = two_sample_ttest_matrix(
                group_a[:, cand], group_b[:, cand], axis=0
            )
            count = int(np.sum(p_cand < alpha))
            candidate_counts.append(count)
            if count > best_count:
                best_count = count
                best_idx = cand
        assert best_idx is not None  # n_candidate_subsets >= 1 guaranteed
        chosen_idx = best_idx
    elif subset_fraction < 1.0:
        chosen_idx = _random_subset(n_features, subset_size, rng)
    else:
        chosen_idx = np.arange(n_features, dtype=np.int64)

    _, p_values = two_sample_ttest_matrix(
        group_a[:, chosen_idx], group_b[:, chosen_idx], axis=0
    )
    p_values = np.asarray(p_values, dtype=np.float64)
    truth_subset = full_truth[chosen_idx].astype(bool)

    reject = _apply_correction(p_values, correction, alpha)

    return {
        "p_values": p_values,
        "reject": reject,
        "truth": truth_subset,
        "n_tested": int(p_values.size),
        "n_reject": int(reject.sum()),
        "false_discoveries": false_discovery_count(reject, truth_subset),
        "true_discoveries": true_discovery_count(reject, truth_subset),
        "fdp": false_discovery_proportion(reject, truth_subset),
        "power": discovery_power(reject, truth_subset),
        "correction": correction,
        "choose_best_subset": bool(choose_best_subset),
        "metadata": {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_signal": int(n_signal),
            "effect_size": float(effect_size),
            "alpha": float(alpha),
            "correlation": float(correlation),
            "subset_fraction": float(subset_fraction),
            "subset_size": int(subset_size),
            "subset_indices": chosen_idx,
            "n_candidate_subsets": (
                int(n_candidate_subsets) if choose_best_subset else 0
            ),
            "candidate_raw_counts": (
                np.asarray(candidate_counts, dtype=np.int64)
                if choose_best_subset
                else np.empty(0, dtype=np.int64)
            ),
            "signal_pattern": signal_pattern,
            "seed": int(seed),
        },
    }


# --------------------------------------------------- sweep_false_positives


def _validate_methods(methods: Iterable[str]) -> tuple[str, ...]:
    methods_t = tuple(methods)
    bad = [m for m in methods_t if m not in CORRECTION_METHODS]
    if bad:
        raise ValueError(
            f"unknown correction(s) {bad}; expected from {CORRECTION_METHODS}"
        )
    return methods_t


def sweep_false_positives(
    feature_counts: Sequence[int],
    alpha: float,
    n_repeats: int,
    seed: int,
    correction_methods: Iterable[str] = CORRECTION_METHODS,
    n_samples: int = 40,
) -> dict[str, Any]:
    """Pure-null sweep of false-discovery counts vs feature count.

    For each ``m`` in ``feature_counts`` and each correction method,
    runs ``n_repeats`` independent two-group analyses on pure-null data
    and records the false-discovery count per repeat. Returns the
    mean, median, 5th and 95th percentiles, and the probability of at
    least one false discovery.
    """
    methods = _validate_methods(correction_methods)
    feature_counts_arr = np.asarray(feature_counts, dtype=np.int64)
    M = feature_counts_arr.size
    results: dict[str, dict[str, NDArray[np.float64]]] = {
        m: {
            "mean": np.empty(M, dtype=np.float64),
            "median": np.empty(M, dtype=np.float64),
            "p05": np.empty(M, dtype=np.float64),
            "p95": np.empty(M, dtype=np.float64),
            "any_fp": np.empty(M, dtype=np.float64),
        }
        for m in methods
    }

    rng = np.random.default_rng(seed)
    for j, m in enumerate(feature_counts_arr):
        # Generate one batch of (group_a, group_b, p_values) per repeat,
        # then evaluate every correction method on the same p-values
        # so the comparison is paired and cheap.
        per_method_counts: dict[str, list[int]] = {meth: [] for meth in methods}
        for _ in range(n_repeats):
            data_seed = int(rng.integers(1 << 30))
            data = make_two_group_data(
                n_a=n_samples, n_b=n_samples, n_features=int(m),
                n_signal=0, effect_size=0.0, seed=data_seed,
            )
            _, p_values = two_sample_ttest_matrix(
                data["group_a"], data["group_b"], axis=0
            )
            p_values = np.asarray(p_values, dtype=np.float64)
            for meth in methods:
                rej = _apply_correction(p_values, meth, alpha)
                per_method_counts[meth].append(int(rej.sum()))
        for meth in methods:
            arr = np.asarray(per_method_counts[meth], dtype=np.float64)
            results[meth]["mean"][j] = float(arr.mean())
            results[meth]["median"][j] = float(np.median(arr))
            results[meth]["p05"][j] = float(np.percentile(arr, 5))
            results[meth]["p95"][j] = float(np.percentile(arr, 95))
            results[meth]["any_fp"][j] = float(np.mean(arr >= 1))

    return {
        "feature_counts": feature_counts_arr,
        "alpha": float(alpha),
        "n_repeats": int(n_repeats),
        "n_samples": int(n_samples),
        "results": results,
    }


# ----------------------------------------------------- sweep_power_fdr


def sweep_power_fdr(
    effect_sizes: Sequence[float],
    n_signal_values: Sequence[int],
    n_features: int,
    n_repeats: int,
    seed: int,
    correction_methods: Iterable[str] = CORRECTION_METHODS,
    n_samples: int = 60,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Sparse-signal sweep of mean FDP and mean power.

    Two-dimensional grid over ``effect_sizes`` (E entries) and
    ``n_signal_values`` (S entries). Each cell averages ``n_repeats``
    independent runs.
    """
    methods = _validate_methods(correction_methods)
    es_arr = np.asarray(effect_sizes, dtype=np.float64)
    ns_arr = np.asarray(n_signal_values, dtype=np.int64)
    E, S = es_arr.size, ns_arr.size
    results: dict[str, dict[str, NDArray[np.float64]]] = {
        m: {
            "fdp_mean": np.empty((E, S), dtype=np.float64),
            "power_mean": np.empty((E, S), dtype=np.float64),
        }
        for m in methods
    }

    rng = np.random.default_rng(seed)
    for i, es in enumerate(es_arr):
        for j, ns in enumerate(ns_arr):
            per_method_fdp: dict[str, list[float]] = {meth: [] for meth in methods}
            per_method_pow: dict[str, list[float]] = {meth: [] for meth in methods}
            for _ in range(n_repeats):
                data_seed = int(rng.integers(1 << 30))
                data = make_two_group_data(
                    n_a=n_samples, n_b=n_samples, n_features=n_features,
                    n_signal=int(ns), effect_size=float(es),
                    seed=data_seed, signal_pattern="block",
                )
                _, p_values = two_sample_ttest_matrix(
                    data["group_a"], data["group_b"], axis=0
                )
                p_values = np.asarray(p_values, dtype=np.float64)
                truth = data["truth"]
                for meth in methods:
                    rej = _apply_correction(p_values, meth, alpha)
                    per_method_fdp[meth].append(
                        false_discovery_proportion(rej, truth)
                    )
                    per_method_pow[meth].append(
                        discovery_power(rej, truth)
                    )
            for meth in methods:
                results[meth]["fdp_mean"][i, j] = float(
                    np.mean(per_method_fdp[meth])
                )
                results[meth]["power_mean"][i, j] = float(
                    np.mean(per_method_pow[meth])
                )

    return {
        "effect_sizes": es_arr,
        "n_signal_values": ns_arr,
        "n_features": int(n_features),
        "n_samples": int(n_samples),
        "n_repeats": int(n_repeats),
        "alpha": float(alpha),
        "results": results,
    }


# ------------------------------------------------ simulate_p_hacking_curve


def simulate_p_hacking_curve(
    n_choices: Sequence[int],
    n_samples: int,
    n_features: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Pure-null p-hacking curve.

    Generates one shared (group_a, group_b) at random. For each
    ``K`` in ``n_choices``, draws ``K`` random feature subsets
    (each of size ``n_features // 2``), runs the t-test on each, and
    reports:

    - ``best_raw[i]``: the maximum number of naive ``p < alpha``
      rejections seen across the ``K`` candidate subsets.
    - ``best_corrected[i]``: how many of those rejections survive a
      Bonferroni-at-the-choice-level correction (``p < alpha / K``)
      in the **best** subset.
    """
    rng = np.random.default_rng(seed)
    data_seed = int(rng.integers(1 << 30))
    data = make_two_group_data(
        n_a=n_samples, n_b=n_samples, n_features=n_features,
        n_signal=0, effect_size=0.0, seed=data_seed,
    )
    group_a = data["group_a"]
    group_b = data["group_b"]
    subset_size = max(1, n_features // 2)

    n_choices_arr = np.asarray(n_choices, dtype=np.int64)
    best_raw = np.empty(n_choices_arr.size, dtype=np.int64)
    best_corrected = np.empty(n_choices_arr.size, dtype=np.int64)

    for j, K in enumerate(n_choices_arr):
        K_int = int(K)
        if K_int <= 0:
            raise ValueError("each n_choices entry must be > 0")
        sub_rng = np.random.default_rng(seed + 1 + j)
        best_count = -1
        best_pvals: NDArray[np.float64] | None = None
        for _ in range(K_int):
            cand = _random_subset(n_features, subset_size, sub_rng)
            _, p_cand = two_sample_ttest_matrix(
                group_a[:, cand], group_b[:, cand], axis=0
            )
            p_cand = np.asarray(p_cand, dtype=np.float64)
            count = int(np.sum(p_cand < alpha))
            if count > best_count:
                best_count = count
                best_pvals = p_cand
        assert best_pvals is not None
        best_raw[j] = best_count
        best_corrected[j] = int(np.sum(best_pvals < alpha / K_int))

    return {
        "n_choices": n_choices_arr,
        "best_raw": best_raw,
        "best_corrected": best_corrected,
        "alpha": float(alpha),
        "subset_size": int(subset_size),
        "metadata": {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "seed": int(seed),
        },
    }
