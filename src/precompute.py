"""Offline precompute step.

Generates the small artefacts the marimo notebook loads at startup.
Two profiles:

- ``--quick`` &mdash; tiny repeat counts, used for development and CI.
- ``--full`` &mdash; stable artefacts for the submitted notebook.

Artefacts are saved as compressed ``.npz`` under ``results/``:

- ``lead_false_positive_sweep.npz``  &mdash; pure-null sweep over
  feature counts, with raw / Bonferroni / BH-FDR results.
- ``sparse_signal_sweep.npz``  &mdash; (effect_size x n_signal) grid
  of mean FDP and power by correction method.
- ``p_hacking_sweep.npz``  &mdash; best-of-K naive discoveries and the
  probability that at least one survives the Bonferroni-at-choice-
  level threshold ``alpha / K``.
- ``default_heatmaps.npz``  &mdash; representative pure-null and
  sparse-signal heatmaps with raw / Bonferroni / BH-FDR rejection
  masks.

Each run also writes ``results/manifest.json`` with timestamps,
seeds, and parameters. No personal information is recorded.

Usage::

    python -m src.precompute --quick
    python -m src.precompute --full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Make ``python -m src.precompute`` work without an installed package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.researcher_freedom import (  # noqa: E402
    simulate_p_hacking_curve,
    sweep_false_positives,
    sweep_power_fdr,
)
from src.stats_tests import (  # noqa: E402
    benjamini_hochberg,
    bonferroni,
    two_sample_ttest_matrix,
)
from src.synthetic_data import (  # noqa: E402
    make_two_group_data,
)


RESULTS_DIR = _REPO_ROOT / "results"
ALPHA_DEFAULT = 0.05


# --------------------------------------------------------- profiles


def _profile(quick: bool) -> dict[str, Any]:
    """Return tunable parameters for the chosen profile."""
    if quick:
        return {
            "name": "quick",
            "lead": {
                "feature_counts": [20, 50, 100, 200, 500],
                "n_repeats": 50,
                "n_samples": 40,
                "seed": 1,
            },
            "sparse": {
                "n_features": 200,
                "n_signal_values": [0, 5, 20],
                "effect_sizes": [0.0, 0.5, 1.0],
                "n_repeats": 20,
                "n_samples": 40,
                "seed": 2,
            },
            "p_hacking": {
                "n_choices": [1, 2, 5, 10, 20],
                "n_samples": 40,
                "n_features": 80,
                "n_repeats": 50,
                "seed": 3,
            },
            "heatmaps": {
                "height": 24, "width": 24, "n_signal_blocks": 2,
                "effect_size": 1.0, "n_samples_per_pixel": 30,
                "seed_null": 4, "seed_signal": 5,
            },
        }
    return {
        "name": "full",
        "lead": {
            "feature_counts": [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
            "n_repeats": 500,
            "n_samples": 40,
            "seed": 1,
        },
        "sparse": {
            "n_features": 1000,
            "n_signal_values": [0, 5, 20, 50],
            "effect_sizes": [0.0, 0.25, 0.5, 0.75, 1.0],
            "n_repeats": 100,
            "n_samples": 60,
            "seed": 2,
        },
        "p_hacking": {
            "n_choices": [1, 2, 5, 10, 20, 50, 100],
            "n_samples": 60,
            "n_features": 200,
            "n_repeats": 200,
            "seed": 3,
        },
        "heatmaps": {
            "height": 32, "width": 32, "n_signal_blocks": 3,
            "effect_size": 1.0, "n_samples_per_pixel": 40,
            "seed_null": 4, "seed_signal": 5,
        },
    }


# ---------------------------------------------------- artifact builders


def _build_lead(params: dict[str, Any], alpha: float) -> dict[str, np.ndarray]:
    out = sweep_false_positives(
        feature_counts=params["feature_counts"], alpha=alpha,
        n_repeats=params["n_repeats"], seed=params["seed"],
        n_samples=params["n_samples"],
    )
    methods = ("none", "bonferroni", "bh_fdr")
    feature_counts = out["feature_counts"]
    M = feature_counts.size

    def _stack(stat: str) -> np.ndarray:
        return np.stack([out["results"][m][stat] for m in methods], axis=0)

    return {
        "feature_counts": feature_counts.astype(np.int64),
        "methods": np.asarray(methods, dtype=object),
        "mean_false_discoveries": _stack("mean").astype(np.float64),
        "median_false_discoveries": _stack("median").astype(np.float64),
        "p05": _stack("p05").astype(np.float64),
        "p95": _stack("p95").astype(np.float64),
        "prob_any_false_discovery": _stack("any_fp").astype(np.float64),
        "expected_naive": (alpha * feature_counts.astype(np.float64)),
        "alpha": np.float64(alpha),
        "n_repeats": np.int64(params["n_repeats"]),
        "n_samples": np.int64(params["n_samples"]),
    }


def _build_sparse(params: dict[str, Any], alpha: float) -> dict[str, np.ndarray]:
    out = sweep_power_fdr(
        effect_sizes=params["effect_sizes"],
        n_signal_values=params["n_signal_values"],
        n_features=params["n_features"],
        n_repeats=params["n_repeats"],
        seed=params["seed"],
        n_samples=params["n_samples"],
        alpha=alpha,
    )
    methods = ("none", "bonferroni", "bh_fdr")

    def _stack(stat: str) -> np.ndarray:
        return np.stack([out["results"][m][stat] for m in methods], axis=0)

    return {
        "effect_sizes": out["effect_sizes"].astype(np.float64),
        "n_signal_values": out["n_signal_values"].astype(np.int64),
        "methods": np.asarray(methods, dtype=object),
        "fdp_mean": _stack("fdp_mean"),
        "power_mean": _stack("power_mean"),
        "n_features": np.int64(params["n_features"]),
        "n_samples": np.int64(params["n_samples"]),
        "n_repeats": np.int64(params["n_repeats"]),
        "alpha": np.float64(alpha),
    }


def _build_p_hacking(
    params: dict[str, Any], alpha: float
) -> dict[str, np.ndarray]:
    """Multi-trial p-hacking sweep.

    Each trial regenerates the data with a different seed and runs the
    same ``simulate_p_hacking_curve`` over the K grid. We report the
    mean ``best_raw`` count and the probability that the
    family-wise-corrected count (``best_corrected >= 1``) crosses
    "publishable" significance under the null.
    """
    n_repeats = int(params["n_repeats"])
    seed_root = int(params["seed"])
    rng = np.random.default_rng(seed_root)

    n_choices_arr = np.asarray(params["n_choices"], dtype=np.int64)
    K_count = n_choices_arr.size

    raw_counts = np.empty((n_repeats, K_count), dtype=np.int64)
    corrected_counts = np.empty((n_repeats, K_count), dtype=np.int64)

    for r in range(n_repeats):
        trial_seed = int(rng.integers(1 << 30))
        out = simulate_p_hacking_curve(
            n_choices=list(n_choices_arr),
            n_samples=int(params["n_samples"]),
            n_features=int(params["n_features"]),
            alpha=float(alpha),
            seed=trial_seed,
        )
        raw_counts[r] = out["best_raw"]
        corrected_counts[r] = out["best_corrected"]

    return {
        "n_choices": n_choices_arr,
        "best_raw_mean": raw_counts.mean(axis=0).astype(np.float64),
        "best_raw_median": np.median(raw_counts, axis=0).astype(np.float64),
        "best_raw_p05": np.percentile(raw_counts, 5, axis=0).astype(np.float64),
        "best_raw_p95": np.percentile(raw_counts, 95, axis=0).astype(np.float64),
        "prob_publishable": np.mean(corrected_counts >= 1, axis=0).astype(np.float64),
        "alpha": np.float64(alpha),
        "n_repeats": np.int64(n_repeats),
        "n_samples": np.int64(params["n_samples"]),
        "n_features": np.int64(params["n_features"]),
        "subset_size": np.int64(int(params["n_features"]) // 2),
    }


def _build_heatmaps(
    params: dict[str, Any], alpha: float
) -> dict[str, np.ndarray]:
    """Two pixelwise grids: pure null and sparse signal.

    Each pixel runs an independent two-group t-test with
    ``n_samples_per_pixel`` rows per group. Sparse-signal grid plants
    several 4x4 blocks at random locations.
    """
    h = int(params["height"])
    w = int(params["width"])
    n_pix = h * w
    n_per = int(params["n_samples_per_pixel"])
    eff = float(params["effect_size"])

    # ---- pure null heatmap
    rng_null = np.random.default_rng(params["seed_null"])
    a_null = rng_null.standard_normal((n_per, n_pix))
    b_null = rng_null.standard_normal((n_per, n_pix))
    _, p_null = two_sample_ttest_matrix(a_null, b_null, axis=0)
    p_null = np.asarray(p_null, dtype=np.float64).reshape(h, w)
    truth_null = np.zeros((h, w), dtype=bool)

    # ---- sparse-signal heatmap
    rng_sig = np.random.default_rng(params["seed_signal"])
    a_sig = rng_sig.standard_normal((n_per, n_pix))
    b_sig = rng_sig.standard_normal((n_per, n_pix))
    truth_sig = np.zeros((h, w), dtype=bool)
    block_h, block_w = 4, 4
    for _ in range(int(params["n_signal_blocks"])):
        i = int(rng_sig.integers(0, max(h - block_h, 1)))
        j = int(rng_sig.integers(0, max(w - block_w, 1)))
        truth_sig[i:i + block_h, j:j + block_w] = True
    # Add the planted shift to group_b at signal pixels.
    truth_flat = truth_sig.reshape(-1)
    b_sig[:, truth_flat] += eff
    _, p_sig = two_sample_ttest_matrix(a_sig, b_sig, axis=0)
    p_sig = np.asarray(p_sig, dtype=np.float64).reshape(h, w)

    def _masks(p_grid: np.ndarray) -> dict[str, np.ndarray]:
        flat = p_grid.reshape(-1)
        naive = (flat < alpha).reshape(p_grid.shape)
        bonf = bonferroni(flat, alpha=alpha)["reject"].reshape(p_grid.shape)
        bh = benjamini_hochberg(flat, alpha=alpha)["reject"].reshape(p_grid.shape)
        return {"naive": naive, "bonferroni": bonf, "bh_fdr": bh}

    null_masks = _masks(p_null)
    sig_masks = _masks(p_sig)

    return {
        "null_p_values": p_null,
        "null_truth": truth_null,
        "null_naive_reject": null_masks["naive"],
        "null_bonferroni_reject": null_masks["bonferroni"],
        "null_bh_fdr_reject": null_masks["bh_fdr"],
        "signal_p_values": p_sig,
        "signal_truth": truth_sig,
        "signal_naive_reject": sig_masks["naive"],
        "signal_bonferroni_reject": sig_masks["bonferroni"],
        "signal_bh_fdr_reject": sig_masks["bh_fdr"],
        "alpha": np.float64(alpha),
        "height": np.int64(h),
        "width": np.int64(w),
        "n_samples_per_pixel": np.int64(n_per),
        "effect_size": np.float64(eff),
    }


# ----------------------------------------------------------- driver


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path.stat().st_size


def precompute(quick: bool = False, results_dir: Path = RESULTS_DIR,
               alpha: float = ALPHA_DEFAULT) -> dict[str, Any]:
    profile = _profile(quick)
    sizes: dict[str, int] = {}
    timings: dict[str, float] = {}

    for name, build, fname in (
        ("lead", _build_lead, "lead_false_positive_sweep.npz"),
        ("sparse", _build_sparse, "sparse_signal_sweep.npz"),
        ("p_hacking", _build_p_hacking, "p_hacking_sweep.npz"),
        ("heatmaps", _build_heatmaps, "default_heatmaps.npz"),
    ):
        t0 = time.perf_counter()
        arrays = build(profile[name], alpha)
        path = results_dir / fname
        size = _save_npz(path, arrays)
        timings[name] = time.perf_counter() - t0
        sizes[fname] = size
        print(f"  {fname:<36} {size / 1024:>8.1f} KB   "
              f"({timings[name]:.2f} s)")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile["name"],
        "alpha": alpha,
        "seeds": {
            "lead": profile["lead"]["seed"],
            "sparse": profile["sparse"]["seed"],
            "p_hacking": profile["p_hacking"]["seed"],
            "heatmaps_null": profile["heatmaps"]["seed_null"],
            "heatmaps_signal": profile["heatmaps"]["seed_signal"],
        },
        "parameters": {
            "lead": {k: v for k, v in profile["lead"].items() if k != "seed"},
            "sparse": {k: v for k, v in profile["sparse"].items() if k != "seed"},
            "p_hacking": {k: v for k, v in profile["p_hacking"].items() if k != "seed"},
            "heatmaps": {
                k: v for k, v in profile["heatmaps"].items()
                if k not in ("seed_null", "seed_signal")
            },
        },
        "artifacts": list(sizes.keys()),
        "artifact_sizes_bytes": sizes,
        "build_seconds": {k: round(v, 3) for k, v in timings.items()},
    }
    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  manifest.json                        "
          f"{manifest_path.stat().st_size / 1024:>8.1f} KB")

    total = sum(sizes.values()) + manifest_path.stat().st_size
    print(f"  total                                {total / 1024:>8.1f} KB")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.precompute",
        description="Precompute notebook artefacts under results/.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="small repeat counts, for development and CI")
    mode.add_argument("--full", action="store_true",
                      help="stable artefacts for the submitted notebook")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                        help="significance threshold (default 0.05)")
    args = parser.parse_args(argv)

    quick = args.quick or not args.full
    print(f"precompute profile: {'quick' if quick else 'full'}  "
          f"(alpha={args.alpha})")
    precompute(quick=quick, alpha=args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
