"""I/O helpers.

Two responsibilities:

- Locate the repo's ``results/`` directory across runtimes (local
  ``marimo edit``, ``marimo run``, molab cloud, WASM via ``pyodide``)
  without hard-coding any private machine path.
- Load all precomputed ``.npz`` artefacts and the manifest into a
  small dict that the notebook can consume.

No personal information is read or written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# Filenames written by ``src.precompute``.
ARTIFACT_FILENAMES: tuple[str, ...] = (
    "lead_false_positive_sweep.npz",
    "sparse_signal_sweep.npz",
    "p_hacking_sweep.npz",
    "default_heatmaps.npz",
)
MANIFEST_FILENAME = "manifest.json"


def find_results_dir(start: Path | None = None) -> Path:
    """Locate the ``results/`` directory.

    Search order:

    1. ``DEAD_SALMONS_RESULTS`` environment variable, if set.
    2. ``<start>/results`` walking up the parents of ``start`` (which
       defaults to this file's directory).
    3. ``./results`` and ``../results`` relative to the current
       working directory.

    Returns the first existing directory. Raises ``FileNotFoundError``
    otherwise.
    """
    env = os.environ.get("DEAD_SALMONS_RESULTS")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    candidates: list[Path] = []
    here = (start or Path(__file__).resolve().parent)
    for parent in [here, *here.parents]:
        candidates.append(parent / "results")
    candidates.append(Path.cwd() / "results")
    candidates.append(Path.cwd().parent / "results")

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand

    raise FileNotFoundError(
        "Could not locate results/ directory. Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    """Load a ``.npz`` file into a plain dict of numpy arrays."""
    with np.load(path, allow_pickle=True) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def load_artifacts(
    results_dir: Path | None = None,
    artifacts: Iterable[str] = ARTIFACT_FILENAMES,
) -> dict[str, Any]:
    """Load every precomputed artefact and the manifest.

    Returns a dict with one entry per filename (without the ``.npz``
    suffix) plus ``"manifest"`` and ``"results_dir"``.
    """
    rd = results_dir or find_results_dir()
    out: dict[str, Any] = {"results_dir": rd}
    for fname in artifacts:
        key = fname.removesuffix(".npz")
        out[key] = load_npz_dict(rd / fname)
    manifest_path = rd / MANIFEST_FILENAME
    out["manifest"] = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {}
    )
    return out


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dict to a JSON file with stable indentation."""
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
