# Dead Salmons, False Discoveries, and Interpretability

[![Open in molab](https://marimo.io/shield.svg)](https://molab.marimo.io/github/nairakhils/dead-salmons-molab/blob/main/notebooks/walkthrough.py)

Author: Akhil Nair

A marimo molab notebook on multiple testing and false discoveries in
AI interpretability, after Méloux, Dirupo, Portet & Peyrard,
*The Dead Salmons of AI Interpretability*
([arXiv:2512.18792](https://arxiv.org/abs/2512.18792), 2025). Naive
significance testing can produce apparent interpretability findings
from pure noise. Multiple-comparison correction changes the
conclusion because the relevant object is the family of tests, not
the most exciting feature. The notebook lets the reader change the
number of features and analysis choices to see the false-discovery
mechanism directly.

## What the notebook demonstrates

The notebook isolates one statistical mechanism behind dead-salmon-
style findings: independent significance tests over a large family of
features produce false discoveries at a predictable rate under any
null. Live sliders show the rate scaling, the effect of Bonferroni
and Benjamini-Hochberg correction, two synthetic heatmaps (pure null
and sparse signal), and a ranked discovery table where many
high-ranked features are noise. Every figure is driven by a small
precomputed sweep so the lead view loads instantly.

## Original extension

Section 5 adds a researcher-degrees-of-freedom sandbox not in the
paper. The user picks the number of analysis configurations to try
on the same null data and sees the best-of-K result inflate with K.
This is the same multiple-comparisons mechanism applied to the
choice of analysis rather than the choice of feature.

## Open in molab

[https://molab.marimo.io/github/nairakhils/dead-salmons-molab/blob/main/notebooks/walkthrough.py](https://molab.marimo.io/github/nairakhils/dead-salmons-molab/blob/main/notebooks/walkthrough.py)

## Run locally

```
git clone <repo-url> dead-salmons-molab
cd dead-salmons-molab
uv sync           # or:  uv pip install -e .  (or:  pip install -e .)
marimo edit notebooks/walkthrough.py    # editable view
marimo run notebooks/walkthrough.py     # read-only app view
```

Python 3.11 or newer.

## Regenerate artifacts

The notebook reads four precomputed files from `results/`. They are
checked in for the molab cloud workflow but can be rebuilt locally:

```
python -m src.precompute --quick    # development profile (~1 s)
python -m src.precompute --full     # submission profile (under 30 s)
```

`--quick` uses small repeat counts and is suitable for CI; `--full`
produces the stable artifacts shipped with the notebook. Both write
`results/manifest.json` with timestamps, seeds, and parameter blocks.

## Test

```
pytest
marimo check notebooks/walkthrough.py
```

`pytest` runs the unit-test suite over the math layer (multiple-
comparison corrections, t-tests, synthetic-data generators,
researcher-freedom sweeps, plotting smoke tests). `marimo check`
verifies that the notebook is well-formed.

## Reference

Méloux, M., Dirupo, G., Portet, F., & Peyrard, M. (2025).
*The Dead Salmons of AI Interpretability.*
[arXiv:2512.18792](https://arxiv.org/abs/2512.18792).

## Limits

The demonstrations use synthetic data only; no interpretability
method is run on a real network. The notebook does not reproduce the
empirical studies in the paper; it isolates the multiple-comparisons
mechanism that those studies invoke. Neither the paper nor this
notebook claims that all interpretability work is invalid. The
sandbox covers one statistical failure mode out of several discussed
in the paper.

## Privacy note

The repository intentionally excludes local caches, scratch files,
downloaded papers, virtual environments, and any private metadata.
The committed tree contains only the source needed to run the
notebook plus precomputed artifacts. The author is identified by name
only; no affiliation, workplace, advisor, email, or private link
appears anywhere in the repository.
