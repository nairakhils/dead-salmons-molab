# Implementation notes

Working notes for the Dead Salmons molab notebook. The committed
[README.md](../README.md) is the public-facing summary; this file is
the engineering record.

## Module map

| Module | Responsibility | Notes |
|---|---|---|
| [src/synthetic_data.py](../src/synthetic_data.py) | Null-only feature matrices, two-group sparse-signal datasets, 2-D heatmap grids, and zero-padded feature labels. | Pure NumPy. Seed-deterministic. |
| [src/stats_tests.py](../src/stats_tests.py) | Per-feature Welch t-tests, Bonferroni, Benjamini-Hochberg, +1/+1 Monte Carlo p-values, label-permutation p-values, discovery metrics. | NumPy + SciPy. No statsmodels. |
| [src/researcher_freedom.py](../src/researcher_freedom.py) | `run_single_analysis`, `sweep_false_positives`, `sweep_power_fdr`, `simulate_p_hacking_curve`. | Reuses the math layer; no new statistics. |
| [src/precompute.py](../src/precompute.py) | Builds the four `.npz` artefacts and the manifest. CLI: `python -m src.precompute --quick / --full`. | One-command harness. |
| [src/plots.py](../src/plots.py) | Matplotlib figure helpers used by both the notebook and the precompute thumbnail. Returns `Figure` objects; never calls `plt.show`. | Single-shot rcParams setter for consistent style. |
| [src/io_utils.py](../src/io_utils.py) | `find_results_dir` (env-var → walk-up → CWD), `load_artifacts`, JSON read/write helpers. | No private paths. |

## Notebook structure (current)

Sections in [notebooks/walkthrough.py](../notebooks/walkthrough.py):

0. Imports and artifact loading.
1. Lead figure: false discoveries vs feature count, three corrections.
2. Paper claim, notebook demonstration, extension (three callouts).
3. Dead-salmon heatmap on a pure-null grid.
4. Sparse signal among nulls: power / FDP at the selected cell, plus
   a side-by-side correction comparison.
5. Researcher-degrees-of-freedom sandbox (extension).
6. Ranked feature table from one sparse-signal dataset.
7. Practical guardrails.
8. Limits.
9. References.

Reactivity: every figure in §§1–4 reads from the precomputed `.npz`
files. §5's curve is precomputed; the optional live run is gated
behind a `mo.ui.run_button`. §6 runs a small (~50 ms) live simulation
on each slider change. No cell exceeds the 5 s budget.

## Random seeds

Seed-determinism is enforced at three levels:

- **Module level.** Every public function in `src/` accepts a `seed`
  argument and constructs its own `np.random.default_rng(seed)`. No
  shared global RNG, no module-level `np.random.seed` call.
- **Sub-seed derivation.** Where a function generates several
  independent components (e.g. `make_two_group_data` needs three
  independent RNGs for group A, group B, and the signal pattern), it
  derives sub-seeds via `rng.integers(1 << 30)` from the parent seed.
  This keeps the seeds fully reproducible from a single integer
  without correlating the components.
- **Artifact level.** [src/precompute.py](../src/precompute.py) pins
  one seed per artefact. The values land in `results/manifest.json`
  next to the parameters used to build that artefact.

Default artefact seeds:

| Artifact | Seed |
|---|---|
| `lead_false_positive_sweep.npz` | 1 |
| `sparse_signal_sweep.npz`       | 2 |
| `p_hacking_sweep.npz`           | 3 |
| `default_heatmaps.npz` (null)   | 4 |
| `default_heatmaps.npz` (signal) | 5 |

Cross-machine byte-determinism of the `.npz` files is not promised:
LAPACK reduction order can shift bits between Linux CI runs at
single-thread BLAS. The pytest suite is what verifies correctness.

## Artifact generation

`python -m src.precompute --quick` builds the four artefacts in under
a second and is suitable for development and CI. `--full` produces
the artefacts shipped with the notebook and runs in well under thirty
seconds on a laptop.

| File | Contents | Loaded by |
|---|---|---|
| `lead_false_positive_sweep.npz` | Pure-null sweep over feature counts $m \in \{20, 50, 100, 200, 500, 1000, 2000, 5000, 10000\}$ for three correction methods. Per (method, m) cell: mean / median / 5th & 95th percentile of false-discovery counts plus `prob_any_false_discovery`. Includes the analytic reference `expected_naive = α·m`. | §1. |
| `sparse_signal_sweep.npz` | $E \times S$ grid of mean FDP and mean power per correction method. Full profile uses $n\_features = 1000$, effect sizes $\{0, 0.25, 0.5, 0.75, 1.0\}$, n_signal $\{0, 5, 20, 50\}$, $n\_repeats = 100$. | §4. |
| `p_hacking_sweep.npz` | For each $K \in \{1, 2, 5, 10, 20, 50, 100\}$ the mean / median / 5th / 95th percentile of best-of-K naive-discovery count under the global null, plus the probability that Bonferroni-at-choice-level (`p < α / K`) leaves at least one survivor. | §5. |
| `default_heatmaps.npz` | Two pre-rolled $32 \times 32$ grids (pure-null, sparse-signal): per-pixel raw p-values, truth mask, and naive / Bonferroni / BH-FDR rejection masks. Each pixel is an independent two-group Welch t-test with `n_samples_per_pixel` rows per group; signal pixels live in $4 \times 4$ blocks at random locations. | §3 (null grid), §4 (signal grid). |

The harness also writes `results/manifest.json` with: UTC timestamp,
profile name, α, the five seeds above, the full parameter blocks per
artefact, the build wall-clock per artefact, and the byte size of
every output. No machine hostname, username, working directory, or
environment variable is captured.

## Correction methods

Three correction methods are implemented from scratch in
[src/stats_tests.py](../src/stats_tests.py):

- **No correction (naive `p < α`).** Each test is calibrated at level
  α, so the per-test rejection probability under the null is α. With
  m independent tests, expected false discoveries equal α·m and the
  family-wise probability of any rejection is `1 − (1 − α)^m`.
- **Bonferroni.** Multiplies each p-value by m and clips at 1.
  Equivalently, rejects only at threshold α/m. Controls the
  family-wise error rate (FWER) at level α; conservative when m is
  large or the tests are correlated.
- **Benjamini-Hochberg (BH-FDR).** Step-up procedure on sorted p-values:
  finds the largest rank k with $p_{(k)} \le k/m \cdot \alpha$ and
  rejects all tests at rank ≤ k. Controls the expected proportion of
  false rejections among all rejections (FDR) at level α under
  independence (and under positive dependence). Adjusted p-values
  are the cumulative-min `m · p / k` from the right, capped at 1.
  NaN p-values are excluded from the family and returned as
  `reject = False`, `p_adjusted = NaN`.

A fourth helper, **`monte_carlo_p`**, computes the +1/+1 corrected
permutation p-value $\hat p = (1 + \sum_b \mathbb{1}\{T^{(b)}_{\text{null}} \ge T_{\text{obs}}\}) / (B + 1)$
following North, Curtis & Sham (2002) and Phipson & Smyth (2010).
This is the construction the paper recommends in its Appendix A. It
is used inside `permutation_pvalues` (label-shuffle null) and in the
notebook's narrative; the sandbox in §5 demonstrates the inflation
the construction is designed to defend against.

## Why no real model is used

The notebook deliberately uses synthetic Gaussian data and Welch
t-tests rather than activations from a real network for three
reasons:

- **Pedagogical isolation.** The dead-salmon mechanism is a property
  of the family of tests, not of any particular interpretability
  method. Any data on which the per-feature test statistic is
  calibrated under the null reproduces it. Synthetic Gaussians make
  the mechanism visible without a layer of architecture-specific
  questions about probe choice, layer choice, or activation
  preprocessing.
- **Reproducibility and runtime.** The full notebook runs in well
  under a second on a laptop and uses only NumPy, SciPy, Matplotlib,
  and pandas. There is no GPU dependency, no model checkpoint to
  ship, and no PyTorch / TensorFlow runtime in the molab cloud.
- **Honest scope.** Running probes on a real network would invite the
  reader to interpret the notebook as a reproduction of the paper's
  empirical studies. It is not. The notebook isolates one statistical
  mechanism that those studies invoke; it does not reproduce the
  studies themselves and does not extend their conclusions.

The trade-off: the notebook cannot speak to the paper's identifiability
or overdetermination arguments, both of which require a model with
internal structure. Those mechanisms are flagged in the §8 limits
section and in [paper_summary.md](paper_summary.md).

## Limitations

- **Synthetic data only.** No interpretability method is run on a
  real network. The notebook is a statistical demonstration, not an
  empirical replication.
- **One mechanism out of several.** The paper discusses
  underspecification, overdetermination, non-identifiability, and
  several method-specific failure modes. The notebook covers only
  the multiple-comparisons / weak-null mechanism.
- **Independent-test assumption.** The `1 − (1 − α)^m` family-wise
  formula and the BH-FDR theorem assume independent (or positively
  dependent) tests. Real interpretability features are correlated.
  `make_null_feature_matrix` exposes a `correlation` knob for
  exploration, but the precomputed sweeps use independent tests.
- **No identifiability content.** The non-identifiability argument
  that occupies most of the paper's §3 is not demonstrated here.
- **No model-randomisation null.** The paper's Appendix A test draws
  null statistics from `k = 20` re-randomised networks. The notebook
  uses label-permutation as a structurally similar but cheaper null;
  this is an analogue, not a port.
- **Fixed α and fixed n_samples in artefacts.** Both are exposed as
  parameters in `precompute.py` and could be re-run, but the shipped
  manifest fixes α = 0.05.

## Local-only files (not committed)

- `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/` — caches
  and environments.
- `paper/` — downloaded PDF of the paper (gitignored).
- `scratch/`, `tmp/`, `local/`, `downloads/` — working directories.
- `docs/paper_summary.md` and `docs/implementation_notes.md` — kept
  in `docs/` for the author's reference; whether to commit them is a
  per-fork choice.
