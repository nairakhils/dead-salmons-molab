# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import asyncio
    import io
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # WASM detection. The molab cloud runtime gives us a real
    # filesystem; the /wasm runtime via Pyodide does not, so we have
    # to fetch artefacts over HTTPS in that case.
    try:
        import pyodide  # noqa: F401
        IN_WASM = True
    except ImportError:
        IN_WASM = False

    # Make the repo's src/ importable when the notebook is launched
    # from either ./ or ./notebooks/, locally or in cloud molab. We
    # never use private machine paths.
    _here = Path.cwd()
    for _candidate in (_here, _here.parent, _here.parent.parent):
        if (_candidate / "src" / "__init__.py").exists():
            if str(_candidate) not in sys.path:
                sys.path.insert(0, str(_candidate))
            break
    return IN_WASM, asyncio, io, np, pd


@app.cell(hide_code=True)
def _():
    # Project modules. Imported in one cell so reactive cells below
    # depend on them directly.
    from src import plots
    from src.researcher_freedom import run_single_analysis
    from src.stats_tests import (
        benjamini_hochberg,
        bonferroni,
    )
    from src.synthetic_data import feature_labels

    return (
        benjamini_hochberg,
        bonferroni,
        feature_labels,
        plots,
        run_single_analysis,
    )


@app.cell(hide_code=True)
def _(plots):
    plots.set_competition_style()
    return


@app.cell(hide_code=True)
async def _(IN_WASM, asyncio, io, np):
    # Dual-path artefact loader: filesystem first (local + cloud
    # molab), HTTPS fallback for /wasm (Pyodide). Mirrors the
    # geometry-of-noise-molab pattern.
    import json as _json

    DATA_BASE_REMOTE = (
        "https://raw.githubusercontent.com/nairakhils/"
        "dead-salmons-molab/main/results"
    )
    DATA_NAMES = (
        "lead_false_positive_sweep.npz",
        "sparse_signal_sweep.npz",
        "p_hacking_sweep.npz",
        "default_heatmaps.npz",
    )

    async def _fetch_bytes(name):
        if not IN_WASM:
            for _prefix in ("results", "../results"):
                try:
                    with open(f"{_prefix}/{name}", "rb") as _f:
                        return _f.read()
                except FileNotFoundError:
                    continue
        if IN_WASM:
            import pyodide.http
            _resp = await pyodide.http.pyfetch(f"{DATA_BASE_REMOTE}/{name}")
            return await _resp.bytes()
        import urllib.request
        return urllib.request.urlopen(f"{DATA_BASE_REMOTE}/{name}").read()

    async def _fetch_npz(name):
        _b = await _fetch_bytes(name)
        with np.load(io.BytesIO(_b), allow_pickle=True) as _z:
            return {_k: np.asarray(_z[_k]) for _k in _z.files}

    _loaded = await asyncio.gather(*[_fetch_npz(_n) for _n in DATA_NAMES])
    (
        lead_sweep,
        sparse_sweep,
        p_hacking_sweep,
        default_heatmaps,
    ) = _loaded

    try:
        _manifest_bytes = await _fetch_bytes("manifest.json")
        manifest = _json.loads(_manifest_bytes.decode("utf-8"))
    except Exception:
        manifest = {"profile": "?", "alpha": 0.05}
    return (
        default_heatmaps,
        lead_sweep,
        manifest,
        p_hacking_sweep,
        sparse_sweep,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dead Salmons, False Discoveries, and Interpretability

    *Akhil Nair*

    Naive significance testing can produce apparent interpretability findings from pure noise. Multiple-comparison correction changes the conclusion because the relevant object is the family of tests, not the most exciting feature. The sandbox below lets the reader change the number of features and analysis choices to see the false-discovery mechanism directly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Finding signal in pure noise

    Each point in the figure below comes from a precomputed sweep of independent two-group t-tests on data with no real effect. The y-axis is the count of features that crossed the chosen threshold, averaged over many repeats per setting.
    """)
    return


@app.cell(hide_code=True)
def _(lead_sweep, mo):
    _feature_counts = [int(m) for m in lead_sweep["feature_counts"]]
    _default_m = 10000 if 10000 in _feature_counts else _feature_counts[-1]

    lead_m_slider = mo.ui.slider(
        steps=_feature_counts,
        value=_default_m,
        label="number of features tested  m",
        show_value=True,
    )
    lead_correction = mo.ui.dropdown(
        options={
            "naive (no correction)": "none",
            "Bonferroni": "bonferroni",
            "BH-FDR": "bh_fdr",
        },
        value="naive (no correction)",
        label="correction method",
    )
    lead_show_expected = mo.ui.checkbox(
        value=True, label="show expected α·m line",
    )
    return lead_correction, lead_m_slider, lead_show_expected


@app.cell(hide_code=True)
def _(
    lead_correction,
    lead_m_slider,
    lead_show_expected,
    lead_sweep,
    mo,
    np,
    plots,
):
    _fig = plots.plot_lead_false_positive(
        sweep_data=lead_sweep,
        selected_n_features=int(lead_m_slider.value),
        correction_method=str(lead_correction.value),
        show_expected=bool(lead_show_expected.value),
    )

    _feat = np.asarray(lead_sweep["feature_counts"])
    _methods = np.asarray(lead_sweep["methods"]).reshape(-1)
    _idx_m = int(np.argmin(np.abs(_feat - int(lead_m_slider.value))))
    _idx_method = int(np.where(_methods == str(lead_correction.value))[0][0])
    _mean = float(lead_sweep["mean_false_discoveries"][_idx_method, _idx_m])
    _alpha = float(lead_sweep["alpha"])
    _expected = float(_alpha * _feat[_idx_m])

    _correction_blurb = {
        "none": (
            "Under the null, each test rejects with probability α. With "
            "m independent tests, the expected count of rejections is α·m, "
            "shown here as ≈ **{exp:.0f}**. None of those rejections "
            "reflect a real effect."
        ),
        "bonferroni": (
            "Bonferroni divides α by m before testing. The family-wise "
            "error rate (the probability of any false rejection) stays "
            "below α, and the average rejection count drops to ≈ "
            "**{mean:.1f}** under the null."
        ),
        "bh_fdr": (
            "Benjamini-Hochberg controls the expected proportion of "
            "rejections that are false at q = α. Under the global null "
            "the procedure leaves ≈ **{mean:.1f}** rejections on average."
        ),
    }[str(lead_correction.value)].format(mean=_mean, exp=_expected)

    mo.vstack([
        _fig,
        mo.hstack(
            [lead_m_slider, lead_correction, lead_show_expected],
            justify="start", gap=2, wrap=True,
        ),
        mo.md(
            f"At m = {int(_feat[_idx_m]):,} and α = {_alpha:.2f}, naive "
            f"testing on pure noise expects about **{_expected:.0f}** false "
            f"discoveries. " + _correction_blurb
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**Raw p-values are not enough when the family of tests is "
            "large.** Each individual test is calibrated at α, but the "
            "probability that at least one test in m independent tests "
            "crosses α is `1 - (1 - α)^m`, which approaches 1 as m grows. "
            "The relevant unit of inference is the family, not the single "
            "test."
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Paper claim, notebook demonstration, extension

    The three blocks below separate what the paper says, what this notebook actually demonstrates, and where the notebook adds material beyond the paper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.callout(
            mo.md(
                "**Paper claim.** Méloux, Dirupo, Portet & Peyrard "
                "(arXiv:2512.18792, 2025) report that several "
                "interpretability methods, applied to randomly initialised "
                "networks, return findings that look statistically "
                "significant under standard analysis. They identify "
                "non-identifiability of interpretability queries as a root "
                "cause and recommend testing against meaningful null "
                "models with explicit uncertainty quantification."
            ),
            kind="neutral",
        ),
        mo.callout(
            mo.md(
                "**Notebook demonstration.** This notebook isolates one "
                "statistical mechanism behind such findings: independent "
                "significance tests over a large family of features "
                "produce false discoveries at a predictable rate under "
                "any null. The demonstrations use synthetic data and "
                "Welch t-tests; they do not reproduce the empirical "
                "studies in the paper, and they do not show every "
                "failure mode the paper discusses."
            ),
            kind="info",
        ),
        mo.callout(
            mo.md(
                "**Extension.** Section 5 adds a researcher-degrees-of-"
                "freedom sandbox not in the paper. The user picks the "
                "number of analysis configurations to try on the same "
                "null data and sees the best-of-K result inflate with K. "
                "This is the same multiple-comparisons mechanism applied "
                "to the choice of analysis rather than the choice of "
                "feature."
            ),
            kind="success",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. The dead-salmon heatmap

    In the 2009 Bennett et al. study a dead Atlantic salmon was placed in an MRI scanner and the standard analysis returned voxels significantly correlated with social-emotional images, before multiple-comparison correction. The grid below replays the same shape: every pixel is an independent two-group t-test on data with no real effect.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    salmon_correction = mo.ui.dropdown(
        options={
            "naive (no correction)": "naive",
            "Bonferroni": "bonferroni",
            "BH-FDR": "bh_fdr",
        },
        value="naive (no correction)",
        label="correction method",
    )
    salmon_display = mo.ui.dropdown(
        options={"−log₁₀(p)": "neglog", "raw p": "raw"},
        value="−log₁₀(p)",
        label="display",
    )
    return salmon_correction, salmon_display


@app.cell(hide_code=True)
def _(default_heatmaps, mo, plots, salmon_correction, salmon_display):
    _correction = str(salmon_correction.value)
    _mask_key = {
        "naive": "null_naive_reject",
        "bonferroni": "null_bonferroni_reject",
        "bh_fdr": "null_bh_fdr_reject",
    }[_correction]

    _p = default_heatmaps["null_p_values"]
    _reject = default_heatmaps[_mask_key]

    # The display-mode toggle controls the title; the heatmap helper
    # always uses −log₁₀(p) since raw p compresses interesting cells.
    _title = {
        "naive": "Pure noise — naive p < 0.05",
        "bonferroni": "Pure noise — Bonferroni",
        "bh_fdr": "Pure noise — BH-FDR",
    }[_correction]
    if str(salmon_display.value) == "raw":
        _title += "  (display: raw p)"

    _fig = plots.plot_discovery_heatmap(
        p_values=_p, reject_mask=_reject, truth=None, title=_title,
    )

    n_disc = int(_reject.sum())
    n_pix = int(_p.size)
    mo.vstack([
        _fig,
        mo.hstack(
            [salmon_correction, salmon_display],
            justify="start", gap=2,
        ),
        mo.md(
            f"Out of {n_pix} pixels of pure noise, the selected analysis "
            f"marks **{n_disc}** as discoveries. The pixels themselves "
            f"are independent draws under the null; the differences "
            f"between methods come entirely from the rejection rule."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Sparse signal among nulls

    The previous section showed pure-null data. This section plants a small number of true signal features and asks how each correction trades discovery power against false-discovery proportion.
    """)
    return


@app.cell(hide_code=True)
def _(mo, sparse_sweep):
    _es = [float(x) for x in sparse_sweep["effect_sizes"]]
    _ns = [int(x) for x in sparse_sweep["n_signal_values"]]

    sparse_effect = mo.ui.dropdown(
        options={f"{e:.2f}": e for e in _es},
        value=f"{_es[len(_es) // 2]:.2f}",
        label="effect size",
    )
    sparse_n_signal = mo.ui.dropdown(
        options={str(n): n for n in _ns},
        value=str(_ns[1] if len(_ns) > 1 else _ns[0]),
        label="number of true signals",
    )
    sparse_correction = mo.ui.dropdown(
        options={
            "naive (no correction)": "none",
            "Bonferroni": "bonferroni",
            "BH-FDR": "bh_fdr",
        },
        value="BH-FDR",
        label="correction shown in heatmap",
    )
    return sparse_correction, sparse_effect, sparse_n_signal


@app.cell(hide_code=True)
def _(
    default_heatmaps,
    mo,
    np,
    plots,
    sparse_correction,
    sparse_effect,
    sparse_n_signal,
    sparse_sweep,
):
    _power_fig = plots.plot_power_fdr(
        sparse_signal_sweep=sparse_sweep,
        selected_effect_size=float(sparse_effect.value),
        selected_n_signal=int(sparse_n_signal.value),
    )

    _comparison_fig = plots.plot_correction_comparison({
        "p_values": default_heatmaps["signal_p_values"],
        "naive_reject": default_heatmaps["signal_naive_reject"],
        "bonferroni_reject": default_heatmaps["signal_bonferroni_reject"],
        "bh_fdr_reject": default_heatmaps["signal_bh_fdr_reject"],
        "truth": default_heatmaps["signal_truth"],
    })

    _es = np.asarray(sparse_sweep["effect_sizes"])
    _ns = np.asarray(sparse_sweep["n_signal_values"])
    _methods = np.asarray(sparse_sweep["methods"]).reshape(-1)
    _i_es = int(np.argmin(np.abs(_es - float(sparse_effect.value))))
    _i_ns = int(np.argmin(np.abs(_ns - int(sparse_n_signal.value))))
    _fdp = sparse_sweep["fdp_mean"]
    _pwr = sparse_sweep["power_mean"]

    def _row(method: str) -> tuple[float, float]:
        k = int(np.where(_methods == method)[0][0])
        return float(_fdp[k, _i_es, _i_ns]), float(_pwr[k, _i_es, _i_ns])

    _f_none, _p_none = _row("none")
    _f_bonf, _p_bonf = _row("bonferroni")
    _f_bh, _p_bh = _row("bh_fdr")

    mo.vstack([
        _power_fig,
        mo.hstack(
            [sparse_effect, sparse_n_signal, sparse_correction],
            justify="start", gap=2, wrap=True,
        ),
        mo.md(
            f"At effect size {_es[_i_es]:.2f} with {int(_ns[_i_ns])} true "
            f"signals out of {int(sparse_sweep['n_features'])} features, "
            f"the three corrections show their typical trade-off: naive "
            f"testing recovers more true signals but mixes in many false "
            f"ones; Bonferroni keeps false discoveries near zero at the "
            f"cost of power; Benjamini-Hochberg sits between the two.\n\n"
            f"- naive: FDP = **{_f_none:.2f}**, power = **{_p_none:.2f}**\n"
            f"- Bonferroni: FDP = **{_f_bonf:.2f}**, power = **{_p_bonf:.2f}**\n"
            f"- BH-FDR: FDP = **{_f_bh:.2f}**, power = **{_p_bh:.2f}**\n"
        ),
        _comparison_fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Researcher degrees-of-freedom sandbox

    Trying more analyses on the same null data inflates the best-observed result. The curve below tracks the maximum naive-rejection count across K random feature subsets on a single pure-null dataset, swept over K.
    """)
    return


@app.cell(hide_code=True)
def _(mo, p_hacking_sweep):
    _choices = [int(c) for c in p_hacking_sweep["n_choices"]]
    rdof_n_choices = mo.ui.dropdown(
        options={str(c): c for c in _choices},
        value=str(_choices[len(_choices) // 2]),
        label="number of analysis choices  K",
    )
    rdof_subset_search = mo.ui.checkbox(
        value=True, label="use subset search",
    )
    rdof_correction = mo.ui.dropdown(
        options={
            "naive (no correction)": "none",
            "Bonferroni": "bonferroni",
            "BH-FDR": "bh_fdr",
        },
        value="naive (no correction)",
        label="correction (applied to live run)",
    )
    rdof_run = mo.ui.run_button(label="run live analysis")
    return rdof_correction, rdof_n_choices, rdof_run, rdof_subset_search


@app.cell(hide_code=True)
def _(mo, np, p_hacking_sweep, plots, rdof_n_choices):
    _fig = plots.plot_p_hacking_sweep(
        p_hacking_sweep=p_hacking_sweep,
        selected_n_choices=int(rdof_n_choices.value),
    )
    _choices = np.asarray(p_hacking_sweep["n_choices"])
    _idx = int(np.argmin(np.abs(_choices - int(rdof_n_choices.value))))
    _best_mean = float(p_hacking_sweep["best_raw_mean"][_idx])
    _best_at_one = float(p_hacking_sweep["best_raw_mean"][0])

    mo.vstack([
        _fig,
        mo.md(
            f"With K = {int(rdof_n_choices.value)} analysis configurations "
            f"tried on the same null data, the best run reports about "
            f"**{_best_mean:.1f}** rejections on average, against "
            f"**{_best_at_one:.1f}** for a single analysis. The data has no "
            f"real effect; the inflation comes from picking the maximum "
            f"over more candidate analyses."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(
    mo,
    rdof_correction,
    rdof_n_choices,
    rdof_run,
    rdof_subset_search,
    run_single_analysis,
):
    # Live analysis is gated behind the run button so default loads
    # stay snappy. Ignored until the user clicks.
    if rdof_run.value:
        _result = run_single_analysis(
            n_samples=60,
            n_features=200,
            n_signal=0,
            effect_size=0.0,
            alpha=0.05,
            correction=str(rdof_correction.value),
            seed=int(rdof_run.value),
            subset_fraction=0.5,
            choose_best_subset=bool(rdof_subset_search.value),
            n_candidate_subsets=int(rdof_n_choices.value),
        )
        _summary = (
            f"Live run: {_result['n_tested']} features tested, "
            f"**{_result['n_reject']}** rejected, FDP = "
            f"**{_result['fdp']:.2f}**, power = "
            f"**{_result['power']:.2f}**. Compare these numbers to the "
            f"sweep curve above at the same K."
        )
    else:
        _summary = (
            "_Click **run live analysis** to draw a fresh dataset under "
            "the current settings and apply the chosen correction._"
        )

    mo.vstack([
        mo.hstack(
            [rdof_n_choices, rdof_subset_search, rdof_correction, rdof_run],
            justify="start", gap=2, wrap=True,
        ),
        mo.md(_summary),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Ranked feature table

    The table below is a single sparse-signal dataset rendered as the kind of ranked discovery list that often accompanies an interpretability claim. The `true_signal` column shows the planted ground truth so the reader can check which top rows are real and which are not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    table_effect = mo.ui.slider(
        start=0.0, stop=1.5, step=0.1, value=0.4,
        label="planted effect size",
    )
    table_n_signal = mo.ui.slider(
        start=0, stop=20, step=1, value=4,
        label="planted true signals",
    )
    table_n_features = mo.ui.slider(
        start=20, stop=200, step=20, value=80,
        label="number of features",
    )
    table_seed = mo.ui.slider(
        start=0, stop=20, step=1, value=0,
        label="dataset seed",
    )
    return table_effect, table_n_features, table_n_signal, table_seed


@app.cell(hide_code=True)
def _(
    benjamini_hochberg,
    bonferroni,
    feature_labels,
    pd,
    table_effect,
    table_n_features,
    table_n_signal,
    table_seed,
):
    from src.stats_tests import two_sample_ttest_matrix as _tt
    from src.synthetic_data import make_two_group_data as _mk

    _alpha = 0.05
    _data = _mk(
        n_a=60, n_b=60, n_features=int(table_n_features.value),
        n_signal=int(table_n_signal.value),
        effect_size=float(table_effect.value),
        seed=int(table_seed.value), signal_pattern="block",
    )
    _t, _p = _tt(_data["group_a"], _data["group_b"], axis=0)
    _bonf = bonferroni(_p, alpha=_alpha)
    _bh = benjamini_hochberg(_p, alpha=_alpha)

    feature_df = pd.DataFrame({
        "feature": feature_labels(_p.size, kind="feature"),
        "raw_p": _p,
        "bonferroni_p": _bonf["p_adjusted"],
        "bh_fdr_p": _bh["p_adjusted"],
        "naive_reject": _p < _alpha,
        "bonferroni_reject": _bonf["reject"],
        "bh_fdr_reject": _bh["reject"],
        "true_signal": _data["truth"],
        "effect_size": _data["effect"],
    }).sort_values("raw_p").reset_index(drop=True)
    return (feature_df,)


@app.cell(hide_code=True)
def _(
    feature_df,
    mo,
    table_effect,
    table_n_features,
    table_n_signal,
    table_seed,
):
    def _counts(col: str) -> tuple[int, int]:
        rej = feature_df[col]
        true_pos = int((rej & feature_df["true_signal"]).sum())
        false_pos = int((rej & ~feature_df["true_signal"]).sum())
        return true_pos, false_pos

    _t_naive, _f_naive = _counts("naive_reject")
    _t_bonf, _f_bonf = _counts("bonferroni_reject")
    _t_bh, _f_bh = _counts("bh_fdr_reject")
    _n_signal = int(feature_df["true_signal"].sum())

    mo.vstack([
        mo.hstack(
            [table_effect, table_n_signal, table_n_features, table_seed],
            justify="start", gap=2, wrap=True,
        ),
        mo.md(
            f"True signals planted: {_n_signal} of {len(feature_df)} "
            f"features. The three rows below count rejections under each "
            f"correction, split into true and false discoveries against "
            f"the planted ground truth.\n\n"
            f"- naive: **{_t_naive}** true / **{_f_naive}** false\n"
            f"- Bonferroni: **{_t_bonf}** true / **{_f_bonf}** false\n"
            f"- BH-FDR: **{_t_bh}** true / **{_f_bh}** false\n"
        ),
        mo.ui.table(feature_df, page_size=15, selection=None),
        mo.callout(
            mo.md(
                "**A clean-looking ranked list can still be dominated by "
                "null features.** Sort by `raw_p` and read down the "
                "`true_signal` column. Naive rejection accepts many top "
                "rows that have no planted effect. Bonferroni and BH-FDR "
                "trim the list to rows that beat the multiplicity-aware "
                "threshold."
            ),
            kind="warn",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Practical guardrails

    Six items that cover most failure modes shown above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Define the **family of tests** before looking at any p-values: the relevant denominator is the count of tests considered, not the count reported.
    - Report **all tests run**, or state the correction scope precisely.
    - Use **negative controls** such as randomised labels or randomly initialised networks; the paper's hypothesis test is one such control.
    - Use **held-out validation** so the same data are not used to both pick and confirm a finding.
    - Report **effect sizes** alongside p-values so the reader can judge practical relevance.
    - Prefer a **multiple-comparison correction** appropriate to the question (FWER for a single confirmation, FDR for a screening list) over a raw p-value table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**Corrections answer a specific question; they do not fix "
            "bad hypotheses or bad null models.** Bonferroni and BH-FDR "
            "control false discoveries within the family you declare and "
            "under the null you assume. They do not rescue an analysis "
            "whose alternative hypothesis is vague, whose null model is "
            "wrong, or whose feature space was selected after seeing the "
            "data. The paper's central recommendation is to test against "
            "a meaningful null, not only to count more tests."
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Limits

    A short list of what this notebook does not do, to keep its scope honest.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The demonstrations use synthetic data only; no interpretability method is run on a real network.
    - The notebook does not reproduce the empirical studies in the paper. It isolates the multiple-comparisons mechanism that those studies invoke.
    - Neither the paper nor this notebook claims that all interpretability work is invalid.
    - The sandbox covers one statistical failure mode. Non-identifiability, overdetermination, and several other mechanisms discussed in the paper are not addressed here.
    """)
    return


@app.cell(hide_code=True)
def _(manifest, mo):
    _profile = manifest.get("profile", "?")
    _alpha = manifest.get("alpha", "?")
    mo.md(
        rf"""
    ## 9. References

    - Méloux, Dirupo, Portet & Peyrard, *The Dead Salmons of AI Interpretability*, [arXiv:2512.18792](https://arxiv.org/abs/2512.18792), 2025.
    - Bennett, Miller & Wolford, *Neural correlates of interspecies perspective taking in the post-mortem Atlantic salmon*, 2009.
    - Benjamini & Hochberg, *Controlling the false discovery rate*, JRSS-B 1995.
    - North, Curtis & Sham, *A note on the calculation of empirical p-values from Monte Carlo procedures*, AJHG 2002.
    - Phipson & Smyth, *Permutation p-values should never be zero*, 2010.

    *Akhil Nair* &middot; precompute profile `{_profile}`, &alpha; = `{_alpha}`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
