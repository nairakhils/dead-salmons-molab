# Paper summary: *The Dead Salmons of AI Interpretability*

**Reference.** Méloux, M., Dirupo, G., Portet, F., & Peyrard, M. (2025).
*The Dead Salmons of AI Interpretability.* arXiv:2512.18792 (21 Dec 2025), 24 pages.

This file is local-only extraction notes. It is gitignored. Do not
commit it. The committed `README.md` and notebook prose are the
public-facing summary.

---

## 1. Central thesis

**Dead-salmon analogy.** In 2009, Bennett, Miller & Wolford placed a
post-mortem Atlantic salmon in an fMRI scanner, showed it photographs
of humans in social situations, and "asked" it to judge their
emotions. The standard analysis pipeline of the time returned brain
voxels significantly predictive of emotional content. The salmon
was dead; the discovery was a multiple-comparisons artifact in a
mass-univariate test over many voxels. The paper's title invokes this
demonstration as the prototype of a "dead-salmon artifact": a
plausible-looking discovery that survives standard analysis even when
the underlying system has no signal.

**Specific warning for AI interpretability.** The authors argue that
contemporary interpretability methods routinely produce findings
that look like real explanations even when applied to randomly
initialised networks (paraphrased). They demonstrate (Figure 1) that
standard sentiment-classification probes on a fully randomised
BERT-base report highly significant correlations and nontrivial
cross-validated probe accuracy across layers. The deeper claim is that this is not a single bug but
the surface of a structural problem: **non-identifiability**.
Interpretability queries usually fail to uniquely determine an
explanation from observed computational traces, which manifests as
high variance, poor generalisation, and false discovery.

**What goes wrong without false-discovery control.** The paper lists
three concrete consequences (§3, "The Deeper Statistical Issue"):
(i) **poor generalisation** &mdash; among the multiple explanations
that fit equally well, no principled rule says which transfers off-
distribution; (ii) **sensitivity to design choices** &mdash; different
algorithmic choices (datasets, optimisation, hyper-parameters) produce
different "explanations" of the same network; (iii) **false discovery**
&mdash; as the hypothesis space grows, the probability of recovering
a *spurious* explanation that happens to fit the data grows with it.

---

## 2. Interpretability targets discussed

The paper organises §2 ("The Statistical Fragility of AI
Interpretability") by method family. Each entry below is explicitly
named and critiqued.

| Method family | Critique cited in the paper |
|---|---|
| **Feature attribution / saliency** | Saliency maps remain visually plausible after model weights are randomised (Adebayo et al. 2018). Gradient-based attributions can be adversarially perturbed without changing predictions (Dombrowski et al. 2019). Attributions are unstable under minor data transformations (Ghorbani et al. 2019). Bilodeau et al. 2024 prove no attribution method can simultaneously satisfy intuitive desiderata across broad model classes. |
| **Probing classifiers** | Probes recover information from randomly initialised contextual embeddings (Conneau et al. 2018; Hewitt & Manning 2019). Syntactic probes do not generalise (Hall Maudslay & Cotterell 2021). Probes can extract features that are present but causally irrelevant (Ravichander et al. 2021). |
| **Sparse autoencoders (SAEs)** | SAEs recover apparently interpretable components from randomly initialised transformers (Heap et al. 2025). They fail to generalise across settings or tasks (Heindrich et al. 2025; Kantamneni et al. 2025) and are sensitive to adversarial input perturbations (Li et al. 2025a). |
| **Concept-based explanations (TCAV, ACE, concept-bottleneck)** | "Interpretability illusion" where individual neurons in BERT spuriously appear to encode a concept (Bolukbasi et al. 2021). Concept activation scores produce highly inconsistent explanations (Nicolson et al. 2025). Poor generalisation and high sensitivity to dataset choice (Ramaswamy et al. 2023). |
| **Circuit discovery** | "Hydra effect" &mdash; ablating components labelled as causally important fails to change behaviour because of redundant pathways (McGrath et al. 2023). Circuit explanations fail to generalise (Wang et al. 2022). Sensitive to minor experimental choices (Méloux et al. 2025). |
| **Causal interventions / causal mediation analysis** | Sensitive to experimental design (Zhang & Nanda 2024; Canby et al. 2025). The estimand of mediation analysis is identifiable; the *interpretive claim* that the high-effect component is the *locus* of the mechanism is **not**, because of overdetermination. |
| **Activation patching** | Subspace activation patching can produce "interpretability illusions" by activating alternate pathways (Makelov et al. 2023) &mdash; a problem of overdetermination. |
| **Mechanistic interpretability (where-then-what and what-then-where families)** | Toy-model studies recover multiple incompatible explanations even for *random* networks (Méloux et al. 2025). Sutter et al. 2025 prove that, in general, existing causal-abstraction methods can produce explanations for random networks. |
| **Natural-language explanations / chain-of-thought self-explanations** | LLM-generated explanations can be systematically unfaithful (Ajwani et al. 2024); CoT rationales are typically unfaithful to the model's actual computation (Lanham et al. 2023; Arcuschin et al. 2025; Turpin et al. 2023). |

The paper does **not** discuss: representation engineering, gradient
ascent visualisations, network dissection in detail (it is cited but
not critiqued at length), or LLM-as-judge interpretability.

---

## 3. Statistical failure modes named

Items below are explicitly named in the paper. Items the prompt asked
us to check for that are *not* discussed are flagged at the end.

- **Multiple comparisons (the dead-salmon root cause).** §1 frames the
  original 2009 demonstration as a failure to correct for multiple
  comparisons. §A.1 frames the proposed remedy in those terms.
- **Statistical fragility under small perturbations.** §2: "small
  perturbations to inputs (Ghorbani 2019; Kindermans 2019; Zhang 2025)
  or changes in random initialisation (Adebayo 2018; Zafar 2021) can
  radically change explanations".
- **Failure to generalise off-distribution.** §2: "explanations often
  fail to generalise to new settings and input distributions
  (Hoelscher-Obermaier 2023)".
- **Underspecification.** §3: "multiple, distinct explanations can
  equally well account for the same input/output patterns".
  Inherited from D'Amour et al. 2022.
- **Overdetermination.** §3 and §2: "multiple redundant, independently
  sufficient causal pathways exist", which makes the localisation
  claim in causal mediation and circuit discovery ill-posed.
- **Non-identifiability.** §3: the formal label the paper attaches to
  the union of underspecification and overdetermination.
- **High variance / low reliability of estimators.** §3, §5.2:
  "different algorithmic choices traverse the manifold differently,
  producing different explanations".
- **Confabulation in natural-language explanations.** §2: NL
  explanations are "particularly susceptible to confabulation, and
  thus, to false positives".

The prompt asked us to check for several other failure modes; none of
the following appear by name in the paper:
- selective reporting *per se* (the replication-crisis literature is
  cited &mdash; Simmons et al. 2011, Ioannidis 2005, Schimmack 2020 &mdash;
  but the paper does not coin a specific failure mode for AI work);
- garden of forking paths (cited via the same replication-crisis block,
  not named as a specific AI-interpretability mode);
- researcher degrees of freedom (same);
- post-hoc hypothesis formation (not named);
- weak null models (the paper *does* argue for stronger nulls &mdash;
  randomly reinitialised networks &mdash; but does not coin "weak null");
- insufficient negative controls (covered implicitly by the
  randomised-baseline argument);
- overclaiming from visual evidence (saliency maps are critiqued, but
  the paper does not generalise to a "visual-evidence overclaim"
  category).

---

## 4. Positive recommendations

The paper's prescriptive contribution is in §4 and §A. They are
**framework-level** rather than recipe-level: the central
recommendation is to reframe interpretability as statistical-causal
inference. The concrete tools they endorse:

- **Hypothesis testing against randomised null models.** §4 and §A.1.
  Construct a family of null models by *full weight randomisation*,
  *random orthogonal transformations of representations*, or
  *label shuffling*. Compute the test statistic on the trained
  network and on each null draw. Use the **Monte Carlo p-value with
  the +1/+1 correction**:
  $\hat p = \dfrac{1 + \sum_b \mathbb{1}\{T^{(b)}_{\mathrm{null}} \ge T_{\mathrm{obs}}\}}{B + 1}$
  citing North, Curtis & Sham 2002 and Phipson & Smyth 2010 for the
  +1/+1 correction.
- **Effect sizes alongside p-values.** §A.1 reports "effect sizes
  relative to random guessing and to randomised models", not just
  significance.
- **Cross-validated estimates.** §A.1: 10-fold cross-validation on the
  probe accuracy in their three demonstrations.
- **Confidence sets / credible sets / posterior distributions.** §5.1
  and §5.3: "interpretability methods should report not just point
  estimates but confidence sets or posterior distributions over
  explanations". §5.3 sketches a Bayesian-interpretability programme.
- **Pre-specifying the interpretability task** as the triple
  $(\mu, \mathcal{E}, D)$ &mdash; query distribution, hypothesis class,
  discrepancy &mdash; ahead of analysis. §4.3.
- **Identifiability-by-design.** §5.2: choose $\mu$, $\mathcal{E}$, $D$
  so the population risk has a unique minimiser (up to acceptable
  symmetries).
- **Pre-registration, registered reports, and open data sharing.**
  §5.3 argues that standardised effect-size measures, pre-registered
  analyses, and open sharing of computational traces would let
  interpretability accumulate as a cumulative science (paraphrased).
- **Stronger null baselines.** Throughout: the *standard* null in the
  paper's experiments is a *randomly reinitialised* version of the
  same architecture (k = 20 reinits in their setup), plus a
  random-guessing baseline, plus a majority-vote baseline.

The paper does **not** explicitly recommend Bonferroni or
Benjamini-Hochberg by name. It frames the problem as needing better
*null hypotheses* and *uncertainty quantification* rather than as a
multiple-comparisons-correction problem per se. The MC p-value above
is the closest formal recommendation it makes.

---

## 5. Empirical demonstrations in the paper

The empirical content lives in **Figure 1** (in the main text) and
**§A.1, Figure 4** (in Appendix A). All experiments are probing
experiments, designed to illustrate the proposed Monte Carlo
hypothesis test.

**Figure 1 (main text): "Minimal dead salmon artifacts".**
- *Model:* randomly initialised BERT-base (no training of any kind).
- *Dataset:* 300 IMDb sentences with binary sentiment labels.
- *Method A:* take token representations, average over sequence length,
  compute correlation of each principal component with the sentiment
  label. Several components are coloured (p < 0.05).
- *Method B:* train a linear probe on the per-layer representations.
  The probe achieves nontrivial cross-validated accuracy across
  layers despite the model being random.

**§A.1 / Figure 4: three full probing experiments with the proposed
hypothesis test.**

| Sub-experiment | Model | Dataset | Probe | Null | Result |
|---|---|---|---|---|---|
| (A) Sentiment Analysis | `BERT-base-uncased` | 1000 IMDb sentences, 10-fold CV | linear probe per layer to predict binary sentiment | k = 20 random reinitialisations + random-guessing baseline | All pretrained layers beat random guessing with large effect sizes; under the new test, *no* layer is statistically distinguishable from the randomised models. Later layers show an upward trend in effect size relative to random. |
| (B) Syntax / POS tagging | `BERT-base-uncased` | CoNLL-2003 subset, 500 sentences, 10-fold CV | logistic regression probe per layer per POS tag | k = 20 random reinitialisations + majority baseline | Consistent with prior work, accuracy peaks in middle layers; under the new test, only middle layers remain statistically above the random baseline, and effect sizes are substantially reduced. |
| (C) World Models (space) | `pythia-160m` | "world places" dataset of Gurnee & Tegmark 2024 | linear ridge regression on residual-stream activations to predict latitude/longitude | k = 20 baselines with transformer blocks randomised but embeddings fixed | Raw embeddings already contain $R^2 \approx 0.12$ spatial structure ($Z \approx 100$ vs random guessing). Passing them through *randomised* transformer blocks degrades $R^2$ to $\approx 0.38$. The pretrained model's final layers reach $Z \approx 25$ above the *random-block* baseline, confirming that the model genuinely learns spatial structure beyond what the embeddings contain. |

**Numeric thresholds and recommended values that appear in the paper:**
- $B = k = 20$ random reinitialisations as the null family in their
  experiments.
- 10-fold cross-validation throughout.
- Significance threshold not pinned to a single $\alpha$; effect-size
  $Z$-scores are reported instead. The implied threshold in their
  reading is roughly $|Z| > 2$ (standard).
- 300 sentences for Figure 1; 1000 for §A.1 (A); 500 for §A.1 (B).

The paper does **not** report empirical results on SAEs, circuits,
saliency, concept-based methods, or causal mediation in its own
experiments &mdash; those are surveyed via the literature in §2 only.

---

## 6. Notebook scope

### In scope

- Live simulation of false positives on synthetic data
  (lead bar chart, §1 of notebook; closed-form vs Monte Carlo curves).
- Multiple-comparison correction: Bonferroni and Benjamini-Hochberg,
  implemented in pure NumPy, applied to live p-value vectors.
- Sparse signal vs many null features: §4 of notebook plants a weak
  shared signal in a $200 \times 24$ activation matrix and shows that
  several distinct probes still recover it without agreeing on the
  weights.
- Researcher-degrees-of-freedom sandbox: §3 of notebook lets the user
  vary $K$ (number of probes), $k$ (random reinits), and the
  threshold $\alpha = q$, watching survivors shrink as the test
  tightens.
- Interpretability-flavoured toy examples: tiny pure-NumPy random MLP
  whose hidden representations stand in for activations; ridge-
  regularised linear probe with permutation p-values; randomisation
  test against random reinits as a literal port of the paper's Eq. 4.

### Out of scope

- Reproducing every empirical critique from the paper. The notebook
  does not run probes on real BERT, Pythia, GPT-2, or any other
  pretrained network.
- Running real interpretability tools on large models (no SAEs, no
  circuit discovery, no activation patching, no causal mediation).
- Claiming that all interpretability work is invalid. The paper itself
  does not claim this; the notebook closes (§5) with the paper's
  constructive reframing.
- Reproducing §A.1's Figures 4(A)/(B)/(C) numerically. Citing them is
  enough to anchor the demonstration.

### Original contribution

An interactive marimo sandbox where the user can manufacture a
"finding" by changing analysis choices &mdash; number of tests, $\alpha$,
random seed, probe configuration &mdash; while the notebook tracks how
many discoveries survive Bonferroni, BH-FDR, and the paper's
randomisation test side by side. The non-identifiability section (§4
of the notebook) is the only one that goes beyond the paper's
demonstrations: four probes on the *same* synthetic data give four
different "explanations", which is the paper's underspecification
argument made tangible in five seconds.

---

## 7. Ambiguities

1. **What exactly is the recommended null family?**
   *Ambiguous:* §4 lists "full weight randomization, random orthogonal
   transformations of representations, or label shuffling
   (recovering the standard permutation test)" without ranking them.
   *Why it matters:* These nulls are not equivalent. Full weight
   randomisation is the strictest; label shuffling only tests the
   probe-vs-label link, not the probe-vs-network link.
   *How the notebook avoids overclaiming:* §2 of the notebook uses
   label permutation (matches the synthetic-label setup); §3 uses
   random reinitialisation of the activation matrix as an analogue of
   full weight randomisation. The text labels both as "Eq. 4 in
   spirit", not as a literal port of the BERT experiment.

2. **What is the value of $k$ that should be used in practice?**
   *Ambiguous:* The paper uses $k = 20$ in §A.1 without justification.
   Whether this is "enough" for stable type-I error control is not
   discussed.
   *Why it matters:* For a Monte Carlo p-value with $B = k$, the
   resolution is $1 / (B + 1)$. With $B = 20$, the smallest
   reportable p-value is $\approx 0.048$, right at the conventional
   threshold.
   *How the notebook avoids overclaiming:* §3 of the notebook makes
   $k$ a slider (5 &ndash; 30) so the user can see the resolution
   limitation directly. §2 uses $B = 99 \ldots 499$ for the label-
   permutation null since shuffling is cheaper than reinitialisation.

3. **Does the paper recommend Bonferroni or BH at all?**
   *Ambiguous:* Neither is named explicitly. §2 acknowledges that the
   2009 dead-salmon finding was a multiple-comparisons oversight
   correctable by standard adjustments (paraphrased), which implies
   acceptance, but the constructive recommendation is the
   randomisation test, not a correction-of-p-values approach.
   *Why it matters:* The notebook's lead figure is a Bonferroni / BH
   demonstration, which is *not* the paper's headline recommendation.
   *How the notebook avoids overclaiming:* §3 of the notebook puts
   raw, Bonferroni, BH, and randomisation-test side by side, and
   the §5 closing prose explicitly says "the randomisation test in
   §3 is the simplest of these tools; the harder work is choosing
   $(\mu, \mathcal{E}, D)$ so the answer is well-posed in the first
   place."

4. **Is the dead-salmon finding really a one-bit summary of all the
   failure modes the paper surveys?**
   *Ambiguous:* The paper distinguishes (i) the surface dead-salmon
   artifact, fixable by stronger nulls, from (ii) the deeper non-
   identifiability problem, which is *not* fixable by hypothesis
   tests alone. The title and abstract foreground (i); the body
   foregrounds (ii).
   *Why it matters:* A notebook that only demonstrates (i) would
   misrepresent the paper.
   *How the notebook avoids overclaiming:* §4 is explicitly about (ii)
   and the §5 closing prose names underspecification, overdetermination,
   and identifiability as the harder problems.

5. **What level of pretrained-vs-random separation counts as a "real"
   discovery?**
   *Ambiguous:* The paper reports $Z$-scores against random reinits
   (e.g. $Z \approx 25$ for spatial probing of pythia-160m's final
   layers) without committing to a $Z$-threshold.
   *Why it matters:* The notebook needs to choose a threshold for the
   table column "BH-FDR survives".
   *How the notebook avoids overclaiming:* The threshold is a slider
   ($q$ from 0.01 to 0.20); the prose does not assert any single value
   is "right".

6. **Does the paper view interpretability as fundamentally salvageable?**
   *Ambiguous:* §5 reads as constructive (a research programme),
   while §3 reads as pessimistic ("non-identifiability is the common
   root cause"). Cynthia Rudin (cited) takes the more pessimistic
   view that interpretability of black-box models is the wrong
   research target; the paper situates itself as more sympathetic to
   the pragmatic stance of Páez 2019.
   *Why it matters:* The notebook's tone should not be more pessimistic
   than the paper.
   *How the notebook avoids overclaiming:* §5 of the notebook ends with
   the three constructive commitments (alternative hypotheses,
   uncertainty quantification, identifiability by design) rather than
   "interpretability is hopeless".

---

## 8. Confidence

Per-group confidence on a 1&ndash;5 scale (1 = guess, 5 = directly
quoted or derived from the paper text).

| Group | Confidence | Notes |
|---|---|---|
| Thesis | 5 | Directly from abstract, §1, §3. |
| Methods critiqued | 5 | §2 is an explicit method-by-method survey; cross-checked against the bibliography. |
| Failure modes | 4 | The paper's *named* failure modes are extracted with confidence 5; the prompt's checklist of broader replication-crisis failure modes I marked as "not named" with confidence 4 (could be present in passing prose I did not re-read). |
| Recommendations | 4 | Eq. 4 and the surrogate-models framework are direct quotations. The Bonferroni/BH question is genuinely ambiguous in the paper; this entry is downgraded by one. |
| Empirical demonstrations | 5 | Numbers and methods come from §A.1 directly (BERT-base, IMDb 1000, CoNLL-2003 500, pythia-160m, $k = 20$, 10-fold CV). |
| Notebook scope | 5 | Authored by us in agreement with the user's brief; no extraction risk. |
