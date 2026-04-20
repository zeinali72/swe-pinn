# Experiment 1 — Ablation Study Design

> **Purpose.** A controlled, discovery-driven sequence of ablations on the
> flat-channel analytical dam-break problem (Hunter 2005, `experiment_1`)
> that isolates and quantifies the contribution of four methodological
> choices — input/output non-dimensionalization, loss functional (MSE vs
> L2 norm with epsilon stabilization), collocation-point density, and
> floating-point precision — in the order in which each was motivated by
> the previous finding. Results form the methods baseline for all
> subsequent experiments (2–11).

---

## 1. Narrative framing

Rather than presenting a flat factorial of configurations, the ablation is
structured as a **discovery story**: each step is motivated by an
observation made in the previous step. This reflects how the methodology
was actually developed in practice, and makes every design choice
traceable to a diagnosed problem rather than a guessed improvement.

The chain, in order:

1. **Observe** a training plateau under the current best MSE configuration.
2. **Diagnose** the plateau as gradient-flatness intrinsic to MSE as
   losses approach zero.
3. **Propose** replacing MSE with an L2-norm functional whose gradient
   stays linear near the optimum.
4. **Encounter** a new numerical issue (divergent `sqrt` gradient at
   exactly zero) and **remedy** it with a Charbonnier-style epsilon.
5. **Verify** the result with an orthogonal confounder check
   (collocation-point density) to confirm L2 is the load-bearing change.
6. **Confirm** that single precision (`float32`) is sufficient once the
   non-dim + L2 recipe is in place.

---

## 2. Research questions

- **RQ1.** Does non-dimensionalizing the SWE inputs **and** outputs via
  `SWEScaler` reduce inter-term loss-scale spread and produce measurably
  better training dynamics than input-only normalization?
- **RQ2.** Under a non-dim MSE baseline, is the observed training plateau
  caused by MSE's quadratic flatness near zero, or by some other factor
  (e.g. undersampling)?
- **RQ3.** Does replacing MSE with a per-batch L2 norm `sqrt(Σ rᵢ²)`
  eliminate the plateau, and is the epsilon stabilizer `sqrt(Σ rᵢ² + ε)`
  necessary for numerical stability of the gradient?
- **RQ4.** Is the L2 + ε breakthrough load-bearing on its own, or does it
  require the accompanying increase in collocation-point density to
  reach the reported NSE ceiling?
- **RQ5.** Once the non-dim + L2 + ε + dense-sampling baseline is fixed,
  does `float64` deliver measurable accuracy benefit over `float32`, and
  at what wall-clock cost?

---

## 3. Hypotheses and falsifiable predictions

| ID  | Hypothesis                                                                                                                  | Falsifiable prediction |
|-----|------------------------------------------------------------------------------------------------------------------------------|------------------------|
| H1  | Full non-dim reduces per-term loss spread → better conditioned optimization under identical weights.                         | A1 per-term ratios closer to 1 than A0; ΔNSE ≥ 0. |
| H2  | Under non-dim MSE, training plateaus because `∂MSE/∂θ ∝ r · ∂r/∂θ` vanishes quadratically as residual `r → 0`.                 | Per-epoch gradient-norm trace on A1 collapses by ≥ 1 decade at the plateau epoch. |
| H3a | Replacing MSE with L2 norm preserves gradient magnitude near zero and eliminates the plateau.                                 | A3 continues improving past A1's best epoch; ΔNSE ≥ 0.10. |
| H3b | Unmodified `sqrt(Σ r²)` has a divergent derivative at zero; ε under the sqrt is a *numerical* necessity, not a regularizer. | A2 (L2 without ε) shows gradient spikes or NaNs in late training; A3 (with ε) shows neither; final NSE unchanged to within noise. |
| H4  | The ~0.98 NSE reached with dense sampling is a joint effect of L2 and sampling, not either alone.                              | A5 (MSE + dense sampling) < 0.75; A3 (L2 + sparse) < 0.85; A4 (L2 + dense) ≥ 0.95. |
| H5  | For a non-dim, O(1)-residual problem, `float32` suffices.                                                                    | A6 NSE within 0.01 of A4; wall-clock ≥ 2× slower. |

---

## 4. Methodological framing

### 4.1 Common configuration (control variables)

Fixed across every ablation row:

- Domain: 1200 × 100 m, `T_final = 3600 s`.
- Architecture: MLP, width 512, depth 4.
- Optimizer: Adam, `lr = 1e-4`, `clip_norm = 1.0`, reduce-on-plateau
  schedule (factor 0.5, patience 50, rtol 1e-3).
- Training: 2000 epochs, batch size 256, seed 42, early-stop disabled
  (`min_epochs = 4000 > epochs`).
- Loss weights: `pde=1, ic=1, bc=10, neg_h=1`; data-free.
- Validation: 20,000-point analytical grid, NSE on water depth `h` as
  primary selection metric.
- Reference: analytical Hunter (2005) solution.

### 4.2 Ordered ablation chain

Each row **inherits** all changes from rows above, changing exactly one
variable at a time. Variables that change are bolded.

| ID | Config (target)                                       | Loss  | ε  | n_pde | dtype   | Motivation                                              |
|----|--------------------------------------------------------|-------|-----|-------|---------|---------------------------------------------------------|
| A0 | `experiment_1.yaml`                                    | MSE   | —   | 10k   | float32 | Current practice: input-normalized only. Baseline.      |
| A1 | `experiment_1_nondim.yaml`                             | MSE   | —   | 10k   | float32 | **+ output non-dim** → RQ1. Expected to reveal plateau. |
| A2 | `experiment_1_nondim_l2_noeps.yaml` *(new)*            | **L2**| **0**| 10k  | float32 | **MSE → L2, no ε** → RQ3a (does L2 help?) and RQ3b first half (does pure sqrt have numerical issues?). |
| A3 | `experiment_1_nondim_l2.yaml`                           | L2    | **1e-12** | 10k | float32 | **+ ε stabilization** → RQ3b second half. Expected: same accuracy as A2, strictly smoother gradients. |
| A4 | `experiment_1_nondim_l2_dense.yaml` *(new)*             | L2    | 1e-12 | **100k** | float32 | **+ dense sampling** → final headline result (~0.98 NSE reported empirically). |
| A5 | `experiment_1_nondim_mse_dense.yaml` *(new, decoupling)* | **MSE** | — | 100k | float32 | **Confounder isolation**: MSE with dense sampling. Tests whether L2 or sampling is load-bearing (RQ4). |
| A6 | `experiment_1_nondim_l2_f64.yaml` *(new)*               | L2    | 1e-12 | 100k  | **float64** | **+ precision** → RQ5. |

Note: **A5 is a decoupling run, not an inherited step**. It branches
from A1 (not A4) to isolate the effect of point count under MSE. Without
A5, any claim that L2 caused the breakthrough is confounded with the
simultaneous change in sampling density. A5 is the single most important
run in this design.

### 4.3 Replication protocol

Primary reporting uses **a single seed (42)** per row. The analytical
reference is deterministic and the measured deltas between adjacent
rows (~0.2 NSE) are much larger than any plausible seed-to-seed noise
on this verification problem.

**Sensitivity spot-check:** one row — the headline A4 (L2 + ε + dense) —
is additionally run with seeds 123 and 2024, to empirically bound
seed variance. If `std(NSE) < 0.02` across the 3 seeds of A4, the
single-seed protocol for the other rows is justified in text. If
`std(NSE) ≥ 0.02`, the protocol is escalated to 3 seeds for all rows.

Total runs: **7** (6 inherited + A5 decoupler) + **2 sensitivity seeds**
on A4 = **9 training runs**.

### 4.4 Metrics

| Group       | Metric                                         | Purpose in this study |
|-------------|-------------------------------------------------|-----------------------|
| Primary     | NSE on `h` (global, best-of-run)               | Headline convergence quality |
| Primary     | Epoch at which best NSE is reached              | Diagnoses plateau vs continued improvement |
| Diagnostic  | Per-term loss trajectories (`pde`, `ic`, `bc`, `neg_h`) | Inter-term scale balance (RQ1) |
| Diagnostic  | Global gradient norm ‖∇θ L‖ per epoch          | Gradient-flatness evidence for H2; spike detection for H3b |
| Supporting  | NSE on `hu`, `hv`                              | Velocity field fidelity |
| Supporting  | RMSE, MAE, Rel L2 on `h`                       | Error magnitude and localization |
| Supporting  | Mass balance E_mass (max, final)               | Physical plausibility |
| Cost        | Total training wall-clock (s)                  | RQ5 cost comparison |

All metrics computed by the shared `evaluation/metrics/` modules specified
in `docs/experimental_programme_reference.md`.

---

## 5. Expected outcome table

Format for the headline results table in the write-up:

| Ablation | Change                  | NSE (h) | Best epoch | ‖∇θ L‖ at plateau | Wall-clock |
|----------|-------------------------|--------:|-----------:|------------------:|-----------:|
| A0       | baseline (input-norm)   |    —    |      —     |         —         |    1.0×    |
| A1       | + non-dim               |    —    |      —     |         —         |    ≈1.0×   |
| A2       | + L2 (ε = 0)            |    —    |      —     |         —         |    ≈1.0×   |
| A3       | + ε = 1e-12             |    —    |      —     |         —         |    ≈1.0×   |
| A4       | + dense sampling (100k) |    —    |      —     |         —         |    ~10×    |
| A5       | MSE + dense (decoupling)|    —    |      —     |         —         |    ~10×    |
| A6       | + float64               |    —    |      —     |         —         |    ~25×    |

Empirical observations to date (informal, pre-ablation):

- A1 ≈ 0.55 (plateau at epoch ~270).
- A3 ≈ 0.76 (at 10k points, with ε).
- A4 ≈ 0.98 (at 100k points, with ε).

These numbers must be **re-run under the protocol above** before they
enter the manuscript. They are recorded here only to set expectation
scale and to flag that A5 — currently missing — is the critical
confounder check.

---

## 6. Write-up structure (thesis chapter)

The chapter follows the discovery narrative, not the ablation row order:

1. **Problem setup.** Hunter (2005) analytical dam-break, domain, BCs, IC.
2. **Common training protocol.** §4.1, verbatim.
3. **Observation: a training plateau.** Present A0 → A1. Show the plateau
   at epoch ~270 despite non-dim and show per-term loss trajectories.
   (H1 pass/fail on non-dim's benefit; H2 plateau observation.)
4. **Diagnosis: gradient flatness of MSE.** Analytic derivation of
   `∂MSE/∂θ` magnitude near `r ≈ 0`; empirical gradient-norm trace
   from A1 confirming the predicted collapse. (H2 verdict.)
5. **Remedy: L2 norm.** Derivation of `∂‖r‖₂/∂θ` showing the gradient
   magnitude is bounded below as long as the residual vector is non-zero.
   Present A2 (pure L2): show the plateau is broken but also show any
   observed gradient spikes / NaNs that motivate the next step.
   (H3a verdict; first half of H3b.)
6. **Numerical stabilization: the ε trick.** The `sqrt` divergence at
   zero; Charbonnier loss analogy; A3 as A2 + `ε = 1e-12`. Show gradient
   trace is strictly smoother and final NSE is preserved. (H3b verdict.)
7. **Confounder check: does sampling density explain the breakthrough?**
   Present A5 (MSE + dense sampling) and A4 (L2 + dense sampling) side
   by side. Decompose the 0.55 → 0.98 jump into an L2 component and a
   sampling component; show neither alone reaches 0.98. (H4 verdict.)
8. **Precision: do we need float64?** A4 vs A6. Quantify the cost/benefit.
   (H5 verdict.)
9. **Summary.** Which choices carry forward as defaults for Experiments
   2–11 and why.

---

## 7. Deliverables

### 7.1 Code artifacts (to be produced)

- [ ] Config `configs/experiment_1/experiment_1_nondim_l2_noeps.yaml`
      (A2) — copy of `experiment_1_nondim_l2.yaml` with a new
      `training.l2_eps: 0.0` key.
- [ ] Config `configs/experiment_1/experiment_1_nondim_l2_dense.yaml`
      (A4) — copy with `sampling.n_points_pde: 100000` (and
      `n_points_ic`, `n_points_bc_domain` matched to 100000).
- [ ] Config `configs/experiment_1/experiment_1_nondim_mse_dense.yaml`
      (A5) — copy of `experiment_1_nondim.yaml` (MSE path) with the
      same dense-sampling settings as A4. **Critical decoupling run.**
- [ ] Config `configs/experiment_1/experiment_1_nondim_l2_f64.yaml`
      (A6) — copy of A4 with `device.dtype: float64`.
- [ ] Refactor `experiments/experiment_1/train_nondim_l2.py` to read
      `training.l2_eps` from config (default 1e-12) so A2 and A3 share
      one training script, and the A2 vs A3 comparison is literally a
      one-line config diff.
- [ ] Add **gradient-norm logging** (global ‖∇θ L‖ per epoch) to
      `train_nondim_l2.py` output and/or tracker. This is the primary
      diagnostic evidence for H2 and H3b — the ablation story does not
      work without it. (If adding to the shared training loop is out
      of scope, log it in the local custom loop.)
- [ ] `scripts/ablation_exp1.sh` — runs A0–A6 + 2 sensitivity seeds
      sequentially.
- [ ] `scripts/aggregate_exp1_ablation.py` — collects per-run
      `training_history.json` + best-NSE stats and emits §5 table as
      both CSV and LaTeX.

### 7.2 Figures (publication-quality, Exeter + Blue Heart palette, 300 DPI)

- **F1.** Per-term loss trajectories A0 vs A1 (log y, 4 subplots).
  Evidence for H1.
- **F2.** Validation NSE(epoch) overlay for A1, A2, A3, A4 on one axis.
  The headline "L2 escapes the plateau" figure.
- **F3.** Global gradient norm ‖∇θ L‖(epoch) overlay for A1 (MSE), A2
  (L2 no ε), A3 (L2 with ε). Evidence for H2 and H3b.
- **F4.** Bar decomposition of the 0.55 → 0.98 gap into L2-component
  (A3 − A1), sampling-component (A5 − A1), and joint-excess
  (A4 − A3 − A5 + A1). Evidence for H4.
- **F5.** Bar chart: final NSE and wall-clock for A4 vs A6. Evidence for
  H5.

### 7.3 Tables

- **T1.** §5 headline results (7 rows + seed-sensitivity footnote on A4).
- **T2.** Per-term loss ratios (pde:ic:bc) at epoch 500 for A0 vs A1.
- **T3.** Hypothesis verdict table (H1–H5, pass/fail, evidence pointer).

---

## 8. Threats to validity and mitigations

- **Single analytical problem.** Exp 1 is effectively 1D and friction-only.
  Mitigation: scope claims explicitly to "verification regime"; re-test
  the L2 + ε recipe on Exp 2 (2D obstacle) before stating it as a
  universal default.
- **HPO coupling.** The tuned weights (`bc_weight=10`) were found under
  MSE. L2 may have a different optimum weight balance. Mitigation: for
  A3 and A4, additionally report NSE under uniform `{1,1,1,1}` weights
  as a robustness check. If L2 is insensitive to weight choice, that is
  itself a finding.
- **Single seed.** Larger variance could invalidate small deltas.
  Mitigation: the A4 sensitivity spot-check (§4.3). Report the A4
  3-seed std explicitly and state the single-seed protocol is only
  justified if that std < 0.02.
- **Gradient-norm interpretation.** A collapsing global gradient is
  consistent with both "loss is near zero" (good) and "loss is stuck on
  a plateau" (bad). Mitigation: always plot gradient norm **alongside**
  the loss value, so the reader can distinguish "converged" from
  "stalled" visually.
- **Early-stop must be disabled.** Confirm `early_stop_min_epochs >
  epochs` in every config; otherwise A3/A4's later best-epoch could be
  clipped prematurely.
- **`numerics.eps = 1e-6` is not the same as `training.l2_eps`.** The
  former is a water-depth floor inside `SWEPhysics` (prevents `1/0` in
  flux computations). The latter is the Charbonnier stabilizer inside
  `sqrt`. Never conflate them in the write-up; they remedy different
  failure modes.

---

## 9. Relationship to the broader programme

This ablation is scoped deliberately narrowly: it answers *"what is the
correct default training recipe for a PINN on SWE in a verification
regime"* and nothing more. The deliverable is a **methods baseline**
that Experiments 2–11 inherit without re-litigating. Choices that this
study deliberately does **not** investigate (adaptive weighting,
architecture selection, sampling strategy, slope terms, data-driven
training) are explicitly deferred to Experiments 2–11 as specified in
`docs/experimental_programme_reference.md`.
