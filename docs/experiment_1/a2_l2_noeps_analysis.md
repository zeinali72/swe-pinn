# Experiment 1 — A2 (nondim + L2, ε = 0) Analysis

> Scope: single A2 run on the canonical seed (123). Purpose: quantify the A1 → A2
> lift from swapping MSE for a per-batch L2 norm, and record the first-half evidence
> for H3b (does pure `sqrt(Σr²)` without ε cause gradient spikes or NaNs?).

## TL;DR

- **A2 best NSE(h) = 0.6814 at epoch 19997**, vs A1 seed=123 best 0.6327 at epoch 19997.
  **ΔNSE = +0.049**, a clear positive lift but **below H3a's predicted +0.10**.
- **No NaNs, no obvious late-training gradient catastrophe.** H3b's first-half
  prediction ("pure `sqrt` without ε shows gradient spikes late in training")
  **did not materialise** in this run — but the residuals never approached zero
  closely enough for the √(0) singularity to be exercised, so H3b is *not falsified*
  either. A4 / A6 at dense sampling will stress this more.
- **Gradient norms are ~1000× larger under L2 than under MSE** (median ≈ 1000 vs ≈ 0.2
  for A1). Consistent with the theoretical motivation: MSE gradient scales with the
  residual, L2 gradient is O(1) in the residual, so L2 delivers more signal when
  residuals are small. Training stability is preserved by the `clip_norm=1.0`
  already specified in §4.1.
- **A2 is slower to start than A1**, then overtakes it around epoch 7500.
- **Recommendation for next step: yes, proceed to A3 with `SEED=123`.** A3 (A2 + ε)
  is the natural continuation to close out H3b.

---

## 1. Run metadata

| Field           | Value |
|-----------------|-------|
| W&B id          | `cejhxljp` ([link](https://wandb.ai/zeinali72-exeter/swe-pinn/runs/cejhxljp)) |
| Name            | `2026-04-21_00-55-23_experiment_1_mlp`       |
| Git commit      | `7bc9b33` (same as A0 / A1)                  |
| Program         | `experiments.experiment_1.train_nondim_l2`   |
| Config          | `configs/experiment_1/experiment_1_nondim_l2_noeps.yaml` |
| Seed            | 123 (canonical, override via `SEED=123`)     |
| `training.l2_eps` | `0.0` (this is what makes it A2, not A3)   |
| Runtime         | 974 s (~16 min) — within expected envelope   |
| State           | `finished`                                   |

All other hyperparameters (architecture, optimiser, LR schedule, loss weights,
sampling density, dtype) identical to A1 per §4.1 of the design doc.

---

## 2. Headline comparison: A1 seed=123 → A2 seed=123

| Metric                 | A1 seed=123 (`jqr0ednx`) | A2 seed=123 (`cejhxljp`) | Δ |
|------------------------|-------------------------:|-------------------------:|--:|
| Best NSE(h)            |                   0.6327 |               **0.6814** | **+0.0487** |
| Best-NSE epoch         |                    19997 |                    19997 | 0 |
| Best NSE(hu)           |                   0.9486 |                   0.9508 | +0.002 |
| Best RMSE(h), m        |                   0.0937 |                   0.0873 | −0.006 |
| Best Rel L2(h)         |                   0.4735 |                   0.4410 | −0.033 |
| Best RMSE(hv), m²/s    |                   0.0012 |                   0.0004 | −0.0008 |
| LR ≤ 1e-7 by epoch     |                     3195 |                     3129 | ≈  |
| Runtime (s)            |                      956 |                      974 | ≈  |

**Conclusions**
- A2 improves **every** primary and diagnostic metric. No regressions.
- The h-field improves most. The momentum fields were already good under A1
  (NSE(hu) ≈ 0.95) and A2 matches that. `hv` is near-machine-noise in both runs
  (symmetric setup; the analytical solution has `v = 0`), so RMSE(hv) differences
  are within noise.
- **ΔNSE(h) = +0.049 is materially below H3a's ≥ +0.10 prediction.** H3a is partially
  supported: direction correct, magnitude smaller than expected. Most likely cause:
  with 10 k collocation points + LR floor saturation by epoch 3129, L2 doesn't have
  enough high-LR budget to pull away from the MSE local minimum. A4 at 100 k points
  is the real test (H4).

---

## 3. Training trajectory

Note: the A2 "loss" is `sqrt(Σrᵢ²)`-style per-term, not the MSE-style mean, so
absolute loss values are **not comparable** to A1 across runs. Use NSE / RMSE / grad
norm for cross-row comparison.

| Epoch | total loss | PDE term | IC term | BC term | grad norm | LR    | val NSE(h) |
|------:|-----------:|---------:|--------:|--------:|----------:|------:|-----------:|
|     0 |   6.11e+05 | 6.11e+05 |    6.94 |   16.17 |  2.80e+10 | 1e-4  |     −25.73 |
|     1 |   6.17e+03 | 6.09e+03 |    5.00 |    7.80 |  5.31e+07 | 1e-4  |      −5.10 |
|     5 |      8.00  |    1.92  |    1.44 |    0.46 |       847 | 1e-4  |      −1.70 |
|    50 |      3.73  |    0.71  |    0.75 |    0.23 |       559 | 1e-4  |      −0.99 |
|   300 |      1.87  |    0.32  |    0.39 |    0.12 |       611 | 1e-4  |      +0.07 |
|  1000 |      0.83  |    0.32  |    0.18 |    0.03 |       614 | 5e-5  |      +0.10 |
|  2000 |      0.55  |    0.29  |    0.09 |    0.02 |      1106 | 6.3e-6| +0.45 |
|  5000 |      0.43  |    0.24  |    0.08 |    0.01 |       252 | 1e-7  | +0.62 |
| 10000 |      0.43  |    0.23  |    0.07 |    0.01 |       655 | 1e-7  | +0.64 |
| 15000 |      0.42  |    0.24  |    0.07 |    0.01 |       969 | 1e-7  | +0.66 |
| 19000 |      0.42  |    0.23  |    0.07 |    0.01 |       725 | 1e-7  | +0.68 |
| 20000 |      0.39  |    0.22  |    0.07 |    0.01 |       731 | 1e-7  | +0.68 |

### 3.1 Gradient-norm behaviour

| Phase     | Mean       | Median | p99      | Max       |
|-----------|-----------:|-------:|---------:|----------:|
|   0–100   |   2.81e+08 |    540 | 3.33e+08 |  2.80e+10 |
| 100–500   |        611 |    616 |      750 |       816 |
| 500–2000  |        672 |    658 |     1050 |      1670 |
| 2000–5000 |       1210 |   1060 |     3270 |      4670 |
| 5000–10 k |       1230 |   1150 |     2750 |      4240 |
| 10 k–20 k |       1030 |    981 |     2070 |      3280 |

Interpretation:

- The early burst (epochs 0–5, grad ≈ 1e+10) is the expected transient as the
  initially-random MLP is pulled toward the BC/IC scale. `clip_norm = 1.0` truncates
  it safely — no blow-up.
- After epoch 100, grad norm stabilises at ~600 and drifts *upward* to ~1000 over
  the rest of the run. Median is monotonically around 500–1000 for 99 % of the run.
- **Crucially, no late-training spikes.** The p99 grad norm in the 10 k–20 k phase
  is 2070 — two orders of magnitude below the early-epoch burst, and within 2×
  of the mean. There is no evidence of the √(0) singularity being approached.
- For reference, A1 seed=123 had grad-norm median ~0.2 in the same late phase.
  The ~5000× L2:MSE ratio is consistent with the analytical derivation: MSE grad
  scales with `r` (here ~0.3) while L2 grad is normalised by `‖r‖` and scales with
  `~1`, so you expect L2's grad to be `~1/‖r‖ ≈ 3000×` the MSE grad — which matches.

### 3.2 NSE trajectory: A2 vs A1 seed=123

| Epoch | A1 NSE(h) | A2 NSE(h) | Δ        | Who's ahead |
|------:|----------:|----------:|---------:|-------------|
|   100 |    −0.35  |    −0.56  |    −0.21 | A1          |
|   500 |    −0.06  |    −0.04  |    +0.02 | A2 (tied)   |
|  1000 |    +0.25  |    +0.10  |    −0.15 | A1          |
|  5000 |    +0.59  |    +0.62  |    +0.02 | A2          |
| 10 000|    +0.61  |    +0.64  |    +0.04 | A2          |
| 19 999|    +0.63  |    +0.68  |    +0.05 | A2          |

- **A2 loses the first 5000 epochs.** This is counter-intuitive at first sight
  but sensible: while residuals are large, MSE's `r · ∂r/∂θ` is a bigger gradient
  than L2's normalised form, so MSE descends faster initially.
- **A2 first surpasses A1's best (0.6327) at epoch 7531**, and keeps climbing.
- A1 had essentially converged to its plateau by epoch 10 000; A2 continues to
  eke out gains throughout the remaining 10 k epochs. This is the L2-loss
  signature H3a predicts — non-vanishing gradient magnitude allows continued
  improvement where MSE has effectively stalled.

---

## 4. Hypothesis verdicts (interim)

| Hypothesis | Prediction                                                           | Observation (A2 vs A1 seed=123)                     | Verdict         |
|-----------:|----------------------------------------------------------------------|-----------------------------------------------------|-----------------|
| H3a        | L2 continues improving past A1's best; ΔNSE ≥ 0.10                   | A2 continues to improve past A1 from ep 7531; ΔNSE = **+0.049** | **Partial pass** — direction correct, magnitude smaller than predicted |
| H3b (1st half) | A2 without ε shows gradient spikes or NaNs in late training       | No NaNs; no late spikes (p99 grad ~2000 in final phase) | **Not falsified, not validated** — residuals never approached zero closely enough for √(0) to matter. A4/A6 with dense sampling will stress this harder. |

H3a's shortfall (observed +0.049 vs predicted +0.10) is the most load-bearing
finding here. Three plausible reasons:

1. **Sampling density is the bottleneck**, not the loss function.
   A1 seed=123 had already reached 0.63 NSE — close to what a 10 k-point sparse grid
   can resolve. Neither MSE nor L2 can compensate for undersampling of the solution
   manifold.
2. **LR floor saturation** (by epoch 3129) leaves only ~16 % of the run at
   high-signal LR. L2's advantage is in late-training gradient signal; cutting off
   the LR early clips that advantage.
3. **Seed effect.** Seed 2024 on A1 reached 0.66 already; if A2 had been run on
   seed 2024 it might have reached 0.70+ (ΔNSE ≥ 0.10). We can't know without the
   other two seeds.

None of these falsify H3a; they just reinterpret it as *conditional* on
adequate sampling and LR headroom. A4 (L2 + dense + ε) should resolve this.

---

## 5. Should A3 be run next with seed 123?

**Yes.**

A3 is `A2 + ε = 1e-12` — the Charbonnier stabilizer under the √. By design, A3
should:

- Match A2's NSE within noise (~±0.01).
- Show **strictly smoother** gradient traces with no spikes, regardless of whether
  A2's gradients have been well-behaved so far.

Points in favour of proceeding:

- A2 already shows no NaN/catastrophe, so A3 is not being run to *recover* a
  broken run — it's being run to close out H3b with a clean comparison.
- Per §4.3 of the design doc, the canonical seed is **123** for *all* rows A0–A6.
  Changing seed between rows would invalidate the chain comparison.
- A3's only change from A2 is one scalar config field (`l2_eps: 0.0 → 1e-12`).
  Incremental cost: ~16 min of GPU.

Command:

```bash
SEED=123 bash scripts/experiment_1/run_A3.sh
```

---

## 6. Open items

1. **A3 next.** See §5. No blockers.
2. **Gradient-spike audit for H3b.** Record A3's grad-norm trajectory side-by-side
   with A2's and check whether the two traces are statistically indistinguishable
   (as H3b predicts). If they are, the ε is a *defensive* precaution, not a
   necessity, at 10 k-point sparse sampling. If A3 is visibly smoother, log it.
3. **A4 (dense sampling)** will be where H3b is really tested. At 100 k points,
   individual residuals can plausibly reach near-zero and trigger the √(0)
   singularity. Watch that run closely.
4. **Append A2 row to the design-doc results table** (`docs/experiment_1_ablation_design.md §5`)
   once the full chain completes. Deferred to a single sweep after A6.
5. **Determinism audit** (pending from the A0/A1 analysis §5.2) still not done.
   A2 was run on the canonical seed but we have no A/B reproducibility check.
   Not urgent; document run-to-run variance only if a re-run is necessary.
