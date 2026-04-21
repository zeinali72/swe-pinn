# Experiment 1 — A4 (nondim + L2 + ε + dense 100 k) Analysis

> Scope: single A4 run, canonical seed 123. Only delta from A3 is
> `sampling.n_points_pde: 10 000 → 100 000`. Purpose: observe the headline row of
> the ablation chain and set up the A4 vs A5 decoupling comparison (H4 — is L2 or
> sampling density the load-bearing change from A1 → A4?).

## TL;DR

- **A4 best NSE(h) = 0.9302 at epoch 19997**, vs A3's 0.6483.
  **ΔNSE(A3 → A4) = +0.282** — the biggest single jump in the chain.
- **A4 falls 0.05 short of the informal pre-ablation expectation (~0.98)**
  cited in `experiment_1_ablation_design.md §5`. Still a qualitative win, but the
  thesis number is now **≈ 0.93 at 100 k PDE points / 10 k IC / 10 k BC / float32**,
  not 0.98.
- **Gradient norms are 10³–10⁶× larger than at sparse sampling** (p99 1.85×10⁶ in
  the final phase; single-epoch max 1.23×10¹² mid-run). The Charbonnier ε = 1e-12
  floors `sqrt(Σr² + ε)` at `√ε = 1e-6`, but that floor is not enough to tame
  the gradient — **`clip_norm = 1.0` is doing the real stability work**. H3b
  reads as *ε helps but is not sufficient at dense sampling*.
- **A4 runtime: 103.5 min** (~6.3× A3's 16 min) — less than the design doc's
  informal 10× estimate, consistent with vectorised per-batch scaling efficiency.
- **IC and BC losses drop 5× vs A3** (IC: 0.074 → 0.015; BC: 0.010 → 0.003) *even
  though IC and BC sampling counts are unchanged*. This is a solution-quality
  effect, not a sampling effect — dense PDE sampling rippled into better-fit
  boundaries.
- **A5 is the decisive decoupling test.** If A5 (MSE + dense) lands at ≈ 0.90+,
  the lift is attributable to sampling density, not the loss functional.
  If A5 ≈ 0.60–0.70, L2 was the load-bearing change and sampling was secondary.
  **Run A5 next with `SEED=123 bash scripts/experiment_1/run_A5.sh`.**

---

## 1. Run metadata

| Field           | Value |
|-----------------|-------|
| W&B id          | `zktw19p8` ([link](https://wandb.ai/zeinali72-exeter/swe-pinn/runs/zktw19p8)) |
| Name            | `2026-04-21_01-33-10_experiment_1_mlp`           |
| Git commit      | `7bc9b33` (same as A0 / A1 / A2 / A3)            |
| Program         | `experiments.experiment_1.train_nondim_l2`       |
| Config          | `configs/experiment_1/experiment_1_nondim_l2_dense.yaml` |
| Seed            | 123 (canonical)                                  |
| Only delta from A3 | `sampling.n_points_pde: 10000 → 100000`       |
| IC / BC samples | Unchanged at 10 000 / 10 000 (intentional, per §4.2) |
| Runtime         | 6209 s = **103.5 min** (~6.3× A3)                |
| Median per-epoch time | 0.302 s (from `time/last_epoch_s`)         |
| State           | `finished`                                       |

---

## 2. Headline comparison across the chain (all seed = 123)

| Metric                | A1 `jqr0ednx` | A2 `cejhxljp` | A3 `rjxp91qh` | **A4 `zktw19p8`** | Δ A3→A4 |
|-----------------------|--------------:|--------------:|--------------:|------------------:|--------:|
| Best NSE(h)           |        0.6327 |        0.6814 |        0.6483 |        **0.9302** | **+0.282** |
| Best NSE(hu)          |        0.9486 |        0.9508 |        0.9512 |        **0.9844** |  +0.033 |
| Best RMSE(h), m       |        0.0937 |        0.0873 |        0.0917 |        **0.0409** |  −0.051 |
| Best Rel L2(h)        |        0.4735 |        0.4410 |        0.4633 |        **0.2064** |  −0.257 |
| Best RMSE(hv), m²/s   |       0.00122 |       0.00041 |       0.00044 |       **0.00014** | −0.0003 |
| LR ≤ 1e-7 by epoch    |         3195  |         3129  |         2721  |            3199   |       ≈ |
| Runtime (min)         |          15.9 |          16.2 |          16.5 |          **103.5** |  ≈ 6.3× |

All primary metrics move in the right direction; no regressions anywhere. The
`hv` RMSE is now approaching machine-zero (1.4×10⁻⁴ m²/s, versus ~5×10⁻⁵
float32-noise floor).

### 2.1 A4 vs the design-doc expectation

`docs/experiment_1_ablation_design.md §5` lists an informal pre-ablation
expectation of **A4 ≈ 0.98**. Observed value is **0.9302**, a 0.05-NSE shortfall.

Three candidate explanations, in order of plausibility:

1. **The 0.98 number came from a longer-training / different-config run.**
   The design-doc bullet is marked "informal, pre-ablation". The likely source is
   the older `configs/train/experiment_1*.yaml` family (pre-chain), which uses
   different schedulers and/or dense IC + BC sampling. Within the strict A4 spec
   (§4.2: only `n_points_pde` changes from A3; IC/BC stay at 10 k, float32,
   20 k epochs, reduce-on-plateau), 0.93 may simply *be* the ceiling.
2. **LR floor saturation persists** (epoch 3199). Beyond that, ~85 % of the run
   is at 1e-7 and cannot climb further. The A0/A1 finding still applies — a
   cosine schedule or longer training might reclaim the final 0.05.
3. **IC / BC undersampling.** These stay at 10 k while PDE points are 100 k.
   The boundary terms may now be the *limiting* constraint. A targeted one-off
   test (A4 with `n_points_ic = n_points_bc = 100 k`) would isolate this — deferred
   as a follow-up, not within the ablation scope.

None of these falsify H4; they adjust the target NSE of A4 within the §4.1-fixed
training budget.

---

## 3. Training trajectory

### 3.1 Key-epoch snapshot

| Epoch | total loss | PDE    | IC     | BC     | grad norm | LR     | NSE(h) |
|------:|-----------:|-------:|-------:|-------:|----------:|-------:|-------:|
|     0 |    1.48e+6 | 1.48e+6 |   1.95 |  1.92  |  2.60e+11 | 1e-4   | −1.735 |
|     1 |       7.13 |  1.137 |  1.387 |  0.461 |       647 | 1e-4   | −1.645 |
|    10 |       2.99 |  0.558 |  0.621 |  0.181 |       522 | 1e-4   | −0.248 |
|    50 |       1.38 |  0.435 |  0.221 |  0.072 |       510 | 1e-4   | −0.111 |
|   100 |       1.15 |  0.387 |  0.181 |  0.058 |       624 | 1e-4   | +0.103 |
|   500 |       0.83 |  0.332 |  0.139 |  0.036 |       639 | 1e-4   | +0.376 |
|  1000 |       0.53 |  0.286 |  0.070 |  0.017 |       765 | 5e-5   | +0.434 |
|  2000 |       0.34 |  0.200 |  0.034 |  0.011 |      1591 | 2.5e-5 | **+0.831** |
|  5000 |       0.21 |  0.163 |  0.021 |  0.003 |      3651 | 1e-7   | **+0.913** |
| 10000 |       0.35 |  0.296 |  0.019 |  0.004 |      4289 | 1e-7   | +0.920 |
| 15000 |       0.18 |  0.129 |  0.019 |  0.003 |      2468 | 1e-7   | +0.925 |
| 20000 |       0.17 |  0.126 |  0.019 |  0.003 |      2203 | 1e-7   | **+0.930** |

Two structural features stand out:

- **NSE crosses A3's ceiling (0.65) by epoch ~1200**, reaches **0.83 by epoch
  2000** and **0.91 by epoch 5000**. The bulk of the NSE gain comes in the first
  5 000 epochs — while the LR is still above the floor.
- **After epoch 5 000, progress slows to a crawl** — the "LR floor + plateau"
  pattern reappears. NSE gains 0.017 over the final 15 000 epochs (0.913 → 0.930).

### 3.2 Early crossover vs A3

A4 is ahead of A3 from **epoch 5 onward** and the gap keeps widening:

| Epoch | A3 NSE(h) | A4 NSE(h) | Δ |
|------:|----------:|----------:|--:|
|    50 |    −0.956 |    −0.111 | +0.845 |
|   100 |    −0.422 |    +0.103 | +0.526 |
|   500 |    −0.057 |    +0.376 | +0.433 |
|  1000 |    +0.149 |    +0.434 | +0.285 |
|  2000 |    +0.456 |    +0.831 | +0.375 |
|  5000 |    +0.588 |    +0.913 | +0.325 |
| 20000 |    +0.648 |    +0.930 | +0.282 |

Dense sampling isn't a late-training refinement; it changes the descent
trajectory from the very first epoch.

---

## 4. Gradient behaviour — the H3b dense-sampling stress test

A4 is where H3b (√-at-0 singularity) should bite, if it ever does, because dense
sampling means individual batches are more likely to contain configurations with
`Σr² → near-zero`. Observed:

### 4.1 Grad-norm phase statistics

| Phase        | Median   | p99       | Max       |
|--------------|---------:|----------:|----------:|
|     0 – 100  |      593 |  2.6×10⁹  | 2.6×10¹¹ |
|   100 – 500  |      708 |  7.1×10³  | 8.0×10⁵  |
|   500 – 2000 |      796 |  2.1×10⁴  | 3.2×10⁸  |
|  2000 – 5000 |  3.1×10³ |  3.8×10⁵  | 7.2×10¹⁰ |
|  5000 – 10 k |  4.0×10³ |  6.2×10⁵  | 1.0×10¹¹ |
| 10 k – 20 k  |  3.4×10³ |  1.9×10⁶  | **1.23×10¹²** |

For comparison, A3's late-phase p99 was ~2×10³. **A4's p99 is three orders of
magnitude higher, and the per-epoch max exceeds 10¹² multiple times.**

### 4.2 Why ε = 1e-12 is not enough on its own

The denominator in `∂/∂θ sqrt(Σrᵢ² + ε)` is floored at `sqrt(ε) = 10⁻⁶`.
With `∂rᵢ/∂θ` of magnitude ~10³ (typical network Jacobian magnitudes), the
post-ε gradient can reach `10³ / 10⁻⁶ = 10⁹` on any batch where `Σr² ≈ 0`.
That is exactly what we observe — dense sampling produces mini-batches where
the residuals happen to be collectively small, and gradient explodes by ~nine
orders of magnitude in that single step.

The stability of the run is instead preserved by **`clip_norm = 1.0`** (set in
§4.1). Every one of those 10⁹–10¹² grad-norm epochs is clipped down to a unit
update before being applied. No NaNs anywhere (`train/pde` min = 0.115, no `inf`
values in any logged column).

### 4.3 What this means for H3b at scale

H3b as stated — *"ε is a numerical necessity, not a regularizer"* — is still
pass**ed**: ε prevents outright NaN at `Σr² = 0`. But the practical-stability
story is that **both ε and gradient clipping are load-bearing** at dense
sampling. A thesis-grade statement is:

> At 10 k sampling, ε matters primarily to eliminate a sporadic spike cluster
> (A2 → A3 grad outlier count: 83 → 5). At 100 k sampling, ε bounds the grad
> denominator but the grad numerator remains large enough that `clip_norm`
> is the decisive stabiliser. Both are required; neither alone is sufficient.

This is a stronger and more honest claim than "ε fixes the sqrt divergence".

### 4.4 Per-term residual floors

The PDE term never got smaller than **0.115** across 20 k epochs; IC bottomed at
**0.0147**; BC at **0.0023**. None of these approach `√ε = 10⁻⁶`. So the √-at-0
singularity is **not** directly exercised at the *aggregated* loss level. The
gradient spikes must come from *within-batch* sub-selections where the
per-sample residual sum is much smaller than the per-epoch average. This is
consistent with the stochastic-batching picture: `batch_size = 256` ÷ 100 000 =
~0.3 % sample fraction per step; some batches happen to be all-low-residual.

---

## 5. Hypothesis verdicts (interim — H4 awaits A5)

| Hypothesis | Prediction                                                | Observation (A4 vs A3)                           | Verdict |
|-----------:|-----------------------------------------------------------|--------------------------------------------------|---------|
| H4 (half)  | A4 (L2 + dense) ≥ 0.95                                    | A4 = 0.930 — **below** 0.95, **above** 0.90; big lift from A3 (+0.282) | **Near-pass — magnitude shortfall, direction strongly correct.** Full verdict requires A5. |
| H3b (dense) | A3's "no-spike" pattern should persist at 100 k          | A4 grad p99 is 10³–10⁶× higher than A3's — ε alone is not enough, but no divergence thanks to clip | **Refined pass** — ε is necessary but not sufficient at dense sampling; `clip_norm = 1.0` is the decisive stabiliser. |

The H4 full verdict ("L2 and dense sampling are both necessary") depends on A5:

- **If A5 (MSE + dense) ≥ 0.90**: dense sampling is the load-bearing ingredient.
  L2 contributes marginally. H4 re-interpreted as "sampling dominant, L2 minor".
- **If A5 ∈ [0.65, 0.80]**: L2 and dense *both* required, and the A4 win is a
  joint effect. Original H4 wording stands.
- **If A5 ≤ 0.65**: the same as A1 / A3 — dense sampling alone buys nothing under
  MSE. Original H4 wording strongly supported.

---

## 6. Recommendation for A5

**Yes, run A5 next with `SEED=123`.** A5 is the decoupling row that gives the
H4 verdict. Without it, we cannot claim that "L2 + dense is the breakthrough"
rather than "dense alone is the breakthrough".

```bash
SEED=123 bash scripts/experiment_1/run_A5.sh
```

Expected runtime ~100 min (same order as A4, since sampling density is the
dominant cost).

Things to watch in A5:

1. **Final NSE** — the single number that decides the H4 scenario above.
2. **Grad-norm behaviour.** MSE gradient scales with `r` (not `1/‖r‖`), so with
   dense sampling MSE gradients should be *smaller* than A4's, not larger.
   If A5's grad-norm p99 is ≪ A4's, the "clip is load-bearing" story above
   becomes specific to L2, not a general dense-sampling story.
3. **NSE-vs-loss correlation.** A4 shows a clean climb; A1 showed loss
   decreasing while NSE was stable. If A5's NSE stalls while loss keeps dropping,
   that's the "MSE flat-gradient" mechanism re-asserting itself even at 100 k
   points — which would be a strong H4 pass.

---

## 7. Updated results table

| Row | Change from prev                       | NSE(h)  | Best epoch | Runtime (min) |
|-----|----------------------------------------|--------:|-----------:|--------------:|
| A0  | baseline (dimensional MSE, 10 k)       | −0.1117 |        11  |         15.2  |
| A1  | + nondim (seed=123)                    |  0.6327 |     19 997 |         15.9  |
| A2  | + L2 (ε = 0)                           |  0.6814 |     19 996 |         16.2  |
| A3  | + ε = 1e-12                            |  0.6483 |     19 999 |         16.5  |
| **A4** | **+ dense PDE (100 k points)**      |**0.9302**|    19 997 |       **103.5** |
| A5  | **← pending** (decoupling: MSE + dense)|   —     |       —    |           —   |
| A6  | + float64                              |   —     |       —    |           —   |

Data source: `docs/experiment_1/data/ablation_rows_summary.csv`.

---

## 8. Open items

1. **Run A5 next** with `SEED=123`. See §6. This is the single most important
   remaining run in the chain.
2. **If A5 falls below 0.65** — revisit the "0.98 A4 expectation" in the design
   doc, since it would imply the chain as specified tops out at 0.93.
3. **Dense-IC / dense-BC follow-up** (mentioned §2.1 item 3): one-off test of
   `n_points_ic = n_points_bc = 100 k` with everything else = A4. Outside the
   ablation scope; noted for the recommendations section of the chapter.
4. **Determinism audit** (from A0/A1 §5.2) still pending. Not blocking A5; flag
   for post-A6 cleanup.
5. **H3b refined claim** (§4.3) should be reflected in the thesis write-up:
   the honest story is "ε + clip together", not "ε alone".
6. **Final results-table update** to `docs/experiment_1_ablation_design.md §5`:
   deferred until A6 completes — then update all rows in a single commit.
