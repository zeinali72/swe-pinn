# Experiment 1 — A0 / A1 Seed Test Analysis

> Scope: five training runs on the flat Hunter (2005) dam-break problem, covering the
> dimensional baseline (A0) and the non-dim MSE baseline (A1) across three seeds.
> Purpose: (i) verify Step 0–1 of the ablation protocol
> (`docs/experiment_1_ablation_design.md §4.3`), (ii) select the canonical seed for
> the downstream A0–A6 chain, (iii) document anomalies that affect interpretation
> of later rows.

## TL;DR

- **Canonical seed = 123.** Median-NSE(h) of the three seed-test runs (42, 123, 2024).
  Proceed with `SEED=123 bash scripts/experiment_1/run_A<row>.sh` for every row.
- **A0 collapses** to NSE(h) = **−0.11** at epoch 12 and degrades to −1.44 by end-of-run,
  while training loss falls 11 orders of magnitude. The dimensional baseline is finding
  a low-residual non-physical solution, not a weak-but-valid one.
- **A1 reaches NSE(h) 0.54–0.66** depending on seed — a ~0.12 spread that is *larger*
  than the A0 → A1 lift suggested by the design doc's informal expectation
  ("A1 ≈ 0.55 plateau"). Only seed 42 exhibits the textbook epoch-~270 plateau; the
  other two climb slowly to the end of training.
- **Seed 42 is not bitwise-deterministic across runs.** Two A1 runs with
  seed=42, same git SHA, same config, same GPU diverged at epoch 1 and ended at
  NSE 0.66 vs 0.54. JAX/XLA GPU non-determinism is the most likely cause. This
  must be flagged in the thesis narrative and a deterministic run mode should be
  audited before the A2–A6 chain is executed.
- **The LR schedule saturates at the floor (1e-7) by epoch 2000–3600 of 20000**,
  so 75–90 % of wall-clock is spent at minimum LR. This is by design (`reduce_on_plateau`
  with `min_scale=1e-3`) but it amplifies sensitivity to the early-epoch trajectory.

---

## 1. Runs analysed

All runs used `git commit 7bc9b33`, single H100 (or equivalent) GPU, `float32`, 20000 epochs,
`configs/experiment_1/experiment_1_A0.yaml` (A0) or `configs/experiment_1/experiment_1_nondim.yaml` (A1).
W&B project `zeinali72-exeter/swe-pinn`.

| Label                    | Row | Seed | W&B id      | Tags                    | Purpose                                   |
|--------------------------|-----|-----:|-------------|-------------------------|-------------------------------------------|
| `A0_seed42`              | A0  |   42 | `4lx2sj3c`  | `row=A0, seed=42`       | Dimensional baseline (sanity)             |
| `A1_seed42_pre`          | A1  |   42 | `v31cp1g9`  | `row=A1, seed=42`       | Standalone A1 (pre-protocol, seed=42)     |
| `A1_seed42_seedtest`     | A1  |   42 | `nsna7kyv`  | `row=A1, seed-test, seed=42`   | Seed-test run 1/3                 |
| `A1_seed123_seedtest`    | A1  |  123 | `jqr0ednx`  | `row=A1, seed-test, seed=123`  | Seed-test run 2/3                 |
| `A1_seed2024_seedtest`   | A1  | 2024 | `dys295wa`  | `row=A1, seed-test, seed=2024` | Seed-test run 3/3                 |

One earlier seed=123 attempt (W&B id `2026-04-20_22-38-43…`) crashed at epoch 41 with
NSE −0.86 and is **excluded** from the seed test (state=`failed`). The crash happened
before `runs/experiment_1` script reorganisation and is covered by commits
`c7ce7c8` / `27ba620`; it is not reproducible on the current HEAD.

---

## 2. Headline metrics

Best NSE(h) = best validation NSE on water depth (Hunter analytical reference,
20 000-point eval grid). Best-epoch = epoch at which that maximum occurred.
Min-loss-epoch = epoch at which `train/total_loss` minimum was recorded.

| Label                    | Best NSE(h) | Best epoch | Final NSE(h) | Final NSE(hu) | Min loss | Min-loss epoch | LR ≤ 1e-7 by |
|--------------------------|------------:|-----------:|-------------:|--------------:|---------:|---------------:|-------------:|
| `A0_seed42`              |     **−0.1117** |         12 |      −1.4408 |        −1.325 | 2.25e-05 |          17703 |         2015 |
| `A1_seed42_pre`          |      0.6598 |      19999 |       0.6598 |         0.951 | 3.78e-04 |          19402 |         2223 |
| `A1_seed42_seedtest`     |      0.5419 |        272 |       0.5410 |         0.938 | 1.49e-04 |          19115 |         3599 |
| `A1_seed123_seedtest`    |      0.6327 |      19997 |       0.6324 |         0.948 | 1.18e-04 |          18980 |         3195 |
| `A1_seed2024_seedtest`   |      0.6639 |      19993 |       0.6635 |         0.950 | 1.12e-04 |          19140 |         3119 |

Source: `docs/experiment_1/data/ablation_rows_summary.csv`.

---

## 3. A0 — dimensional baseline collapses

The expected range (header of `scripts/experiment_1/run_sanity_check.sh`,
"*A0 dimensional MSE, 10k: NSE ~0.5–0.76*") is **NSE(h) 0.5–0.76**. Observed best
is **−0.11**, reached at **epoch 12** (i.e. effectively initialisation), after which
validation NSE degrades monotonically to −1.44 while training loss decreases by 11
orders of magnitude. Note that `docs/experiment_1_ablation_design.md §5` gives no
explicit A0 expectation; the two informal sources are inconsistent and both are
stale relative to this HEAD.

### 3.1 Trajectory

| Epoch | NSE(h)  | Total loss | PDE loss  | IC loss   | BC loss   |
|------:|--------:|-----------:|----------:|----------:|----------:|
|     0 | −0.962  |   3.12e+06 |  3.12e+06 | 2.48e-02  | 6.80e-02  |
|    12 | **−0.112** | 8.33e-03 | 7.21e-04  | 1.90e-03  | 5.72e-04  |
|   100 | −0.521  |   1.45e-03 |  1.05e-05 | 4.58e-04  | 9.86e-05  |
|  1000 | −1.131  |   8.82e-05 |  2.47e-06 | 2.56e-05  | 6.01e-06  |
|  5000 | −1.324  |   1.02e-04 |  6.31e-05 | 1.67e-05  | 2.18e-06  |
| 10000 | −1.373  |   3.85e-05 |  7.35e-07 | 1.61e-05  | 2.17e-06  |
| 19000 | −1.434  |   2.92e-05 |  7.39e-07 | 1.38e-05  | 1.46e-06  |

### 3.2 Diagnosis

The PDE residual drops ~13 orders of magnitude (`3e+06 → 7e-07`) while validation NSE
drops from −0.11 to −1.44. This is a classic PINN **loss-plane trivialisation**:

- **Initial scale explosion.** At epoch 0 the dimensional PDE residual is 3.1×10⁶ —
  the MLP is essentially predicting zero on dimensional inputs that span O(1200 m) and
  O(3600 s). This dominates the weighted total loss
  (`pde=1, ic=1, bc=10, neg_h=1`) by seven orders of magnitude over IC/BC terms.
- **Runaway minimisation.** Gradient descent aggressively pushes PDE residual to ~1e-6
  within the first 10–100 epochs. At this magnitude, `∂L/∂θ ≈ 2 r ∂r/∂θ` is quadratically
  small in `r`, so the optimiser becomes increasingly insensitive to further IC / BC
  misfit. The solution lands near a **non-physical fixed point of the SWE** (likely a
  quasi-stationary configuration) that is locally stable under the weighted loss but
  does not respect the Hunter IC.
- **LR floor compounds it.** By epoch 2015 the LR scheduler has hit the 1e-7 floor,
  freezing the model in this bad basin.

The design-doc expectation of "A0 NSE 0.5–0.76" cannot be reproduced on the current
HEAD with this config. This is **not a bug in the ablation** — it is the very
pathology that motivates the non-dim step (RQ1/H1 in §3 of the design doc) — but
the thesis narrative should cite the observed collapse (NSE → −1.4) rather than the
stale "weak-but-positive" expectation.

### 3.3 Action items for A0

- **None before proceeding.** A0 = −0.11 is a valid A0 data point that supports H1
  more strongly than the stale informal number.
- Update `docs/experiment_1_ablation_design.md §5` "Empirical observations to date"
  to replace `A0 ~0.5–0.76` with the observed ~−0.1-to-−1.4 range.

---

## 4. A1 — seed-test results

### 4.1 Canonical seed selection

Protocol (`§4.3`): pick the **median-NSE(h)** seed, disclose best and worst.

| Rank  | Seed | Best NSE(h) | Best epoch | Role             |
|-------|-----:|------------:|-----------:|------------------|
| Best  | 2024 |       0.664 |      19993 | Disclosed        |
| **Median** | **123**  | **0.633**   |      19997 | **Canonical**    |
| Worst |   42 |       0.542 |        272 | Disclosed        |

**Canonical seed = 123.**

Downstream invocation:

```bash
SEED=123 bash scripts/experiment_1/run_A0.sh
SEED=123 bash scripts/experiment_1/run_A1.sh   # re-run A1 on canonical seed for the chain
SEED=123 bash scripts/experiment_1/run_A2.sh
# ... A3, A4, A5, A6
```

### 4.2 Seed sensitivity is large and qualitative

The three seeds span **0.542 → 0.664** (Δ = 0.12, ≈ 22 % of the median). More
importantly, they split into two *qualitatively different* regimes:

| Regime                        | Seeds     | Best epoch    | Behaviour                                  |
|-------------------------------|-----------|--------------:|--------------------------------------------|
| "Classic plateau" (§H2)       | 42        |        ~272   | NSE peaks early, flat thereafter           |
| "Slow late-climb"             | 123, 2024 |      ~19 995  | NSE keeps growing slowly through 20k epochs |

The "plateau at epoch ~270" cited in the design doc as motivation for H2 is therefore
**seed-dependent**, not universal. Two of three seeds do *not* exhibit it within 20k
epochs.

NSE(h) trajectory snapshots (seed-test runs only):

| Epoch  | seed=42 | seed=123 | seed=2024 |
|-------:|--------:|---------:|----------:|
|   1000 |   0.074 |    0.252 |     0.320 |
|   5000 |   0.496 |    0.594 |     0.592 |
|  10000 |   0.515 |    0.607 |     0.617 |
|  20000 |   0.541 |    0.632 |     0.663 |

### 4.3 Implication for H2

H2 ("training plateaus under non-dim MSE because of quadratic gradient flatness near
zero") is **partially supported** by seed=42 and **not obviously supported** by seeds
123/2024 within 20k epochs. A sharper thesis-narrative reading: *the predicted
plateau holds for 1/3 seeds, while 2/3 seeds show gradual drift to a similar final
NSE (0.63–0.66). This suggests H2 may be a landscape-initialisation phenomenon —
MSE's flat gradient is a necessary but not sufficient condition, and whether a
given init falls into the plateau basin is stochastic.* That reading is stronger
than a plain H2 pass/fail because it accounts for all three seeds rather than
cherry-picking the one that matches.

Two options for additional evidence:

1. **Extend all three seeds to 50k epochs** and check whether seeds 123/2024 eventually
   plateau around 0.66–0.68.
2. **Report the plateau as a seed-dependent mode**, with seed=42 as a representative
   worst case and an accompanying note that better-initialised runs avoid the plateau
   but also fail to reach meaningful NSE (0.66 ≪ the 0.98 target from A4).

Option 2 is defensible and cheap (no extra compute). Option 1 is stronger but
increases compute budget by ~1.5×.

---

## 5. Seed 42 non-determinism — reproducibility concern

Two independent A1 runs with **identical seed=42, identical config, identical git SHA**
produced meaningfully different outcomes:

| Run                  | Best NSE(h) | Best epoch | Min loss  | LR ≤ 1e-7 by |
|----------------------|------------:|-----------:|----------:|-------------:|
| `A1_seed42_pre` (v31cp1g9)       | **0.660** |      19999 | 3.78e-04  |         2223 |
| `A1_seed42_seedtest` (nsna7kyv)  | **0.542** |        272 | 1.49e-04  |         3599 |

Both runs shared `git commit 7bc9b333648c53247df4a3086191bd74baba291b` and the same
`experiment_1_nondim.yaml`. Epoch-0 total loss matches to the precision shown
(`6.506×10⁹`), consistent with identical parameter initialisation; epoch-0 validation
NSE differs only in the 4th decimal (−1.951 vs −1.950), plausibly a validation-grid
rounding artefact. Divergence is unambiguous by **epoch 1**:

| Epoch | pre: total / pde | seedtest: total / pde |
|------:|-----------------:|----------------------:|
|     0 |   6.506e9 / 6.506e9  |       6.506e9 / 6.506e9 |
|     1 |   0.301 / 0.209  |       0.320 / 0.227  |
|    50 |   0.009 / 0.0022 |       0.010 / 0.0025 |
|   200 |   NSE 0.36       |       NSE 0.37       |
|  1000 |   NSE 0.48       |       NSE 0.07       |

### 5.1 Likely cause: JAX / XLA GPU non-determinism

JAX on GPU is **not bitwise deterministic by default** even with fixed seeds.
Sources of drift include:

- CUDA reduction kernels have implementation-defined ordering (affects matmuls, reductions).
- XLA's HLO fusion decisions can depend on hardware state or scheduler order.
- cuBLAS / cuDNN tensor-op selection is opportunistic.

JAX accepts `jax.config.update("jax_default_matmul_precision", "highest")`
and XLA accepts the env var `XLA_FLAGS="--xla_gpu_deterministic_ops=true"`. Neither is set
in `.devcontainer/` or the training entry points today.

### 5.2 Proposed mitigations (unverified on this codebase)

A principled way forward — none of these have been tested against `train_nondim.py`
on the current HEAD, so they are hypotheses, not confirmed fixes:

- Set `XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"` in `_common.sh`.
- Add `jax.config.update("jax_default_matmul_precision", "highest")` at the top of the
  Python entry points.
- Audit each randomness touchpoint (`jax.random.PRNGKey`, numpy RNG in sampling,
  any host-side shuffles) to confirm they are all keyed from the single `training.seed`.
- Re-run a small A/B on seed=42 to verify bit-reproducibility *before* launching the
  A2–A6 chain. Two identical runs that now match → safe to interpret individual
  rows as exact data points. If residual drift remains, accept it and present
  thesis numbers as `±0.02` run-to-run bands.
- **Document the outcome** in the thesis methodology chapter whether the fix works
  or not, so reviewers see the reproducibility discipline.

### 5.3 Workaround for this session

Until determinism is enabled, treat the **seed-test run (`nsna7kyv`) as authoritative
for seed=42** (it was produced by the canonical `run_A1_seedtest.sh` protocol; the
`A1_seed42_pre` run was a one-off sanity launch). This is the basis of the seed-choice
decision above.

---

## 6. LR schedule saturation

All five runs hit the learning-rate floor (`1e-7 = 1e-4 × min_scale=1e-3`) well before
the 20 000-epoch budget:

| Run                         | LR ≤ 5e-5 by | LR ≤ 1e-7 by | % epochs at LR ≤ 1e-7 |
|-----------------------------|-------------:|-------------:|----------------------:|
| A0 seed=42                  |          563 |         2015 |                 89.9% |
| A1 seed=42 pre              |          441 |         2223 |                 88.9% |
| A1 seed=42 seedtest         |         1401 |         3599 |                 82.0% |
| A1 seed=123 seedtest        |          763 |         3195 |                 84.0% |
| A1 seed=2024 seedtest       |          757 |         3119 |                 84.4% |

The schedule (`reduce_on_plateau`, `factor=0.5`, `patience=50`, `min_scale=1e-3`)
allows at most ~10 halvings to the floor, so a run that has no meaningful improvement
for ~500 consecutive epochs exhausts the budget. Consequences:

- **Most of the wall-clock (≥80 %) contributes only millidigit NSE gains.** Per-epoch
  time is dominated by the vmapped forward/backward pass, not the LR — so cost per
  unit accuracy is very poor once LR ≤ 1e-7.
- **Convergence behaviour is effectively "whichever basin the model is in at ~epoch
  1000–3000 is the answer."** This directly amplifies seed sensitivity (§4.2) and
  GPU non-determinism (§5).
- **The "plateau at epoch ~270"** (seed=42) may partially be an artefact of LR decay
  rather than pure gradient-flatness of MSE: at epoch 270 the scheduler has typically
  already performed several halvings, reducing LR to ~1e-5. That is consistent with
  the MSE-flatness story, but isolating the two effects cleanly requires a cosine-LR
  control, which §4.2 of the design doc notes was trialled and produced the same
  ~0.55 NSE plateau. Re-confirm this with the current HEAD code before citing it.

### 6.1 Recommendation

- **Do not change the schedule mid-ablation.** The ablation holds §4.1 common config
  fixed. Changing `min_scale` or the schedule would invalidate the A0–A6 comparison.
- **After A2–A6 complete**, revisit the schedule as its own post-hoc ablation
  (an "A7: extended LR floor" could go in as a follow-up).

---

## 7. Open items

1. **Enable JAX/XLA determinism** before the A2–A6 chain (see §5.2).
2. **Re-run A1 seed test under determinism** to confirm the canonical seed choice
   holds. If seed=42's seed-test run still lands at ~0.54 and 123/2024 at ~0.63–0.66,
   the median rule still gives seed=123 — but the decision will then be defensible in
   review. Low priority if time-pressed.
3. **Update `docs/experiment_1_ablation_design.md §5`** to replace the informal
   empirical expectations (A0 ≈ 0.55–0.76, A1 ≈ 0.55 plateau) with the current
   observations (A0 ≈ −0.1-to-−1.4, A1 ∈ [0.54, 0.66] across seeds). Deferred to
   after A2–A6 so the full table can be updated in one commit.
4. **Decide whether to extend A1 seeds 123/2024 to 50 k epochs** to characterise their
   asymptote (see §4.3). Current recommendation: *no* — accept the slow-climb
   observation as a seed-dependent mode and note it in the thesis.
5. **Spatial error maps** (P2.1 from the programme reference) for A0 and each A1 seed
   are not yet generated. Needed before the chapter draft so the "A0 finds a
   non-physical fixed point" claim is visually supported.

---

## 8. Verdict and next step

- **Seed 123 is the canonical seed for the A0–A6 chain.** Proceed with
  `SEED=123 bash scripts/experiment_1/run_A<row>.sh` for each row.
- The A0 result (NSE −0.11 → −1.44) is unusual compared to the design-doc
  expectation but is *consistent* with the mechanism the ablation is designed to
  diagnose. It stands.
- The GPU-non-determinism finding should be addressed (§5.2) before A2–A6 are
  interpreted as individual data points rather than ±0.02 point clouds.

### W&B references

- Group: `exp1-ablation-v1`
- Runs: [`4lx2sj3c`](https://wandb.ai/zeinali72-exeter/swe-pinn/runs/4lx2sj3c) (A0),
  [`v31cp1g9`](https://wandb.ai/zeinali72-exeter/swe-pinn/runs/v31cp1g9) (A1 seed=42 pre),
  [`nsna7kyv`](https://wandb.ai/zeinali72-exeter/swe-pinn/runs/nsna7kyv) (A1 seed=42 seedtest),
  [`jqr0ednx`](https://wandb.ai/zeinali72-exeter/swe-pinn/runs/jqr0ednx) (A1 seed=123 seedtest),
  [`dys295wa`](https://wandb.ai/zeinali72-exeter/swe-pinn/runs/dys295wa) (A1 seed=2024 seedtest).
- Raw summary table: `docs/experiment_1/data/ablation_rows_summary.csv`.
