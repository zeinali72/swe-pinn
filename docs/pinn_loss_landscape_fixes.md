# PINN Loss-Landscape Fixes for Experiment 1

> Context: Experiment 1 (1200×100 m flat channel, Hunter 2005, T=3600 s, data-free)
> shows high seed-to-seed variance in NSE_h (0.55 ± lottery, single lucky run at
> 0.77). FP64 eliminated the numerical-noise hypothesis — the problem is the
> multi-objective loss landscape itself. The three options below attack it
> structurally rather than by weight tuning.

---

## Option 1 — Causal Training (Wang, Sankaran & Perdikaris, 2022)

### Problem
The SWE PDE is time-dependent on `t ∈ [0, 3600]`. Uniform `(x,t)` sampling lets
the optimizer minimise the residual at `t = 3600` before `t = 0` is learned —
a violation of physical causality. The network converges to a solution that
satisfies the PDE *everywhere at once* in a non-causal way, which is an easy
local minimum far from the true trajectory.

### Idea
Weight each collocation residual by an exponential of the *cumulative residual
at earlier times*. A point at time `t_i` contributes to the loss only once the
network has driven residuals at all `t_j < t_i` close to zero.

### Formulation
Bin collocation points by time into `N_t` chunks `t_1 < t_2 < ... < t_Nt`.
Define the mean residual within chunk `i`:

```
L_r(t_i) = mean over (x,y) in chunk i of |R_pde(x, y, t_i; θ)|²
```

Causal weight:

```
w_i = exp( −ε · Σ_{k<i} L_r(t_k) )       (stop-gradient on the sum)
```

Weighted PDE loss:

```
L_pde_causal = (1 / N_t) · Σ_i w_i · L_r(t_i)
```

`ε ≈ 100` is the Wang 2022 default for normalised PDEs; for Experiment 1 start
with `ε ∈ {1, 10, 100}` and pick the one where `w_i` at the final time reaches
`≈1` only in the last 20% of training.

### Expected impact
On 1D/2D time-dependent PDEs in the PINN literature this routinely converts
NSE from 0.5 → 0.95. It directly targets the exact failure mode we see: the
"bad" seeds plateau at epoch ~275 because they found a non-causal minimum and
the loss gradient gives them no reason to leave.

### Cost
- One new hyperparameter `ε`
- ~30 LOC in `src/training/losses.py`: sort collocation points by `t`, compute
  chunked residual, apply `jax.lax.stop_gradient` to the cumulative sum,
  multiply, mean.
- No architectural change.

### Applies to
Experiments 1, 4, 6–9, 11 (all time-dependent PINNs). Does **not** help
steady-state problems.

---

## Option 2 — Hard Initial & Boundary Condition Constraints

### Problem
Soft constraints put IC, BC, and PDE into a single weighted sum:

```
L = λ_pde · L_pde + λ_ic · L_ic + λ_bc · L_bc + λ_neg · L_neg_h
```

This is a multi-objective problem with a Pareto front. The seed determines
*which point on the front* the optimizer converges to — that's the basin
lottery. Worse, the IC constraint (a 1D manifold at `t=0`) has measure zero
against the PDE points, so `λ_ic` must be large to be felt — but then it
fights the PDE term.

### Idea
Bake IC and BC into the ansatz so they are satisfied *by construction*,
regardless of network weights. The network then only has to minimise `L_pde`
(and `L_neg_h`, which Option 3 also removes). Single objective, no lottery.

### Formulation for Experiment 1

Let `NN_h, NN_u, NN_v : (x, y, t) → ℝ` be the raw network outputs
(post non-dim scaling).

**Initial condition** (`h(x,y,0) = h_0(x,y)`, `u = u_const`, `v = 0`):

```
h_hat(x,y,t)  = h_0(x,y)         +  φ_t(t) · NN_h(x,y,t)
hu_hat(x,y,t) = h_0(x,y) · u_0   +  φ_t(t) · NN_hu(x,y,t)
hv_hat(x,y,t) = 0                +  φ_t(t) · NN_hv(x,y,t)
```

where `φ_t(t) = 1 − exp(−t / τ)` is a smooth gate that is exactly 0 at `t=0`
and approaches 1 for `t ≫ τ`. Choose `τ ≈ t_final / 20 = 180 s` so the gate
opens well within the training window.

**Boundary conditions.** For the flat channel:
- Left inflow `x = 0`: `h = h_in`, `hu = hu_in` prescribed
- Right outflow `x = L_x`: soft (open) — leave it in the loss or use `∂h/∂x=0`
- Top/bottom `y = 0, L_y`: slip walls `hv = 0`

Inflow can be hard-enforced with a spatial gate `φ_x(x) = 1 − exp(−x/λ)`:

```
h_hat  = h_in               +  φ_t · φ_x · NN_h
hu_hat = hu_in              +  φ_t · φ_x · NN_hu
```

Top/bottom slip `hv = 0` can be hard-enforced with
`φ_y(y) = y · (L_y − y) / (L_y/2)²`:

```
hv_hat = 0                  +  φ_t · φ_y · NN_hv
```

After these substitutions, `L_ic ≡ 0`, `L_bc_left ≡ 0`, `L_bc_top_bottom ≡ 0`
*by construction*. Only the outflow BC (if you choose to keep it soft) and
`L_pde` remain in the loss.

### Expected impact
Removes IC and most BC losses → collapses the Pareto front from 4D to 1D → the
seed lottery largely disappears. In the Raissi et al. and Lu et al. studies,
hard constraints typically add **+0.1 to +0.2 NSE** and cut seed variance by
3–5×. It is the single most reliable "works on my first try" intervention for
small-domain PINNs.

### Cost
- ~50 LOC in `src/models/`: wrap the MLP output with an ansatz module that
  applies the gates. Keep the raw MLP unchanged so you can A/B.
- One new hyperparameter `τ` (and optionally `λ` for the spatial inflow gate).
- The gates must be differentiable and smooth; `1 − exp(−t/τ)` is both.
- **Gotcha:** the IC function `h_0(x,y)` must itself be smooth and
  `jax.grad`-able. For Experiment 1 Hunter 2005 it's a simple closed form.

### Applies to
Experiments 1, 2, 4, 7, 8, 9, 10, 11 with varying complexity. For irregular
geometries (Exp 10, 11) the spatial gates become level-set distance functions,
which is harder but still standard.

### Reference
Lu, Pestourie, Yao, Wang, Verdugo, Johnson (2021), *Physics-informed neural
networks with hard constraints for inverse design* — canonical reference for
the ansatz construction.

---

## Option 3 — Structural Positivity via Output Parameterization

### Problem
Water depth `h` must satisfy `h ≥ 0` everywhere. A vanilla MLP output can
(and does) go slightly negative near wet/dry regions, triggering the soft
penalty:

```
L_neg_h = mean( relu(−h)² )
```

Evidence from the three reference runs: all had `train/neg_h ≈ 0.0156`
persistently. Our FP64 run drove this to `1.9e-6` (still nonzero). Any nonzero
`L_neg_h` produces a gradient that *opposes* the PDE gradient, because the
PDE near the wet/dry boundary wants `h → 0` from above while the penalty
wants `h` strictly positive.

### Idea
Parameterize the *raw* output through a monotone positive function so that
`h` is non-negative by construction:

```
h(x,y,t) = softplus(NN_h_raw(x,y,t)) + eps
```

`softplus(z) = log(1 + exp(z))` is smooth, differentiable, monotone, and
approaches `z` for large `z`, `0` for very negative `z`. Adding a small
`eps` (e.g. `1e-6`) keeps the PDE residual well-conditioned at the wet/dry
interface (division by `h` appears in friction / Froude terms).

An alternative: `h = eps + |NN_h_raw|`. Simpler but non-smooth at `NN=0`, so
softplus is preferred.

### Momentum variables
For `hu, hv` there is no sign constraint — keep them linear. But if you
enforce `h` via softplus, compute velocities safely:

```
u = hu / (h + eps)
v = hv / (h + eps)
```

### Expected impact
- Removes the `L_neg_h` term from the loss entirely (one fewer weight to tune,
  one fewer Pareto axis).
- Removes gradient conflict near wet/dry interfaces — particularly important
  for Experiments 4, 8, 10, 11 which have true wet/dry fronts.
- Expected NSE gain alone: +0.02 to +0.05 for Experiment 1 (it's a small
  effect on a flat channel that never really goes dry), but much larger
  (+0.1+) for Experiments 4, 8, 11.

### Cost
- ~5 LOC in the model forward pass.
- No new hyperparameter.
- **Gotcha:** the IC ansatz in Option 2 assumes `NN_h` is the raw output. If
  combining Options 2 and 3, apply softplus *before* the ansatz gating:
  `h_hat = h_0 + φ_t · (softplus(NN_h_raw) − h_0)` — this preserves both hard
  IC and structural positivity simultaneously.

### Applies to
Experiments 1–11 (universal).

---

## Combined Recommendation

Implement in this order — each is independent and each provides a measurable
step-change:

1. **Option 3** first (5 LOC, universal, no new hyperparameter) — establishes
   that `neg_h` pathology is gone in all future runs.
2. **Option 2** second (hard IC + left/top/bottom BCs) — this alone should
   collapse the seed lottery on Experiment 1. At this point run 5 seeds; you
   should see mean ≳ 0.85 NSE with std < 0.05.
3. **Option 1** last, only if (1)+(2) do not reach the target NSE, or when
   moving to longer-horizon experiments (4, 6–9, 11) where causality becomes
   the binding constraint.

All three are additive and architecturally compatible with the existing
`MLP`, `Fourier`, and `DGM` backbones in `src/models/`.
