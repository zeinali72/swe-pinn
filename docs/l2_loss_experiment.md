# L2-Norm Loss for Experiment 1 (non-dim)

## Motivation

Under standard MSE losses, Experiment 1 training plateaus around NSE ≈ 0.55
near epoch ~270 and fails to improve — regardless of learning-rate schedule
(reduce-on-plateau, cosine) or non-dimensionalization. Diagnosis: as each
term approaches its optimum, MSE goes **quadratically flat**, so gradients
vanish before the network has actually converged on the solution.

Switching from MSE to the **L2 norm** of the residual (i.e. `sqrt(sum(r²))`)
replaces the quadratic bowl with a linear one near zero. The gradient
magnitude stays on the same order as the residual itself, so training keeps
making progress past the point where MSE gradients die out.

**Observed effect:** NSE jumped from ~0.55 (MSE) to >0.76 (L2) within the
same epoch budget on Experiment 1 non-dim, config `experiment_1_nondim_l2.yaml`.

## The equations

Given a residual vector `r = (r₁, ..., r_N)`:

| Quantity       | Formula                     |
|----------------|-----------------------------|
| L2 norm        | `‖r‖₂ = sqrt(Σ rᵢ²)`         |
| Squared L2     | `‖r‖₂² = Σ rᵢ²`              |
| MSE            | `(1/N) Σ rᵢ²`                |
| RMSE           | `sqrt(MSE) = ‖r‖₂ / sqrt(N)` |

Note that L2 and RMSE differ by a constant `sqrt(N)`. For Experiment 1
(fixed batch size across terms) that constant is absorbable into the LR,
but once batch counts differ between loss terms the distinction matters —
L2 rewards terms that are sampled more densely, RMSE hides the sample count.

## The epsilon trick

`sqrt` has an infinite gradient at zero:

```
d/dx sqrt(x) = 1 / (2·sqrt(x))  →  ∞   as x → 0
```

If any loss term approaches zero (common for IC in data-free PINN runs),
the autodiff gradient explodes and destabilizes training. Fix:

```python
_SQRT_EPS = 1e-12
L = sqrt(sum_of_squares + _SQRT_EPS)
```

The epsilon sits inside the root, so the gradient near zero caps at
`1 / (2·sqrt(eps)) ≈ 5e5` instead of diverging. For values `>> 1e-12` the
output is numerically identical to `sqrt(sum_of_squares)`.

This is standard practice in numerical optimization (Charbonnier loss,
pseudo-Huber, etc.) and is cheap — one add per term per step.

## Implementation pattern

All shared loss helpers in `src/losses/` return **MSE**
(i.e. `jnp.mean(residual²)`). To get true L2 norm at the call site:

```python
_SQRT_EPS = 1e-12

def _l2(sum_sq):
    return jnp.sqrt(sum_sq + _SQRT_EPS)

# Recover sum from mean by multiplying by N before the sqrt:
n_pde = pde_batch.shape[0]
terms['pde'] = _l2(compute_pde_loss(...) * n_pde)
```

For terms assembled from multiple sub-residuals (e.g. 4-wall BC), sum the
sums before a single sqrt — this treats the boundary as one concatenated
residual vector and applies a single L2 norm.

## When to use L2 vs MSE

**Prefer L2** when:
- Training plateaus while individual loss terms still look non-zero.
- You suspect gradient-flatness is the bottleneck, not loss imbalance.
- Different terms have comparable sample counts (or you want to expose
  sample-count differences rather than hide them).

**Keep MSE** when:
- Losses are already large and training is progressing normally — MSE's
  quadratic amplification of big errors is helpful early on.
- You're running a weight-balancing scheme (ReLoBRaLo, SoftAdapt) that
  assumes smooth loss ratios; L2 and MSE both work but the ratios behave
  differently and you may need to re-tune temperature / EMA.

## Files

- `experiments/experiment_1/train_nondim_l2.py` — training script with the
  `_l2` helper and true-L2 `compute_losses`.
- `configs/experiment_1/experiment_1_nondim_l2.yaml` — matching config.

## Run

```bash
WANDB_MODE=disabled python -m experiments.experiment_1.train_nondim_l2 \
    --config configs/experiment_1/experiment_1_nondim_l2.yaml
```
