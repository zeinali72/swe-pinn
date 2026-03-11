---
name: pinn-experiment
description: Set up and run PINN training experiments using the JAX/Flax/Optax stack for solving the 2D Shallow Water Equations. Use this skill whenever the user asks to set up a new experiment, write a training loop, configure the PINN solver, implement a loss function, add a sampling strategy, debug training convergence, or modify the experiment pipeline. Also trigger for "training loop", "collocation points", "loss function", "physics residual", "boundary condition", "experiment setup", "SWE residual", "PINN training", or any request about the core training pipeline.
---

# PINN Experiment — JAX Training Loop and Experiment Setup

This skill encodes the complete PINN training pipeline for solving the 2D Shallow Water Equations (SWE) using JAX, Flax, and Optax.

## Framework Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Differentiation & JIT | JAX | `jax.grad`, `jax.vmap`, `@jax.jit` |
| Neural networks | Flax Linen | `nn.Module`, explicit parameter management |
| Optimisation | Optax | Adam, reduce-on-plateau, global norm clipping |
| HPO | Optuna | 100 Sobol + 50 TPE trials, SQLite backend |
| Tracking | Aim | Defer to `pinn-training-logger` skill |

## Architectures

Three architectures, all outputting `(h, hu, hv)`:

### 1. MLP
Standard feedforward network. Configurable depth and width. Default activation: `tanh`.

### 2. Fourier-MLP
Fourier feature mapping at the input layer (overcomes spectral bias), followed by a standard MLP. Key hyperparameters: `mapping_size`, `frequency_scale`.

### 3. DGM (Deep Galerkin Method)
Highway-style gating layers that inject raw coordinates `(x, y, t)` at every hidden layer. Particularly effective for sharp gradients.

Architecture is selected via `config.model.name` and dynamically imported from `src/models.py`.

## Loss Function Structure

The composite loss for the conservative-form SWE:

```
L_total = w_pde * L_pde + w_ic * L_ic + w_bc * L_bc + w_data * L_data
```

| Term | Description | Active |
|------|-------------|--------|
| `L_pde` | MSE of SWE residuals (continuity + x-momentum + y-momentum) at collocation points | Always |
| `L_ic` | MSE of initial condition (h=0, hu=0, hv=0 at t=0 for dry-bed) | Always |
| `L_bc` | MSE of BCs: slip (u·n=0 at walls), inflow (prescribed discharge) | Always |
| `L_data` | MSE against ICM observational data | Experiments 3-8 only |

Weights `w_pde, w_ic, w_bc, w_data` are hyperparameters that can be static or adaptive (GradNorm, SoftAdapt, NTK).

Read `references/loss_functions.py` for the template implementation of the SWE residual and composite loss.

## Sampling Strategies

Sampling depends on domain geometry:

| Experiments | Domain | Strategy |
|------------|--------|----------|
| 1-6 | Rectangular | Latin Hypercube Sampling (LHS) |
| 7-8 | Irregular | Area-weighted triangle sampling (interior), CDF-weighted segment sampling (boundary) |
| 2 | Rectangular + building | LHS with Boolean masking to exclude building interior |
| 8 | Urban mesh | Buildings excluded at triangulation stage (no masking) |

**All experiments**: collocation points are dynamically resampled every epoch.

Read `references/sampling.py` for implementations of LHS, area-weighted triangle sampling, and CDF-weighted segment sampling.

## Training Loop Pattern

```python
# 1. Initialise
params = model.init(rng, dummy_input)
opt_state = optimizer.init(params)

# 2. JIT-compile the training step
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    grads = optax.clip_by_global_norm(grads, max_norm)  # or via optimizer chain
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 3. Training loop
for epoch in range(max_epochs):
    batch = resample_collocation_points(rng)  # dynamic resampling
    params, opt_state, loss = train_step(params, opt_state, batch)

    if epoch % log_interval == 0:
        log_metrics(epoch, loss)

    if epoch % val_interval == 0:
        val_metrics = validate(params)
        check_convergence(val_metrics)
        update_lr_scheduler(val_metrics)

    save_best_checkpoint(params, val_metrics)
```

## Key Conventions

- **Precision**: float32 (established in Experiment 1)
- **Manning's n**: 0.05 unless otherwise specified
- **Bed slope**: bilinear interpolation from DEM (Experiments 3+)
- **Coordinates**: spatial in metres, temporal in seconds
- **JIT**: compile the training step function
- **vmap**: use for batched PDE residual computation
- **Gradient clipping**: global norm clipping via Optax chain

## Data Ratio Methodology (Experiments 3-8)

```
total_dataset_size = (simulation_duration / temporal_output_interval) * n_spatial_nodes
```

The training fraction is determined empirically in Experiment 3 and applied proportionally to subsequent experiments.

## Loss Weighting Strategies

| Strategy | Module | Description |
|----------|--------|-------------|
| Static | config | Fixed weights from YAML |
| GradNorm | `src/gradnorm.py` | Balances gradient magnitudes across loss terms; requires separate optimizer for weights |
| SoftAdapt | `src/softadapt.py` | Rate-of-change-based adaptive weighting |
| NTK | `src/ntk.py` | Weights based on Neural Tangent Kernel traces |

## Convergence Criteria

Training stops when:
1. Loss plateau detected (learning rate scheduler reduces below threshold), OR
2. Maximum epochs reached, OR
3. Early stopping triggered (validation metric hasn't improved for N epochs)

## Reference Files

- `references/loss_functions.py` — Template for composite loss with SWE residuals
- `references/sampling.py` — LHS, area-weighted triangle, CDF-weighted segment sampling
- `references/architectures.py` — Flax module definitions for MLP, Fourier-MLP, DGM
