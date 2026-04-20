# Developer Guide

This guide is for contributors who want to **extend or modify** the codebase, not just run it. For installation and quick-start commands see the [README](../README.md). For the authoritative specification of every experiment's metrics, plots, and logged values see the [Experimental Programme Reference](experimental_programme_reference.md).

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [How the subsystems wire together](#2-how-the-subsystems-wire-together)
3. [Config schema](#3-config-schema)
4. [Subsystem reference](#4-subsystem-reference)
   - [Config](#41-config--srcconfigpy)
   - [Models](#42-models--srcmodels)
   - [Physics](#43-physics--srcphysics)
   - [Losses](#44-losses--srclosses)
   - [Data](#45-data--srcdata)
   - [Training](#46-training--srctraining)
   - [Metrics](#47-metrics--srcmetrics)
   - [Prediction & Inference](#48-prediction--inference)
   - [Monitoring & Checkpointing](#49-monitoring--checkpointing)
   - [Loss Balancing (advanced)](#410-loss-balancing--srcbalancing)
5. [Extension recipes](#5-extension-recipes)
   - [Add a new experiment](#51-add-a-new-experiment)
   - [Add a new loss term](#52-add-a-new-loss-term)
   - [Add a new model architecture](#53-add-a-new-model-architecture)
   - [Add a new evaluation metric](#54-add-a-new-evaluation-metric)
6. [Critical constraints & pitfalls](#6-critical-constraints--pitfalls)
7. [Reading these docs](#7-reading-these-docs)

---

## 1. Architecture overview

```
configs/               ← YAML hyperparameters (no Python, no logic)
experiments/           ← Per-experiment entry points. Import from src/. Produce a model.
src/                   ← Reusable library. Zero experiment-specific logic.
scripts/               ← Consume models/results to produce something else (plots, benchmarks).
test/                  ← Unit tests for src/ and smoke tests for experiments.
optimisation/          ← Optuna HPO drivers.
docs/                  ← This file and design references.
```

**Golden rule for where code lives:**
| Produces a model → | `experiments/` |
|---|---|
| Consumes a model/results → | `scripts/` |
| Reusable pure logic → | `src/` |

---

## 2. How the subsystems wire together

Training data flow (left to right):

```
YAML config
    │
    ▼
src.config.load_config()         ← sets global DTYPE and EPS; returns dict
    │
    ├──► src.training.setup.init_model_from_config()   → (model, params)
    │
    ├──► src.training.optimizer.create_optimizer()     → optax chain
    │
    ├──► src.data.*  (sampling, batching, loading)     → collocation batches
    │
    ▼
src.training.loop.run_training_loop()   ← epoch loop orchestrator
    │
    ├── each epoch:
    │       src.training.step.train_step_jitted()
    │           │
    │           ├── experiment's compute_losses_fn()
    │           │       ├── src.losses.pde.compute_pde_loss()
    │           │       │       └── src.physics.swe.SWEPhysics (flux Jacobians + source)
    │           │       ├── src.losses.boundary.*
    │           │       └── src.losses.data_loss.compute_data_loss()
    │           │
    │           └── src.losses.composite.total_loss()  → weighted scalar
    │
    ├── every N epochs:
    │       validation via src.metrics.accuracy.compute_validation_metrics()
    │       src.checkpointing.saver.CheckpointManager.update()
    │       src.monitoring.wandb_tracker.WandbTracker.log()
    │
    └── end of training:
            src.inference.runner.run_inference()   (if called separately)
                ├── src.predict.predictor.Predictor  → batched forward pass
                ├── src.metrics.*                    → all evaluation metrics
                └── src.inference.reporting.*        → YAML / CSV / plots
```

---

## 3. Config schema

**Always load configs via `src.config.load_config(path)`** — never parse YAML directly. This function sets the global `DTYPE` and `EPS` that every module reads via `get_dtype()` / `get_eps()`.

Full annotated schema:

```yaml
# ── Experiment identity ─────────────────────────────────────────────────────
experiment:
  name: experiment_1          # logical name, used for output dirs and W&B tags
  scenario: flat_channel      # optional override for SCENARIO_ASSET_ALIASES

# ── Domain geometry ─────────────────────────────────────────────────────────
domain:
  lx: 1200.0                  # channel length (m)
  ly: 100.0                   # channel width (m)
  t_final: 3600.0             # simulation end time (s)
  x_min: 0.0                  # optional origin offset
  y_min: 0.0

# ── Collocation point counts ─────────────────────────────────────────────────
sampling:
  n_points_pde: 2000          # interior physics residual points
  n_points_ic: 500            # initial-condition points (t=0)
  n_points_bc_domain: 200     # domain boundary points (walls)
  n_points_bc_inflow: 200     # inflow boundary points

# ── Model architecture ───────────────────────────────────────────────────────
model:
  name: FourierPINN           # FourierPINN | MLP | DGMNetwork | DeepONet
  width: 128                  # neurons per hidden layer
  depth: 4                    # number of hidden layers
  ff_dims: 256                # Fourier feature output dimension (FourierPINN only)
  fourier_scale: 1.0          # frequency scale for Fourier features
  output_dim: 3               # always 3: [h, hu, hv]
  bias_init: 0.0              # constant initialiser for bias terms

# ── Physics ──────────────────────────────────────────────────────────────────
physics:
  g: 9.81                     # gravitational acceleration (m/s²)
  n_manning: 0.03             # Manning roughness coefficient
  inflow: null                # scalar source (m²/s) or null

# ── Loss weights ─────────────────────────────────────────────────────────────
loss_weights:
  pde: 1.0
  ic: 1.0
  bc: 1.0
  data: 0.0                   # set > 0 only when training data is present
  neg_h: 1.0                  # penalty for negative water depths

# ── Training optimiser ───────────────────────────────────────────────────────
training:
  seed: 42
  epochs: 5000
  learning_rate: 1.0e-3
  batch_size: 512
  num_batches: 4              # gradient steps per epoch (= scan length)
  clip_norm: 1.0              # gradient clipping threshold
  validation_interval: 100    # epochs between validation metric computations
  reduce_on_plateau:
    factor: 0.5               # LR multiplier on plateau
    patience: 20              # number of intervals with no improvement
    min_lr: 1.0e-6

# ── Data files ───────────────────────────────────────────────────────────────
data:
  validation_file: val_lhs_points.npy    # path relative to data dir, or absolute
  training_file: null                    # optional ICM training data

# ── Precision ────────────────────────────────────────────────────────────────
device:
  dtype: float32              # float32 | float64 (float64 enables JAX x64)

numerics:
  eps: 1.0e-6                 # water-depth threshold for masking

# ── W&B tracking ─────────────────────────────────────────────────────────────
wandb:
  project: swe-pinn
  enable: true
  tags: []                    # extra string tags appended to every run
```

String floats (`"1e-5"`) are converted to Python `float` automatically by `load_config`.

---

## 4. Subsystem reference

### 4.1 Config — `src/config.py`

```python
from src.config import load_config, get_dtype, get_eps

cfg = load_config("configs/experiment_1/config.yaml")
# Returns a plain dict. Sets module-level DTYPE and EPS as side effects.

dtype = get_dtype()   # jnp.float32 or jnp.float64
eps   = get_eps()     # e.g. 1e-6
```

**Why getters instead of a direct import?** HPO runs multiple trials in one process, each calling `load_config` with a different config. A `from src.config import DTYPE` captures the value at import time and won't see updates. The getter functions always return the current module-level value.

---

### 4.2 Models — `src/models/`

All models are Flax `nn.Module` subclasses. They accept a `config: FrozenDict` in `setup()` and expose a single `__call__(x, train=True) -> jnp.ndarray` method.

**Input:** `x` of shape `(N, 3)` — columns are `[x, y, t]`.  
**Output:** shape `(N, 3)` — columns are `[h, hu, hv]`.

| Class | File | Notes |
|---|---|---|
| `FourierPINN` | `pinn.py` | Fourier feature encoding → dense + tanh. Best general-purpose choice. |
| `MLP` | `pinn.py` | Standard fully-connected PINN without feature encoding. |
| `DGMNetwork` | `pinn.py` | Deep Galerkin method with LSTM-style gating. |
| `DeepONet` | `deeponet.py` | Operator learning: branch net (params) + trunk net (coords). |
| `FourierDeepONet` | `deeponet.py` | DeepONet with Fourier-enhanced trunk. |

**Layer building blocks** (`src/models/layers.py`):

| Class | Purpose |
|---|---|
| `Normalize` | Maps `(x, y, t)` to `[-1, 1]` using domain bounds from config |
| `FourierFeatures` | Projects inputs to `[sin, cos]` frequency space |
| `NTKDense` | Dense layer with NTK parameterisation |
| `DGMLayer` | LSTM-inspired gating layer used by `DGMNetwork` |

**Initialising a model:**

```python
from src.models.factory import init_model
from src.models.pinn import FourierPINN
from flax.core import freeze
import jax

model, params = init_model(
    model_class=FourierPINN,
    key=jax.random.PRNGKey(42),
    config=freeze(cfg),
)
# params is {'params': {...}}  — pass to model.apply({'params': params['params']}, x)
```

Or use the convenience wrapper in `src/training/setup.py`:

```python
from src.training.setup import init_model_from_config
model, params = init_model_from_config(cfg)
```

---

### 4.3 Physics — `src/physics/`

#### `SWEPhysics` (`swe.py`)

Encapsulates the 2D Shallow Water Equations in conservative form.

```python
from src.physics.swe import SWEPhysics

physics = SWEPhysics(U_pred, eps=1e-6)
# U_pred: jnp.ndarray of shape (N, 3) — [h, hu, hv]

JF, JG = physics.flux_jac(g=9.81)
# JF, JG: shape (N, 3, 3) — Jacobians of x- and y-fluxes w.r.t. U

S = physics.source(
    g=9.81,
    n_manning=0.03,
    inflow=None,          # or scalar external source
    bed_grad_x=...,       # shape (N,) from bathymetry
    bed_grad_y=...,
    Cf=None,              # optional friction coefficient override
)
# S: shape (N, 3) — source terms [R, -gh(S0x+Sfx), -gh(S0y+Sfy)]
```

Water depth is regularised internally: `h_safe = max(h, eps)` to prevent division by zero in momentum flux terms.

#### Analytical solutions (`analytical.py`)

Exact solutions for the 1D dam-break problem, used as reference in Experiment 1.

```python
from src.physics.analytical import h_exact, hu_exact, hv_exact

h  = h_exact(x, t, n_manning=0.03, u_const=0.5)   # shape (N,)
hu = hu_exact(x, t, n_manning=0.03, u_const=0.5)  # = h * u_const
hv = hv_exact(x, t)                                # = 0 (1D problem)
```

---

### 4.4 Losses — `src/losses/`

#### PDE residual (`pde.py`)

```python
from src.losses.pde import compute_pde_loss, compute_neg_h_loss, compute_ic_loss

loss_pde = compute_pde_loss(
    model, params,
    pde_batch,       # (N, 3) — [x, y, t] collocation points
    config,
    pde_mask=None,   # optional boolean (N,) mask; True = include point
)
# Returns scalar MSE of the SWE residual, masked where h < eps.
```

Internally: computes `jax.jacfwd(U_fn)(pde_batch)` to get `∂U/∂x`, `∂U/∂y`, `∂U/∂t`, then evaluates `∂U/∂t + ∇·F(U) - S(U)`.

```python
# Penalty for negative water depths:
loss_neg_h = compute_neg_h_loss(model, params, pde_points)

# Zero initial condition (still water at t=0):
loss_ic = compute_ic_loss(model, params, ic_batch)   # ic_batch has t=0
```

#### Boundary losses (`boundary.py`)

```python
from src.losses.boundary import (
    loss_boundary_dirichlet,
    loss_boundary_neumann_outflow_x,
    loss_boundary_wall_horizontal,
)
```

#### Composite (`composite.py`)

```python
from src.losses.composite import total_loss

total = total_loss(
    terms={'pde': 0.4, 'ic': 0.1, 'bc': 0.2},
    weights={'pde': 1.0, 'ic': 1.0, 'bc': 1.0, 'neg_h': 1.0},
)
# Weighted sum: sum(weights[k] * terms.get(k, 0.0) for k in weights)
```

#### How `compute_losses_fn` works

`train_step` is generic — it knows nothing about IC/BC specifics. Each experiment provides a closure:

```python
def make_compute_losses(ic_points, bc_points, inflow_fn):
    def compute_losses_fn(model, params, batch, config, data_free):
        return {
            'pde':  compute_pde_loss(model, params, batch['pde'], config),
            'ic':   my_ic_loss(model, params, ic_points),
            'bc':   my_bc_loss(model, params, bc_points, inflow_fn),
            'neg_h': compute_neg_h_loss(model, params, batch['pde']),
        }
    return compute_losses_fn
```

---

### 4.5 Data — `src/data/`

#### Sampling collocation points

```python
from src.data.sampling import sample_points, sample_lhs

# Uniform random sampling:
pts = sample_points(key, n=1000, x_min=0, x_max=1200, y_min=0, y_max=100, t_min=0, t_max=3600)
# Returns (N, 3) array [x, y, t]

# Latin Hypercube Sampling (better space-filling, JIT-compatible):
pts = sample_lhs(key, n=1000, bounds=[(0,1200),(0,100),(0,3600)])
```

#### Batching

```python
from src.data.batching import get_batches, get_batches_tensor

batches = get_batches(key, data, batch_size=512)
# Returns list of (batch_size, ...) arrays — shuffled each call

batches = get_batches_tensor(key, data, batch_size=512, total_batches=4)
# Returns (total_batches, batch_size, ...) — for use with lax.scan
```

#### Validation data and boundary conditions

```python
from src.data.loading import load_validation_data, load_boundary_condition

val_data = load_validation_data("path/to/val.npy", dtype=jnp.float32)
# Returns array with columns [x, y, t, h, hu, hv]

inflow_fn = load_boundary_condition("path/to/bc.csv")
# Returns interpolation function: t (scalar) → inflow value
```

#### Bathymetry (DEM)

```python
from src.data import bathymetry_fn  # set by load_bathymetry()
from src.data.bathymetry import load_bathymetry

load_bathymetry("path/to/dem.npy")
# Sets module-level bathymetry_fn. After this call:
# elevation, grad_x, grad_y = bathymetry_fn(x_batch, y_batch)
```

#### Irregular domain sampling (Experiments 7–8)

```python
from src.data.irregular import IrregularDomainSampler

sampler = IrregularDomainSampler(mesh_path="triangulation.npz")
interior_pts = sampler.sample_interior(key, n=2000)   # (N, 3)
boundary_pts, normals = sampler.sample_boundary(key, n=500)
```

---

### 4.6 Training — `src/training/`

#### Single gradient step

```python
from src.training.step import train_step_jitted

new_params, new_opt_state, metrics, loss_val = train_step_jitted(
    model=model,
    optimiser=optimiser,
    params=params,
    opt_state=opt_state,
    batch=batch,                       # dict of named jnp arrays
    config=config,                     # plain dict — NOT frozen here
    data_free=True,                    # False if data loss is active
    compute_losses_fn=compute_losses_fn,
    weights_dict=weights_dict,         # FrozenDict matching keys in compute_losses_fn
)
# metrics: dict[str, float] including '_grad_norm'
# loss_val: scalar float
```

`train_step_jitted` is `jax.jit` with `static_argnames=['model','optimiser','config','compute_losses_fn','weights_dict','data_free']` — these must be hashable or frozen.

#### Optimiser

```python
from src.training.optimizer import create_optimizer

optimiser = create_optimizer(cfg, num_batches=4)
# Returns optax chain: gradient clipping → Adam → ReduceOnPlateau
```

#### Full training loop

```python
from src.training.loop import run_training_loop

results = run_training_loop(
    cfg=frozen_cfg,
    cfg_dict=cfg,
    model=model,
    params=params,
    opt_state=opt_state,
    train_key=jax.random.PRNGKey(0),
    optimiser=optimiser,
    generate_epoch_data_jit=generate_fn,     # JIT-compiled; takes key → batches
    scan_body=scan_body_fn,                  # (carry, batch) → (carry, (terms, total))
    num_batches=4,
    experiment_name="experiment_1",
    trial_name="trial_20240101_120000",
    results_dir="results/experiment_1/trial_...",
    model_dir="models/experiment_1/trial_...",
    config_path="configs/experiment_1/config.yaml",
    validation_fn=my_val_fn,                 # optional; returns {'nse_h': float, ...}
    selection_metric_key="nse_h",            # checkpoint selection criterion
)
# returns dict with 'best_nse', 'best_loss', 'final_params', 'opt_state', ...
```

---

### 4.7 Metrics — `src/metrics/`

Always compute metrics per variable. See [Experimental Programme Reference](experimental_programme_reference.md) for the full spec of each metric and which experiments use them.

#### Accuracy

```python
from src.metrics.accuracy import (
    nse, rmse, mae, relative_l2,
    compute_all_metrics,        # single variable
    compute_all_accuracy,       # dict interface: {var: array}
    compute_validation_metrics, # flat dict, used inside training loop
)

# Single variable:
metrics = compute_all_metrics(h_pred, h_ref)
# {'nse': float, 'rmse': float, 'mae': float, 'rel_l2': float}

# All variables at once (training loop):
flat = compute_validation_metrics(U_pred, U_true)
# {'nse_h': float, 'rmse_h': float, ..., 'nse_hu': float, ..., 'nse_hv': float, ...}
# U_pred, U_true shape (N, 3) — columns [h, hu, hv]

# Dict interface (inference pipeline):
per_var = compute_all_accuracy(
    y_pred={'h': h_pred, 'hu': hu_pred, 'hv': hv_pred},
    y_ref= {'h': h_ref,  'hu': hu_ref,  'hv': hv_ref},
)
# {'h': {'nse': ..., 'rmse': ...}, 'hu': {...}, 'hv': {...}}
```

---

### 4.8 Prediction & Inference

#### Batched forward pass

```python
from src.predict.predictor import Predictor

predictor = Predictor(model, batch_size=4096, min_depth=0.0)
U_pred = predictor.predict_full(params, coords_flat)
# coords_flat: (N, 3) — [x, y, t]
# U_pred: (N, 3) — [h, hu, hv], with h zeroed below min_depth
```

`Predictor` splits `coords_flat` into chunks of `batch_size` to avoid OOM on large grids, then concatenates.

#### Full inference pipeline

```python
python scripts/infer.py \
  --config configs/experiment_1/config.yaml \
  --checkpoint models/experiment_1/<trial>/checkpoints/best_nse \
  --checkpoints best_nse
```

Or call programmatically:

```python
from src.inference.runner import run_inference

run_inference(
    config_path="configs/experiment_1/config.yaml",
    checkpoint_path="models/experiment_1/<trial>/checkpoints",
    checkpoints=["best_nse", "best_loss"],
)
```

Outputs written to `results/<trial>/`:
- `summary_metrics.yaml` — all metrics flat
- `predictions_<checkpoint>.npy` — raw `(N, 6)` array: `[x, y, t, h, hu, hv]`
- Per-variable CSV tables (spatial/temporal decomposition)
- Plot PNGs (time series, spatial maps)

---

### 4.9 Monitoring & Checkpointing

#### W&B logging

```python
from src.monitoring.wandb_tracker import WandbTracker

tracker = WandbTracker(cfg_dict, trial_name, enable=True)
tracker.log({'loss_pde': 0.003, 'nse_h': 0.94, 'lr': 1e-4})
tracker.log_flags({'architecture': 'FourierPINN'})
tracker.finish()
```

All JAX/NumPy types are automatically converted to Python natives before logging.

#### Checkpointing

```python
from src.checkpointing.saver import CheckpointManager
from src.checkpointing.loader import load_checkpoint

manager = CheckpointManager(model_dir=model_dir)
manager.update(
    epoch=500,
    params=params,
    opt_state=opt_state,
    val_metrics={'nse_h': 0.93},
    losses={'pde': 0.002, 'ic': 0.001},
    config=cfg,
)
# Saves to model_dir/checkpoints/{best_nse, best_loss, final}/

params, metadata = load_checkpoint("model_dir/checkpoints/best_nse")
```

Three checkpoints are always maintained: `best_nse` (highest `nse_h`), `best_loss` (lowest total loss), and `final` (last epoch).

---

### 4.10 Loss Balancing — `src/balancing/`

These are optional advanced strategies. Standard experiments use static weights from the config.

#### ReLoBRaLo (adaptive weights)

```python
from src.balancing.relobralo import ReLoBRaLo

balancer = ReLoBRaLo(
    loss_keys=['pde', 'ic', 'bc'],
    alpha=0.999,           # EMA smoothing factor
    rho=0.999,
)
weights = balancer.update(current_losses)
# Returns updated weight dict; call each epoch.
```

#### Importance sampling

```python
from src.balancing.importance_sampling import (
    compute_sampling_probs,
    sample_from_pool,
)

probs = compute_sampling_probs(model, params, pool_points, config)
active_pts = sample_from_pool(key, pool_points, probs, n_active=1000)
```

Implements Algorithm 2 from Wu et al. (2021): residual-proportional resampling from a fixed collocation pool. Used in `experiments/experiment_1/train_imp_samp.py` and `experiment_8/train_imp_samp.py`.

---

## 5. Extension recipes

### 5.1 Add a new experiment

1. Create `experiments/experiment_N/train.py`.
2. Follow this skeleton:

```python
import argparse
import jax
from src.config import load_config
from src.training.setup import init_model_from_config, create_output_dirs
from src.training.optimizer import create_optimizer
from src.training.loop import run_training_loop
from src.losses.pde import compute_pde_loss, compute_neg_h_loss
from src.losses.boundary import loss_boundary_wall_horizontal  # or others
from src.monitoring.wandb_tracker import WandbTracker

def main(config_path):
    cfg = load_config(config_path)
    model, params = init_model_from_config(cfg)
    optimiser = create_optimizer(cfg, num_batches=cfg['training']['num_batches'])
    trial_name, results_dir, model_dir = create_output_dirs(cfg, "experiment_N")
    opt_state = optimiser.init(params)

    # --- define your compute_losses_fn ---
    def compute_losses_fn(model, params, batch, config, data_free):
        return {
            'pde':   compute_pde_loss(model, params, batch['pde'], config),
            'ic':    ...,   # your IC loss
            'bc':    ...,   # your BC loss
            'neg_h': compute_neg_h_loss(model, params, batch['pde']),
        }

    run_training_loop(
        cfg=cfg, cfg_dict=cfg,
        model=model, params=params, opt_state=opt_state,
        train_key=jax.random.PRNGKey(cfg['training']['seed']),
        optimiser=optimiser,
        generate_epoch_data_jit=...,   # your data generator
        scan_body=...,                 # your scan body
        num_batches=cfg['training']['num_batches'],
        experiment_name="experiment_N",
        trial_name=trial_name,
        results_dir=results_dir,
        model_dir=model_dir,
        config_path=config_path,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(parser.parse_args().config)
```

3. Create a corresponding config in `configs/experiment_N/`.
4. Register the experiment in `src/inference/experiment_registry.py` (`EXPERIMENT_REGISTRY` dict) so the inference pipeline knows its domain type, validation file, and DEM path.

---

### 5.2 Add a new loss term

1. Implement a function in `src/losses/`:

```python
# src/losses/my_new_loss.py
def compute_my_loss(model, params, points, config):
    U_pred = model.apply({'params': params['params']}, points, train=False)
    # ... compute scalar loss
    return scalar_loss
```

2. Add the key to the experiment's `compute_losses_fn` return dict:

```python
return {
    'pde': ...,
    'ic':  ...,
    'my_new': compute_my_loss(model, params, batch['my_points'], config),
}
```

3. Add a matching weight to the config:

```yaml
loss_weights:
  my_new: 1.0
```

4. The `weights_dict` in `train_step` is built from the config's `loss_weights` block — no other changes needed.

---

### 5.3 Add a new model architecture

1. Create a Flax `nn.Module` in `src/models/pinn.py` (or a new file).
2. Accept `config: FrozenDict` and expose `__call__(x, train=True)`.
3. Input shape: `(N, 3)` — `[x, y, t]`. Output shape: `(N, 3)` — `[h, hu, hv]`.
4. Apply `Normalize` from `src.models.layers` to map coordinates to `[-1, 1]`.
5. Apply `apply_output_scaling` at the end if you want output denormalization.
6. Register the class name in `src/models/factory.py`'s `init_model()` dispatch:

```python
MODEL_REGISTRY = {
    'FourierPINN': FourierPINN,
    'MLP': MLP,
    'DGMNetwork': DGMNetwork,
    'MyNewModel': MyNewModel,   # add here
}
```

7. Set `model.name: MyNewModel` in the YAML config.

---

### 5.4 Add a new evaluation metric

1. Implement in the appropriate `src/metrics/` file (or create one):

```python
def my_metric(pred: jnp.ndarray, ref: jnp.ndarray) -> float:
    ...
    return float(result)
```

2. Call it in `src/inference/runner.py` alongside the existing metric calls, and add its key to the `InferenceContext` results dict.
3. Add it to `src/inference/reporting.py`'s `save_yaml_summary()` so it appears in `summary_metrics.yaml`.
4. Consult [Experimental Programme Reference](experimental_programme_reference.md) for the exact definition and which experiments should use it.

---

## 6. Critical constraints & pitfalls

| Pitfall | Rule |
|---|---|
| JAX JIT side effects | No Python side effects (print, list append, dict update) inside `@jax.jit` or `lax.scan` bodies — first call traces, subsequent calls replay the trace. |
| Float precision | Float64 physics require `config.device.dtype: float64`. This triggers `jax.config.update("jax_enable_x64", True)` in `load_config`. Never enable x64 manually after config is loaded. |
| `DTYPE` / `EPS` import | Import via `get_dtype()` / `get_eps()`, not `from src.config import DTYPE`. The latter captures the value at import time and misses HPO trial updates. |
| Water depth masking | Always apply `h >= eps` mask before dividing by h in physics terms. `SWEPhysics` handles this internally; custom loss terms must apply it manually. |
| Large validation data | Use `np.load(..., mmap_mode='r')` for multi-GB `.npy` files. Full loads will OOM on large grids. |
| Config mutability | `train_step_jitted` requires `config` as a static arg. Pass a plain `dict` (not FrozenDict) to the training step; Flax `FrozenDict` is only needed for model init. |
| Checkpoint selection | The `selection_metric_key` in `run_training_loop` defaults to `"nse_h"`. Override if your experiment does not compute h separately. |
| Building masks | For experiments with obstacle geometry, call `src.utils.domain.mask_points_inside_building()` when generating PDE collocation points to exclude interior building cells. |

---

## 7. Reading these docs

All docs in this repo are plain Markdown. Three ways to read them:

**1. GitHub** — push to a remote and GitHub renders all `.md` files natively with math (via MathJax). No setup required.

**2. VS Code** — open any `.md` file and press `Ctrl+Shift+V` (or `Cmd+Shift+V` on Mac) to open a live preview pane. Math rendering requires the **Markdown Math** extension (`goessner.mdmath`).

**3. MkDocs (local hosted site)** — for a fully navigable documentation site:

```bash
pip install mkdocs mkdocs-material
mkdocs serve   # runs at http://127.0.0.1:8000
```

Create a `mkdocs.yml` at the repo root:

```yaml
site_name: SWE-PINN Developer Docs
theme:
  name: material
nav:
  - Home: README.md
  - Developer Guide: docs/developer_guide.md
  - Experimental Programme: docs/experimental_programme_reference.md
  - Scaling Reference: docs/scaling_reference.md
  - Ablation Design (Exp 1): docs/experiment_1_ablation_design.md
docs_dir: .        # serve from repo root so README.md is included
```

```bash
mkdocs build   # generates static site in site/
```

No `mkdocs.yml` is committed to this repo; the above is optional and local-only.
