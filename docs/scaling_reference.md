# SWE Non-Dimensionalization Reference

This document describes the non-dimensionalization algorithm implemented in
`src/physics/scaling.py` and applied in the experiment training scripts.

## 1. Scale Computation (one-time at startup)

```
Inputs from config:
  g_phys = scaling.g           (default 9.81 m/s^2)
  H0     = scaling.H0          (default 1.0 m)
  n      = physics.n_manning
  Lx     = domain.lx
  Ly     = domain.ly

Computed constants (stored in SWEScaler, never updated):
  L0  = max(Lx, Ly)
  U0  = sqrt(g_phys * H0)          # shallow-water celerity
  T0  = L0 / U0                    # advective timescale
  HU0 = H0 * U0                    # output scale for hu, hv
  Cf  = g_phys * n^2 * L0 / H0^(4/3)   # dimensionless friction number
```

**Key**: `g_phys` comes from `scaling.g`, NOT `physics.g`. The scaler
always uses the true physical gravity regardless of what `physics.g` is
set to in the config.

## 2. Config Transformation

`nondim_physics_config()` produces a config for the PDE loss:

| Key | Original | Nondim |
|-----|----------|--------|
| `physics.g` | any | `1.0` (absorbed into scaling) |
| `physics.n_manning` | 0.03 | `0.0` (absorbed into Cf) |
| `physics.Cf` | absent | `10.59` (dimensionless friction) |
| `physics.dimensional.g` | absent | `9.81` (preserved for analytical BCs) |
| `physics.dimensional.n_manning` | absent | `0.03` (preserved) |
| `physics.dimensional.u_const` | absent | `0.29` (preserved) |

## 3. Scaling Rules

Exactly two operations. Every variable passes through exactly one, never both.

### SCALE (dimensional -> non-dimensional)

| Variable | Formula | When |
|----------|---------|------|
| x, y | x* = x / L0 | Before network forward pass |
| t | t* = t / T0 | Before network forward pass |
| z_b | z* = z_b / H0 | Once during data prep |
| h | h* = h / H0 | BC/IC/data targets entering loss |
| hu | (hu)* = hu / HU0 | BC/IC/data targets entering loss |
| hv | (hv)* = hv / HU0 | BC/IC/data targets entering loss |

### UNSCALE (non-dimensional -> dimensional)

| Variable | Formula | When |
|----------|---------|------|
| h* | h = h* * H0 | Network output -> validation/plots |
| (hu)* | hu = (hu)* * HU0 | Network output -> validation/plots |
| (hv)* | hv = (hv)* * HU0 | Network output -> validation/plots |

## 4. Data Flow

### Training

```
SAMPLING:
  Ranges in non-dim space (computed once before JIT):
    x_range = (0, Lx/L0),  y_range = (0, Ly/L0),  t_range = (0, t_final/T0)
  All sample_domain() calls use scaled ranges.
  Points are born in non-dim space.

FORWARD PASS:
  (x*, y*, t*) -> network -> (h*, (hu)*, (hv)*)

PDE RESIDUAL (all in * space, g=1, Cf only):
  Autodiff: dU*/dx*, dU*/dy*, dU*/dt*
  Flux Jacobians with g=1.0
  Source: Cf replaces n^2, g=1.0
  Loss = mean(|R * h_mask|^2)

BC LOSS:
  1. Recover dimensional time: t_dim = t* * T0
  2. Evaluate analytical solution: h_dim = h_exact(0, t_dim, n, u_const)
  3. Scale target: h_target = h_dim / H0, hu_target = hu_dim / HU0
  4. Compare: loss = mean((network_output - target)^2)
```

### Validation

```
  1. Forward pass: U*_pred = model(val_pts_nd)
  2. UNSCALE: h_pred = h* * H0, hu_pred = (hu)* * HU0
  3. Compare with DIMENSIONAL ground truth
  4. Metrics in SI units (RMSE in metres, NSE dimensionless)
```

### Plotting

```
  1. Build points in dimensional space
  2. Scale inputs for network
  3. Forward pass -> U*
  4. Unscale outputs
  5. Plot with dimensional axes (m, m^2/s, s)
```

## 5. Non-Dimensional PDE

Continuity:
```
R_cont = dh*/dt* + d(hu)*/dx* + d(hv)*/dy*
```

x-momentum:
```
R_xmom = d(hu)*/dt*
       + d[ (hu)*^2/h* + h*^2/2 ]/dx*
       + d[ (hu)*(hv)*/h* ]/dy*
       + h* dz*/dx*
       + Cf (hu)* sqrt((hu)*^2 + (hv)*^2) / h*^(7/3)
```

y-momentum:
```
R_ymom = d(hv)*/dt*
       + d[ (hu)*(hv)*/h* ]/dx*
       + d[ (hv)*^2/h* + h*^2/2 ]/dy*
       + h* dz*/dy*
       + Cf (hv)* sqrt((hu)*^2 + (hv)*^2) / h*^(7/3)
```

**g does not appear anywhere.** The pressure flux is `h*^2/2`, not `g*h*^2/2`.

### Friction implementation note

The code computes friction using primitive variables (u, v) derived from
conservative variables:

```python
u = hu / max(h, eps)
v = hv / max(h, eps)
vel = sqrt(u^2 + v^2)
sfx = Cf * u * vel / h^(4/3)
S_xmom = -1.0 * h * sfx
```

This is mathematically equivalent to the conservative form:

```
-Cf * (hu) * sqrt((hu)^2 + (hv)^2) / h^(7/3)
```

Proof: substitute u = hu/h into `h * Cf * u * vel / h^(4/3)`:
```
= Cf * hu * vel / h^(4/3)
= Cf * hu * sqrt(u^2 + v^2) / h^(4/3)
= Cf * hu * sqrt((hu)^2 + (hv)^2) / (h * h^(4/3))
= Cf * hu * sqrt((hu)^2 + (hv)^2) / h^(7/3)
```

## 6. Identity Mode

When `scaling.enabled: false` (default):

```
L0 = H0 = U0 = T0 = HU0 = 1.0
Cf = physics.g * n^2   (dimensional)
nondim_physics_config returns original config unchanged
```

The pipeline runs identically to before the feature existed.

## 7. File Locations

| Component | File | Entry point |
|-----------|------|-------------|
| Scaler | `src/physics/scaling.py` | `SWEScaler` |
| SWE source with Cf | `src/physics/swe.py` | `SWEPhysics.source(Cf=...)` |
| PDE loss | `src/losses/pde.py` | reads `config["physics"]["Cf"]` |
| IS residual | `src/balancing/importance_sampling.py` | same pattern |
| Config | `configs/experiment_1/best_trial_51_config.yaml` | `scaling.*` |

## 8. Config Example

```yaml
scaling:
  enabled: true    # false = dimensional mode (identity scaler)
  H0: 1.0          # reference depth [m]
  # g: 9.81        # physical gravity [m/s^2] (default, rarely changed)
```
