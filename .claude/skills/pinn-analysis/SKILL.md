---
name: pinn-analysis
description: Compute validation metrics and perform spatial/temporal error decomposition for PINN flood prediction experiments. Use this skill whenever the user asks to compute metrics, evaluate model performance, analyse results, perform error decomposition, compare experiments, assess accuracy, check mass conservation, or prepare a results table for the thesis. Also trigger for "compute NSE", "RMSE", "CSI", "error map", "results table", "compare architectures", or any post-training evaluation request.
---

# PINN Analysis — Metrics and Error Decomposition

This skill handles everything between "training is done" and "here are the results". It computes validation metrics, decomposes errors spatially and temporally, and produces thesis-ready comparison tables.

## Validation Metrics

Always compute this full set unless the user specifies otherwise. Use NumPy for all computations (no sklearn dependency).

### Per-Variable Metrics (h, hu, hv)

| Metric | Formula | Units |
|--------|---------|-------|
| NSE | `1 - sum((pred - obs)^2) / sum((obs - mean(obs))^2)` | dimensionless |
| RMSE | `sqrt(mean((pred - obs)^2))` | m (h), m²/s (hu, hv) |
| MAE | `mean(abs(pred - obs))` | m (h), m²/s (hu, hv) |
| R² | `1 - SS_res / SS_tot` | dimensionless |
| Peak error | `abs(max(pred) - max(obs))` | same as variable |

### Flood Extent Metrics

Compute Critical Success Index (CSI) at multiple depth thresholds:

```
CSI = TP / (TP + FP + FN)
```

where:
- TP: both prediction and observation exceed threshold
- FP: prediction exceeds but observation does not
- FN: observation exceeds but prediction does not

Standard thresholds: `[0.01, 0.05, 0.1, 0.3]` metres.

### Mass Conservation Deficit

Percentage difference between total inflow volume and volume present in domain at final time step:

```
deficit_pct = 100 * abs(V_inflow - V_domain) / V_inflow
```

For Experiment 8: use the triangulated evaluation mesh for volume integration (not rectangular control volumes).

## Spatial Error Decomposition

Classify each evaluation point into one of three categories, then report metrics separately for each.

### Category 1 — Shock / Wet-Dry Front Region
Points where the spatial gradient of h exceeds a threshold (default: 90th percentile of |nabla h|).

```python
grad_h = np.sqrt(np.gradient(h, axis=0)**2 + np.gradient(h, axis=1)**2)
threshold = np.percentile(grad_h, 90)
cat1_mask = grad_h > threshold
```

Category 1 takes priority: if a point also satisfies Category 2, it stays in Category 1.

### Category 2 — Boundary Interaction Region
Points within a specified proximity distance (default: 3-5 grid spacings) of any solid boundary. This includes:
- Domain walls (all experiments)
- Building perimeters (Experiments 2 and 8)

Use Shapely `distance()` against boundary geometries for proximity computation.

### Category 3 — Smooth Interior
All remaining points not in Category 1 or 2.

### Output for Each Category
- All per-variable metrics (NSE, RMSE, MAE, R², peak error)
- Fraction of total points in the category
- Point count

## Temporal Error Decomposition

Divide the simulation into characteristic phases and report metrics for each:

| Phase | Definition |
|-------|-----------|
| Rising limb | t where dh/dt > threshold (flow arriving) |
| Peak | neighbourhood of max(h) within ±N time steps |
| Recession | t where dh/dt < -threshold (flow receding) |
| Steady state | t where abs(dh/dt) < threshold (if applicable) |

Phase thresholds should be derived from the data (e.g., percentile-based) rather than hardcoded.

## Output Formats

Always produce results as **both**:

1. **pandas DataFrame saved as CSV** — for further processing
2. **Formatted console summary** — for quick inspection

When comparing multiple experiments or architectures, additionally produce a **LaTeX-ready table**:

```python
df.to_latex(
    float_format="%.4f",
    caption="Comparison of ...",
    label="tab:...",
    escape=False,
)
```

## Data Loading Conventions

- **Aim run data**: loaded via `aim.Repo` and `aim.Run`
- **ICM baseline**: CSV exports with columns `x, y, t, h, hu, hv`
- **PINN predictions**: forward pass over the evaluation grid using the trained model

### Handling Edge Cases
- Dry cells where h=0 in both prediction and reference: exclude from h-error computation but count as correct in CSI (true negative)
- NaN values: exclude from metric computation, report count of excluded points
- For Experiment 8: the evaluation mesh is triangulated and independent of both the training mesh and the ICM mesh

## Reference Files

- `references/metrics.py` — NumPy implementations of all metric functions, spatial/temporal decomposition, and LaTeX table generation. Read this file for the exact function signatures and usage patterns.
