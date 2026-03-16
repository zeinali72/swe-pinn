# Experimental Programme Reference

> **What this document is:** The authoritative specification for all 11 experiments, their validation metrics, inference plots, and training-tracked values. All evaluation code should be modular and reusable across experiments. When asked to implement any metric, plot, or evaluation pipeline, refer to this document for definitions, formulas, units, and experiment applicability.

---

## Experimental Programme Overview

### Phase 1 — Baseline Verification and Architecture Selection (Objective 1)

| Exp | Title | Domain | Primary Question |
|-----|-------|--------|-----------------|
| 1 | Flat domain verification | 1200x100m flat channel, Hunter (2005), T=3600s | Can a vanilla MLP solve the simplest SWE? |
| 2 | Building obstacle — spectral bias | 1200x100m + 50x50m obstacle, ICM baseline | Where does spectral bias break the MLP? Does DGM work on 2D SWE? |
| 3 | Hyperparameter optimisation | Exp 1 + Exp 2 domains, TPE Bayesian | Optimal configs and sensitivity? |

### Phase 2 — Topographic Complexity, EA Benchmark Suite (Objective 2)

| Exp | Title | Domain | Primary Question |
|-----|-------|--------|-----------------|
| 4 | Sloped terrain — physics-only attempt | EA Test 1: 700x100m, disconnected water body, T=20hrs | Can Phase 1 configs handle slopes physics-only? |
| 5 | Sloped terrain — HPO on sloped domain | EA Test 1: same domain, targeted HPO | Is the failure a hyperparameter problem? |
| 6 | Sloped terrain — training regime comparison | EA Test 1: same domain, best HPO configs | How much does data help? How much does physics help? |
| 7 | Egg-box topography — full 2D | EA Test 2: egg-box terrain, T=48hrs | Scale to 2D gravity-driven flows? Which architecture wins? |
| 8 | Momentum conservation — overtopping | EA Test 5: raised obstruction, two depressions | Can PINN resolve momentum-driven overtopping? |
| 9 | Extended floodplain — scale and cost | EA Test 6: 1000x2000m, breach inflow | Scale to largest domain? What does it cost? |

### Phase 3 — Irregular Geometry and Real-World Application (Objective 3)

| Exp | Title | Domain | Primary Question |
|-----|-------|--------|-----------------|
| 10 | Irregular domain sampling | Separate irregular EA catchment, no buildings | Can triangle + CDF sampling handle irregular geometry? |
| 11 | Eastbourne — full urban catchment | Blue Heart Project, real DEM, buildings, real BCs | Can the PINN framework replace a numerical solver? |

### Assumptions Deferred to Recommendations

- Wall boundary representation for buildings (vs terrain or friction alternatives)
- PDE residual as post-training verification tool (requires operator learning)
- Adaptive / importance sampling (assume LHS + dynamic resampling)
- Transfer learning to unseen catchments (Objective 4 removed from thesis)

---

## Validation Metrics

All metrics computed post-training on a dense evaluation grid independent of training collocation points. Every accuracy metric reported **separately for h, hu, hv**.

### Group A: Prediction Accuracy

**A1. Nash-Sutcliffe Efficiency (NSE)** — Primary goodness-of-fit.
```
NSE = 1 - sum((Y_obs - Y_pred)^2) / sum((Y_obs - mean(Y_obs))^2)
```
Range: (-inf, 1]. Units: dimensionless. Reported per gauge + global aggregate.
Used in: Exp 1-11. Reference = analytical (Exp 1) or ICM (Exp 2-11).

**A2. Root Mean Squared Error (RMSE)** — Error magnitude, outlier-sensitive.
```
RMSE = sqrt(mean((Y_obs - Y_pred)^2))
```
Units: m (h), m²/s (hu, hv). When RMSE >> MAE, errors are localised.
Used in: Exp 1-11.

**A3. Mean Absolute Error (MAE)** — Typical error, complement to RMSE.
```
MAE = mean(|Y_obs - Y_pred|)
```
Units: m (h), m²/s (hu, hv). Always report alongside RMSE.
Used in: Exp 1-11.

**A4. Relative L2 Error** — Normalised spatial field comparison per time step.
```
Rel_L2 = ||Y_obs - Y_pred||_2 / ||Y_obs||_2
```
Units: dimensionless. Enables comparison with PINN literature (Raissi et al., 2019).
Used in: Exp 1-11.

**A5. Peak Depth Error** — Difference in maximum water depth.
```
E_peak = max(h_pred) - max(h_ref)    per gauge point
```
Units: m. Critical for flood warning applications.
Used in: Exp 4-11.

**A6. Time-to-Peak Error** — Temporal offset of flood peak.
```
E_ttp = t(max(h_pred)) - t(max(h_ref))    per gauge point
```
Units: s or hrs. Positive = late; negative = early.
Used in: Exp 4-11.

**A7. Critical Success Index (CSI)** — Flood extent accuracy.
```
CSI = Hits / (Hits + Misses + False Alarms)
```
Wet/dry threshold: h = 0.01 m. Range: [0, 1].
Used in: Exp 10-11.

### Group B: Mass Conservation

**B1. Domain-Integral Volume Balance (E_mass)**
```
V_domain(t_k) = integral(h_hat dA)
E_mass(t_k) = |V_domain - (V_inflow - V_outflow)| / V_inflow * 100%
```
Units: %. Report: max(E_mass), E_mass(t_final), full time series, ICM comparison.
Rectangular domains: trapezoidal rule. Irregular domains: area-weighted centroid sum.
Used in: Exp 1-11.

**B2. Local Control-Volume Flux Balance (r_j)** — Per-triangle conservation residual.
```
dV_j/dt + sum_edges(integral(hu*n_x + hv*n_y) dl) = r_j
```
Units: m³/s. Reported as spatial map. Requires purpose-built evaluation mesh.
Used in: Exp 11 only.

**B3. Pointwise Continuity Residual (R_mass)** — Via AD on trained network.
```
R_mass = dh/dt + d(hu)/dx + d(hv)/dy
```
Units: m/s. Presented alongside B2 for pointwise vs integral comparison.
Used in: Exp 11 only.

### Group C: Boundary Condition Enforcement

**C1. Slip Boundary Violation (V_slip)**
```
V_slip = |u*n_x + v*n_y|    at boundary evaluation points
```
Units: m/s. Report mean + max along each boundary category.
Used in: Exp 1-11 (walls), Exp 2 + 11 (buildings).

**C2. Inflow Boundary Error**
```
E_inflow = RMSE(h_pred - h_prescribed)    at inflow points over all time steps
```
Units: m or m²/s. Used in: Exp 1-11.

**C3. Outflow Boundary Gradient Residual**
```
E_outflow = |dh/dn|    at outflow points via AD
```
Units: dimensionless. Used in: Exp 1 only (open outflow).

**C4. Initial Condition Error**
```
E_IC = RMSE(h_pred(t=0) - h_prescribed(t=0))
```
Units: m. Used in: Exp 1-11. Critical for Exp 4 (non-trivial IC at 9.7m).

### Group D: Computational Cost

**D1. Data Preparation Cost** — ICM wall-clock time. Units: s. Used in: Exp 2-11.

**D2. Training Cost** — Wall-clock from epoch 1 to convergence. Units: s. Report with GPU spec. Used in: Exp 1-11.

**D3. Inference Cost** — Single forward pass over full evaluation grid. Units: s. Report + speedup ratio vs ICM. Used in: Exp 1-11, primary focus Exp 9 + 11.

**D4. Break-Even Query Count**
```
N_break = (T_data_prep + T_training) / (T_ICM - T_inference)
```
Units: integer. Used in: Exp 9, 11.

### Group E: HPO-Specific

**E1.** Best NSE per architecture. Used in: Exp 3, 5.
**E2.** Hyperparameter importance (fANOVA via Optuna). Used in: Exp 3, 5.
**E3.** Sensitivity shift — % change in optimal HP values from flat to sloped domain. Used in: Exp 5 only.

### Group F: Data Experiment

**F1. Data Fraction** — % of total ICM spatiotemporal points used for training. Used in: Exp 6, 11.
**F2. Data Efficiency Ratio** — Ratio of fractions at which PINN+data and data-only reach equivalent accuracy. Used in: Exp 6, 11.

---

## Inference Plots

All plots: Exeter palette (Deep Green #003C3C, Teal #007D69, Mint #00C896) + Blue Heart palette (Navy #0D2B45, Ocean #1B5E8A, Sky #4FA3D1). Arial. 300 DPI.

### P1: Time Series

| ID | Plot | Description | Used in |
|----|------|-------------|---------|
| P1.1 | Gauge time series | h_pred vs h_ref at each gauge over full simulation. NSE in legend. Variants: single model, multi-architecture overlay, multi-regime overlay, multi-fraction overlay. | Exp 1-11 |
| P1.2 | Mass balance time series | E_mass(t) for PINN and ICM (dashed). Annotate max and final values. | Exp 1-11 |
| P1.3 | Training loss curves | All loss components over epochs (log scale). LR on secondary axis. | Exp 1-11 |
| P1.4 | Validation NSE during training | NSE at gauges evaluated periodically. | Exp 1-11 |

### P2: Spatial Maps

| ID | Plot | Description | Used in |
|----|------|-------------|---------|
| P2.1 | Spatial error map | 2D heatmap of \|h_pred - h_ref\| at 3-5 key time steps. | Exp 1-11 |
| P2.2 | Spatial depth map | Side-by-side: PINN prediction vs ICM reference. Same colourbar. | Exp 1-11 |
| P2.3 | Flood extent map (CSI) | Hits (blue), Misses (red), False Alarms (orange). | Exp 10-11 |
| P2.4 | Conservation residual map | Triangulated domain coloured by \|r_j\|. | Exp 11 |
| P2.5 | Continuity residual map | Pointwise \|R_mass\| on evaluation grid. Present alongside P2.4. | Exp 11 |
| P2.6 | Slip violation map | V_slip along building perimeters as coloured line segments. | Exp 2, 11 |
| P2.7 | Spatial error decomposition | Points classified by category (shock/boundary/smooth), coloured by error. | Exp 1-11 |
| P2.8 | Collocation point distribution | Scatter of D_PDE, D_IC, D_BC, D_build points coloured by type. | Exp 10-11 |
| P2.9 | Coverage density heatmap | KDE of collocation point spatial distribution. | Exp 10-11 |

### P3: Comparison and Analysis

| ID | Plot | Description | Used in |
|----|------|-------------|---------|
| P3.1 | Precision comparison bar | NSE + training time for float64/32/bfloat16. | Exp 1 |
| P3.2 | Architecture comparison bar | NSE, RMSE, training time for MLP/Fourier/DGM. | Exp 2, 7 |
| P3.3 | Frequency spectrum | FFT of pred vs ref along transect through obstacle wake. Shows spectral bias. | Exp 2 |
| P3.4 | Data fraction curve | NSE vs data % for PINN+data and data-only. Log x-axis. Annotate crossover + efficiency ratio. | Exp 6, 11 |
| P3.5 | Gauge vs random comparison | Accuracy comparison: fixed gauge locations vs random spatiotemporal at same fraction. | Exp 6 |
| P3.6 | Cost breakdown bar | Stacked: data prep + training + inference vs ICM total. | Exp 9, 11 |
| P3.7 | Break-even crossover | Cumulative cost vs N queries. Two lines crossing at N_break. | Exp 9, 11 |

### P4: HPO

| ID | Plot | Description | Used in |
|----|------|-------------|---------|
| P4.1 | Optimisation history | NSE vs trial number. | Exp 3, 5 |
| P4.2 | Parallel coordinate | Trials as lines through HP axes, coloured by NSE. | Exp 3, 5 |
| P4.3 | HP importance bar | fANOVA importance ranking. | Exp 3, 5 |
| P4.4 | Pairwise interaction heatmap | 2D grid of HP pair interactions. | Exp 3 |
| P4.5 | Sensitivity shift bar | Phase 1 vs Phase 2 optimal HP values. | Exp 5 |

---

## Values Tracked During Training (via Aim)

### T1: Loss Components (every epoch)

| ID | Value | Experiments |
|----|-------|-------------|
| T1.1 | L_total (weighted composite) | All |
| T1.2 | L_PDE (mass + x-momentum + y-momentum) | All |
| T1.3 | L_BC | All |
| T1.4 | L_IC | All |
| T1.5 | L_build | Exp 2, 11 |
| T1.6 | L_data | Exp 6-11 (when data used) |
| T1.7 | L_phy | All |

### T2: Optimisation State (every epoch)

| ID | Value | Experiments |
|----|-------|-------------|
| T2.1 | Learning rate | All |
| T2.2 | Gradient norm | All |
| T2.3 | Epoch wall-clock (s) | All |
| T2.4 | Gradient clipping event count | All |

### T3: Validation Checks (every N epochs)

| ID | Value | Experiments |
|----|-------|-------------|
| T3.1 | NSE at validation gauges | All |
| T3.2 | RMSE at validation gauges | All |
| T3.3 | Best epoch (NSE checkpoint) | All |

### T4: HPO Trial Tracking (per trial)

| ID | Value | Experiments |
|----|-------|-------------|
| T4.1 | Trial NSE (objective) | Exp 3, 5 |
| T4.2 | Trial total epochs | Exp 3, 5 |
| T4.3 | Trial wall-clock (s) | Exp 3, 5 |
| T4.4 | Pruned/completed status | Exp 3, 5 |
| T4.5 | All HP values | Exp 3, 5 |

---

## Experiment-to-Module Mapping

| Exp | Accuracy | Conservation | Boundary | Cost | HPO | Data | Time Series | Spatial Maps | Comparison | HPO Plots |
|-----|----------|-------------|----------|------|-----|------|-------------|-------------|------------|-----------|
| 1 | A1-A4 | B1 | C1-C4 | D2 | | | P1.1-4 | P2.1,2,7 | P3.1 | |
| 2 | A1-A4 | B1 | C1,C2,C4 | D1,D2 | | | P1.1-4 | P2.1,2,6,7 | P3.2,3 | |
| 3 | | | | | E1,E2 | | | | | P4.1-4 |
| 4 | A1-A4 | B1 | C1,C2,C4 | D1,D2 | | | P1.1-4 | P2.1,2 | | |
| 5 | | | | | E1-E3 | | | | | P4.1-3,5 |
| 6 | A1-A6 | B1 | C1,C2,C4 | D1,D2 | | F1,F2 | P1.1-4 | P2.1,2,7 | P3.4,5 | |
| 7 | A1-A6 | B1 | C1,C2,C4 | D1,D2 | | | P1.1-4 | P2.1,2,7 | P3.2 | |
| 8 | A1-A6 | B1 | C1,C2,C4 | D1,D2 | | | P1.1-4 | P2.1,2 | | |
| 9 | A1-A6 | B1 | C1,C2,C4 | D1-D4 | | | P1.1-4 | P2.1,2 | P3.6,7 | |
| 10 | A1-A6 | B1 | C1,C2,C4 | D1,D2 | | | P1.1-4 | P2.1-3,7-9 | | |
| 11 | A1-A7 | B1-B3 | C1,C2,C4 | D1-D4 | | F1,F2 | P1.1-4 | P2.1-9 | P3.4,6,7 | |

---

## Suggested Code Module Structure

```
evaluation/
├── metrics/
│   ├── accuracy.py          # A1-A4: NSE, RMSE, MAE, Rel_L2
│   ├── flood_metrics.py     # A5-A7: peak depth, time-to-peak, CSI
│   ├── conservation.py      # B1-B3: volume balance, local flux, continuity residual
│   ├── boundary.py          # C1-C4: slip violation, inflow/outflow, IC error
│   ├── cost.py              # D1-D4: timing, break-even
│   └── data_efficiency.py   # F1-F2: data fraction, efficiency ratio
├── plots/
│   ├── time_series.py       # P1.1-P1.4
│   ├── spatial_maps.py      # P2.1-P2.9
│   ├── comparisons.py       # P3.1-P3.7
│   └── hpo_plots.py         # P4.1-P4.5
├── decomposition/
│   ├── spatial_decomp.py    # Category 1/2/3 classification
│   └── temporal_decomp.py   # Error by simulation phase
└── tracking/
    └── aim_logger.py        # T1-T4: Aim wrapper
```

### Key Function Signatures

```python
# accuracy.py
def compute_nse(y_pred: np.ndarray, y_ref: np.ndarray) -> float
def compute_rmse(y_pred: np.ndarray, y_ref: np.ndarray) -> float
def compute_mae(y_pred: np.ndarray, y_ref: np.ndarray) -> float
def compute_rel_l2(y_pred: np.ndarray, y_ref: np.ndarray) -> float
def compute_all_accuracy(
    y_pred: dict[str, np.ndarray],   # {'h': ..., 'hu': ..., 'hv': ...}
    y_ref: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]     # {'h': {'nse': ..., 'rmse': ...}, ...}

# conservation.py
def compute_volume_balance(
    h_pred: np.ndarray,              # (n_times, n_spatial)
    cell_areas: np.ndarray,          # (n_spatial,)
    inflow_volume: np.ndarray,       # (n_times,) cumulative
    outflow_volume: np.ndarray,      # (n_times,) cumulative
) -> dict                            # {'e_mass': array, 'max_error': float, 'final_error': float}

def compute_local_flux_balance(
    pinn_model,                      # trained JAX model
    eval_mesh,                       # triangulated evaluation mesh
    time_steps: np.ndarray,
) -> np.ndarray                      # (n_triangles, n_times)

# time_series.py
def plot_gauge_timeseries(
    t: np.ndarray,
    predictions: dict[str, np.ndarray],   # {'MLP': h_pred, 'DGM': h_pred, ...} or {'PINN': ..., 'Data-only': ...}
    h_ref: np.ndarray,
    gauge_name: str,
    metrics: dict[str, float] | None = None,   # {'NSE': 0.94, ...} shown in legend
    save_path: str | None = None,
) -> matplotlib.figure.Figure

# spatial_maps.py
def plot_error_map(
    x: np.ndarray, y: np.ndarray,
    error: np.ndarray,
    time_label: str,
    domain_type: str = 'rectangular',   # or 'triangulated'
    buildings: gpd.GeoDataFrame | None = None,
    save_path: str | None = None,
) -> matplotlib.figure.Figure

# comparisons.py
def plot_data_fraction_curve(
    fractions: list[float],
    nse_hybrid: list[float],
    nse_data_only: list[float],
    nse_physics_only: float | None = None,   # horizontal dashed line if converged
    save_path: str | None = None,
) -> matplotlib.figure.Figure

def plot_break_even(
    t_data_prep: float, t_training: float, t_inference: float, t_icm: float,
    max_queries: int = 100,
    save_path: str | None = None,
) -> matplotlib.figure.Figure
```
