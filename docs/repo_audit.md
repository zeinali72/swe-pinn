# SWE-PINN Repository Audit

**Date:** 2026-03-11
**Scope:** Every file, folder, module, and script in the repository

---

## Table of Contents

1. [File & Module Map](#1-file--module-map)
2. [Dependency Chains per Experiment](#2-dependency-chains-per-experiment)
3. [Config Completeness](#3-config-completeness)
4. [Data Pipeline Readiness](#4-data-pipeline-readiness)
5. [Reproducibility & Precision](#5-reproducibility--precision)
6. [Checkpoint & Output Paths](#6-checkpoint--output-paths)
7. [Dead Code, Orphans & Inconsistencies](#7-dead-code-orphans--inconsistencies)
8. [Critical Issues Summary](#8-critical-issues-summary)

---

## 1. File & Module Map

### 1.1 Core Source (`src/`)

| File | Purpose | Used By | Naming |
|------|---------|---------|--------|
| `__init__.py` | Package marker (empty) | Python import system | OK |
| `train.py` | Unified training entry point; `main(config_path)` orchestrates full pipeline | All experiments via `python src/train.py <config>` | OK |
| `models.py` | 7 architectures: FourierPINN, MLP, DGMNetwork, DeepONet, FourierDeepONet, NTK_MLP, FourierNTK_MLP; `init_model()`, `init_deeponet_model()` | All experiments | OK |
| `losses.py` | PDE residual, IC, BC (Dirichlet/Neumann/slip), building BC, data, neg-h losses; `total_loss()` | All experiments | OK |
| `physics.py` | `SWEPhysics` class (flux Jacobians, source terms), `h_exact()` analytical solution | losses.py, utils.py, ntk.py, experiment scripts | OK |
| `gradnorm.py` | GradNorm adaptive loss weighting; `GradNormState`, `init_gradnorm()`, `update_gradnorm_weights()` | train.py (conditional on config) | OK |
| `softadapt.py` | SoftAdapt loss weighting; `softadapt_update()` | **UNUSED — never imported anywhere** | OK |
| `ntk.py` | NTK trace computation; `compute_ntk_traces()`, `update_ntk_weights_algo1()` | `analytical_ntk.py` only | OK |
| `data.py` | Sampling (`sample_points`, `sample_domain`, `sample_lhs`), batching (`get_batches`, `get_batches_tensor`), bathymetry loading, BC loading, `IrregularDomainSampler`, `DeepONetParametricSampler` | All experiments | OK |
| `config.py` | YAML loading, string-float conversion, sets global `DTYPE`/`EPS` | All experiments, HPO | OK |
| `utils.py` | Metrics (NSE, RMSE), plotting, `save_model()`, `mask_points_inside_building()`, `ask_for_confirmation()` | All experiments | OK |
| `reporting.py` | Console logging (`print_epoch_stats`, `print_final_summary`), Aim tracking (`log_metrics`) | All experiments | OK |

### 1.2 Experiment Scripts (`src/scenarios/`)

| File | Experiment | Purpose | Imports From |
|------|-----------|---------|-------------|
| `experiment_1/experiment_1.py` | Exp 1 | Analytical dam-break baseline | config, data, models, losses, utils, physics, reporting |
| `experiment_1/analytical_ntk.py` | Exp 1 variant | NTK weight adaptation | Above + `src.ntk` |
| `experiment_1/experiment_1_deeponet.py` | Exp 1 variant | DeepONet operator learning | Above + `src.losses.compute_operator_*` (**BROKEN — functions don't exist**) |
| `experiment_1/experiment_1_lbfgs_finetune.py` | Exp 1 variant | L-BFGS fine-tuning from checkpoint | config, data, models, losses, utils, physics |
| `experiment_1/train_deeponet.py` | Exp 1 variant | DeepONet + GradNorm | `src.operator_learning.*` (**BROKEN — module doesn't exist**) |
| `experiment_2/experiment_2.py` | Exp 2 | Building obstacle scenario | config, data, models, losses (incl. building_bc), utils, physics, reporting |
| `experiment_3/experiment_3.py` | Exp 3 | X-slope + bathymetry | config, data (incl. bathymetry, LHS), losses (atomic BCs), utils, reporting |
| `experiment_3/pix2pix_experiment_3.py` | Exp 3 variant | Pix2Pix GAN surrogate | config, data (bathymetry only), utils, reporting; custom Flax models in-file |
| `experiment_4/experiment_4.py` | Exp 4 | X+Y slope | Same as Exp 3 |
| `experiment_5/experiment_5.py` | Exp 5 | Synthetic complexity stage 1 | Same as Exp 3 |
| `experiment_6/experiment_6.py` | Exp 6 | Synthetic complexity stage 2 | Same as Exp 3 |
| `experiment_7/experiment_7.py` | Exp 7 | Irregular boundaries, mesh | config, data (IrregularDomainSampler), models, losses (slip_wall_generalized), utils, reporting |
| `experiment_8/experiment_8.py` | Exp 8 | Real urban domain (Eastbourne) | Same as Exp 7 + building boundaries |
| `experiment_8/experiment_8_imp_samp.py` | Exp 8 variant | Importance sampling | Above + `src.physics.SWEPhysics` directly |

**Missing `__init__.py`:** None of the 8 `experiment_N/` directories contain `__init__.py`. The top-level `src/scenarios/__init__.py` exists but is empty. Running as `python -m src.scenarios.experiment_1.experiment_1` may fail depending on Python version and import resolution.

### 1.3 Configuration Files (31 total)

| Directory | Count | Purpose |
|-----------|-------|---------|
| `configs/` | 13 | Experiment-specific YAML configs |
| `configs/train/` | 5 | HPO-optimised per-architecture configs |
| `optimisation/configs/exploration/` | 6 | Sobol exploration HPO configs |
| `optimisation/configs/exploitation/` | 6 | TPE exploitation HPO configs |
| `test/test_assets/` | 1 | Minimal test config |

**Notable:** No `configs/experiment_2.yaml` or `configs/experiment_6.yaml` exist. Experiment 2 uses `configs/train/fourier_experiment_2.yaml` or `configs/train/DGM_experiment_2.yaml`. Experiment 6 has no dedicated config file in `configs/` — only referenced via `experiment_8.yaml` (which points to `benchmark_test_4`).

### 1.4 HPO System (`optimisation/`)

| File | Purpose |
|------|---------|
| `run_optimization.py` | Main HPO entry point; creates Optuna study with TPE + MedianPruner |
| `run_sensitivity_analysis.py` | Exploration phase; QMCSampler + NopPruner |
| `objective_function.py` | Config-driven hyperparameter suggestion; mutates config per trial |
| `optimization_train_loop.py` | Single trial training loop (adapted from train.py) |
| `extract_best_params.py` | Post-processing: extract best trials from SQLite DBs |
| `analyze_importance.py` | Parameter importance via fANOVA |
| `run_all_exploitations.sh` | Shell driver for all exploitation runs |
| `database/exploration/` | 6 Optuna SQLite DBs (one per arch x scenario) |
| `database/exploitation/` | 6 Optuna SQLite DBs |
| `results/` | 13 result directories with best trial YAML configs |

### 1.5 Scripts (`scripts/`)

| File | Purpose | Used By |
|------|---------|---------|
| `create_samples.py` | Generate training/validation samples from validation_tensor.npy (GPU-accelerated, memory-mapped) | All data-driven experiments |
| `filter_sample.py` | Post-hoc time filtering on validation_sample.npy | Ad-hoc analysis |
| `convert_bin_to_npy.py` | Convert C++ binary output to NumPy | Data pipeline after C++ preprocessing |
| `preprocess_irregular.py` | Triangulation + boundary detection for irregular domains | Exp 7-8 |
| `extract_gauges.py` | Extract observation point time-series from tensors | Benchmark validation |
| `process_gauges_split.py` | Merge InfoWorks gauge CSVs, train/val split | Exp 7-8 (real-world gauges) |
| `process_test2_gauges.py` | Merge gauge CSVs (no split) | Benchmark test 2 |
| `render_video.py` | Animate predicted water depth over mesh domain | Visualisation (has hardcoded paths) |
| `run_preprocess.sh` | CMake build + run C++ preprocessor | Data pipeline |
| `jobs/run_job.sh` | Parallel HPO on H100 cluster (3 jobs, NVMe staging) | HPO exploration |
| `jobs/run_job_L40.sh` | Sequential HPO on L40S GPU | HPO exploitation |
| `jobs/run_all_hpo.sh` | All 6 arch x scenario HPO runs sequentially | Full HPO sweep |
| `jobs/run_jobs_temp.sh` | Partial HPO (2 studies) | Testing |
| `utils/gpu_debug.py` | GPU diagnostics (PyTorch/CuPy/JAX detection) | Debug |
| `cpp/preprocess.cpp` | C++ CSV merger (angle/depth/speed → binary tensor) | Data pipeline |

### 1.6 Tests (`test/`)

| File | Purpose | Coverage |
|------|---------|----------|
| `test_train.py` | Validates unified training pipeline | Config loading, model init, training loop, loss computation |
| `test_train_gradnorm.py` | 6 test cases covering all mode combinations | data_free x gradnorm x building (2x2x2 minus 2) |
| `test_assets/test_config.yaml` | Minimal config for tests | FourierPINN, 2 epochs, small grid |

### 1.7 Other Files

| File/Dir | Purpose |
|----------|---------|
| `pyproject.toml` | Package metadata (swe_pinn 0.1.0) |
| `README.md` | Project documentation |
| `CLAUDE.md` | AI assistant guidance |
| `.claudeignore` | Excludes data/, models/, results/ from AI analysis |
| `.gitignore` | Standard Python + large data exclusions |
| `.devcontainer/` | Docker dev container (NVIDIA JAX base, GDAL, requirements.txt) |
| `.github/workflows/docker-publish.yml` | CI/CD: manual Docker build → GHCR |
| `notebook/` | 5 Jupyter notebooks (analytical_viz, benchmark_test_1/5/6, data_preprocessing) |
| `notes/` | 3 markdown files (main, optimisation, one_buildings_recom) |
| `data/` | InfoWorks ICM data (benchmark_test_1-5, experiment_6, one_building_DEM_zero) |
| `models/` | 185+ trained checkpoints across all experiments |

---

## 2. Dependency Chains per Experiment

### 2.1 Experiment 1 — Analytical Dam-Break (WORKING)

```
experiment_1.py
  ├─ load_config(yaml) → config dict; sets global DTYPE, EPS
  ├─ init_model(model_class, key, config) → model, params
  ├─ sample_domain(key, n, bounds) → [x, y, t] points
  ├─ get_batches_tensor(key, data, batch_size, total) → batched arrays
  ├─ Training loop:
  │   ├─ compute_pde_loss(model, params, batch, config)
  │   ├─ compute_ic_loss(model, params, batch)
  │   ├─ compute_bc_loss(model, params, L, R, B, T, config)
  │   ├─ compute_neg_h_loss(model, params, batch)
  │   └─ total_loss(terms, weights)
  ├─ Validation: h_exact(x, t, n_manning, u_const) → ground truth; nse(), rmse()
  └─ save_model(params, model_dir, trial_name) → {trial_name}_params.pkl
```

**Status:** All imports resolve. No external data files needed. Fully functional.

### 2.2 Experiment 2 — Building Obstacle (WORKING)

Same as Exp 1 plus:
- `compute_building_bc_loss()` for obstacle walls
- `mask_points_inside_building()` for masking
- `plot_comparison_scatter_2d()` for 2D visualisation
- Loads `data/{scenario}/validation_sample.npy` and optionally `training_dataset_sample.npy`

**Status:** All imports resolve. Requires data files.

### 2.3 Experiments 3-6 — Topographic Complexity (WORKING)

Same core chain plus:
- `load_bathymetry(dem_path)` → sets global `_BATHYMETRY_FN`
- `bathymetry_fn(x, y)` → `(z, dz/dx, dz/dy)` used in PDE loss
- `load_boundary_condition(csv_path)` → time-varying BC function
- `sample_lhs()` for Latin Hypercube Sampling (Exp 3)
- Atomic loss functions: `loss_boundary_dirichlet_h`, `loss_boundary_wall_horizontal/vertical`
- Loads validation/training `.npy` files from `data/{scenario}/`

**Status:** All imports resolve. Requires data + bathymetry files.

### 2.4 Experiments 7-8 — Irregular/Real-World Domains (WORKING)

Different sampling chain:
- `IrregularDomainSampler(config)` → loads `domain_artifacts.npz`
- `loss_slip_wall_generalized(model, params, batch)` for arbitrary wall normals
- Building boundaries treated as additional wall BCs (Exp 8)
- `experiment_8_imp_samp.py` adds `SWEPhysics` for per-point residual computation

**Status:** All imports resolve. Requires mesh artifacts and data files.

### 2.5 Experiment 1 — DeepONet Variant (BROKEN)

```
experiment_1_deeponet.py
  └─ from src.losses import compute_operator_pde_loss, ...  ← DOES NOT EXIST
```

**`src/losses.py` has no `compute_operator_*` functions.** This script will crash on import with `ImportError`.

### 2.6 Experiment 1 — train_deeponet.py (BROKEN)

```
train_deeponet.py
  ├─ from src.operator_learning.losses_op import ...  ← MODULE DOES NOT EXIST
  ├─ from src.operator_learning.physics_op import h_exact  ← MODULE DOES NOT EXIST
  └─ References undefined variable `val_param_names` at line 671 (should be `param_names`)
```

**`src/operator_learning/` directory does not exist.** This script will crash on import.

### 2.7 gradnorm.py Import Chain (BROKEN at module level)

```
gradnorm.py (line 18)
  └─ from src.operator_learning.losses_op import ...  ← MODULE DOES NOT EXIST
```

**This means `import src.gradnorm` fails, which means `src/train.py` (line 37) fails.** However, this is at the top-level import — if the import actually triggers, all training is blocked.

**Investigation:** Checking whether this import is currently guarded or whether train.py actually runs:

The import at `gradnorm.py:18` is unconditional. This means **`python src/train.py` should fail on startup**. If training currently works, it's because experiments use their own scripts in `src/scenarios/` rather than `src/train.py`, and those scripts import from `src.losses` directly without going through `gradnorm.py`.

### 2.8 Experiment 1 — NTK Variant (WORKING)

```
analytical_ntk.py
  └─ from src.ntk import compute_ntk_traces, update_ntk_weights_algo1  ← EXISTS
```

**Status:** Imports resolve. `src/ntk.py` exists and exports these functions.

---

## 3. Config Completeness

### 3.1 Required Fields Checklist

Every config must define: `training` (learning_rate, epochs, batch_size, seed), `model` (name, width, depth, output_dim), `domain` (lx, ly, t_final), `physics` (g, n_manning), `loss_weights` (pde_weight, ic_weight, bc_weight, neg_h_weight), `device` (dtype), `numerics` (eps).

### 3.2 Completeness Matrix

| Config File | training | model | domain (lx,ly) | physics | weights | dtype | eps | Status |
|-------------|----------|-------|-----------------|---------|---------|-------|-----|--------|
| experiment_1_fourier.yaml | OK | OK | 1200, 100 | OK | 7 weights | float32 | 1e-6 | **COMPLETE** |
| experiment_1_ntk_config.yaml | OK | OK | 1200, 100 | OK | 4 weights | float32 | 1e-6 | **COMPLETE** |
| experiment_1_deeponet.yaml | OK | OK | 1200, 100 | OK | 5 weights | float32 | 1e-6 | **COMPLETE** |
| experiment_3.yaml | OK | OK | 300, 100 | OK | 5 weights | float32 | 1e-6 | **COMPLETE** |
| experiment_4.yaml | OK | OK | 1000, 2000 | OK | 5 weights | float32 | 1e-6 | **COMPLETE** |
| experiment_5.yaml | OK | OK | **MISSING lx, ly** | OK | 5 weights | float32 | 1e-6 | **INCOMPLETE** |
| experiment_7.yaml | OK | OK | **MISSING lx, ly** | OK | 5 weights | float32 | 1e-6 | **INCOMPLETE** |
| experiment_8.yaml | OK | OK | 1000, 2000 | OK | 5 weights | float32 | 1e-6 | **COMPLETE** |
| pix2pix_experiment_3.yaml | OK | OK | Custom | OK | 4 weights | float32 | 1e-6 | **COMPLETE** |
| dgm_datafree_static_exp1.yaml | OK | OK | 1200, 100 | OK | 6 weights | float32 | 1e-6 | **COMPLETE** |
| dgm_datafree_gradnorm_exp1.yaml | OK | OK | 1200, 100 | OK | 6 weights | float32 | 1e-6 | **COMPLETE** |
| config_operatornet_exp1.yaml | OK | OK | 1200, 100 | OK | 4 weights | float32 | 1e-6 | **COMPLETE** |
| train/mlp_experiment_1.yaml | OK | OK | 1200, 100 | OK | Full | float32 | 1e-6 | **COMPLETE** |
| train/fourier_experiment_1.yaml | OK | OK | 1200, 100 | OK | Full | float32 | 1e-6 | **COMPLETE** |
| train/fourier_experiment_2.yaml | OK | OK | 1200, 100 | OK | Full | float32 | 1e-6 | **COMPLETE** |
| train/DGM_no_experiment_1.yaml | OK | OK | 1200, 100 | OK | Full | float32 | 1e-6 | **COMPLETE** |
| train/DGM_experiment_2.yaml | OK | OK | 1200, 100 | OK | Full | float32 | 1e-6 | **COMPLETE** |

**Missing configs:** No `configs/experiment_2.yaml` or `configs/experiment_6.yaml` exist. Experiment 2 uses architecture-specific configs in `configs/train/`. Experiment 6 has no config file at all.

### 3.3 Config Issues

| Issue | File(s) | Severity |
|-------|---------|----------|
| Missing `domain.lx` and `domain.ly` | experiment_5.yaml, experiment_7.yaml | HIGH — `config["domain"]["lx"]` will KeyError |
| No config for Experiment 6 | (missing file) | MEDIUM — experiment_6.py has no matching config |
| No config for Experiment 2 | (missing file) | LOW — uses train/ configs instead |
| All seeds hardcoded to 42 | All 17 configs | OK (intentional for reproducibility) |
| All dtypes are float32 | All 17 configs | OK (consistent) |

---

## 4. Data Pipeline Readiness

### 4.1 Data Path Configuration

Experiment scripts construct data paths as `data/{scenario}/` where `scenario` comes from `config["scenario"]`. The pipeline is:

```
Config YAML  →  scenario name  →  data/{scenario}/validation_sample.npy
                                   data/{scenario}/training_dataset_sample.npy
                                   data/{scenario}/validation_plotting_t_{t}s.npy
```

Paths are **configurable** via the `scenario` key, not hardcoded.

### 4.2 Data Format

All `.npy` data files use the format `[t, x, y, h, u, v]` (6 columns). Scripts reorder to `[x, y, t]` for model input via `data[:, [1, 2, 0]]`. This reordering is repeated in every experiment script — no shared utility function.

### 4.3 Shape Compatibility

- Model input: `(batch, 3)` for `[x, y, t]`
- Model output: `(batch, 3)` for `[h, hu, hv]`
- Data columns 3-5 provide `[h, u, v]` ground truth
- **No shape validation exists** — if data has wrong columns, errors are silent or cryptic.

### 4.4 Bathymetry Loading

`load_bathymetry()` sets a module-level global `_BATHYMETRY_FN`. If not called, `bathymetry_fn(x, y)` silently returns `(0, 0, 0)` — treating the domain as flat with **no warning**.

### 4.5 Irregular Domain Artifacts

Exp 7-8 require `domain_artifacts.npz` produced by `scripts/preprocess_irregular.py`. The `IrregularDomainSampler` raises `FileNotFoundError` if this file is missing.

---

## 5. Reproducibility & Precision

### 5.1 Random Seeds

All configs set `seed: 42`. Within scripts, the seed is split for model init, training, and validation using `jax.random.split()`. Consistent across all experiments.

### 5.2 Float Precision

- **All configs specify `float32`** — no float64 configs exist.
- `DTYPE` is set globally in `config.py` and imported by `models.py`, `data.py`, `losses.py`.
- `physics.py` uses `jnp` operations that inherit the global dtype.
- **No untyped operations found** that would silently promote to float64.

### 5.3 Machine Epsilon

All configs use `eps: 1.0e-06` consistently. This is used in `SWEPhysics` for `h_safe = max(h, eps)` to prevent division by zero.

---

## 6. Checkpoint & Output Paths

### 6.1 Naming Convention

```python
# utils.py
trial_name = f"{timestamp}_{config_filename}"
model_path = f"{save_dir}/{trial_name}_params.pkl"
```

Produces: `models/2026-03-11_14-30_experiment_1_fourier/{trial_name}_params.pkl`

### 6.2 Output Structure

```
models/{trial_name}/
  └── {trial_name}_params.pkl

results/{trial_name}/
  ├── validation_scatter_*.png
  └── training_loss_*.png
```

### 6.3 render_video.py Hardcoded Paths

```python
TRIAL_DIR_NAME = "2026-02-10_17-41_experiment_6"   # Hardcoded
CONFIG_NAME = "experiment_6.yaml"                    # Hardcoded (this config doesn't exist!)
SHAPEFILE_PATH = "data/experiment_6/2D Zones.shp"    # Hardcoded
```

This script will fail because `configs/experiment_6.yaml` does not exist.

### 6.4 extract_best_params.py Hardcoded Paths

```python
db_dir = Path("/workspaces/swe-pinn/optimisation/database/exploration")       # Absolute
output_base_dir = Path("/workspaces/swe-pinn/optimisation/sensivity_analysis_output/best_parameters")  # Absolute
```

Not portable to other machines or clone locations.

---

## 7. Dead Code, Orphans & Inconsistencies

### 7.1 Dead Code — Unused Modules

| Module | Lines | Evidence | Recommendation |
|--------|-------|----------|----------------|
| `src/softadapt.py` | 58 | `grep -r "softadapt\|from src.softadapt" --include="*.py"` returns **zero** matches outside itself | Remove or integrate with config flag |
| `src/ntk.py` (partially) | 90 | Only imported by `analytical_ntk.py`; never used by main training or HPO | Document as Exp 1 variant only |

### 7.2 Dead Code — Missing Modules Referenced

| Missing Module | Referenced By | Impact |
|----------------|-------------|--------|
| `src/operator_learning/losses_op.py` | `gradnorm.py:18`, `train_deeponet.py:37` | **ImportError** — breaks `gradnorm.py` import chain, which breaks `train.py` |
| `src/operator_learning/physics_op.py` | `train_deeponet.py:41` | **ImportError** — breaks train_deeponet.py |
| `src.losses.compute_operator_*` functions | `experiment_1_deeponet.py:33-35` | **ImportError** — these functions do not exist in losses.py |

### 7.3 Orphaned Files

| File | Issue |
|------|-------|
| `configs/experiment_6.yaml` | **Does not exist** but is referenced by `render_video.py` |
| `configs/experiment_2.yaml` | **Does not exist** — Exp 2 relies on `configs/train/` architecture-specific configs |
| `notes/` directory (3 files) | Not referenced anywhere in code; purely documentation |
| `optimisation/run_all_exploitations.sh` | Shell driver; may be stale if individual exploitation configs changed |

### 7.4 Inconsistent Naming

| Item | Issue |
|------|-------|
| `sensivity_analysis_output/` | Typo: should be `sensitivity_analysis_output` |
| `experiment_8.yaml` → `scenario: benchmark_test_4` | Config named for Exp 8 but points to benchmark_test_4 scenario |
| `experiment_5.yaml` → `scenario: benchmark_test_3` | Config named for Exp 5 but points to benchmark_test_3 scenario |
| Loss weight keys use `_weight` suffix in config but are stripped in code | Intentional but undocumented convention |

### 7.5 Inconsistent Function Signatures

**`loss_slip_wall_generalized()` in `src/losses.py:128`** uses a different `model.apply()` pattern than all other loss functions:

```python
# All other losses (e.g., line 88):
U_pred = model.apply({'params': params['params']}, batch, train=False)

# loss_slip_wall_generalized (line 136):
U = model.apply(params, coords, train=True)
```

This means the caller must pass `params` in a different format for this one function. If GradNorm or any generic wrapper iterates over all loss functions assuming a uniform signature, this will break.

### 7.6 Duplicated Logic

| Duplication | Location | Lines |
|-------------|----------|-------|
| `update_gradnorm_weights()` vs `update_gradnorm_weights_operatornet()` | gradnorm.py:107-193 vs 300-389 | ~87 lines, 95% identical |
| Data reordering `[:, [1, 2, 0]]` | Every experiment script (Exp 2-8) | ~1 line per script, 7 occurrences |
| `sample_points()` vs `sample_domain()` | data.py:10-31 vs 49-67 | Two functions doing nearly the same thing with different APIs |

### 7.7 Global Mutable State

| Location | Variable | Risk |
|----------|----------|------|
| `config.py:31-33` | `DTYPE`, `EPS` | Multiple `load_config()` calls overwrite; not thread-safe |
| `data.py:169` | `_BATHYMETRY_FN` | Multiple `load_bathymetry()` calls overwrite; silent zero-fallback |

### 7.8 Silent Fallbacks

| Location | Behavior | Risk |
|----------|----------|------|
| `data.py:228-231` | If bathymetry not loaded, returns `(0, 0, 0)` — flat domain | Bug goes unnoticed if bathymetry file path is wrong |
| Experiment scripts data loading | If training data file missing, silently switches to data-free mode | User unaware training is physics-only |

### 7.9 Undefined Variable

| File | Line | Variable | Should Be |
|------|------|----------|-----------|
| `train_deeponet.py` | 671 | `val_param_names` | `param_names` (defined at line 143) |

### 7.10 Deviations from CLAUDE.md Structure

| CLAUDE.md Says | Actual |
|----------------|--------|
| `configs/experiment_6.yaml` exists | **Does not exist** |
| `configs/experiment_2.yaml` implied | **Does not exist** |
| `src/scenarios/experiment_1/train_deeponet.py` works | **Broken imports** |
| `src/train.py` is unified entry point | **Broken** — cannot import due to gradnorm.py dependency on missing module |

---

## 8. Critical Issues Summary

### CRITICAL (Blocks Execution)

| # | Issue | Location | Impact | Fix |
|---|-------|----------|--------|-----|
| C1 | Missing `src/operator_learning/` module | `gradnorm.py:18-21`, `train_deeponet.py:37-41` | `src/train.py` fails on import; both DeepONet scripts crash | Guard with `try/except ImportError` or create the module |
| C2 | Missing `compute_operator_*` functions in losses.py | `experiment_1_deeponet.py:33-35` | Script crashes on import | Add functions to losses.py or fix import path |
| C3 | Undefined variable `val_param_names` | `train_deeponet.py:671` | Runtime crash when printing validation metrics | Change to `param_names` |
| C4 | Missing domain bounds `lx`, `ly` | `experiment_5.yaml`, `experiment_7.yaml` | `KeyError` at runtime when accessing `config["domain"]["lx"]` | Add lx, ly values to these configs |

### HIGH (Functional Risk)

| # | Issue | Location | Impact | Fix |
|---|-------|----------|--------|-----|
| H1 | No `configs/experiment_6.yaml` | `configs/` | Exp 6 has no config; `render_video.py` references it | Create config or fix render_video.py |
| H2 | `loss_slip_wall_generalized` signature inconsistency | `losses.py:136` | Uses `model.apply(params, ...)` instead of `model.apply({'params': params['params']}, ...)` | Standardize signature |
| H3 | Missing `__init__.py` in experiment directories | `src/scenarios/experiment_*/` | Module import may fail | Add empty `__init__.py` files |
| H4 | `softadapt.py` completely unused | `src/softadapt.py` | Dead code, maintenance burden | Remove or integrate |

### MEDIUM (Quality/Maintainability)

| # | Issue | Location | Impact | Fix |
|---|-------|----------|--------|-----|
| M1 | Hardcoded absolute paths | `extract_best_params.py:7-8` | Not portable | Use `Path(__file__).parent` relative paths |
| M2 | Hardcoded paths in render_video.py | `render_video.py:32-35` | Breaks for any other trial/config | Add CLI arguments |
| M3 | Silent bathymetry fallback | `data.py:228-231` | Bugs go unnoticed | Add warning or raise error |
| M4 | Global mutable state (DTYPE, EPS, _BATHYMETRY_FN) | `config.py`, `data.py` | Not thread-safe, hard to test | Use dependency injection |
| M5 | Duplicated gradnorm update functions | `gradnorm.py:107-389` | 87 lines duplicated | Refactor to single function with switchable loss map |
| M6 | `sensivity_analysis_output/` typo | `optimisation/` | Confusing | Rename to `sensitivity_analysis_output/` |
| M7 | No data shape validation | All experiment scripts | Wrong data format gives cryptic errors | Add shape assertions on load |
| M8 | Repeated `[:, [1, 2, 0]]` column reorder | All experiment scripts | Duplication; error-prone | Create `load_validation_data()` utility |
| M9 | Non-descriptive and mismatched naming across data folders, configs, and scenario keys | See Section 9 below | Cannot tell which experiment a folder/config/scenario serves without cross-referencing multiple files | Adopt consistent descriptive naming |

---

## 9. Non-Descriptive and Mismatched Naming of Data Folders, Configs, and Scenario Keys

### 9.1 The Problem

Three naming systems exist for the same experiments, and none of them align:

| Experiment | Config File(s) | `scenario:` Key in Config | Data Folder |
|------------|---------------|--------------------------|-------------|
| Exp 1 (no building) | `experiment_1_fourier.yaml`, `experiment_1_ntk_config.yaml`, `experiment_1_deeponet.yaml` | *(none — no scenario key)* | *(none — analytical only)* |
| Exp 1 (no building, HPO) | `configs/train/mlp_experiment_1.yaml`, `fourier_experiment_1.yaml`, `DGM_no_experiment_1.yaml` | `no_building_scenario` | *(no matching data folder)* |
| Exp 1 (building, DGM) | `dgm_datafree_static_experiment_1.yaml`, `dgm_datafree_gradnorm_experiment_1.yaml` | `one_building_DEM_zero` | *(no matching data folder by this name)* |
| Exp 2 (building, HPO) | `configs/train/fourier_experiment_2.yaml`, `DGM_experiment_2.yaml` | `one_building_DEM_zero` | `data/experiment_2/` |
| Exp 3 | `experiment_3.yaml` | `benchmark_test_1` | `data/experiment_3/` |
| Exp 3 (Pix2Pix) | `pix2pix_experiment_3.yaml` | `benchmark_test_1` | `data/experiment_3/` |
| Exp 4 | `experiment_4.yaml` | `benchmark_test_2` | `data/experiment_4/` |
| Exp 5 | `experiment_5.yaml` | `benchmark_test_3` | `data/experiment_5/` |
| Exp 6 | *(no config file)* | — | `data/experiment_6/` |
| Exp 7 | `experiment_7.yaml` | `benchmark_test_5` | `data/experiment_7/` |
| Exp 8 | `experiment_8.yaml` | `benchmark_test_4` | `data/experiment_8/` |

### 9.2 Specific Issues

**Issue 1: `scenario:` keys are opaque legacy names that don't match any experiment or data folder.**

- `benchmark_test_1` through `benchmark_test_5` are numbered independently of experiments. Without a lookup table, there's no way to know that `benchmark_test_1` serves Experiment 3 while `benchmark_test_5` serves Experiment 7.
- `no_building_scenario` and `one_building_DEM_zero` are descriptive of geometry but don't identify which experiment they belong to.

**Issue 2: Config filenames don't follow a single convention.**

| Pattern | Examples | Problem |
|---------|----------|---------|
| `experiment_N.yaml` | `experiment_3.yaml` ... `experiment_8.yaml` | OK, but not used for Exp 1 or 2 |
| `experiment_N_<variant>.yaml` | `experiment_1_fourier.yaml`, `experiment_1_deeponet.yaml` | Good for Exp 1 only |
| `<arch>_experiment_N.yaml` | `DGM_experiment_2.yaml`, `fourier_experiment_2.yaml` | Inverts the naming order vs above |
| `<arch>_<mode>_<weighting>_experiment_N.yaml` | `dgm_datafree_static_experiment_1.yaml` | Very different from the rest |
| `config_<arch>_experiment_N.yaml` | `config_operatornet_experiment_1.yaml` | Yet another pattern with `config_` prefix |
| `pix2pix_experiment_N.yaml` | `pix2pix_experiment_3.yaml` | Model-first naming |
| `experiment_N_ntk_config.yaml` | `experiment_1_ntk_config.yaml` | Adds `_config` suffix (redundant for a .yaml) |

Six different naming patterns for 18 config files.

**Issue 3: Data folders use `experiment_N/` naming, but code constructs paths from the `scenario:` key — not the experiment number.**

The code does `data/{scenario}/validation_sample.npy`, so when `experiment_3.yaml` says `scenario: benchmark_test_1`, it looks for `data/benchmark_test_1/`, not `data/experiment_3/`. This means either:
- The data folders have been renamed from `benchmark_test_*` to `experiment_*` without updating configs, or
- There are symlinks or the scenario key is overridden at runtime.

Either way, **the config `scenario:` key and the actual data folder name are mismatched**, which will cause `FileNotFoundError` unless there's a runtime workaround.

**Issue 4: HPO configs use yet another naming layer.**

- Exploration: `hpo_<arch>_datafree_static_<BUILDING|NOBUILDING>.yaml`
- Exploitation: `hpo_exploitation_<arch>_<building|nobuilding>.yaml`
- Inconsistent casing: `BUILDING` vs `building`, `NOBUILDING` vs `nobuilding`

### 9.3 Recommended Naming Convention

Adopt a single scheme: `experiment_<N>_<arch>_<variant>.yaml` for configs, `experiment_<N>/` for data folders, and `experiment_<N>` as the `scenario:` key.

**Configs:**
```
configs/
├── experiment_1_fourier.yaml          (already good)
├── experiment_1_mlp.yaml              (rename from train/mlp_experiment_1.yaml)
├── experiment_1_dgm.yaml              (rename from train/DGM_no_experiment_1.yaml)
├── experiment_1_dgm_datafree.yaml     (rename from dgm_datafree_static_experiment_1.yaml)
├── experiment_1_dgm_gradnorm.yaml     (rename from dgm_datafree_gradnorm_experiment_1.yaml)
├── experiment_1_deeponet.yaml         (already good)
├── experiment_1_ntk.yaml              (rename from experiment_1_ntk_config.yaml)
├── experiment_1_operatornet.yaml      (rename from config_operatornet_experiment_1.yaml)
├── experiment_2_fourier.yaml          (rename from train/fourier_experiment_2.yaml)
├── experiment_2_dgm.yaml              (rename from train/DGM_experiment_2.yaml)
├── experiment_3.yaml                  (already good)
├── experiment_3_pix2pix.yaml          (rename from pix2pix_experiment_3.yaml)
├── experiment_4.yaml                  (already good)
├── experiment_5.yaml                  (already good)
├── experiment_6.yaml                  (CREATE — currently missing)
├── experiment_7.yaml                  (already good)
└── experiment_8.yaml                  (already good)
```

**Scenario keys in configs:**
```yaml
# Before (opaque):
scenario: benchmark_test_1     # Which experiment is this?
scenario: one_building_DEM_zero  # Which experiment?

# After (self-documenting):
scenario: experiment_3         # Matches data/experiment_3/
scenario: experiment_2         # Matches data/experiment_2/
```

**Data folders:** Already use `experiment_N/` — this is the target convention. Configs should match.

**HPO configs:** Standardize casing and prefix:
```
hpo_exploration_<arch>_<scenario>.yaml    (e.g., hpo_exploration_dgm_building.yaml)
hpo_exploitation_<arch>_<scenario>.yaml   (e.g., hpo_exploitation_dgm_building.yaml)
```

### 9.4 Cross-Reference Mismatch Summary

| Config's `scenario:` Key | Expected Data Path | Actual Data Folder | Match? |
|--------------------------|-------------------|--------------------|--------|
| `benchmark_test_1` | `data/benchmark_test_1/` | `data/experiment_3/` | **NO** |
| `benchmark_test_2` | `data/benchmark_test_2/` | `data/experiment_4/` | **NO** |
| `benchmark_test_3` | `data/benchmark_test_3/` | `data/experiment_5/` | **NO** |
| `benchmark_test_4` | `data/benchmark_test_4/` | `data/experiment_8/` | **NO** |
| `benchmark_test_5` | `data/benchmark_test_5/` | `data/experiment_7/` | **NO** |
| `one_building_DEM_zero` | `data/one_building_DEM_zero/` | `data/experiment_2/` | **NO** |
| `no_building_scenario` | `data/no_building_scenario/` | *(none — analytical)* | N/A |

**Every data-dependent config has a scenario key that doesn't match the current data folder name.** This is the highest-impact naming issue: it means either data folders were renamed without updating configs, or there's a runtime mapping layer not visible in the YAML.

---

*End of audit.*
