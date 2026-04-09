# GEMINI.md

This file provides context, engineering standards, and operational guidelines for the Gemini CLI agent working within the SWE-PINN codebase. These instructions take absolute precedence over the general workflows and defaults.

## Project Overview

SWE-PINN is a **Physics-Informed Neural Network (PINN) framework for urban flood prediction**. It solves the **2D Shallow Water Equations (SWE)** as a neural PDE solver: the network takes spatiotemporal coordinates `(x, y, t)` as input and predicts water depth `h` and specific discharges `hu`, `hv`. Physics is enforced through a composite loss that penalizes PDE residuals (via automatic differentiation), initial condition violations, boundary condition violations, and deviation from observational data produced by InfoWorks ICM numerical simulations.

The codebase is built in Python with JAX and Flax, using Optuna for hyperparameter optimization.

### Neural Network Architectures
- **MLP** — standard multi-layer perceptron baseline
- **Fourier-MLP** — projects inputs into a higher-dimensional frequency space to overcome spectral bias
- **DGM** (Deep Galerkin Method) — injects raw coordinates into every hidden layer via gated sub-networks

## Tech Stack

- **ML Framework**: JAX + Flax (neural networks) + Optax (optimizers)
- **Language**: Python 3.8+
- **GPU**: CUDA via NVIDIA JAX Docker image (`nvcr.io/nvidia/jax:25.01-py3`)
- **Hyperparameter Optimization**: Optuna
- **Experiment Tracking**: Weights & Biases (W&B)
- **Visualization**: Matplotlib, Seaborn
- **Configuration**: YAML files loaded via `src/config.py`

## Repository Layout

The source code is organised into subpackages under `src/`:

| Package | Purpose |
|---------|---------|
| `src/balancing/` | Importance sampling, Relobralo adaptive balancing |
| `src/checkpointing/` | Model checkpoint save/load |
| `src/data/` | Batching, sampling, loading, bathymetry |
| `src/inference/` | Post-training inference pipeline |
| `src/losses/` | PDE, BC, IC, data, composite losses |
| `src/metrics/` | NSE, RMSE, conservation, flood extent, peak errors |
| `src/models/` | FourierPINN, MLP, DGM, DeepONet, NTK (`pinn.py`) |
| `src/monitoring/` | W&B tracker, console logger, diagnostics |
| `src/physics/` | SWE equations (`swe.py`), analytical solutions |
| `src/plots/` | Time series, spatial maps, comparisons, HPO plots |
| `src/predict/` | Batched prediction wrapper |
| `src/training/` | Training loop, epoch, step, optimizer, setup |
| `src/utils/` | Domain, I/O, naming, plotting, profiling |

Experiment training scripts live in `experiments/experiment_N/train.py` (run as `python -m experiments.experiment_N.train --config <config>`).

## Architecture and Design Patterns

### Configuration-Driven Training
All hyperparameters are specified in YAML config files (located in the `configs/` directory). The config structure includes sections like `training`, `model`, `domain`, `grid`, `physics`, `loss_weights`, `device`, and `numerics`. Always use `src.config.load_config()` to load configurations.

### Loss Weighting Strategies
- **Static**: Fixed weights from config
- **Importance Sampling** (`src/balancing/importance_sampling.py`): Adaptive point weighting based on loss magnitude
- **Relobralo** (`src/balancing/relobralo.py`): Relative loss balancing with random lookback

### Data Modes
- **Data-free**: Physics-only training using PDE residuals (no reference data)
- **Data-driven**: Combines physics losses with observational data losses
This is controlled via the `data_weight` config parameter and data file presence.

## Code Conventions

- **JAX functional style**: Heavy use of `@jax.jit` for JIT compilation and pure functions.
- **Flax linen modules**: All neural networks use `nn.Module` from `flax.linen`.
- **Type hints**: Strictly used in function signatures throughout the codebase.
- **Imports**: Utilize standard JAX ecosystem imports (`jax`, `jax.numpy as jnp`, `flax.linen as nn`, `optax`).
- **Naming conventions**: `snake_case` for functions/variables, `PascalCase` for classes.

## Testing Conventions

- **Test framework**: Python `unittest`. Tests are located in the `test/` directory.
- **Execution**: Tests MUST force JAX to CPU by setting `os.environ["JAX_PLATFORM_NAME"] = "cpu"` at the top of test files for reproducibility.
- **Mocking**: Use `unittest.mock` for mocking file I/O and data. setUp/tearDown should manage temporary directories.
- **Manual Smoke-tests**: When verifying a training code change, run a smoke test with **200 epochs**.

## Common Pitfalls

- **JAX requires explicit device management**: Use `jax.devices()` and config `dtype` settings.
- **JIT compilation**: First call is slow due to tracing; subsequent calls are fast. Strictly avoid Python side effects inside JIT-compiled functions.
- **Memory**: Large validation datasets use memory-mapped numpy arrays (`np.load(..., mmap_mode='r')`). Do not load massive arrays entirely into memory.
- **Float precision**: Certain physics computations require `float64`. This is set via `config.device.dtype`.

## Experimental Programme

The project spans experiments across 3 phases. The authoritative specification for all metrics, plots, tracked values, and module structures lives in:
**`docs/experimental_programme_reference.md`**

**ALWAYS consult this file when:**
- Implementing any evaluation metric (definitions, formulas, units)
- Creating any inference plot (plot type, axes, color conventions)
- Setting up W&B tracking for a training run
- Building or modifying any module in `src/metrics/` or `src/plots/`

**Key Visual Conventions:**
- Accuracy metrics (NSE, RMSE, MAE, Rel L2) must be reported separately for `h`, `hu`, `hv`.
- **Colour palette**: Exeter (Deep Green #003C3C, Teal #007D69, Mint #00C896) + Blue Heart (Navy #0D2B45, Ocean #1B5E8A, Sky #4FA3D1). Font: Arial. Resolution: 300 DPI.

## Key Commands

### Running Training
```bash
# Per-experiment training scripts
python -m experiments.experiment_1.train --config configs/experiment_1/experiment_1.yaml
python -m experiments.experiment_3.train --config configs/experiment_3.yaml
# ... same pattern for experiments 2–8
```

### Running Tests
```bash
# Run all tests
python -m unittest discover test

# Run a specific test module
python -m unittest test.test_train
```

### Hyperparameter Optimization (Optuna)
```bash
python optimisation/run_optimization.py --config <hpo_config> --n_trials 100
python optimisation/run_sensitivity_analysis.py
python optimisation/extract_best_params.py
```
