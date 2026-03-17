# CLAUDE.md

This file provides guidance for AI assistants working with the SWE-PINN codebase.

## Project Overview

SWE-PINN is a **Physics-Informed Neural Network (PINN) framework for urban flood prediction**. It solves the **2D Shallow Water Equations (SWE)** as a neural PDE solver: the network takes spatiotemporal coordinates `(x, y, t)` as input and predicts water depth `h` and specific discharges `hu`, `hv`. Physics is enforced through a composite loss that penalises PDE residuals (via automatic differentiation), initial condition violations, boundary condition violations, and deviation from observational data produced by InfoWorks ICM numerical simulations.

The codebase is built in Python with JAX and Flax, using Optuna for hyperparameter optimisation. Three neural network architectures are compared:
- **MLP** — standard multi-layer perceptron baseline
- **Fourier-MLP** — projects inputs into a higher-dimensional frequency space to overcome spectral bias
- **DGM** (Deep Galerkin Method) — injects raw coordinates into every hidden layer via gated sub-networks

### Research Phases and Experiments

**Phase 1 — Baseline Verification and Architecture Selection**
- **Experiment 1**: Verifies the framework against an analytical dam-break solution on a flat domain.
- **Experiment 2**: Introduces a building obstacle; motivates Fourier-MLP and DGM adoption.

**Phase 2 — Topographic Complexity on Synthetic Domains**
- **Experiment 3**: Introduces terrain slope in the x-direction via bi-linear interpolation.
- **Experiment 4**: Extends slope to both x and y directions.
- **Experiment 5**: Further synthetic complexity to validate robustness.

**Phase 3 — Domain Generalisation and Real-World Application**
- **Experiment 7**: Irregular boundaries using triangulated mesh-based sampling and wall normals.
- **Experiment 8**: Real urban subcatchment in Eastbourne, UK (Blue Heart Project).

## Repository Structure

```
swe-pinn/
├── src/                            # Core source code
│   ├── train.py                    # Unified training script (main entry point)
│   ├── models.py                   # Neural network architectures (FourierPINN, MLP, DGMNetwork)
│   ├── losses.py                   # PDE, IC, BC loss functions for SWE
│   ├── physics.py                  # SWE physics computations and Jacobians
│   ├── softadapt.py                # SoftAdapt adaptive loss weighting
│   ├── ntk.py                      # Neural Tangent Kernel trace computation
│   ├── data.py                     # Data sampling, batching, and validation loading
│   ├── config.py                   # YAML configuration loading
│   ├── utils.py                    # Metrics (NSE, RMSE), plotting, model saving
│   ├── reporting.py                # Training stats logging and Aim integration
│   └── scenarios/                  # Per-experiment training scripts
│       ├── experiment_1/           # Phase 1: analytical dam-break, flat domain
│       │   ├── experiment_1.py
│       │   ├── analytical_ntk.py
│       │   └── experiment_1_lbfgs_finetune.py
│       ├── experiment_2/           # Phase 1: building obstacle
│       │   └── experiment_2.py
│       ├── experiment_3/           # Phase 2: x-direction terrain slope
│       │   └── experiment_3.py
│       ├── experiment_4/           # Phase 2: x+y terrain slope
│       │   └── experiment_4.py
│       ├── experiment_5/           # Phase 2: synthetic complexity
│       │   └── experiment_5.py
│       ├── experiment_6/           # Phase 2: synthetic complexity stage 2
│       │   └── experiment_6.py
│       ├── experiment_7/           # Phase 3: irregular boundaries, mesh-based sampling
│       │   └── experiment_7.py
│       └── experiment_8/           # Phase 3: real urban domain (Eastbourne)
│           ├── experiment_8.py
│           └── experiment_8_imp_samp.py    # Importance sampling variant
├── configs/                        # Experiment configuration YAML files
│   ├── experiment_<N>.yaml             # Per-experiment configs (3–8)
│   ├── experiment_<N>_<arch>.yaml      # Per-architecture variant configs
│   └── train/                          # Final HPO-optimized configs for production runs
│       └── experiment_<N>_<arch>_final.yaml
├── test/                           # Unit tests
│   ├── test_train.py               # Main training script validation
│   └── test_assets/
│       └── test_config.yaml        # Minimal test configuration
├── scripts/                        # Data processing and utility scripts
│   ├── run_preprocess.sh           # Stage 1: Build & run C++ CSV→binary converter
│   ├── binary_to_numpy.py          # Stage 2: Binary→.npy conversion
│   ├── generate_training_data.py   # Stage 3: .npy→training/validation/plotting datasets
│   ├── process_gauge_csvs.py       # Gauge CSV processing (depth/angle/speed→.npy)
│   ├── extract_gauge_timeseries.py # Extract gauge time series from tensor
│   ├── filter_by_time.py           # Filter .npy by maximum time
│   ├── preprocess_irregular.py     # Mesh preprocessing for irregular domains (Exp 7/8)
│   ├── render_video.py             # Render solution animations (CLI-driven)
│   ├── infer.py                    # Post-training inference CLI wrapper
│   ├── lidar_download.py           # Download LIDAR elevation data from UK gov WCS
│   └── cpp/                        # C++ CSV→binary converter
│       └── preprocess.cpp
├── optimisation/                   # Hyperparameter optimisation (Optuna)
│   ├── run_optimization.py         # Main HPO entry point
│   ├── run_sensitivity_analysis.py
│   ├── extract_best_params.py
│   ├── objective_function.py       # Optuna objective
│   ├── optimization_train_loop.py
│   ├── configs/                    # HPO-specific config files
│   │   ├── exploration/            # Sobol exploration configs
│   │   └── exploitation/           # TPE exploitation configs
│   └── sensitivity_analysis_output/  # Importance reports + best params
├── data/                           # Reference simulation data (InfoWorks ICM output)
│   └── experiment_N/               # Per-experiment data folders
├── notebook/                       # Jupyter notebooks for analysis
├── .devcontainer/                  # Docker dev container setup (NVIDIA JAX + CUDA)
├── .github/workflows/              # CI/CD: Docker image build/publish to GHCR
├── pyproject.toml                  # Package metadata and dependencies
└── README.md                       # Project documentation
```

## Tech Stack

- **ML Framework**: JAX + Flax (neural networks) + Optax (optimizers)
- **Language**: Python 3.8+
- **GPU**: CUDA via NVIDIA JAX Docker image (`nvcr.io/nvidia/jax:25.01-py3`)
- **Hyperparameter Optimization**: Optuna
- **Experiment Tracking**: Aim
- **Visualization**: Matplotlib, Seaborn
- **Configuration**: YAML files loaded via `src/config.py`
- **Build System**: setuptools (via `pyproject.toml`)

## Key Commands

### Running Training

```bash
# Main unified training script (takes config path as argument)
python src/train.py <config_path>

# Scenario-specific scripts (run as modules)
python -m src.scenarios.experiment_1.experiment_1 <config_path>
python -m src.scenarios.experiment_2.experiment_2 <config_path>
# ... same pattern for experiments 3–8
```

### Running Tests

```bash
# Run all tests
python -m unittest discover test

# Run specific test files
python -m unittest test.test_train
```

Tests force JAX to CPU (`JAX_PLATFORM_NAME=cpu`) for reproducibility, use mock data and file I/O, and clean up temporary directories in teardown.

### Hyperparameter Optimization

```bash
python optimisation/run_optimization.py --config <hpo_config> --n_trials 100
python optimisation/run_sensitivity_analysis.py
python optimisation/extract_best_params.py
```

### Experiment Tracking

```bash
aim up
```

### Installing Dependencies

```bash
pip install -r .devcontainer/requirements.txt
```

## Architecture and Design Patterns

### Configuration-Driven Training

All hyperparameters are specified in YAML config files. The config structure includes:

| Section | Key Parameters |
|---------|---------------|
| `training` | `learning_rate`, `epochs`, `batch_size`, `seed` |
| `model` | `name` (FourierPINN/MLP/DGMNetwork), `width`, `depth`, `output_dim` |
| `domain` | `lx`, `ly`, `t_final` (spatial/temporal bounds) |
| `grid` | `nx`, `ny`, `nt` (sampling grid points) |
| `physics` | `u_const`, `n_manning`, `g`, `inflow` |
| `loss_weights` | `pde_weight`, `ic_weight`, `bc_weight`, `building_bc_weight`, `neg_h_weight`, `data_weight` |
| `device` | `dtype` (float32/float64), early stopping parameters |
| `numerics` | `eps` (machine epsilon) |

### Neural Network Architectures (`src/models.py`)

1. **FourierPINN** - Fourier feature encoding + dense layers with tanh activation
2. **MLP** - Standard multi-layer perceptron
3. **DGMNetwork** - Deep Galerkin Method with LSTM-like gates

### Loss Weighting Strategies

- **Static**: Fixed weights from config
- **SoftAdapt** (`src/softadapt.py`): Rate-of-change-based adaptive weighting
- **NTK** (`src/ntk.py`): Weights based on Neural Tangent Kernel traces

### Physics Implementation

- `src/physics.py`: `SWEPhysics` class computes flux Jacobians and source terms for the 2D SWE
- `src/losses.py`: PDE residual losses, initial condition losses, boundary condition losses (inflow, zero-gradient, no-flux)
- Supports Manning friction, gravity, and inflow conditions
- Water depth is masked with h >= eps for numerical stability

### Data Modes

- **Data-free**: Physics-only training using PDE residuals (no reference data)
- **Data-driven**: Combines physics losses with observational data losses
- Controlled via the `data_weight` config parameter and data file presence

## Code Conventions

- **JAX functional style**: Heavy use of `@jax.jit` for JIT compilation and pure functions
- **Flax linen modules**: All neural networks use `nn.Module` from `flax.linen`
- **No formal linter configuration**: No `.flake8`, `pylintrc`, or pre-commit hooks are set up
- **Type hints**: Used in function signatures throughout the codebase
- **Imports**: JAX ecosystem imports (`jax`, `jax.numpy as jnp`, `flax.linen as nn`, `optax`)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Config loading**: Always use `src.config.load_config()` to load YAML configs

## Testing Conventions

- Test framework: Python `unittest`
- Tests located in `test/` directory
- Test configs in `test/test_assets/`
- Force CPU execution: `os.environ["JAX_PLATFORM_NAME"] = "cpu"` at top of test files
- Use `unittest.mock` for mocking file I/O and data
- setUp/tearDown for temporary directory management
- Test both data-free and data-driven modes
- **Manual training smoke-tests**: When verifying a training code change, run with **200 epochs**.

## CI/CD

- **GitHub Actions** (`.github/workflows/docker-publish.yml`): Builds and publishes the Docker dev container image to GitHub Container Registry (`ghcr.io/zeinali72/swe-pinn:latest`)
- Triggered manually via `workflow_dispatch`
- Builds for `linux/amd64` platform with GitHub Actions caching

## Common Pitfalls

- **JAX requires explicit device management**: Use `jax.devices()` and config `dtype` settings
- **JIT compilation**: First call is slow due to tracing; subsequent calls are fast. Avoid Python side effects inside JIT-compiled functions.
- **Memory**: Large validation datasets (multi-GB) use memory-mapped numpy arrays. Use `np.load(..., mmap_mode='r')` for big files.
- **Float precision**: Some physics computations require `float64`. Set via `config.device.dtype`.


# Experimental Programme
This project runs 11 experiments across 3 phases. The authoritative specification for all metrics, plots, tracked values, and module structure lives in:
docs/experimental_programme_reference.md
Always consult this file when:

Implementing any evaluation metric (definitions, formulas, units, which experiments use it)
Creating any inference plot (plot type, axes, colour conventions, which experiments need it)
Setting up Aim tracking for a training run (what to log, at what frequency)
Building or modifying any module in evaluation/
Planning what outputs an experiment run should produce

## Key conventions from that document

All accuracy metrics (NSE, RMSE, MAE, Rel L2) reported separately for h, hu, hv
Metrics are grouped: A (accuracy), B (conservation), C (boundary), D (cost), E (HPO), F (data)
Plots are grouped: P1 (time series), P2 (spatial maps), P3 (comparisons), P4 (HPO)
Tracked values grouped: T1 (losses), T2 (optimisation state), T3 (validation), T4 (HPO trials)
The experiment-to-module mapping table shows exactly which modules each experiment needs
Colour palette: Exeter (Deep Green #003C3C, Teal #007D69, Mint #00C896) + Blue Heart (Navy #0D2B45, Ocean #1B5E8A, Sky #4FA3D1). Arial. 300 DPI.

## Subagent Orchestration

Use specialised subagents for tasks matching their domain. Prefer parallel launches when tasks are independent.

| Agent | Use for |
|-------|---------|
| `ai-engineer` | Architecture decisions, JAX optimisation, model design, training strategy |
| `ml-engineer` | JAX/Flax training pipelines, loss functions, Optuna HPO, model optimisation |
| `debugger` | JAX tracing errors, NaN propagation, shape mismatches, JIT issues |
| `code-reviewer` | Code quality reviews, JIT safety, physics correctness, loss implementations |
| `python-pro` | Python refactoring, type safety, idiomatic patterns |
| `data-scientist` | Statistical analysis of training metrics (NSE, RMSE), experiment interpretation |
| `data-analyst` | Data exploration, visualisation, training log analysis, plotting |
| `docker-expert` | Dockerfile, NVIDIA JAX devcontainer, GPU containers, CI/CD |
| `scientific-literature-researcher` | PINN methods, SWE solvers, spectral bias, adaptive loss weighting |
| `senior-orchestrator` | Multi-step tasks spanning multiple specialist agents |

**Guidelines:**
- For multi-file refactors or new features, start with `ai-engineer` or `senior-orchestrator` to plan, then delegate to specialists.
- After writing code, use `code-reviewer` to check JIT safety and physics correctness.
- For debugging training failures (NaN losses, shape errors), use `debugger` first.
- Use `scientific-literature-researcher` when implementing new PINN techniques or loss formulations.
- Run `data-scientist` and `data-analyst` in parallel when analysing experiment results.