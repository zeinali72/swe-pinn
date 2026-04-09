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
├── src/                              # Core source code (package)
│   ├── config.py                     # YAML configuration loading
│   ├── balancing/                    # Training data balancing strategies
│   │   ├── importance_sampling.py    # Importance sampling
│   │   └── relobralo.py             # Relobralo adaptive balancing
│   ├── checkpointing/               # Model checkpoint management
│   │   ├── loader.py                # Checkpoint loading
│   │   └── saver.py                 # Checkpoint saving
│   ├── data/                        # Data handling
│   │   ├── batching.py              # Batch construction
│   │   ├── bathymetry.py            # DEM / terrain loading
│   │   ├── irregular.py             # Irregular domain support
│   │   ├── loading.py               # Data loading utilities
│   │   ├── paths.py                 # Data path resolution
│   │   └── sampling.py             # Collocation point sampling
│   ├── inference/                   # Post-training inference
│   │   ├── context.py              # Inference context setup
│   │   ├── experiment_registry.py  # Experiment-specific inference configs
│   │   ├── reporting.py            # Inference result reporting
│   │   └── runner.py               # Inference execution
│   ├── losses/                      # Loss functions
│   │   ├── boundary.py             # BC losses (inflow, no-flux, zero-gradient)
│   │   ├── composite.py            # Weighted composite loss
│   │   ├── data_loss.py            # Data-driven loss
│   │   └── pde.py                  # PDE residual loss
│   ├── metrics/                     # Evaluation metrics
│   │   ├── accuracy.py             # NSE, RMSE, MAE, Relative L2
│   │   ├── boundary.py             # BC violation metrics
│   │   ├── conservation.py         # Volume balance, continuity residual
│   │   ├── cost.py                 # Computational cost metrics
│   │   ├── data_efficiency.py      # Data efficiency ratios
│   │   ├── decomposition.py        # Spatial/temporal decomposition
│   │   ├── flood_extent.py         # Flood extent agreement
│   │   ├── negative_depth.py       # Negative depth statistics
│   │   └── peak.py                 # Peak depth/timing errors
│   ├── models/                      # Neural network architectures
│   │   ├── deeponet.py             # DeepONet architecture
│   │   ├── factory.py              # Model factory (name → class)
│   │   ├── layers.py               # Custom layers
│   │   ├── ntk.py                  # Neural Tangent Kernel computation
│   │   └── pinn.py                 # FourierPINN, MLP, DGMNetwork
│   ├── monitoring/                  # Experiment tracking
│   │   ├── console_logger.py       # Console output logger
│   │   ├── diagnostics.py          # Training diagnostics
│   │   └── wandb_tracker.py        # Weights & Biases integration
│   ├── physics/                     # SWE physics
│   │   ├── analytical.py           # Analytical dam-break solutions
│   │   └── swe.py                  # SWE flux Jacobians and source terms
│   ├── plots/                       # Visualisation
│   │   ├── comparisons.py          # Architecture/experiment comparisons
│   │   ├── hpo_plots.py            # HPO analysis plots
│   │   ├── spatial_maps.py         # 2D spatial field plots
│   │   └── time_series.py          # Time series plots
│   ├── predict/                     # Prediction utilities
│   │   └── predictor.py            # Batched prediction wrapper
│   ├── training/                    # Training loop
│   │   ├── data_loading.py         # Training data preparation
│   │   ├── epoch.py                # Single epoch logic
│   │   ├── loop.py                 # Main training loop
│   │   ├── optimizer.py            # Optimizer creation
│   │   ├── setup.py                # Trial/run setup
│   │   └── step.py                 # Single gradient step
│   └── utils/                       # Utilities
│       ├── domain.py               # Domain geometry helpers
│       ├── io.py                   # File I/O utilities
│       ├── naming.py               # Run/trial naming conventions
│       ├── plotting.py             # Shared plotting helpers
│       ├── profiling.py            # Performance profiling
│       └── ui.py                   # Terminal UI helpers
├── experiments/                     # Per-experiment training scripts
│   ├── experiment_1/               # Phase 1: analytical dam-break
│   │   ├── train.py                # Standard training
│   │   ├── train_imp_samp.py       # Importance sampling variant
│   │   ├── train_relobralo.py      # Relobralo variant
│   │   └── postprocess.py          # Post-training analysis
│   ├── experiment_2/               # Phase 1: building obstacle
│   │   └── train.py
│   ├── experiment_3/               # Phase 2: x-direction slope
│   │   └── train.py
│   ├── experiment_4/               # Phase 2: x+y slope
│   │   └── train.py
│   ├── experiment_5/               # Phase 2: synthetic complexity
│   │   └── train.py
│   ├── experiment_6/               # Phase 2: synthetic complexity stage 2
│   │   └── train.py
│   ├── experiment_7/               # Phase 3: irregular boundaries
│   │   └── train.py
│   └── experiment_8/               # Phase 3: real urban domain (Eastbourne)
│       ├── train.py
│       └── train_imp_samp.py       # Importance sampling variant
├── configs/                         # Experiment configuration YAML files
│   ├── experiment_1/               # Experiment 1 configs
│   │   ├── experiment_1.yaml
│   │   ├── experiment_1_imp_samp.yaml
│   │   ├── experiment_1_relobralo.yaml
│   │   └── best_trial_51_config.yaml
│   ├── train/                      # Final HPO-optimised configs
│   │   ├── experiment_1_dgm_final.yaml
│   │   ├── experiment_1_fourier_final.yaml
│   │   ├── experiment_1_mlp_final.yaml
│   │   ├── experiment_2_dgm_final.yaml
│   │   └── experiment_2_fourier_final.yaml
│   ├── postprocess/                # Post-processing configs
│   │   └── experiment_1_postprocess.yaml
│   ├── experiment_3.yaml
│   ├── experiment_4.yaml
│   ├── experiment_5.yaml
│   ├── experiment_6.yaml
│   ├── experiment_7.yaml
│   └── experiment_8.yaml
├── test/                            # Unit tests
│   ├── test_train.py               # Training script validation
│   ├── test_batching.py            # Batch construction tests
│   ├── test_checkpointing.py       # Checkpoint save/load tests
│   ├── test_data_paths.py          # Data path resolution tests
│   ├── test_hpo.py                 # HPO objective tests
│   ├── test_hpo_utils.py           # HPO utility tests
│   ├── test_inference.py           # Inference pipeline tests
│   ├── test_losses.py              # Loss function tests
│   ├── test_models.py              # Architecture tests
│   ├── test_physics.py             # SWE physics tests
│   └── test_setup_trial.py         # Trial setup tests
├── scripts/                         # Data processing and utility scripts
│   ├── infer.py                    # Post-training inference CLI
│   ├── render_video.py             # Solution animation renderer
│   ├── generate_training_data.py   # .npy training/validation dataset generation
│   ├── binary_to_numpy.py          # Binary → .npy conversion
│   ├── preprocess_irregular.py     # Mesh preprocessing for irregular domains
│   ├── process_gauge_csvs.py       # Gauge CSV processing
│   ├── extract_gauge_timeseries.py # Gauge time series extraction
│   ├── filter_by_time.py           # Filter .npy by time
│   ├── lidar_download.py           # LIDAR elevation data download
│   ├── benchmark_*.py              # Performance benchmarks
│   ├── profile_training.py         # Training profiler
│   ├── jobs/                       # HPC job scripts
│   │   ├── run_job.sh
│   │   └── run_job_L40.sh
│   └── cpp/                        # C++ CSV → binary converter
│       ├── preprocess.cpp
│       └── CMakeLists.txt
├── optimisation/                    # Hyperparameter optimisation (Optuna)
│   ├── run_optimization.py         # Main HPO entry point
│   ├── run_sensitivity_analysis.py # Parameter importance analysis
│   ├── extract_best_params.py      # Extract best trial parameters
│   ├── objective_function.py       # Optuna objective function
│   ├── optimization_train_loop.py  # HPO training loop
│   ├── utils.py                    # HPO utilities
│   ├── configs/                    # HPO-specific configs
│   │   ├── exploration/            # Sobol exploration phase configs
│   │   └── exploitation/           # TPE exploitation phase configs
│   ├── database/                   # Optuna study databases (SQLite)
│   ├── results/                    # HPO results and best trial configs
│   ├── sensitivity_analysis_output/ # Importance reports
│   └── legacy/                     # Archived HPO configs and databases
├── docs/                            # Documentation
│   └── experimental_programme_reference.md  # Authoritative experiment spec
├── notebook/                        # Jupyter notebooks for analysis
├── data/                            # Reference simulation data (gitignored)
├── models/                          # Trained model checkpoints (gitignored)
├── results/                         # Experiment outputs (gitignored)
├── .devcontainer/                   # Docker dev container setup
│   ├── Dockerfile
│   ├── devcontainer.json
│   ├── requirements.txt
│   ├── install_dependencies.sh
│   └── install_requirements.sh
├── .github/workflows/               # CI/CD: Docker image build/publish
│   └── docker-publish.yml
└── pyproject.toml                   # Package metadata and dependencies
```

## Tech Stack

- **ML Framework**: JAX + Flax (neural networks) + Optax (optimizers)
- **Language**: Python 3.8+
- **GPU**: CUDA via NVIDIA JAX Docker image (`nvcr.io/nvidia/jax:25.01-py3`)
- **Hyperparameter Optimization**: Optuna
- **Experiment Tracking**: Weights & Biases (W&B)
- **Visualization**: Matplotlib, Seaborn
- **Configuration**: YAML files loaded via `src/config.py`
- **Build System**: setuptools (via `pyproject.toml`)

## Key Commands

### Running Training

```bash
# Per-experiment training scripts (run as modules)
python -m experiments.experiment_1.train --config configs/experiment_1/experiment_1.yaml
python -m experiments.experiment_2.train --config configs/train/experiment_2_fourier_final.yaml
python -m experiments.experiment_3.train --config configs/experiment_3.yaml
# ... same pattern for experiments 4–8

# Experiment 1 variants
python -m experiments.experiment_1.train_imp_samp --config configs/experiment_1/experiment_1_imp_samp.yaml
python -m experiments.experiment_1.train_relobralo --config configs/experiment_1/experiment_1_relobralo.yaml

# Experiment 8 importance sampling
python -m experiments.experiment_8.train_imp_samp --config <config>
```

### Running Tests

```bash
# Run all tests
python -m unittest discover test

# Run specific test files
python -m unittest test.test_train
python -m unittest test.test_losses
python -m unittest test.test_models
```

Tests force JAX to CPU (`JAX_PLATFORM_NAME=cpu`) for reproducibility, use mock data and file I/O, and clean up temporary directories in teardown.

### Hyperparameter Optimization

```bash
python optimisation/run_optimization.py --config <hpo_config> --n_trials 100
python optimisation/run_sensitivity_analysis.py
python optimisation/extract_best_params.py
```

### Inference

```bash
python scripts/infer.py \
  --config configs/experiment_3.yaml \
  --checkpoint models/experiment_3/<trial>/checkpoints/best_nse \
  --checkpoints best_nse
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

### Neural Network Architectures (`src/models/pinn.py`)

1. **FourierPINN** - Fourier feature encoding + dense layers with tanh activation
2. **MLP** - Standard multi-layer perceptron
3. **DGMNetwork** - Deep Galerkin Method with LSTM-like gates

### Loss Weighting Strategies

- **Static**: Fixed weights from config
- **Importance Sampling** (`src/balancing/importance_sampling.py`): Adaptive point weighting based on loss magnitude
- **Relobralo** (`src/balancing/relobralo.py`): Relative loss balancing with random lookback

### Physics Implementation

- `src/physics/swe.py`: SWE flux Jacobians and source terms for the 2D SWE
- `src/losses/`: PDE residual losses, initial condition losses, boundary condition losses (inflow, zero-gradient, no-flux)
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

## Experimental Programme

This project runs experiments across 3 phases. The authoritative specification for all metrics, plots, tracked values, and module structure lives in:
`docs/experimental_programme_reference.md`

Always consult this file when:
- Implementing any evaluation metric (definitions, formulas, units, which experiments use it)
- Creating any inference plot (plot type, axes, colour conventions, which experiments need it)
- Setting up W&B tracking for a training run (what to log, at what frequency)
- Building or modifying any module in `src/metrics/` or `src/plots/`
- Planning what outputs an experiment run should produce

### Key conventions from that document

- All accuracy metrics (NSE, RMSE, MAE, Rel L2) reported separately for h, hu, hv
- Metrics are grouped: A (accuracy), B (conservation), C (boundary), D (cost), E (HPO), F (data)
- Plots are grouped: P1 (time series), P2 (spatial maps), P3 (comparisons), P4 (HPO)
- Tracked values grouped: T1 (losses), T2 (optimisation state), T3 (validation), T4 (HPO trials)
- The experiment-to-module mapping table shows exactly which modules each experiment needs
- Colour palette: Exeter (Deep Green #003C3C, Teal #007D69, Mint #00C896) + Blue Heart (Navy #0D2B45, Ocean #1B5E8A, Sky #4FA3D1). Arial. 300 DPI.
