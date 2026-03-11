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
- **Experiment 1**: Verifies the framework against an analytical dam-break solution on a flat domain using the MLP.
- **Experiment 2**: Introduces a building obstacle; the vanilla MLP fails, motivating Fourier-MLP and DGM adoption. A two-stage HPO (100 Sobol + 50 TPE Bayesian trials) is run across all three architectures on both scenarios to produce optimised configurations used throughout.

**Phase 2 — Topographic Complexity on Synthetic Domains**
- **Experiment 3**: Introduces terrain slope in the x-direction via bi-linear interpolation fed into the SWE momentum equations; establishes a data sampling ratio methodology when physics-only training is insufficient.
- **Experiment 4**: Extends slope to both x and y directions.
- **Experiments 5 & 6**: Apply the framework to further synthetic scenarios of increasing complexity to validate robustness before transitioning to real-world domains.

**Phase 3 — Domain Generalisation and Real-World Application**
- **Experiment 7**: Tackles irregular (non-rectangular) boundaries using triangulated mesh-based sampling, automated boundary detection, and computed wall normals for slip boundary conditions.
- **Experiment 8**: Applies the complete framework to a real urban subcatchment in Eastbourne, UK (Blue Heart Project). Buildings are excluded from the mesh by construction rather than masked post-hoc, and are treated as wall boundaries identical to the irregular domain walls from Experiment 7.

## Repository Structure

```
swe-pinn/
├── src/                            # Core source code
│   ├── train.py                    # Unified training script (main entry point)
│   ├── models.py                   # Neural network architectures (FourierPINN, MLP, DGMNetwork)
│   ├── losses.py                   # PDE, IC, BC loss functions for SWE
│   ├── physics.py                  # SWE physics computations and Jacobians
│   ├── gradnorm.py                 # GradNorm adaptive loss weighting
│   ├── softadapt.py                # SoftAdapt adaptive loss weighting
│   ├── ntk.py                      # Neural Tangent Kernel trace computation
│   ├── data.py                     # Data sampling and batching utilities
│   ├── config.py                   # YAML configuration loading
│   ├── utils.py                    # Metrics (NSE, RMSE), plotting, model saving
│   ├── reporting.py                # Training stats logging and Aim integration
│   └── scenarios/                  # Per-experiment training scripts
│       ├── experiment_1/           # Exp 1: analytical dam-break, flat domain
│       │   ├── experiment_1.py     # Main training script
│       │   ├── analytical_ntk.py   # NTK-based weight adaptation variant
│       │   └── experiment_1_lbfgs_finetune.py  # L-BFGS fine-tuning variant
│       ├── experiment_2/           # Exp 2: building obstacle (motivates Fourier/DGM)
│       │   └── experiment_2.py
│       ├── experiment_3/           # Exp 3: x-direction terrain slope + data sampling ratio
│       │   ├── experiment_3.py
│       │   └── pix2pix_experiment_3.py
│       ├── experiment_4/           # Exp 4: x+y terrain slope
│       │   └── experiment_4.py
│       ├── experiment_5/           # Exp 5: synthetic complexity stage 1
│       │   └── experiment_5.py
│       ├── experiment_6/           # Exp 6: synthetic complexity stage 2
│       │   └── experiment_6.py
│       ├── experiment_7/           # Exp 7: irregular boundaries, mesh-based sampling
│       │   └── experiment_7.py
│       └── experiment_8/           # Exp 8: real urban domain (Eastbourne / Blue Heart)
│           ├── experiment_8.py
│           └── experiment_8_imp_samp.py    # Importance sampling variant
├── configs/                        # Experiment configuration YAML files
│   ├── experiment_1_fourier.yaml
│   ├── experiment_1_ntk_config.yaml
│   ├── dgm_datafree_static_experiment_1.yaml
│   ├── dgm_datafree_gradnorm_experiment_1.yaml
│   ├── experiment_3.yaml
│   ├── experiment_4.yaml
│   ├── experiment_5.yaml
│   ├── experiment_6.yaml
│   ├── experiment_7.yaml
│   ├── experiment_8.yaml
│   ├── pix2pix_experiment_3.yaml
│   └── train/                      # Per-architecture training configs
│       ├── mlp_experiment_1.yaml
│       ├── fourier_experiment_1.yaml
│       ├── fourier_experiment_2.yaml
│       ├── DGM_no_experiment_1.yaml
│       └── DGM_experiment_2.yaml
├── test/                           # Unit tests
│   ├── test_train.py               # Main training script validation
│   ├── test_train_gradnorm.py      # GradNorm mode tests (6 test cases)
│   └── test_assets/
│       └── test_config.yaml        # Minimal test configuration
├── scripts/                        # Data processing and utility scripts
│   ├── create_samples.py           # Generate training samples from simulation output
│   ├── filter_sample.py            # Filter/preprocess samples
│   ├── convert_bin_to_npy.py       # Binary-to-numpy conversion
│   ├── preprocess_irregular.py     # Mesh preprocessing for irregular domains (Exp 7/8)
│   ├── extract_gauges.py           # Extract gauge time-series from simulation output
│   ├── process_gauges_split.py     # Split and align gauge data
│   ├── process_test2_gauges.py     # Gauge processing for test 2
│   ├── render_video.py             # Render solution animations
│   ├── run_preprocess.sh           # Shell driver for preprocessing pipeline
│   ├── jobs/                       # HPC job submission scripts
│   │   ├── run_job.sh
│   │   ├── run_job_L40.sh
│   │   ├── run_all_hpo.sh
│   │   └── run_jobs_temp.sh
│   ├── utils/
│   │   └── gpu_debug.py
│   └── cpp/                        # C++ utilities for data generation
│       └── preprocess.cpp
├── optimisation/                   # Hyperparameter optimisation (Optuna)
│   ├── run_optimization.py         # Main HPO entry point
│   ├── run_sensitivity_analysis.py # Exploration phase (Sobol sampler)
│   ├── extract_best_params.py      # Extract best trial configs
│   ├── analyze_importance.py       # Parameter importance analysis
│   ├── objective_function.py       # Optuna objective
│   ├── optimization_train_loop.py  # Training loop used by HPO
│   ├── run_all_exploitations.sh    # Shell driver for all exploitation runs
│   ├── configs/
│   │   ├── exploration/            # Sobol exploration configs (per arch × scenario)
│   │   └── exploitation/           # TPE exploitation configs (per arch × scenario)
│   ├── database/
│   │   ├── exploration/            # Optuna SQLite DBs for exploration runs
│   │   └── exploitation/           # Optuna SQLite DBs for exploitation runs
│   ├── logs/
│   │   ├── exploitation/           # Logs for exploitation runs
│   │   └── ...
│   ├── results/                    # Best trial YAML configs output by HPO
│   └── sensivity_analysis_output/  # Importance HTML/text reports + best params
├── data/                           # Reference simulation data (InfoWorks ICM output)
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

# Example with a specific config
python src/train.py configs/fourier_pinn_config.yaml

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
python -m unittest test.test_train_gradnorm
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
# Launch Aim dashboard to view experiment logs
aim up
```

### Installing Dependencies

```bash
# Via dev container (recommended) or manually:
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

The model class is dynamically imported based on the `model.name` field in the config.

### Neural Network Architectures (`src/models.py`)

1. **FourierPINN** - Fourier feature encoding + dense layers with tanh activation
2. **MLP** - Standard multi-layer perceptron
3. **DGMNetwork** - Deep Galerkin Method with LSTM-like gates

### Loss Weighting Strategies

- **Static**: Fixed weights from config
- **GradNorm** (`src/gradnorm.py`): Adaptive weights that balance gradient magnitudes across loss terms
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
- Test both data-free and data-driven modes, static and GradNorm weight strategies
- **Manual training smoke-tests**: When verifying a training code change, run with **200 epochs** — this is sufficient to confirm convergence with the existing config files. Do not use the full epoch count (2,000–50,000) for validation of code changes.

## CI/CD

- **GitHub Actions** (`.github/workflows/docker-publish.yml`): Builds and publishes the Docker dev container image to GitHub Container Registry (`ghcr.io/zeinali72/swe-pinn:latest`)
- Triggered manually via `workflow_dispatch`
- Builds for `linux/amd64` platform with GitHub Actions caching

## Common Pitfalls

- **JAX requires explicit device management**: Use `jax.devices()` and config `dtype` settings
- **JIT compilation**: First call is slow due to tracing; subsequent calls are fast. Avoid Python side effects inside JIT-compiled functions.
- **Memory**: Large validation datasets (multi-GB) use memory-mapped numpy arrays. Use `np.load(..., mmap_mode='r')` for big files.
- **Float precision**: Some physics computations require `float64`. Set via `config.device.dtype`.
- **GradNorm**: Requires separate optimizer state for loss weights. See `src/gradnorm.py` for initialization patterns.
