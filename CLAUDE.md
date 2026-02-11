# CLAUDE.md

This file provides guidance for AI assistants working with the SWE-PINN codebase.

## Project Overview

SWE-PINN implements **Physics-Informed Neural Networks (PINNs)** for solving **2D Shallow Water Equations (SWE)** using JAX and Flax. It models water flow in channels, rivers, and coastal areas, optionally with building obstacles. The project is research-grade, configuration-driven, and designed for GPU-accelerated training.

## Repository Structure

```
swe-pinn/
├── src/                        # Core source code
│   ├── train.py                # Unified training script (main entry point)
│   ├── models.py               # Neural network architectures (FourierPINN, MLP, DGMNetwork, DeepONet)
│   ├── losses.py               # PDE, IC, BC loss functions for SWE
│   ├── physics.py              # SWE physics computations and Jacobians
│   ├── gradnorm.py             # GradNorm adaptive loss weighting
│   ├── ntk.py                  # Neural Tangent Kernel trace computation
│   ├── data.py                 # Data sampling and batching utilities
│   ├── config.py               # YAML configuration loading
│   ├── utils.py                # Metrics (NSE, RMSE), plotting, model saving
│   ├── reporting.py            # Training stats logging and Aim integration
│   ├── scenarios/              # Scenario-specific training scripts
│   │   ├── analytical/         # Pure analytical scenarios (no buildings)
│   │   │   ├── analytical.py   # Standard analytical training
│   │   │   ├── analytical_ntk.py   # NTK-based weight adaptation
│   │   │   └── train_deeponet.py   # DeepONet operator learning
│   │   └── building/           # Scenarios with building obstacles
│   │       └── building.py     # Training with spatial masking
│   └── operator_learning/      # DeepONet operator learning modules
│       ├── physics_op.py       # SWE physics for parameter-varying solutions
│       └── losses_op.py        # DeepONet-specific loss functions
├── configs/                    # Experiment configuration YAML files
│   ├── fourier_pinn_config.yaml
│   ├── dgm_datafree_static.yaml
│   ├── dgm_datafree_gradnorm.yaml
│   ├── analytical_ntk_config.yaml
│   └── config_operatornet_analytical.yaml
├── test/                       # Unit tests
│   ├── test_train.py           # Main training script validation
│   ├── test_train_gradnorm.py  # GradNorm mode tests (6 test cases)
│   └── test_assets/
│       └── test_config.yaml    # Minimal test configuration
├── scripts/                    # Data processing and utility scripts
│   ├── create_samples.py       # Generate training samples
│   ├── filter_sample.py        # Filter/preprocess samples
│   ├── convert_bin_to_npy.py   # Binary-to-numpy conversion
│   └── cpp/                    # C++ utilities for data generation
├── optimisation/               # Hyperparameter optimization (Optuna)
│   ├── run_optimization.py     # Main HPO entry point
│   ├── run_sensitivity_analysis.py
│   ├── extract_best_params.py
│   ├── objective_function.py   # Optuna objective
│   ├── optimization_train_loop.py
│   └── configs/                # HPO-specific config files
├── notebook/                   # Jupyter notebooks for analysis
├── .devcontainer/              # Docker dev container setup (NVIDIA JAX + CUDA)
├── .github/workflows/          # CI/CD: Docker image build/publish to GHCR
├── pyproject.toml              # Package metadata and dependencies
└── README.md                   # Project documentation
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
python -m src.scenarios.analytical.analytical <config_path>
python -m src.scenarios.analytical.analytical_ntk <config_path>
python -m src.scenarios.building.building <config_path>
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
| `model` | `name` (FourierPINN/MLP/DGMNetwork/DeepONet), `width`, `depth`, `output_dim` |
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
4. **DeepONet** - Operator learning (branch + trunk networks)

### Loss Weighting Strategies

- **Static**: Fixed weights from config
- **GradNorm** (`src/gradnorm.py`): Adaptive weights that balance gradient magnitudes across loss terms
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
