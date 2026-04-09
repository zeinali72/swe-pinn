# SWE-PINN: Physics-Informed Neural Networks for 2D Shallow Water Equations

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-0.4.13-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.7.2-lightgrey.svg)](https://github.com/google/flax)

SWE-PINN is a Physics-Informed Neural Network framework for urban flood modeling and surrogate simulation. The model takes spatiotemporal coordinates $(x,y,t)$ as input and predicts free-surface state variables:

$$
\mathbf{U}(x,y,t)=\begin{bmatrix}h \\ hu \\ hv\end{bmatrix}
$$

where $h$ is water depth and $hu, hv$ are specific discharges.

Training combines physics residuals from the 2D Shallow Water Equations (SWE), initial and boundary conditions, and optional data loss from numerical simulations.

## Table of Contents

- [Overview](#overview)
- [Physics Formulation](#physics-formulation)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Entry Points](#training-entry-points)
- [Inference](#inference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Hyperparameter Optimization](#hyperparameter-optimization)

## Overview

### Architectures

The codebase supports multiple neural architectures:

- **MLP**: baseline fully-connected PINN.
- **FourierPINN**: Fourier feature encoding for improved high-frequency representation.
- **DGMNetwork**: Deep Galerkin-style architecture.
- **DeepONet**: operator-learning variant for selected workflows.

### Loss Composition

The total objective is a weighted sum:

$$
\mathcal{L}=\lambda_{\mathrm{pde}}\mathcal{L}_{\mathrm{pde}}+\lambda_{\mathrm{ic}}\mathcal{L}_{\mathrm{ic}}+\lambda_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}+\lambda_{\mathrm{data}}\mathcal{L}_{\mathrm{data}}+\lambda_{\mathrm{neg}}\mathcal{L}_{\mathrm{neg}}
$$

with $\mathcal{L}_{\mathrm{data}}$ enabled when training data are present.

## Physics Formulation

The 2D SWE in conservative form are:

$$
\frac{\partial \mathbf{U}}{\partial t}+\frac{\partial \mathbf{F}(\mathbf{U})}{\partial x}+\frac{\partial \mathbf{G}(\mathbf{U})}{\partial y}=\mathbf{S}(\mathbf{U})
$$

with

$$
\mathbf{U}=\begin{bmatrix}h \\ hu \\ hv\end{bmatrix},\quad
\mathbf{F}=\begin{bmatrix}
hu \\
\dfrac{(hu)^2}{h}+\dfrac{1}{2}gh^2 \\
\dfrac{(hu)(hv)}{h}
\end{bmatrix},\quad
\mathbf{G}=\begin{bmatrix}
hv \\
\dfrac{(hu)(hv)}{h} \\
\dfrac{(hv)^2}{h}+\dfrac{1}{2}gh^2
\end{bmatrix}
$$

The source term includes rainfall/inflow, bed slope, and friction effects:

$$
\mathbf{S}=\begin{bmatrix}
R \\
-g h (S_{0x}+S_{fx}) \\
-g h (S_{0y}+S_{fy})
\end{bmatrix}
$$

where $g$ is gravity, $R$ is external source (if provided), and Manning-type friction is used for $S_{fx}, S_{fy}$.

## Repository Structure

```text
swe-pinn/
├── src/                              # Core source package
│   ├── config.py                     # YAML configuration loading
│   ├── balancing/                    # Importance sampling, Relobralo
│   ├── checkpointing/               # Model checkpoint save/load
│   ├── data/                        # Batching, sampling, loading, bathymetry
│   ├── inference/                   # Post-training inference pipeline
│   ├── losses/                      # PDE, BC, IC, data, composite losses
│   ├── metrics/                     # NSE, RMSE, conservation, flood extent, etc.
│   ├── models/                      # FourierPINN, MLP, DGM, DeepONet, NTK
│   ├── monitoring/                  # W&B tracker, console logger, diagnostics
│   ├── physics/                     # SWE equations, analytical solutions
│   ├── plots/                       # Time series, spatial maps, comparisons
│   ├── predict/                     # Batched prediction wrapper
│   ├── training/                    # Training loop, epoch, step, optimizer
│   └── utils/                       # Domain, I/O, naming, plotting, profiling
├── experiments/                      # Per-experiment training scripts
│   ├── experiment_1/                # Phase 1: analytical dam-break
│   │   ├── train.py
│   │   ├── train_imp_samp.py        # Importance sampling variant
│   │   ├── train_relobralo.py       # Relobralo variant
│   │   └── postprocess.py
│   ├── experiment_2/                # Phase 1: building obstacle
│   ├── experiment_3/ – experiment_6/ # Phase 2: topographic complexity
│   ├── experiment_7/                # Phase 3: irregular boundaries
│   └── experiment_8/                # Phase 3: real urban domain (Eastbourne)
│       ├── train.py
│       └── train_imp_samp.py
├── configs/                          # Experiment YAML configurations
│   ├── experiment_1/                # Experiment 1 configs
│   ├── train/                       # Final HPO-optimised configs
│   └── postprocess/                 # Post-processing configs
├── optimisation/                     # Hyperparameter optimisation (Optuna)
│   ├── run_optimization.py
│   ├── objective_function.py
│   ├── configs/                     # Exploration + exploitation configs
│   ├── database/                    # Optuna study databases
│   └── results/                     # Best trial configs
├── scripts/                          # Data processing, inference, benchmarks
│   ├── infer.py                     # Inference CLI wrapper
│   ├── render_video.py              # Solution animation renderer
│   ├── generate_training_data.py    # Training data generation
│   ├── benchmark_*.py               # Performance benchmarks
│   ├── jobs/                        # HPC job scripts
│   └── cpp/                         # C++ CSV → binary converter
├── test/                             # Unit tests (unittest)
├── docs/                             # Documentation
├── notebook/                         # Jupyter notebooks
├── .devcontainer/                    # Docker dev container (NVIDIA JAX + CUDA)
├── .github/workflows/                # CI/CD: Docker image build/publish
└── pyproject.toml                    # Package metadata and dependencies
```

Note: `data/`, `models/`, and `results/` contain large/generated artifacts and are gitignored.

## Installation

### Option 1: Editable Install

```bash
git clone https://github.com/zeinali72/swe-pinn.git
cd swe-pinn
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Option 2: Dev Container

Open the repository in VS Code and reopen in the provided dev container.

## Quick Start

Example: train Experiment 1 using the default configuration.

```bash
python -m experiments.experiment_1.train --config configs/experiment_1/experiment_1.yaml
```

Example: train Experiment 3.

```bash
python -m experiments.experiment_3.train --config configs/experiment_3.yaml
```

## Training Entry Points

Each experiment has a module entry point:

```bash
python -m experiments.experiment_1.train --config <config>
python -m experiments.experiment_2.train --config <config>
python -m experiments.experiment_3.train --config <config>
python -m experiments.experiment_4.train --config <config>
python -m experiments.experiment_5.train --config <config>
python -m experiments.experiment_6.train --config <config>
python -m experiments.experiment_7.train --config <config>
python -m experiments.experiment_8.train --config <config>

# Variants
python -m experiments.experiment_1.train_imp_samp --config <config>
python -m experiments.experiment_1.train_relobralo --config <config>
python -m experiments.experiment_8.train_imp_samp --config <config>
```

## Inference

Use the inference wrapper script:

```bash
python scripts/infer.py \
  --config configs/experiment_3.yaml \
  --checkpoint models/experiment_3/<trial>/checkpoints/best_nse \
  --checkpoints best_nse
```

To evaluate all standard checkpoints in a trial:

```bash
python scripts/infer.py \
  --config configs/experiment_3.yaml \
  --checkpoint models/experiment_3/<trial>/checkpoints \
  --checkpoints all
```

## Configuration

Main YAML blocks:

- `training`: optimizer, epochs, batch size, seed, clipping.
- `model`: architecture type and dimensions.
- `domain`: spatial and temporal bounds.
- `grid` and `sampling`: collocation and boundary sampling sizes.
- `physics`: $g$, Manning coefficient, inflow/source settings.
- `loss_weights`: PDE/IC/BC/data/neg-depth balancing.
- `device` and `numerics`: precision and numerical constants.

## Testing

Run the full test suite:

```bash
python -m unittest discover test
```

Run a single test file:

```bash
python -m unittest test.test_train
```

## Hyperparameter Optimization

Run HPO with Optuna:

```bash
python optimisation/run_optimization.py --config optimisation/configs/<file>.yaml --n_trials 100
```

Run sensitivity analysis and extract best parameters:

```bash
python optimisation/run_sensitivity_analysis.py
python optimisation/extract_best_params.py
```

## License

This project is licensed under the MIT License.
