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

Current structure on this branch:

```text
swe-pinn/
├── CLAUDE.md
├── README.md
├── WORKSPACE_STRUCTURE.md
├── analysis.md
├── pyproject.toml
├── configs/
│   ├── experiment_1.yaml
│   ├── experiment_1_dgm_static.yaml
│   ├── experiment_1_fourier.yaml
│   ├── experiment_3.yaml
│   ├── experiment_4.yaml
│   ├── experiment_5.yaml
│   ├── experiment_6.yaml
│   ├── experiment_7.yaml
│   ├── experiment_8.yaml
│   └── train/
│       ├── experiment_1_dgm_final.yaml
│       ├── experiment_1_fourier_final.yaml
│       ├── experiment_1_mlp_final.yaml
│       ├── experiment_2_dgm_final.yaml
│       └── experiment_2_fourier_final.yaml
├── experiments/
│   ├── experiment_1/train.py
│   ├── experiment_2/train.py
│   ├── experiment_3/train.py
│   ├── experiment_4/train.py
│   ├── experiment_5/train.py
│   ├── experiment_6/train.py
│   ├── experiment_7/train.py
│   └── experiment_8/
│       ├── train.py
│       └── train_imp_samp.py
├── optimisation/
│   ├── objective_function.py
│   ├── optimization_train_loop.py
│   ├── run_optimization.py
│   ├── run_sensitivity_analysis.py
│   ├── extract_best_params.py
│   └── utils.py
├── scripts/
│   ├── infer.py
│   ├── render_video.py
│   ├── generate_training_data.py
│   ├── preprocess_irregular.py
│   ├── binary_to_numpy.py
│   ├── process_gauge_csvs.py
│   ├── filter_by_time.py
│   ├── extract_gauge_timeseries.py
│   ├── lidar_download.py
│   ├── benchmark_*.py
│   ├── jobs/
│   └── cpp/
├── src/
│   ├── config.py
│   ├── checkpointing/
│   │   ├── loader.py
│   │   └── saver.py
│   ├── data/
│   │   ├── batching.py
│   │   ├── bathymetry.py
│   │   ├── irregular.py
│   │   ├── loading.py
│   │   ├── paths.py
│   │   └── sampling.py
│   ├── inference/
│   │   ├── context.py
│   │   ├── experiment_registry.py
│   │   ├── reporting.py
│   │   └── runner.py
│   ├── losses/
│   │   ├── boundary.py
│   │   ├── composite.py
│   │   ├── data_loss.py
│   │   └── pde.py
│   ├── metrics/
│   │   ├── accuracy.py
│   │   ├── boundary.py
│   │   ├── conservation.py
│   │   ├── decomposition.py
│   │   ├── flood_extent.py
│   │   ├── negative_depth.py
│   │   └── peak.py
│   ├── models/
│   │   ├── deeponet.py
│   │   ├── factory.py
│   │   ├── layers.py
│   │   ├── ntk.py
│   │   └── pinn.py
│   ├── monitoring/
│   │   ├── aim_tracker.py
│   │   ├── console_logger.py
│   │   └── diagnostics.py
│   ├── physics/
│   │   ├── analytical.py
│   │   └── swe.py
│   ├── predict/
│   │   └── predictor.py
│   ├── training/
│   │   ├── data_loading.py
│   │   ├── epoch.py
│   │   ├── loop.py
│   │   ├── optimizer.py
│   │   ├── setup.py
│   │   └── step.py
│   └── utils/
│       ├── domain.py
│       ├── io.py
│       ├── naming.py
│       ├── plotting.py
│       ├── profiling.py
│       └── ui.py
├── test/
│   ├── test_batching.py
│   ├── test_checkpointing.py
│   ├── test_data_paths.py
│   ├── test_hpo.py
│   ├── test_hpo_utils.py
│   ├── test_inference.py
│   ├── test_losses.py
│   ├── test_models.py
│   ├── test_physics.py
│   └── test_train.py
└── data/, models/, results/, aim_repo/, notebook/, notes/
```

Note: `data/`, `models/`, and `results/` contain large/generated artifacts and are intentionally not expanded here.

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

Example: train Experiment 1 using the Fourier configuration.

```bash
python -m experiments.experiment_1.train --config configs/experiment_1_fourier.yaml
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
