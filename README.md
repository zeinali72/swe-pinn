# SWE-PINN: Physics-Informed Neural Networks for Shallow Water Equations

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-0.4.13-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.7.2-lightgrey.svg)](https://github.com/google/flax)

A high-performance implementation of Physics-Informed Neural Networks (PINNs) for solving the 2D Shallow Water Equations using JAX and Flax. This project leverages Fourier feature mapping, advanced optimization techniques, and comprehensive experiment tracking to deliver accurate and efficient PDE solutions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Research Phases](#research-phases)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Physics Background](#physics-background)
- [Results and Validation](#results-and-validation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

The Shallow Water Equations (SWE) describe the flow of water in channels, rivers, and coastal areas. Traditional numerical methods for solving these PDEs can be computationally expensive and require careful mesh design. Physics-Informed Neural Networks offer an alternative approach by embedding the physical laws directly into the neural network training process.

This implementation provides:
- **Accurate PDE Solutions**: High-fidelity solutions to 2D SWE with complex boundary conditions
- **Efficient Training**: JAX-based automatic differentiation and GPU acceleration
- **Advanced Features**: Fourier features, hyperparameter optimization, and experiment tracking
- **Modular Design**: Clean architecture for easy extension and customization

## Key Features

- **Physics-Informed Training**: Incorporates SWE physics directly into the loss function
- **Complete SWE Implementation**: Handles 2D shallow water flow with source terms
- **Fourier Feature Mapping**: Enhanced capability for learning high-frequency solutions
- **High Performance**: JAX/Flax backend with GPU acceleration
- **Experiment Tracking**: Integrated Aim logging and visualization
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Comprehensive Metrics**: NSE, RMSE, and physics-based validation
- **Modular Architecture**: Clean separation of physics, models, and training components

## Research Phases

The project is organized into three research phases spanning eight experiments:

### Phase 1 — Baseline Verification and Architecture Selection
- **Experiment 1**: Verifies the framework against an analytical dam-break solution on a flat domain.
- **Experiment 2**: Introduces a building obstacle; motivates Fourier-MLP and DGM adoption. Two-stage HPO (100 Sobol + 50 TPE Bayesian trials) is run across all three architectures on both scenarios.

### Phase 2 — Topographic Complexity on Synthetic Domains
- **Experiment 3**: Introduces terrain slope in the x-direction via bi-linear interpolation; establishes a data sampling ratio methodology.
- **Experiment 4**: Extends slope to both x and y directions.
- **Experiment 5**: Further synthetic complexity to validate robustness.

### Phase 3 — Domain Generalisation and Real-World Application
- **Experiment 7**: Irregular (non-rectangular) boundaries using triangulated mesh-based sampling, automated boundary detection, and wall normals for slip boundary conditions.
- **Experiment 8**: Real urban subcatchment in Eastbourne, UK (Blue Heart Project). Buildings are excluded from the mesh and treated as wall boundaries.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Docker (for Dev Container support)

### Dev Container Setup (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zeinali72/swe-pinn.git
   cd swe-pinn
   ```

2. **Open in VS Code:**
   - Launch VS Code and open the project folder
   - When prompted, click "Reopen in Container"
   - The Dev Container will automatically install all dependencies

### Manual Installation

If you prefer manual setup:

1. **Clone and navigate:**
   ```bash
   git clone https://github.com/zeinali72/swe-pinn.git
   cd swe-pinn
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv swe_pinn_env
   source swe_pinn_env/bin/activate  # On Windows: swe_pinn_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   # Or manually install from requirements
   pip install jax flax optax optuna aim numpy pyyaml matplotlib seaborn
   ```

4. **Verify installation:**
   ```bash
   python -c "import jax; print('JAX version:', jax.__version__)"
   ```

## Quick Start

1. **Configure your experiment:**
   ```bash
   cp configs/experiment_1_fourier.yaml configs/my_experiment.yaml
   # Edit my_experiment.yaml with your parameters
   ```

2. **Run training:**
   ```bash
   python -m src.scenarios.experiment_1.experiment_1 configs/my_experiment.yaml
   ```

3. **Monitor results:**
   ```bash
   aim up
   # Open http://localhost:43800 in your browser
   ```

4. **Visualize results:**
   ```bash
   python scripts/render_video.py --model_dir models/your_model --config configs/my_experiment.yaml --output results/video.mp4
   ```

## Project Structure

```
swe-pinn/
├── src/                            # Core source code
│   ├── config.py                   # YAML configuration loading
│   ├── data.py                     # Data sampling, batching, and validation loading
│   ├── losses.py                   # PDE, IC, BC loss functions for SWE
│   ├── models.py                   # Neural network architectures (FourierPINN, MLP, DGMNetwork)
│   ├── physics.py                  # SWE physics computations and Jacobians
│   ├── gradnorm.py                 # GradNorm adaptive loss weighting
│   ├── softadapt.py                # SoftAdapt adaptive loss weighting
│   ├── ntk.py                      # Neural Tangent Kernel trace computation
│   ├── train.py                    # Unified training script (main entry point)
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
│       ├── experiment_7/           # Phase 3: irregular boundaries
│       │   └── experiment_7.py
│       └── experiment_8/           # Phase 3: real urban domain (Eastbourne)
│           ├── experiment_8.py
│           └── experiment_8_imp_samp.py
├── configs/                        # Experiment configuration YAML files
│   ├── experiment_1_fourier.yaml       # Exp 1 Fourier config
│   ├── experiment_1_ntk.yaml           # Exp 1 NTK config
│   ├── experiment_1_dgm_static.yaml    # Exp 1 DGM static config
│   ├── experiment_1_dgm_gradnorm.yaml  # Exp 1 DGM GradNorm config
│   ├── experiment_3.yaml               # Exp 3 config
│   ├── experiment_4.yaml               # Exp 4 config
│   ├── experiment_5.yaml               # Exp 5 config
│   ├── experiment_6.yaml               # Exp 6 config
│   ├── experiment_7.yaml               # Exp 7 config
│   ├── experiment_8.yaml               # Exp 8 config
│   └── train/                          # Final HPO-optimized production configs
│       ├── experiment_1_mlp_final.yaml
│       ├── experiment_1_fourier_final.yaml
│       ├── experiment_1_dgm_final.yaml
│       ├── experiment_2_fourier_final.yaml
│       └── experiment_2_dgm_final.yaml
├── test/                           # Unit tests
│   ├── test_train.py
│   ├── test_train_gradnorm.py
│   └── test_assets/
│       └── test_config.yaml
├── scripts/                        # Data processing and utility scripts
│   ├── create_samples.py           # Generate training samples from simulation output
│   ├── filter_sample.py            # Filter/preprocess samples
│   ├── convert_bin_to_npy.py       # Binary-to-numpy conversion
│   ├── preprocess_irregular.py     # Mesh preprocessing for irregular domains (Exp 7/8)
│   ├── extract_gauges.py           # Extract gauge time-series
│   ├── process_gauges_split.py     # Split and align gauge data
│   ├── render_video.py             # Render solution animations (CLI-driven)
│   ├── run_preprocess.sh           # Shell driver for preprocessing pipeline
│   └── cpp/                        # C++ utilities for data generation
│       └── preprocess.cpp
├── optimisation/                   # Hyperparameter optimisation (Optuna)
│   ├── run_optimization.py         # Main HPO entry point
│   ├── run_sensitivity_analysis.py
│   ├── extract_best_params.py
│   ├── objective_function.py       # Optuna objective
│   ├── optimization_train_loop.py
│   ├── configs/
│   │   ├── exploration/            # Sobol exploration configs (per arch x scenario)
│   │   └── exploitation/           # TPE exploitation configs (per arch x scenario)
│   └── sensitivity_analysis_output/  # Importance reports + best params
├── data/                           # Reference simulation data (InfoWorks ICM output)
├── notebook/                       # Jupyter notebooks for analysis
├── .devcontainer/                  # Docker dev container setup (NVIDIA JAX + CUDA)
├── .github/workflows/              # CI/CD: Docker image build/publish to GHCR
├── pyproject.toml                  # Package metadata and dependencies
└── README.md
```

## Configuration

Experiments are configured using YAML files in the `configs/` directory. Key configuration sections include:

### Domain Configuration
```yaml
domain:
  lx: 100.0
  ly: 20.0
  t_final: 10.0
```

### Model Architecture
```yaml
model:
  name: "FourierPINN"  # or "MLP", "DGMNetwork"
  width: 128
  depth: 6
  output_dim: 3
```

### Training Parameters
```yaml
training:
  learning_rate: 1e-3
  batch_size: 4096
  epochs: 50000
loss_weights:
  pde_weight: 1.0
  ic_weight: 100.0
  bc_weight: 100.0
```

### Physics Parameters
```yaml
physics:
  g: 9.81
  n_manning: 0.012
  inflow: 1.0
```

## Usage

### Basic Training

```bash
# Run experiment-specific training scripts
python -m src.scenarios.experiment_1.experiment_1 configs/experiment_1_fourier.yaml
python -m src.scenarios.experiment_3.experiment_3 configs/experiment_3.yaml
python -m src.scenarios.experiment_7.experiment_7 configs/experiment_7.yaml
python -m src.scenarios.experiment_8.experiment_8 configs/experiment_8.yaml

# Unified training script (for experiments that support it)
python src/train.py configs/experiment_1_fourier.yaml
```

### Hyperparameter Optimization

```bash
# Run HPO with Optuna
python optimisation/run_optimization.py --config optimisation/configs/exploitation/hpo_exploitation_fourier_nobuilding.yaml --n_trials 100

# Sensitivity analysis (Sobol exploration)
python optimisation/run_sensitivity_analysis.py

# Extract best parameters
python optimisation/extract_best_params.py
```

### Experiment Tracking

```bash
# Start Aim dashboard
aim up
# Open http://localhost:43800 in your browser
```

### Visualization

```bash
# Render solution video
python scripts/render_video.py --model_dir models/trained_model --config configs/experiment_1_fourier.yaml --output results/video.mp4
```

## Physics Background

The 2D Shallow Water Equations describe conservation of mass and momentum for water flow:

### Continuity Equation (Mass Conservation)
```
dh/dt + d(hu)/dx + d(hv)/dy = 0
```

### Momentum Equations
```
d(hu)/dt + d(hu^2 + gh^2/2)/dx + d(huv)/dy = -gh * dz_b/dx - g*n^2*u*sqrt(u^2+v^2)/h^(1/3)
d(hv)/dt + d(huv)/dx + d(hv^2 + gh^2/2)/dy = -gh * dz_b/dy - g*n^2*v*sqrt(u^2+v^2)/h^(1/3)
```

Where:
- `h`: water depth [m]
- `u, v`: depth-averaged velocity components [m/s]
- `g`: gravitational acceleration [m/s^2]
- `z_b`: bed elevation [m]
- `n`: Manning's roughness coefficient [s/m^(1/3)]

### Boundary Conditions

- **Inflow**: Prescribed depth and velocity at upstream boundary
- **Outflow**: Zero-gradient conditions at downstream boundary
- **Walls**: No-flux (slip) conditions at lateral boundaries and building walls
- **Initial**: Flat water surface with zero velocity at t=0

## Results and Validation

The implementation provides comprehensive validation metrics:

### Quantitative Metrics
- **Nash-Sutcliffe Efficiency (NSE)**: Measures model performance
- **Root Mean Square Error (RMSE)**: Absolute error magnitude
- **Physics Residuals**: PDE constraint satisfaction

### Qualitative Validation
- **Solution Videos**: Time-evolution of water depth and velocity
- **Contour Plots**: Spatial distribution at specific times
- **Validation Profiles**: Comparison with analytical/reference solutions

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size in configuration
- Decrease model hidden dimensions

**Poor Convergence**
- Increase Fourier features
- Adjust loss weights (increase PDE weight)
- Try different architecture (FourierPINN vs DGM)

**NaN Losses**
- Check boundary condition implementation
- Verify coordinate ranges
- Reduce learning rate

**Slow Training**
- Ensure CUDA is available: `python -c "import jax; print(jax.devices())"`
- Use `float32` precision where possible

### Getting Help

- Check existing issues on GitHub
- Review Aim logs for training diagnostics

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes and add tests**
4. **Run the test suite:**
   ```bash
   python -m unittest discover test
   ```
5. **Commit your changes:**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch:**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{zeinali_swe_pinn_2024,
  title={{SWE-PINN}: Physics-Informed Neural Networks for Shallow Water Equations},
  author={Zeinali, Farzan},
  url={https://github.com/zeinali72/swe-pinn},
  year={2024},
  version={0.1.0}
}
```

## Acknowledgments

- **JAX/Flax**: For providing the high-performance ML framework
- **Aim**: For experiment tracking and visualization
- **Optuna**: For hyperparameter optimization
- **Scientific Community**: For advancing PINN methodologies
