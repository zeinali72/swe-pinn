# SWE-PINN: Physics-Informed Neural Networks for Shallow Water Equations

This repository implements a Physics-Informed Neural Network (PINN) approach to solve the 2D Shallow Water Equations (SWE) using JAX and Flax. The implementation includes Fourier feature mapping for improved convergence and handles complex boundary conditions.

## Features

- **Physics-Informed Neural Networks**: Solves PDEs by incorporating physical laws into the loss function
- **Shallow Water Equations**: Models water flow in rivers, channels, and coastal areas
- **Fourier Feature Mapping**: Enhances the neural network's ability to learn high-frequency solutions
- **JAX/Flax Backend**: High-performance automatic differentiation and GPU acceleration
- **Experiment Tracking**: Integrated with Aim for logging and visualization
- **Modular Architecture**: Clean separation of physics, models, training, and utilities

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for performance)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/zeinali72/swe-pinn.git
cd swe-pinn
```

2. Install dependencies:
```bash
pip install -r .devcontainer/requirements.txt
```

3. For development in VS Code with dev containers, use the provided `.devcontainer` configuration.

## Project Structure

```
swe-pinn/
├── .devcontainer/          # VS Code dev container configuration
├── aim_repo/              # Experiment tracking repository
├── data/                  # Data storage (if any)
├── experiments/           # Configuration files
│   └── config_1.yaml      # Main experiment configuration
├── hyperopt/              # Hyperparameter optimization results
├── models/                # Saved model checkpoints
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── config.py          # Configuration loading and processing
│   ├── data.py            # Data sampling and batching utilities
│   ├── losses.py          # Physics-informed loss functions
│   ├── models.py          # Neural network architectures
│   ├── physics.py         # Shallow water equations implementation
│   ├── train.py           # Main training script
│   └── utils.py           # Metrics, plotting, and utilities
└── results/               # Training results and plots
```

## Configuration

The main configuration is stored in `experiments/config_1.yaml`. Key parameters include:

- **Domain**: Spatial and temporal domain boundaries
- **Model**: Neural network architecture (width, depth, Fourier features)
- **Training**: Learning rate, batch size, epochs, loss weights
- **Physics**: Manning roughness, gravity, inflow conditions
- **Grid**: Discretization for PDE, IC, and BC sampling

## Usage

### Training
Run the main training script:
```bash
python src/train.py
```

### Experiment Tracking
Results are automatically logged to the `aim_repo` directory. View experiments with:
```bash
aim up
```

### Configuration
Modify `experiments/config_1.yaml` to adjust parameters for different scenarios.

## Physics Background

The Shallow Water Equations describe the flow of water in channels and rivers:

∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = S_mass
∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = S_mom_x
∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = S_mom_y

Where:
- h: water depth
- u, v: velocity components
- g: gravity
- S: source terms (friction, inflow)

## Boundary Conditions

- **Inflow**: Prescribed water depth and velocity at left boundary
- **Zero-gradient**: ∂U/∂x = 0 at right boundary
- **No-flux**: v = 0 at bottom and top boundaries
- **Initial**: h = u = v = 0 at t = 0

## Results

The model outputs include:
- Water depth h(x,y,t)
- Velocity components u(x,y,t), v(x,y,t)
- Training metrics: NSE, RMSE, loss components
- Validation plots comparing predicted vs exact solutions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{swe_pinn,
  title={SWE-PINN: Physics-Informed Neural Networks for Shallow Water Equations},
  author={Zeinali, Farzan},
  url={https://github.com/zeinali72/swe-pinn},
  year={2024}
}
```

## Acknowledgments

- Built with JAX and Flax for high-performance computing
- Experiment tracking powered by Aim
- Inspired by physics-informed neural network literature