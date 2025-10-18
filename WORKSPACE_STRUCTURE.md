# SWE-PINN Workspace Structure

This document describes the workspace structure for the Shallow Water Equations Physics-Informed Neural Network (SWE-PINN) project.

## Overview

This workspace is designed to train and validate physics-informed neural networks for shallow water equations using JAX and GPU acceleration.

## Directory Structure

```
/workspaces/swe-pinn/
├── data/                           # ⚠️ IGNORED IN GIT (Large datasets)
│   ├── one_building_DEM_zero/      # Example scenario
│   │   ├── validation_tensor.npy   # Full validation dataset (~24GB)
│   │   ├── validation_sample.npy   # Sampled validation subset
│   │   ├── training_data.npy       # Training dataset (if exists)
│   │   └── metadata.json           # Scenario metadata
│   └── [other_scenarios]/          # Additional simulation scenarios
│
├── scripts/                        # Utility scripts
│   ├── create_validation_sample.py # Sample validation data efficiently
│   └── [other_scripts].py          # Data processing utilities
│
├── models/                         # ⚠️ IGNORED IN GIT (Trained models)
│   ├── checkpoints/                # Model checkpoints during training
│   └── saved_models/               # Final trained models
│
├── results/                        # ⚠️ IGNORED IN GIT (Outputs)
│   ├── plots/                      # Visualization outputs
│   ├── metrics/                    # Training/validation metrics
│   └── logs/                       # Training logs
│
├── src/                            # Source code
│   ├── model.py                    # Neural network architecture
│   ├── physics.py                  # Physics loss functions (SWE)
│   ├── training.py                 # Training loops
│   └── utils.py                    # Helper functions
│
├── configs/                        # Configuration files
│   └── training_config.yaml        # Training hyperparameters
│
├── notebooks/                      # Jupyter notebooks for exploration
│   └── analysis.ipynb              # Data analysis and visualization
│
├── .devcontainer/                  # Development container config
│   └── devcontainer.json           # Container settings
│
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── WORKSPACE_STRUCTURE.md          # This file
```

## Data Location & Organization

### Primary Dataset Location: `/workspaces/swe-pinn/data/`

**⚠️ THIS DIRECTORY IS GITIGNORED** - Contains large binary files.

#### Dataset Structure Per Scenario

Each scenario (e.g., `one_building_DEM_zero/`) contains:

- **`validation_tensor.npy`**: Full validation dataset
  - Shape: `(N, 6)` where N can be ~1 billion rows
  - Columns: `[t, x, y, h, u, v]`
    - `t`: time (seconds)
    - `x`, `y`: spatial coordinates (meters)
    - `h`: water depth (meters)
    - `u`, `v`: velocity components (m/s)
  - Size: ~24GB for large scenarios
  - **Memory-mapped access recommended**

- **`validation_sample.npy`**: Sampled validation subset
  - Created by `scripts/create_validation_sample.py`
  - Smaller subset for quick validation (e.g., 10k-15k points)
  - Same column structure as validation_tensor.npy

- **`training_data.npy`**: Training dataset (if separate)
  - Similar structure to validation data
  - May include collocation points for physics loss

## Gitignored Directories

The following directories are **excluded from version control** (see `.gitignore`):

1. **`/data/`** - Large datasets (GB to TB scale)
2. **`/models/`** - Trained model checkpoints
3. **`/results/`** - Output files, plots, logs
4. **`__pycache__/`** - Python bytecode
5. **`.ipynb_checkpoints/`** - Jupyter notebook checkpoints

## Key Scripts

### `scripts/create_validation_sample.py`

Creates a smaller validation sample from large validation tensors.

**Usage:**
```bash
python scripts/create_validation_sample.py \
  --scenario one_building_DEM_zero \
  --samples 15000 \
  --time 3600.0 \
  --chunk-size 1000000
```

**Purpose:**
- Avoids loading 24GB files into memory
- Uses memory-mapped files for efficient access
- Samples uniformly from time-filtered data
- GPU-accelerated where possible (JAX)

## Environment

- **OS**: Ubuntu 24.04.1 LTS (in dev container)
- **GPU**: CUDA-enabled (JAX uses GPU_0)
- **Python**: 3.x with JAX, NumPy, and ML libraries

## Data Access Patterns

### For Large Files (>1GB)

1. **Use memory mapping**: `np.load(file, mmap_mode='r')`
2. **Process in chunks**: Avoid loading entire dataset
3. **Sort indices**: Better mmap access performance

### For Training

1. Load data in batches
2. Use JAX data loaders for GPU efficiency
3. Consider preprocessing and caching smaller datasets

## Notes for AI Assistants

- **Dataset location**: Always in `/workspaces/swe-pinn/data/[scenario]/`
- **Large files**: Use mmap and chunking to avoid OOM
- **GPU memory**: ~12-18GB VRAM available, plan accordingly
- **Checkpoints**: Save to `/workspaces/swe-pinn/models/checkpoints/`
- **Results**: Output to `/workspaces/swe-pinn/results/`

## Common Scenarios

The `data/` directory may contain scenarios such as:
- `one_building_DEM_zero/` - Single building simulation
- `urban_flood/` - Urban flooding scenario
- `dam_break/` - Dam break simulation
- Additional custom scenarios

Each scenario follows the same data structure outlined above.
```
