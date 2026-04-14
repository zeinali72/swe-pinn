# GEMINI.md

SWE-PINN: Physics-Informed Neural Network framework for urban flood prediction, solving 2D Shallow Water Equations with JAX/Flax. See `docs/experimental_programme_reference.md` for the authoritative experiment spec.

## Key Commands

```bash
# Training (all experiments follow this pattern)
python -m experiments.experiment_<N>.train --config configs/<config>.yaml

# Example: experiment 1 with variants
python -m experiments.experiment_1.train --config configs/experiment_1/experiment_1.yaml
python -m experiments.experiment_1.train_imp_samp --config configs/experiment_1/experiment_1_imp_samp.yaml
python -m experiments.experiment_1.train_relobralo --config configs/experiment_1/experiment_1_relobralo.yaml

# HPO-optimised configs live in configs/train/
python -m experiments.experiment_2.train --config configs/train/experiment_2_fourier_final.yaml

# Tests
python -m unittest discover test          # all tests
python -m unittest test.test_losses       # single file

# HPO
python optimisation/run_optimization.py --config <hpo_config> --n_trials 100

# Inference
python scripts/infer.py --config <config>.yaml --checkpoint models/<experiment>/<trial>/checkpoints/best_nse --checkpoints best_nse

# Dependencies
pip install -r .devcontainer/requirements.txt
```

## Code Style

- IMPORTANT: Always load configs via `src.config.load_config()` — never parse YAML directly
- Use `jax.numpy as jnp` (not `numpy`) inside JIT-compiled functions
- `snake_case` for functions/variables, `PascalCase` for classes
- Type hints required on all function signatures
- No linter or pre-commit hooks are configured

## Testing

- Tests force CPU: `os.environ["JAX_PLATFORM_NAME"] = "cpu"` at top of test files
- Use `unittest.mock` for file I/O and data; setUp/tearDown for temp directories
- Test both data-free and data-driven modes
- **Manual training smoke-tests: use 200 epochs**

## Common Pitfalls

- **JIT side effects**: No Python side effects inside `@jax.jit` functions — first call traces, subsequent calls replay the trace
- **Float precision**: Some physics computations require `float64`. Set via `config.device.dtype`
- **Large data files**: Use `np.load(..., mmap_mode='r')` for multi-GB validation datasets
- **Water depth masking**: `h >= eps` is required for numerical stability in SWE computations

## Experimental Programme

IMPORTANT: Always consult `docs/experimental_programme_reference.md` when:
- Implementing or modifying any evaluation metric or inference plot
- Setting up W&B tracking (what to log, at what frequency)
- Building or modifying any module in `src/metrics/` or `src/plots/`

Key conventions:
- Accuracy metrics (NSE, RMSE, MAE, Rel L2) reported separately for h, hu, hv
- Colour palette: Exeter (Deep Green #003C3C, Teal #007D69, Mint #00C896) + Blue Heart (Navy #0D2B45, Ocean #1B5E8A, Sky #4FA3D1). Arial font. 300 DPI.
