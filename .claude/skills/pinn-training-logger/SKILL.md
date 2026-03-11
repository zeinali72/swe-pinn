---
name: pinn-training-logger
description: Standardise training console output and Aim experiment tracking for PINN experiments. Use this skill whenever the user asks to set up logging, configure Aim tracking, write a training loop's logging section, format training output, or debug logging issues. Also trigger for "training log", "Aim", "experiment tracking", "console output", "log format", or any request about how training progress should be displayed or recorded.
---

# PINN Training Logger — Standardised Logs and Aim Tracking

Every PINN experiment must produce identically structured console logs and Aim metrics. This ensures experiments are easy to parse, compare, and reference in the thesis.

## Console Output Format

### Run Header (printed once at start)

```
================================================================
EXPERIMENT: {experiment_name}
ARCHITECTURE: {arch_name} (e.g., DGM, Fourier-MLP, MLP)
PHASE: {phase_number} - {phase_name}
DATE: {ISO date}
DEVICE: {device_info} (e.g., GPU: NVIDIA A100 40GB, CUDA 12.x)
PRECISION: {float_type} (e.g., float32)
================================================================
HPO Trial: {trial_id} (if applicable, otherwise omit this line)
Total collocation points: PDE={n_pde}, IC={n_ic}, BC={n_bc}, Data={n_data}
Learning rate: {lr_init}, Scheduler: {scheduler_type}
Max epochs: {max_epochs}, Convergence threshold: {threshold}
================================================================
```

### Epoch Log (every N epochs, default N=100)

```
Epoch {epoch:>6d}/{max_epochs} | Loss: {total:.6e} | PDE: {pde:.6e} | IC: {ic:.6e} | BC: {bc:.6e} | Data: {data:.6e} | LR: {lr:.2e} | Time: {epoch_time:.1f}s | NSE(val): {nse:.4f}
```

### Completion Summary (printed once at end)

```
================================================================
TRAINING COMPLETE
Final epoch: {epoch}
Convergence: {yes/no} (criterion: {criterion_description})
Best validation NSE: {best_nse:.4f} at epoch {best_epoch}
Total training time: {hours}h {minutes}m {seconds}s
Final loss breakdown: PDE={pde:.6e}, IC={ic:.6e}, BC={bc:.6e}, Data={data:.6e}
================================================================
```

## Aim Tracking Standard

### Per-Epoch Metrics
Track these every epoch using `/`-separated names for Aim grouping:
- `loss/total`, `loss/pde`, `loss/ic`, `loss/bc`, `loss/data`
- `lr` — current learning rate
- `epoch_time` — wall-clock seconds for this epoch

### Validation Metrics (every validation interval)
- `metrics/nse_h`, `metrics/nse_hu`, `metrics/nse_hv`
- `metrics/rmse_h`, `metrics/rmse_hu`, `metrics/rmse_hv`

### Run-Level Parameters (set once at start)

**Always store:**
- `experiment_name`, `architecture`, `phase`, `precision`
- `n_pde`, `n_ic`, `n_bc`, `n_data`
- `lr_init`, `scheduler`, `max_epochs`, `convergence_threshold`
- `hidden_layers`, `hidden_units`, `activation`
- `device`, `cuda_version`, `jax_version`, `flax_version`

**Architecture-specific (if applicable):**
- `fourier_mapping_size`, `fourier_scale` (Fourier-MLP only)

**HPO runs additionally store:**
- `optuna_trial_id`, `optuna_study_name`, `hpo_stage` (sobol/tpe)

## Implementation

Use Python's `logging` module — not bare `print()`. This allows log-level control and integration with external handlers.

- Logger name: `pinn.{experiment_name}`
- Log level: `INFO` for standard output, `DEBUG` for per-batch details
- Aim repo path: configurable, default `./aim_repo`

Read `references/logger_setup.py` for the implementation of `setup_training_logger()` and the `AimTracker` class.

### Integration Pattern

```python
from logger_setup import setup_training_logger, AimTracker

logger = setup_training_logger("experiment_8")
tracker = AimTracker(config, repo_path="./aim_repo")

# At start
logger.info(tracker.format_header(config))

# Each epoch
logger.info(tracker.format_epoch(epoch, losses, lr, epoch_time, val_nse))
tracker.track_epoch(epoch, losses, lr, epoch_time)

# At end
logger.info(tracker.format_summary(final_epoch, converged, best_nse, ...))
tracker.close()
```

## Reference Files

- `references/logger_setup.py` — Full implementation of `setup_training_logger()` and `AimTracker` class.
