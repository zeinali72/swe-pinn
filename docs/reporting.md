# Reporting Metrics Schema

This document describes the metrics logged by `src/monitoring/aim_tracker.py` to [Aim](https://aimstack.io/) during training.

## Tracking Axes

Every metric is tracked against **three axes**, enabling flexible visualization in the Aim UI:

| Axis | Description | Aim Parameter |
|------|-------------|---------------|
| **Step** | Global training step (cumulative across all epochs) | `step=step` |
| **Epoch** | Training epoch index (0-based) | `epoch=epoch` |
| **Relative Time** | Wall-clock seconds since training start, logged as `system/elapsed_time` | Queryable via Aim UI |

To plot any metric against wall-clock time in the Aim UI, add `system/elapsed_time` as the x-axis.

## Metric Categories

### System Metrics (`context: {subset: 'system'}`)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `system/elapsed_time` | Seconds since training start | All experiments |
| `optim/epoch_time_sec` | Duration of the current epoch in seconds | All experiments |

### Training Metrics (`context: {subset: 'train'}`)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `optim/lr` | Current optimizer learning rate | All experiments |

### Loss Metrics (`context: {subset: 'train'}`)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `loss/total` | Weighted sum of all loss terms | All experiments |
| `loss/{key}` | Individual unweighted loss terms | All experiments |

Common loss keys: `pde`, `ic`, `bc`, `data`, `neg_h`, `building_bc`. The exact set depends on the experiment configuration and which loss weights are non-zero.

### Validation Metrics (`context: {subset: 'validation'}`)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `val/nse_h` | NSE for water depth h | All experiments |
| `val/rmse_h` | RMSE for water depth h | All experiments |
| `val/nse_hu` | NSE for specific discharge hu | Experiments with hu/hv targets |
| `val/rmse_hu` | RMSE for specific discharge hu | Experiments with hu/hv targets |
| `val/nse_hv` | NSE for specific discharge hv | Experiments with hu/hv targets |
| `val/rmse_hv` | RMSE for specific discharge hv | Experiments with hu/hv targets |

Validation metrics are handled generically: `log_epoch()` iterates all keys in the `val_metrics` dict, so any new validation metric added to an experiment will be automatically tracked.

### Diagnostics (`context: {subset: 'diagnostics'}`)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `diagnostics/negative_h_{key}` | Negative depth statistics (count, fraction, min, mean) | All experiments |

### Best-Model Tracking (no context)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `best/nse_h_value` | NSE at new best-NSE checkpoint | All experiments |
| `best/loss_value` | Loss at new best-loss checkpoint | All experiments |

## Console Output

`ConsoleLogger.print_epoch()` (in `src/monitoring/console_logger.py`) prints a single-line summary per epoch:

```
Epoch    42 | Step    4200 | Elapsed: 123.4s | Epoch Time: 2.50s | Total Loss: 1.2345e-03 | PDE: 8.901e-04 | IC: 3.444e-04 | NSE: 0.9512 | RMSE: 0.0034
```

Loss terms are printed dynamically from a dict; only non-zero terms appear. This avoids placeholder values for experiments that don't use certain loss terms (e.g., `building_bc` is omitted when not applicable).

## Final Summary

`ConsoleLogger.print_completion_summary()` reports the best models found during training:

- **Best NSE Model**: The checkpoint with the highest validation NSE
- **Best Total Loss Model**: The checkpoint with the lowest total weighted loss

Each summary includes the epoch, elapsed time, NSE, RMSE, total loss, and per-term unweighted losses.
