# Reporting Metrics Schema

This document describes the metrics logged by `src/reporting.py` to [Aim](https://aimstack.io/) during training.

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
| `system/epoch_time` | Duration of the current epoch in seconds | All experiments |

### Training Metrics (`context: {subset: 'train'}`)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `training/learning_rate` | Current optimizer learning rate | All experiments |

### Loss Metrics (`context: {subset: 'train'}`)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `losses/epoch_avg/total_weighted` | Weighted sum of all loss terms | All experiments |
| `losses/epoch_avg/unweighted/{key}` | Individual unweighted loss terms | All experiments |

Common loss keys: `pde`, `ic`, `bc`, `data`, `neg_h`, `building_bc`. The exact set depends on the experiment configuration and which loss weights are non-zero.

### GradNorm Metrics (`context: {subset: 'train'}`)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `gradnorm/weight_{key}` | Adaptive loss weight for each term | Experiments using GradNorm |

Only logged when the `gradnorm_weights` key is present in the metrics dict (i.e., when using GradNorm loss weighting strategy).

### Validation Metrics (`context: {subset: 'validation'}`)

| Metric Name | Description | Source |
|-------------|-------------|--------|
| `validation/nse` | Nash-Sutcliffe Efficiency (water depth) | Experiments 1-7 |
| `validation/rmse` | Root Mean Square Error (water depth) | Experiments 1-7 |
| `validation/nse_h` | NSE for water depth h | Experiment 8 |
| `validation/nse_hu` | NSE for specific discharge hu | Experiment 8 |
| `validation/nse_hv` | NSE for specific discharge hv | Experiment 8 |
| `validation/combined_nse` | Weighted combination of component NSEs | Experiment 8 |
| `validation/rmse_h` | RMSE for water depth h | Experiment 8 |

Validation metrics are handled generically: `log_metrics()` iterates all keys in the `validation_metrics` dict, so any new validation metric added to an experiment will be automatically tracked.

## Console Output

`print_epoch_stats()` prints a single-line summary per epoch:

```
Epoch    42 | Step    4200 | Elapsed: 123.4s | Epoch Time: 2.50s | Total Loss: 1.2345e-03 | PDE: 8.901e-04 | IC: 3.444e-04 | NSE: 0.9512 | RMSE: 0.0034
```

Loss terms are printed dynamically from a dict; only non-zero terms appear. This avoids placeholder values for experiments that don't use certain loss terms (e.g., `building_bc` is omitted when not applicable).

## Final Summary

`print_final_summary()` reports the best models found during training:

- **Best NSE Model**: The checkpoint with the highest validation NSE
- **Best Total Loss Model**: The checkpoint with the lowest total weighted loss

Each summary includes the epoch, elapsed time, NSE, RMSE, total loss, and per-term unweighted losses.
