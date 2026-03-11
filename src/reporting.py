# src/reporting.py
import time
from typing import Dict
from aim import Run 

def print_epoch_stats(epoch: int, global_step: int, start_time: float, total_loss: float,
                      losses: Dict[str, float],
                      nse: float, rmse: float,
                      epoch_time: float):
    """Prints the training statistics for the current epoch.

    Args:
        losses: Dict of loss term names to values. Only non-zero terms are printed.
    """
    elapsed_time = time.time() - start_time

    loss_parts = []
    for key, val in losses.items():
        fval = float(val)
        if abs(fval) > 1e-9:
            label = key.upper().replace('_', ' ')
            loss_parts.append(f"{label}: {fval:.3e}")
    losses_str = " | ".join(loss_parts)

    print(
        f"Epoch {epoch+1:5d} | Step {global_step:7d} | "
        f"Elapsed: {elapsed_time:.1f}s | "
        f"Epoch Time: {epoch_time:.2f}s | "
        f"Total Loss: {total_loss:.4e} | "
        f"{losses_str} | "
        f"NSE: {nse:.4f} | "
        f"RMSE: {rmse:.4f}"
    )

def _safe_float(val, default=float('nan')):
    """Helper to convert JAX/Numpy arrays/scalars to standard Python floats."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def sanitize_for_aim(obj):
    """Recursively convert JAX/NumPy arrays to native Python types for Aim.

    Aim's ``Run.__setitem__`` stores values as JSON-like metadata and raises
    on non-native types such as ``jaxlib._jax.ArrayImpl``.  This helper
    walks a nested dict/list and converts every numeric leaf to a Python
    ``float`` or ``int``.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_aim(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_for_aim(v) for v in obj)
    # Try int first (preserves epoch counts etc.)
    try:
        if hasattr(obj, 'item'):          # numpy / jax scalar
            return obj.item()
        if isinstance(obj, float):
            return obj
        if isinstance(obj, int):
            return obj
        return float(obj)
    except (TypeError, ValueError):
        return obj

def log_metrics(aim_run: Run, step: int, epoch: int, metrics: Dict):
    """
    Logs metrics to Aim.

    Every metric is tracked against both *step* and *epoch* so that the
    Aim UI can plot against either axis.  An ``elapsed_time`` key in
    *metrics* is also tracked as ``system/elapsed_time``.
    """
    if not aim_run:
        return

    try:
        # 0. Log elapsed wall-clock time (step + epoch)
        if 'elapsed_time' in metrics:
            aim_run.track(_safe_float(metrics['elapsed_time']), name='system/elapsed_time',
                          step=step, epoch=epoch, context={'subset': 'system'})

        # 1. Log GradNorm Weights (step + epoch)
        if 'gradnorm_weights' in metrics:
            for key, val in metrics.get('gradnorm_weights', {}).items():
                aim_run.track(_safe_float(val), name=f'gradnorm/weight_{key}', step=step, epoch=epoch, context={'subset': 'train'})

        # 2. Log Per-Epoch Average Losses (step + epoch)
        if 'epoch_avg_losses' in metrics:
            for key, val in metrics.get('epoch_avg_losses', {}).items():
                aim_run.track(_safe_float(val), name=f'losses/epoch_avg/unweighted/{key}',
                              step=step, epoch=epoch, context={'subset': 'train'})

            aim_run.track(_safe_float(metrics.get('epoch_avg_total_weighted_loss')),
                          name='losses/epoch_avg/total_weighted',
                          step=step, epoch=epoch, context={'subset': 'train'})

        # 3. Log Per-Epoch Validation Metrics — generic iteration (step + epoch)
        if 'validation_metrics' in metrics:
            for key, val in metrics['validation_metrics'].items():
                default = -float('inf') if 'nse' in key else float('inf')
                aim_run.track(_safe_float(val, default), name=f'validation/{key}',
                              step=step, epoch=epoch, context={'subset': 'validation'})

        # 4. Log Per-Epoch System Metrics (step + epoch)
        if 'system_metrics' in metrics:
            aim_run.track(_safe_float(metrics['system_metrics'].get('epoch_time')),
                          name='system/epoch_time',
                          step=step, epoch=epoch, context={'subset': 'system'})

        # 5. Log Training Metrics (e.g. learning rate) (step + epoch)
        if 'training_metrics' in metrics:
            for key, val in metrics.get('training_metrics', {}).items():
                aim_run.track(_safe_float(val), name=f'training/{key}',
                              step=step, epoch=epoch, context={'subset': 'train'})

    except Exception as e:
        print(f"Warning: Failed to log metrics to Aim at epoch {epoch}: {e}")


def print_final_summary(total_time: float, best_nse_stats: Dict, best_loss_stats: Dict):
    """Prints the final training summary for both best models."""
    print(f"\n--- Training Summary ---")
    print(f"Total time: {total_time:.2f} seconds.")
    
    print("\n--- Best NSE Model (Saved) ---")
    if best_nse_stats and best_nse_stats.get('nse', -float('inf')) > -float('inf'):
        print(f"  NSE:           {best_nse_stats['nse']:.6f}")
        print(f"  Epoch:         {best_nse_stats['epoch'] + 1}")
        print(f"  Time Elapsed:  {best_nse_stats['time_elapsed_seconds']:.2f}s")
        print(f"  RMSE:          {best_nse_stats['rmse']:.4f}")
        print(f"  Total Loss:    {best_nse_stats['total_weighted_loss']:.4e}")
        print(f"  Losses (unweighted):")
        for k, v in best_nse_stats.get('unweighted_losses', {}).items():
            if v > 1e-9 or v < -1e-9: # Only print non-zero losses
                print(f"    - {k:<12}: {v:.4e}")
    else:
        print("  No valid best NSE model was found.")

    print("\n--- Best Total Loss Model ---")
    if best_loss_stats and best_loss_stats.get('total_weighted_loss', float('inf')) < float('inf'):
        print(f"  Total Loss:    {best_loss_stats['total_weighted_loss']:.4e}")
        print(f"  Epoch:         {best_loss_stats['epoch'] + 1}")
        print(f"  Time Elapsed:  {best_loss_stats['time_elapsed_seconds']:.2f}s")
        print(f"  NSE:           {best_loss_stats['nse']:.6f}")
        print(f"  RMSE:          {best_loss_stats['rmse']:.4f}")
        print(f"  Losses (unweighted):")
        for k, v in best_loss_stats.get('unweighted_losses', {}).items():
            if v > 1e-9 or v < -1e-9: # Only print non-zero losses
                print(f"    - {k:<12}: {v:.4e}")
    else:
        print("  No valid best loss model was found.")
    
    print(f"------------------------")