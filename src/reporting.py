# src/reporting.py
import time
from typing import Dict
from aim import Run 

def print_epoch_stats(epoch: int, global_step: int, start_time: float, total_loss: float,
                      pde_loss: float, ic_loss: float, bc_loss: float,
                      building_bc_loss: float, data_loss: float, neg_h_loss: float,
                      nse: float, rmse: float,
                      epoch_time: float):
    """Prints the training statistics for the current epoch."""
    elapsed_time = time.time() - start_time

    building_loss_str = f"BldgBC: {building_bc_loss:.3e} | " if building_bc_loss > 1e-9 else ""
    data_loss_str = f"Data: {data_loss:.3e} | " if data_loss > 1e-9 else ""
    neg_h_str = f"NegH: {neg_h_loss:.3e} | " if neg_h_loss > 1e-9 else ""

    print(
        f"Epoch {epoch+1:5d} | Step {global_step:7d} | "
        f"Elapsed: {elapsed_time:.1f}s | "
        f"Epoch Time: {epoch_time:.2f}s | "
        f"Total Loss: {total_loss:.4e} | "
        f"PDE: {pde_loss:.3e} | "
        f"IC: {ic_loss:.3e} | "
        f"BC: {bc_loss:.3e} | "
        f"{building_loss_str}"
        f"{data_loss_str}"
        f"{neg_h_str}"
        f"NSE: {nse:.4f} | "
        f"RMSE: {rmse:.4f}"
    )

def _safe_float(val, default=float('nan')):
    """Helper to convert JAX/Numpy arrays/scalars to standard Python floats."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def log_metrics(aim_run: Run, step: int, epoch: int, metrics: Dict):
    """
    Logs all provided metrics to Aim against the global_step.
    This single function handles:
    - Per-batch losses (from 'batch_losses')
    - GradNorm weights (from 'gradnorm_weights')
    - Per-epoch average losses (from 'epoch_avg_losses')
    - Per-epoch validation metrics (from 'validation_metrics')
    - Per-epoch system metrics (from 'system_metrics')
    
    This ensures all time-series data shares the 'step' axis in Aim.
    """
    if not aim_run:
        return
    
    try:
        # 1. Log Per-Batch Losses (if provided)
        # These are the "noisy" step-by-step losses
        if 'batch_losses' in metrics:
            for key, val in metrics.get('batch_losses', {}).items():
                aim_run.track(_safe_float(val), name=f'losses/batch/unweighted/{key}', step=step, epoch=epoch, context={'subset': 'train'})
            
            aim_run.track(_safe_float(metrics.get('batch_total_weighted_loss')), name='losses/batch/total_weighted', step=step, epoch=epoch, context={'subset': 'train'})

        # 2. Log GradNorm Weights (if provided)
        if 'gradnorm_weights' in metrics:
            for key, val in metrics.get('gradnorm_weights', {}).items():
                aim_run.track(_safe_float(val), name=f'gradnorm/weight_{key}', step=step, epoch=epoch, context={'subset': 'train'})
        
        # 3. Log Per-Epoch Average Losses (if provided)
        # These are the "smooth" epoch-average losses
        if 'epoch_avg_losses' in metrics:
            for key, val in metrics.get('epoch_avg_losses', {}).items():
                aim_run.track(_safe_float(val), name=f'losses/epoch_avg/unweighted/{key}', step=step, epoch=epoch, context={'subset': 'train'})

            aim_run.track(_safe_float(metrics.get('epoch_avg_total_weighted_loss')), name='losses/epoch_avg/total_weighted', step=step, epoch=epoch, context={'subset': 'train'})

        # 4. Log Per-Epoch Validation Metrics (if provided)
        if 'validation_metrics' in metrics:
            aim_run.track(_safe_float(metrics['validation_metrics'].get('nse'), -float('inf')), name='validation/nse', step=step, epoch=epoch, context={'subset': 'validation'})
            aim_run.track(_safe_float(metrics['validation_metrics'].get('rmse'), float('inf')), name='validation/rmse', step=step, epoch=epoch, context={'subset': 'validation'})
        
        # 5. Log Per-Epoch System Metrics (if provided)
        if 'system_metrics' in metrics:
            aim_run.track(_safe_float(metrics['system_metrics'].get('epoch_time')), name='system/epoch_time', step=step, epoch=epoch, context={'subset': 'system'})

    except Exception as e:
        print(f"Warning: Failed to log metrics to Aim at step {step}: {e}")


def print_final_summary(total_time: float, best_epoch: int, best_nse: float, best_nse_time: float):
    """Prints the final training summary."""
    print(f"\n--- Training Summary ---")
    print(f"Total time: {total_time:.2f} seconds.")
    if isinstance(best_nse, (float, int)) and best_nse > -float('inf'):
        print(
            f"Best NSE: {best_nse:.6f} (achieved at epoch {best_epoch+1} around {best_nse_time:.2f}s)"
        )
    else:
        print(f"No valid best NSE achieved during training (Best NSE: {best_nse}).")
    print(f"------------------------")