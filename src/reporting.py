# src/reporting.py
import time
from typing import Dict
from aim import Run # Keep Aim import

# --- Updated signature to include global_step ---
def print_epoch_stats(epoch: int, global_step: int, start_time: float, total_loss: float,
                      pde_loss: float, ic_loss: float, bc_loss: float,
                      building_bc_loss: float, data_loss: float, neg_h_loss: float, # Reordered for consistency
                      nse: float, rmse: float,
                      epoch_time: float):
    """Prints the training statistics for the current epoch."""
    elapsed_time = time.time() - start_time

    # Conditional strings for optional losses
    building_loss_str = f"BldgBC: {building_bc_loss:.3e} | " if building_bc_loss > 1e-9 else ""
    data_loss_str = f"Data: {data_loss:.3e} | " if data_loss > 1e-9 else ""
    neg_h_loss_str = f"NegH: {neg_h_loss:.3e} | " if neg_h_loss > 1e-9 else ""

    print(
        f"Epoch {epoch+1:5d} | Step {global_step:7d} | " # Added Step
        f"Elapsed: {elapsed_time:.1f}s | "
        f"Epoch Time: {epoch_time:.2f}s | "
        f"Total Loss: {total_loss:.4e} | "
        f"PDE: {pde_loss:.3e} | "
        f"IC: {ic_loss:.3e} | "
        f"BC: {bc_loss:.3e} | "
        f"{building_loss_str}"
        f"{data_loss_str}"
        f"{neg_h_loss_str}"
        f"NSE: {nse:.4f} | "
        f"RMSE: {rmse:.4f}"
    )

def _safe_float(val, default=float('nan')):
    """Helper to convert JAX/Numpy arrays/scalars to standard Python floats."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def log_step_metrics(aim_run: Run, step: int, epoch: int, metrics: Dict):
    """Logs metrics that are tracked per-step (e.g., losses, gradnorm weights)."""
    if not aim_run:
        return
    try:
        # Log all unweighted losses
        for key, val in metrics.get('unweighted_losses', {}).items():
            aim_run.track(_safe_float(val), name=f'losses/unweighted/{key}', step=step, epoch=epoch, context={'subset': 'train'})
        
        # Log total weighted loss
        aim_run.track(_safe_float(metrics.get('total_loss_weighted')), name='losses/total_weighted', step=step, epoch=epoch, context={'subset': 'train'})

        # Log dynamic weights (if present)
        for key, val in metrics.get('gradnorm_weights', {}).items():
            aim_run.track(_safe_float(val), name=f'gradnorm/weight_{key}', step=step, epoch=epoch, context={'subset': 'train'})

    except Exception as e:
        print(f"Warning: Failed to log step metrics to Aim at step {step}: {e}")

def log_epoch_metrics(aim_run: Run, epoch: int, metrics: Dict):
    """Logs metrics that are tracked per-epoch (e.g., validation, system)."""
    if not aim_run:
        return
    try:
        # Log validation metrics
        aim_run.track(_safe_float(metrics.get('nse'), -float('inf')), name='validation/nse', step=epoch, context={'subset': 'validation'})
        aim_run.track(_safe_float(metrics.get('rmse'), float('inf')), name='validation/rmse', step=epoch, context={'subset': 'validation'})
        
        # Log system metrics
        aim_run.track(_safe_float(metrics.get('epoch_time')), name='system/epoch_time', step=epoch, context={'subset': 'system'})

    except Exception as e:
        print(f"Warning: Failed to log epoch metrics to Aim at epoch {epoch}: {e}")


def print_final_summary(total_time: float, best_epoch: int, best_nse: float, best_nse_time: float):
    """Prints the final training summary."""
    print(f"\n--- Training Summary ---")
    print(f"Total time: {total_time:.2f} seconds.")
    # Ensure best_nse is a valid float before formatting
    if isinstance(best_nse, (float, int)) and best_nse > -float('inf'):
        print(
            f"Best NSE: {best_nse:.6f} (achieved at epoch {best_epoch+1} around {best_nse_time:.2f}s)"
        )
    else:
        print(f"No valid best NSE achieved during training (Best NSE: {best_nse}).")
    print(f"------------------------")

# --- REMOVED old log_metrics function ---
# The new functions log_step_metrics and log_epoch_metrics replace it.