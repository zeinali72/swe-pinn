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
    Logs metrics to Aim.
    - GradNorm metrics are logged against 'step' and 'epoch'.
    - All other metrics are logged against 'epoch' ONLY.
    - Batch losses are no longer logged.
    """
    if not aim_run:
        return
    
    try:
        # 1. Log GradNorm Weights (step and epoch)
        if 'gradnorm_weights' in metrics:
            for key, val in metrics.get('gradnorm_weights', {}).items():
                aim_run.track(_safe_float(val), name=f'gradnorm/weight_{key}', step=step, epoch=epoch, context={'subset': 'train'})
        
        # 2. Log Per-Epoch Average Losses (epoch only)
        if 'epoch_avg_losses' in metrics:
            for key, val in metrics.get('epoch_avg_losses', {}).items():
                aim_run.track(_safe_float(val), name=f'losses/epoch_avg/unweighted/{key}', epoch=epoch, context={'subset': 'train'})

            aim_run.track(_safe_float(metrics.get('epoch_avg_total_weighted_loss')), name='losses/epoch_avg/total_weighted', epoch=epoch, context={'subset': 'train'})

        # 3. Log Per-Epoch Validation Metrics (epoch only)
        if 'validation_metrics' in metrics:
            aim_run.track(_safe_float(metrics['validation_metrics'].get('nse'), -float('inf')), name='validation/nse', epoch=epoch, context={'subset': 'validation'})
            aim_run.track(_safe_float(metrics['validation_metrics'].get('rmse'), float('inf')), name='validation/rmse', epoch=epoch, context={'subset': 'validation'})
        
        # 4. Log Per-Epoch System Metrics (epoch only)
        if 'system_metrics' in metrics:
            aim_run.track(_safe_float(metrics['system_metrics'].get('epoch_time')), name='system/epoch_time', epoch=epoch, context={'subset': 'system'})

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