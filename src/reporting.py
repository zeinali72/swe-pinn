# src/reporting.py
import time
from typing import Dict
from aim import Run # Keep Aim import

def print_epoch_stats(epoch: int, start_time: float, total_loss: float,
                      pde_loss: float, ic_loss: float, bc_loss: float,
                      building_bc_loss: float,
                      nse: float, rmse: float,
                      epoch_time: float):
    """Prints the training statistics for the current epoch."""
    elapsed_time = time.time() - start_time

    # --- FIX: Create conditional string separately ---
    building_loss_str = f"BldgBC: {building_bc_loss:.3e} | " if building_bc_loss > 1e-9 else ""

    print(
        f"Epoch {epoch+1:5d} | "
        f"Elapsed: {elapsed_time:.1f}s | "
        f"Epoch Time: {epoch_time:.2f}s | "
        f"Total Loss: {total_loss:.4e} | "
        f"PDE: {pde_loss:.3e} | "
        f"IC: {ic_loss:.3e} | "
        f"BC: {bc_loss:.3e} | "
        f"{building_loss_str}" # <-- Include the separate string here
        f"NSE: {nse:.4f} | "
        f"RMSE: {rmse:.4f}"
    )

def log_metrics(aim_run: Run, metrics: Dict, epoch: int):
    """Logs metrics to Aim."""
    if not aim_run:
        return
    try:
        aim_run.track(metrics.get('total_loss', float('nan')), name='total_loss', step=epoch, context={'subset': 'train'})
        aim_run.track(metrics.get('pde_loss', float('nan')), name='pde_loss', step=epoch, context={'subset': 'train'})
        aim_run.track(metrics.get('ic_loss', float('nan')), name='ic_loss', step=epoch, context={'subset': 'train'})
        aim_run.track(metrics.get('bc_loss', float('nan')), name='bc_loss', step=epoch, context={'subset': 'train'})
        aim_run.track(metrics.get('building_bc_loss', 0.0), name='building_bc_loss', step=epoch, context={'subset': 'train'})
        aim_run.track(metrics.get('nse', float('-inf')), name='nse', step=epoch, context={'subset': 'validation'})
        aim_run.track(metrics.get('rmse', float('inf')), name='rmse', step=epoch, context={'subset': 'validation'})
        aim_run.track(metrics.get('epoch_time', float('nan')), name='epoch_time', step=epoch, context={'subset': 'system'})
    except Exception as e:
        print(f"Warning: Failed to log metrics to Aim in epoch {epoch}: {e}")

def print_final_summary(total_time: float, best_epoch: int, best_nse: float, best_nse_time: float):
    """Prints the final training summary."""
    print(f"\n--- Training Summary ---")
    print(f"Total time: {total_time:.2f} seconds.")
    if best_epoch is not None and best_nse > -float('inf'):
        print(
            f"Best NSE: {best_nse:.6f} (achieved at epoch {best_epoch+1} around {best_nse_time:.2f}s)"
        )
    else:
        print("No valid best NSE achieved during training.")
    print(f"------------------------")