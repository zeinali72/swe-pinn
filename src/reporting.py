# src/reporting.py
import time
from typing import Dict
from aim import Run # Keep Aim import

# --- Updated signature to include data_loss ---
def print_epoch_stats(epoch: int, start_time: float, total_loss: float,
                      pde_loss: float, ic_loss: float, bc_loss: float,
                      building_bc_loss: float, data_loss: float, # Added data_loss
                      nse: float, rmse: float,
                      epoch_time: float):
    """Prints the training statistics for the current epoch."""
    elapsed_time = time.time() - start_time

    # Conditional strings for optional losses
    building_loss_str = f"BldgBC: {building_bc_loss:.3e} | " if building_bc_loss > 1e-9 else ""
    # --- NEW: Conditional string for data loss ---
    data_loss_str = f"Data: {data_loss:.3e} | " if data_loss > 1e-9 else ""
    # --- END NEW ---

    print(
        f"Epoch {epoch+1:5d} | "
        f"Elapsed: {elapsed_time:.1f}s | "
        f"Epoch Time: {epoch_time:.2f}s | "
        f"Total Loss: {total_loss:.4e} | "
        f"PDE: {pde_loss:.3e} | "
        f"IC: {ic_loss:.3e} | "
        f"BC: {bc_loss:.3e} | "
        f"{building_loss_str}"
        # --- NEW: Include data loss string ---
        f"{data_loss_str}"
        # --- END NEW ---
        f"NSE: {nse:.4f} | "
        f"RMSE: {rmse:.4f}"
    )

def log_metrics(aim_run: Run, metrics: Dict, epoch: int):
    """Logs metrics to Aim."""
    if not aim_run:
        return
    try:
        # Helper to safely get metrics, using float('nan') as default
        def _get_metric(name, default=float('nan')):
            val = metrics.get(name, default)
            # Ensure value is a standard float, handle potential JAX types
            try:
                 return float(val)
            except (TypeError, ValueError):
                 return default # Return default if conversion fails

        aim_run.track(_get_metric('total_loss'), name='total_loss', step=epoch, context={'subset': 'train'})
        aim_run.track(_get_metric('pde_loss'), name='pde_loss', step=epoch, context={'subset': 'train'})
        aim_run.track(_get_metric('ic_loss'), name='ic_loss', step=epoch, context={'subset': 'train'})
        aim_run.track(_get_metric('bc_loss'), name='bc_loss', step=epoch, context={'subset': 'train'})
        # Use 0.0 as default for optional losses if they weren't computed
        aim_run.track(_get_metric('building_bc_loss', 0.0), name='building_bc_loss', step=epoch, context={'subset': 'train'})
        # --- NEW: Log data loss ---
        aim_run.track(_get_metric('data_loss', 0.0), name='data_loss', step=epoch, context={'subset': 'train'})
        # --- END NEW ---
        aim_run.track(_get_metric('nse', -float('inf')), name='nse', step=epoch, context={'subset': 'validation'}) # Use -inf default for NSE
        aim_run.track(_get_metric('rmse', float('inf')), name='rmse', step=epoch, context={'subset': 'validation'}) # Use inf default for RMSE
        aim_run.track(_get_metric('epoch_time'), name='epoch_time', step=epoch, context={'subset': 'system'})

        # --- NEW: Log dynamic weights ---
        for key in metrics:
            if key.startswith('weight_'):
                 aim_run.track(_get_metric(key, 1.0), name=key, step=epoch, context={'subset': 'gradnorm'})

    except Exception as e:
        print(f"Warning: Failed to log metrics to Aim in epoch {epoch}: {e}")


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