# src/reporting.py
import time
from typing import Dict
from aim import Run

def print_epoch_stats(epoch: int, start_time: float, total_loss: float, 
                      pde_loss: float, ic_loss: float, bc_loss: float,
                      nse: float, rmse: float):
    """Prints the training statistics for the current epoch."""
    elapsed_time = time.time() - start_time
    print(
        f"Epoch {epoch+1:5d} | "
        f"Time: {elapsed_time:.2f}s | "
        f"Total Loss: {total_loss:.4e} | "
        f"PDE: {pde_loss:.4e} | "
        f"IC: {ic_loss:.4e} | "
        f"BC: {bc_loss:.4e} | "
        f"NSE: {nse:.4f} | "
        f"RMSE: {rmse:.4f}"
    )

def log_metrics(aim_run: Run, metrics: Dict, epoch: int):
    """Logs metrics to Aim."""
    aim_run.track(metrics['total_loss'], name='total_loss', step=epoch, context={'subset': 'train'})
    aim_run.track(metrics['pde_loss'], name='pde_loss', step=epoch, context={'subset': 'train'})
    aim_run.track(metrics['ic_loss'], name='ic_loss', step=epoch, context={'subset': 'train'})
    aim_run.track(metrics['bc_loss'], name='bc_loss', step=epoch, context={'subset': 'train'})
    aim_run.track(metrics['nse'], name='nse', step=epoch, context={'subset': 'validation'})
    aim_run.track(metrics['rmse'], name='rmse', step=epoch, context={'subset': 'validation'})

def print_final_summary(total_time: float, best_epoch: int, best_nse: float, best_nse_time: float):
    """Prints the final training summary."""
    print(f"Training ended. Total time: {total_time:.2f} seconds.")
    if best_epoch is not None:
        print(
            f"Best model from epoch {best_epoch+1} saved with NSE {best_nse:.6f}, "
            f"achieved at {best_nse_time:.2f}s."
        )