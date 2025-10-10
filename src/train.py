# src/train.py
import os
import time
import copy
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import random
import optax
from aim import Repo, Run

# Import from our source code
from src.config import config
from src.data import sample_points, get_batches
from src.models import init_model
from src.losses import compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss
from src.utils import nse, rmse, generate_trial_name, save_model, plot_h_vs_x
from src.physics import h_exact

def train_step(model: Any, params: Dict[str, Any], opt_state: Any,
               pde_batch: jnp.ndarray, ic_batch: jnp.ndarray,
               bc_left_batch: jnp.ndarray, bc_right_batch: jnp.ndarray,
               bc_bottom_batch: jnp.ndarray, bc_top_batch: jnp.ndarray,
               weights_dict: Dict[str, float],
               optimiser: optax.GradientTransformation
               ) -> Tuple[Any, Any, Dict[str, jnp.ndarray]]:
    """Perform a single training step for the PINN model."""
    def loss_and_stats(p):
        pde_loss = compute_pde_loss(model, p, pde_batch)
        ic_loss = compute_ic_loss(model, p, ic_batch)
        bc_loss = compute_bc_loss(
            model, p, bc_left_batch, bc_right_batch, bc_bottom_batch, bc_top_batch
        )
        terms = {'pde': pde_loss, 'ic': ic_loss, 'bc': bc_loss}
        total = total_loss(terms, weights_dict)
        return total, terms

    (loss_val, term_vals), grads = jax.value_and_grad(loss_and_stats, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, term_vals

# JIT-compile the training step as a function call
train_step_jitted = jax.jit(
    train_step,
    # Mark non-array arguments as static to avoid recompilation
    static_argnames=('model', 'optimiser')
)

def main():
    """Main training loop for the PINN."""
    cfg = config  # Use a shorter alias
    key = random.PRNGKey(cfg["training"]["seed"])
    model, params = init_model(key)

    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    
    # Paths are relative to the project root
    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Point to a single, centralized repository for all runs.
    central_aim_dir = "aim_repo" 
    aim_repo = Repo(path=central_aim_dir, init=True)
    aim_run = Run(repo=aim_repo, experiment=trial_name)
    aim_run["hparams"] = cfg

    # --- CORRECTED OPTIMIZER SETUP ---
    # Re-introduce the learning rate scheduler and gradient clipping
    lr = cfg["training"]["learning_rate"]
    
    # Using boundaries similar to your original code
    lr_boundaries = {15000: 0.1, 30000: 0.1} 
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=lr,
        boundaries_and_scales=lr_boundaries
    )

    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    # --- END OF CORRECTION ---
    
    opt_state = optimiser.init(params)
    weights_dict = {
        'pde': cfg["loss_weights"]["pde_weight"],
        'ic': cfg["loss_weights"]["ic_weight"],
        'bc': cfg["loss_weights"]["bc_weight"]
    }

    print("Training started.")
    best_nse: float = -jnp.inf
    best_epoch: int = 0
    best_params: Dict = None
    start_time = time.time()
    
    try:
        for epoch in range(cfg["training"]["epochs"]):
            key, pde_key, ic_key, l_key, r_key, b_key, t_key = random.split(key, 7)
            
            # --- Data Sampling ---
            pde_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["grid"]["nx"], cfg["grid"]["ny"], cfg["grid"]["nt"], pde_key)
            ic_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., 0., cfg["ic_bc_grid"]["nx_ic"], cfg["ic_bc_grid"]["ny_ic"], 1, ic_key)
            left_wall = sample_points(0., 0., 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_left"], cfg["ic_bc_grid"]["nt_bc_left"], l_key)
            right_wall = sample_points(cfg["domain"]["lx"], cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_right"], cfg["ic_bc_grid"]["nt_bc_right"], r_key)
            bottom_wall = sample_points(0., cfg["domain"]["lx"], 0., 0., 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_bottom"], 1, cfg["ic_bc_grid"]["nt_bc_other"], b_key)
            top_wall = sample_points(0., cfg["domain"]["lx"], cfg["domain"]["ly"], cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_top"], 1, cfg["ic_bc_grid"]["nt_bc_other"], t_key)

            # --- Batching ---
            batch_size = cfg["training"]["batch_size"]
            key, pde_b_key, ic_b_key, l_b_key, r_b_key, b_b_key, t_b_key = random.split(key, 7)
            pde_batches = get_batches(pde_b_key, pde_points, batch_size)
            ic_batches = get_batches(ic_b_key, ic_points, batch_size)
            left_batches = get_batches(l_b_key, left_wall, batch_size)
            right_batches = get_batches(r_b_key, right_wall, batch_size)
            bottom_batches = get_batches(b_b_key, bottom_wall, batch_size)
            top_batches = get_batches(t_b_key, top_wall, batch_size)
            num_batches = len(pde_batches)
            
            epoch_losses = {'pde': 0.0, 'ic': 0.0, 'bc': 0.0}
            for i in range(num_batches):
                params, opt_state, batch_losses = train_step_jitted(
                    model, params, opt_state,
                    pde_batches[i],
                    ic_batches[i % len(ic_batches)],
                    left_batches[i % len(left_batches)],
                    right_batches[i % len(right_batches)],
                    bottom_batches[i % len(bottom_batches)],
                    top_batches[i % len(top_batches)],
                    weights_dict, optimiser
                )
                for k in epoch_losses: epoch_losses[k] += batch_losses[k]

            # --- Validation & Logging ---
            avg_pde_loss = float(epoch_losses['pde'] / num_batches)
            avg_ic_loss = float(epoch_losses['ic'] / num_batches)
            avg_bc_loss = float(epoch_losses['bc'] / num_batches)
            avg_total_loss = float(total_loss({'pde': avg_pde_loss, 'ic': avg_ic_loss, 'bc': avg_bc_loss}, weights_dict))
            
            with jax.disable_jit():
                U_pred_val = model.apply({'params': params['params']}, pde_points, train=False)
                h_pred_val = U_pred_val[..., 0]
                h_true_val = h_exact(pde_points[:, 0], pde_points[:, 2])
                nse_val = float(nse(h_pred_val, h_true_val))
                rmse_val = float(rmse(h_pred_val, h_true_val))

            if nse_val > best_nse:
                best_nse, best_epoch, best_params = nse_val, epoch, copy.deepcopy(params)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:5d} | Loss: {avg_total_loss:.4e} | NSE: {nse_val:.4f} | RMSE: {rmse_val:.4f}")

            aim_run.track(avg_total_loss, name='total_loss', step=epoch, context={'subset': 'train'})
            aim_run.track(avg_pde_loss, name='pde_loss', step=epoch, context={'subset': 'train'})
            aim_run.track(nse_val, name='nse', step=epoch, context={'subset': 'validation'})

            # --- Early Stopping ---
            if epoch > cfg["device"]["early_stop_min_epochs"] and (epoch - best_epoch) > cfg["device"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch+1}.")
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        aim_run.close()
        print(f"Training ended. Total time: {time.time() - start_time:.2f} seconds.")
        if best_params is not None:
            save_model(best_params, model_dir, trial_name)
            print(f"Best model from epoch {best_epoch+1} saved with NSE {best_nse:.6f}.")
            
            print("Generating final validation plot...")
            x_val = jnp.linspace(0.0, cfg["domain"]["lx"], cfg["plotting"]["nx_val"])
            pts_val = jnp.stack([
                x_val, 
                jnp.full_like(x_val, cfg["plotting"]["y_const_plot"]), 
                jnp.full_like(x_val, cfg["plotting"]["t_const_val"])
            ], axis=1)
            U_val_pred = model.apply({'params': best_params['params']}, pts_val, train=False)
            plot_path = os.path.join(results_dir, "final_validation_plot.png")
            plot_h_vs_x(x_val, U_val_pred[..., 0], cfg["plotting"]["t_const_val"], cfg["plotting"]["y_const_plot"], plot_path)
        else:
            print("Warning: No best model found or saved.")

if __name__ == "__main__":
    main()