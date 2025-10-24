# src/train_test.py
import os
import time
import copy
import argparse
import importlib
import itertools
from typing import Any, Dict, Tuple
import shutil

import jax
import jax.numpy as jnp
from jax import random
import optax
from aim import Repo, Run
from flax.core import FrozenDict

from src.config import load_config
from src.data import sample_points, get_batches
from src.models import init_model
from src.losses import compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss, compute_building_bc_loss
from src.utils import nse, rmse, generate_trial_name, save_model, plot_h_vs_x, ask_for_confirmation, plot_h_2d_top_view, mask_points_inside_building
from src.physics import h_exact
from src.reporting import print_epoch_stats, log_metrics, print_final_summary


def train_step(model: Any, params: Dict[str, Any], opt_state: Any,
               pde_batch: jnp.ndarray, ic_batch: jnp.ndarray,
               bc_left_batch: jnp.ndarray, bc_right_batch: jnp.ndarray,
               bc_bottom_batch: jnp.ndarray, bc_top_batch: jnp.ndarray,
               building_batches: Dict[str, jnp.ndarray],
               weights_dict: Dict[str, float],
               optimiser: optax.GradientTransformation,
               config: FrozenDict
               ) -> Tuple[Any, Any, Dict[str, jnp.ndarray]]:
    """Perform a single training step for the PINN model."""
    def loss_and_stats(p):
        pde_loss = compute_pde_loss(model, p, pde_batch, config)
        ic_loss = compute_ic_loss(model, p, ic_batch)
        bc_loss = compute_bc_loss(
            model, p, bc_left_batch, bc_right_batch, bc_bottom_batch, bc_top_batch, config
        )
        terms = {'pde': pde_loss, 'ic': ic_loss, 'bc': bc_loss}

        if "building" in config and building_batches: # Ensure building batches are provided
            building_loss = compute_building_bc_loss(
                model, p,
                building_batches['left'],
                building_batches['right'],
                building_batches['bottom'],
                building_batches['top']
            )
            terms['building_bc'] = building_loss

        total = total_loss(terms, weights_dict)
        return total, terms

    (loss_val, term_vals), grads = jax.value_and_grad(loss_and_stats, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, term_vals

train_step_jitted = jax.jit(
    train_step,
    static_argnames=('model', 'optimiser', 'config')
)

# --- MODIFIED: Added run_for_hpo flag ---
def main(config_path: str, run_for_hpo: bool = False):
    """Main training loop for the PINN."""
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)
    has_building = "building" in cfg

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e

    key = random.PRNGKey(cfg["training"]["seed"])
    model, params = init_model(model_class, key, cfg)

    # Use a unique name based on config + timestamp, suitable for HPO trials
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)

    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    # Only create dirs if not running for HPO (to avoid clutter)
    if not run_for_hpo:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    # Initialize Aim Run
    aim_repo = None
    aim_run = None
    run_hash = None
    try:
        aim_repo = Repo(path="aim_repo", init=True)
        aim_run = Run(repo=aim_repo, experiment=trial_name)
        run_hash = aim_run.hash
        # Log hyperparameters (config dictionary)
        aim_run["hparams"] = cfg_dict
    except Exception as e:
        print(f"Warning: Failed to initialize Aim tracking: {e}")
        # Allow training to continue without Aim

    # Optimizer setup
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=cfg["training"]["learning_rate"],
        boundaries_and_scales={15000: 0.1, 30000: 0.1} # Consider making these configurable
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimiser.init(params)
    weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}

    # --- Prepare validation data once ---
    val_points, h_true_val = None, None
    if has_building:
        validation_data_path = os.path.join("data", cfg.get('scenario', 'default_scenario'), "validation_sample.npy")
        if os.path.exists(validation_data_path):
            try:
                validation_data = jnp.load(validation_data_path)
                val_points = validation_data[:, :3]  # x, y, t
                h_true_val = validation_data[:, 3]

                # Mask validation points outside the loop
                mask = mask_points_inside_building(val_points, cfg["building"])
                val_points = val_points[mask]
                h_true_val = h_true_val[mask]
                print(f"Loaded and masked validation data: {val_points.shape[0]} points.")
            except Exception as e:
                print(f"Error loading validation file {validation_data_path}: {e}")
                val_points, h_true_val = None, None # Ensure validation is skipped
        else:
            print(f"Warning: Validation file not found at {validation_data_path}. Validation metrics (NSE/RMSE) will be skipped for building case.")
            val_points, h_true_val = None, None # Ensure validation is skipped

    print(f"\n--- Training Started: {trial_name} ---")
    print(f"Model: {cfg['model']['name']}, Epochs: {cfg['training']['epochs']}, Batch Size: {cfg['training']['batch_size']}")
    best_nse: float = -jnp.inf
    best_epoch: int = 0
    best_params: Dict = None
    best_nse_time: float = 0.0
    start_time = time.time()

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()
            # --- Dynamic Sampling in each epoch ---
            key, pde_key, ic_key, l_key, r_key, b_key, t_key = random.split(key, 7)

            building_keys = {}
            if has_building:
                key, bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(key, 5)
                building_keys = {'left': bldg_l_key, 'right': bldg_r_key, 'bottom': bldg_b_key, 'top': bldg_t_key}

            # Sample points for PDE, IC, and Boundaries
            pde_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["grid"]["nx"], cfg["grid"]["ny"], cfg["grid"]["nt"], pde_key)
            ic_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., 0., cfg["ic_bc_grid"]["nx_ic"], cfg["ic_bc_grid"]["ny_ic"], 1, ic_key)
            left_wall = sample_points(0., 0., 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_left"], cfg["ic_bc_grid"]["nt_bc_left"], l_key)
            right_wall = sample_points(cfg["domain"]["lx"], cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_right"], cfg["ic_bc_grid"]["nt_bc_right"], r_key)
            bottom_wall = sample_points(0., cfg["domain"]["lx"], 0., 0., 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_bottom"], 1, cfg["ic_bc_grid"]["nt_bc_other"], b_key)
            top_wall = sample_points(0., cfg["domain"]["lx"], cfg["domain"]["ly"], cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_top"], 1, cfg["ic_bc_grid"]["nt_bc_other"], t_key)

            # Sample points for Building (if present)
            building_points = {}
            if has_building:
                b_cfg = cfg["building"]
                building_points['left'] = sample_points(b_cfg["x_min"], b_cfg["x_min"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], building_keys['left'])
                building_points['right'] = sample_points(b_cfg["x_max"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], building_keys['right'])
                building_points['bottom'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_min"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], building_keys['bottom'])
                building_points['top'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_max"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], building_keys['top'])

            # --- Create Batches ---
            batch_size = cfg["training"]["batch_size"]
            key, pde_b_key, ic_b_key, l_b_key, r_b_key, b_b_key, t_b_key = random.split(key, 7)

            building_batch_keys = {}
            if has_building:
                key, bldg_l_b_key, bldg_r_b_key, bldg_b_b_key, bldg_t_b_key = random.split(key, 5)
                building_batch_keys = {'left': bldg_l_b_key, 'right': bldg_r_b_key, 'bottom': bldg_b_b_key, 'top': bldg_t_b_key}

            # Get shuffled batches for each set of points
            pde_batches = get_batches(pde_b_key, pde_points, batch_size)
            ic_batches = get_batches(ic_b_key, ic_points, batch_size)
            left_batches = get_batches(l_b_key, left_wall, batch_size)
            right_batches = get_batches(r_b_key, right_wall, batch_size)
            bottom_batches = get_batches(b_b_key, bottom_wall, batch_size)
            top_batches = get_batches(t_b_key, top_wall, batch_size)

            building_batches_dict = {} # Renamed to avoid conflict
            if has_building:
                for wall in ['left', 'right', 'bottom', 'top']:
                    building_batches_dict[wall] = get_batches(building_batch_keys[wall], building_points[wall], batch_size)

            # Use iterators to cycle through smaller batch lists (IC, BC)
            ic_batch_iter = itertools.cycle(ic_batches) if ic_batches else iter(())
            left_batch_iter = itertools.cycle(left_batches) if left_batches else iter(())
            right_batch_iter = itertools.cycle(right_batches) if right_batches else iter(())
            bottom_batch_iter = itertools.cycle(bottom_batches) if bottom_batches else iter(())
            top_batch_iter = itertools.cycle(top_batches) if top_batches else iter(())

            building_batch_iters = {}
            if has_building:
                for wall, batches in building_batches_dict.items():
                    building_batch_iters[wall] = itertools.cycle(batches) if batches else iter(())

            # --- Training Step Loop ---
            num_batches = len(pde_batches)
            epoch_losses = {'pde': 0.0, 'ic': 0.0, 'bc': 0.0}
            if has_building:
                epoch_losses['building_bc'] = 0.0

            for i in range(num_batches):
                current_building_batch_data = {} # Renamed
                if has_building:
                    for wall, iterator in building_batch_iters.items():
                        try:
                           current_building_batch_data[wall] = next(iterator)
                        except StopIteration:
                             # Handle case where building batches might be empty if sampling failed
                             current_building_batch_data[wall] = jnp.array([]) # Or handle appropriately

                # Ensure iterators provide data, handle potential StopIteration if lists are empty
                try:
                    ic_batch_data = next(ic_batch_iter)
                    left_batch_data = next(left_batch_iter)
                    right_batch_data = next(right_batch_iter)
                    bottom_batch_data = next(bottom_batch_iter)
                    top_batch_data = next(top_batch_iter)
                except StopIteration:
                    print(f"Warning: Ran out of IC/BC batches in epoch {epoch}, batch {i}. Check sampling sizes.")
                    # Decide how to handle: skip batch, use last batch, or break epoch?
                    # For simplicity, we might skip the batch or use zeros, but this indicates a potential config issue.
                    continue # Skipping batch for now


                params, opt_state, batch_losses = train_step_jitted(
                    model, params, opt_state,
                    pde_batches[i],
                    ic_batch_data,
                    left_batch_data,
                    right_batch_data,
                    bottom_batch_data,
                    top_batch_data,
                    current_building_batch_data, # Pass the dict of building batches
                    weights_dict, optimiser, cfg
                )
                for k in epoch_losses:
                    epoch_losses[k] += float(batch_losses.get(k, 0.0)) # Ensure float accumulation

            # --- Calculate Average Losses and Validation Metrics ---
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            avg_total_loss = float(total_loss(avg_losses, weights_dict))

            nse_val, rmse_val = -jnp.inf, jnp.inf # Default values if validation skipped

            # Use disable_jit context for potentially non-jittable validation logic
            with jax.disable_jit():
                if has_building:
                    if val_points is not None and h_true_val is not None:
                        # Ensure val_points is not empty before applying model
                        if val_points.shape[0] > 0:
                             U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                             h_pred_val = U_pred_val[..., 0]
                             nse_val = float(nse(h_pred_val, h_true_val))
                             rmse_val = float(rmse(h_pred_val, h_true_val))
                        else:
                            print(f"Warning: Validation points array is empty after masking in epoch {epoch}.")
                    # else: validation skipped due to missing file or load error
                else: # No building case - validate against analytical solution using PDE points
                     if pde_points.shape[0] > 0:
                         U_pred_val_no_building = model.apply({'params': params['params']}, pde_points, train=False)
                         h_pred_val_no_building = U_pred_val_no_building[..., 0]
                         h_true_val_no_building = h_exact(pde_points[:, 0], pde_points[:, 2], cfg["physics"]["n_manning"], cfg["physics"]["u_const"])
                         # Add safety check for denominator in NSE
                         if jnp.sum((h_true_val_no_building - jnp.mean(h_true_val_no_building))**2) > 1e-9:
                             nse_val = float(nse(h_pred_val_no_building, h_true_val_no_building))
                         else:
                             nse_val = -jnp.inf # Avoid division by zero
                         rmse_val = float(rmse(h_pred_val_no_building, h_true_val_no_building))
                     else:
                          print(f"Warning: PDE points array is empty in epoch {epoch}.")


            # Update best NSE score and model parameters
            if nse_val > best_nse:
                best_nse = nse_val
                best_epoch = epoch
                best_params = copy.deepcopy(params) # Deep copy to save the state
                best_nse_time = time.time() - start_time

            # --- Logging and Reporting ---
            epoch_time = time.time() - epoch_start_time
            # Log less frequently to reduce overhead, e.g., every 10 or 100 epochs
            if (epoch + 1) % 100 == 0:
                print_epoch_stats(
                    epoch, start_time, avg_total_loss,
                    avg_losses['pde'], avg_losses['ic'], avg_losses['bc'],
                    avg_losses.get('building_bc', 0.0), # Add building BC loss if present
                    nse_val, rmse_val, epoch_time # Pass epoch time
                )

            # Log metrics to Aim if initialized
            if aim_run:
                try:
                    log_metrics(aim_run, {
                        'total_loss': avg_total_loss, 'pde_loss': avg_losses['pde'],
                        'ic_loss': avg_losses['ic'], 'bc_loss': avg_losses['bc'],
                        'building_bc_loss': avg_losses.get('building_bc', 0.0),
                        'nse': nse_val, 'rmse': rmse_val, 'epoch_time': epoch_time
                    }, epoch)
                except Exception as e:
                    print(f"Warning: Failed to log metrics to Aim in epoch {epoch}: {e}")

            # --- Early Stopping Check ---
            # Ensure keys exist before checking
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))

            if epoch > min_epochs and (epoch - best_epoch) > patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                print(f"Best NSE {best_nse:.6f} achieved at epoch {best_epoch+1}.")
                break

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\n--- An error occurred during training loop: {e} ---")
        import traceback
        traceback.print_exc()
        # Optionally re-raise or handle cleanup

    # --- Final Summary and Optional Saving ---
    finally:
        if aim_run:
            try:
                aim_run.close()
            except Exception as e:
                 print(f"Warning: Error closing Aim run: {e}")

        total_time = time.time() - start_time
        print_final_summary(total_time, best_epoch, best_nse, best_nse_time)

        # --- MODIFIED: Conditional Saving/Plotting Block ---
        if not run_for_hpo: # Only run this if not called from HPO
            # Check if directories were created before asking to save
            if os.path.exists(results_dir) and os.path.exists(model_dir):
                if ask_for_confirmation():
                    if best_params is not None:
                        try:
                            save_model(best_params, model_dir, trial_name)

                            # Plotting logic (only if saving confirmed)
                            if has_building:
                                resolution = cfg.get("plotting", {}).get("plot_resolution", 100) # Default resolution
                                x = jnp.linspace(0, cfg["domain"]["lx"], resolution)
                                y = jnp.linspace(0, cfg["domain"]["ly"], resolution)
                                xx, yy = jnp.meshgrid(x, y)
                                t_const_val = cfg.get("plotting", {}).get("t_const_val", cfg["domain"]["t_final"] / 2.0)
                                t = jnp.full_like(xx, t_const_val)
                                pts_2d = jnp.stack([xx.ravel(), yy.ravel(), t.ravel()], axis=-1)

                                U_val_pred_2d = model.apply({'params': best_params['params']}, pts_2d, train=False)
                                h_val_pred_2d = U_val_pred_2d[..., 0].reshape(resolution, resolution)
                                eps_plot = cfg.get("numerics", {}).get("eps", 1e-6)
                                h_val_pred_2d = jnp.where(h_val_pred_2d < eps_plot, 0.0, h_val_pred_2d)

                                plot_path_2d = os.path.join(results_dir, "final_2d_top_view.png")
                                plot_h_2d_top_view(xx, yy, h_val_pred_2d, cfg, plot_path_2d)
                            else:
                                nx_val_plot = cfg.get("plotting", {}).get("nx_val", 101)
                                y_const_plot = cfg.get("plotting", {}).get("y_const_plot", 0.0)
                                t_const_val_plot = cfg.get("plotting", {}).get("t_const_val", cfg["domain"]["t_final"] / 2.0)
                                x_val = jnp.linspace(0.0, cfg["domain"]["lx"], nx_val_plot)
                                pts_val = jnp.stack([x_val, jnp.full_like(x_val, y_const_plot), jnp.full_like(x_val, t_const_val_plot)], axis=1)
                                U_val_pred = model.apply({'params': best_params['params']}, pts_val, train=False)
                                h_val_pred = U_val_pred[..., 0]
                                eps_plot = cfg.get("numerics", {}).get("eps", 1e-6)
                                h_val_pred = jnp.where(h_val_pred < eps_plot, 0.0, h_val_pred)
                                plot_path = os.path.join(results_dir, "final_validation_plot.png")
                                plot_h_vs_x(x_val, h_val_pred, t_const_val_plot, y_const_plot, cfg, plot_path)

                            print("Artifacts saved.")
                        except Exception as e:
                             print(f"Error during saving/plotting: {e}")
                    else:
                        print("Warning: No best model found (best_params is None). Skipping save.")
                else:
                    print("Save aborted by user. Deleting artifacts...")
                    try:
                        if aim_repo and run_hash:
                            aim_repo.delete_run(run_hash)
                        if os.path.exists(results_dir): shutil.rmtree(results_dir)
                        if os.path.exists(model_dir): shutil.rmtree(model_dir)
                        print("All artifacts successfully deleted.")
                    except Exception as e:
                        print(f"Error during cleanup: {e}")
            else:
                 print("Results/Model directories do not exist, skipping save confirmation.")
        # --- END MODIFIED Conditional Block ---

    # --- MODIFIED: Ensure best_nse is always returned, even if training failed or was interrupted ---
    # Return -inf if best_nse wasn't updated (e.g., failed immediately)
    return best_nse if best_nse > -jnp.inf else -1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a PINN model for SWE.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., experiments/one_building_config.yaml)")
    # Add an optional flag for HPO runs, although the hyper_optimizer script will call main directly
    parser.add_argument("--hpo", action="store_true", help="Run in HPO mode (skips saving prompts/plots).")
    args = parser.parse_args()

    try:
        # Pass the hpo flag to main
        final_nse = main(args.config, run_for_hpo=args.hpo)
        print(f"\nScript finished. Final best NSE reported: {final_nse:.6f}")
    except FileNotFoundError as e:
         print(f"Error: {e}. Please check the config file path.")
    except ValueError as e:
         print(f"Configuration or Model Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
        import traceback
        traceback.print_exc()