# src/train.py
# Unified training script handling scenarios with and without buildings.

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
from aim import Repo, Run # Keep AIM imports, handle potential errors
from flax.core import FrozenDict
import numpy as np # Import numpy for loading

# --- Use DTYPE from config ---
from src.config import load_config, DTYPE
from src.data import sample_points, get_batches
from src.models import init_model
# --- Updated losses import ---
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_building_bc_loss, compute_data_loss # Added compute_data_loss
)
# --- END ---
# --- Re-added imports for no-building case ---
from src.utils import ( # Updated utils import
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation,
    # plot_h_2d_top_view, # <<<--- REMOVED this import
    mask_points_inside_building, plot_h_vs_x,
    plot_h_prediction_vs_true_2d # <<<--- Keep the stacked plot function
)
from src.physics import h_exact
# --- END ---
from src.reporting import print_epoch_stats, log_metrics, print_final_summary


def train_step(model: Any, params: Dict[str, Any], opt_state: Any,
               pde_batch: jnp.ndarray, ic_batch: jnp.ndarray,
               bc_left_batch: jnp.ndarray, bc_right_batch: jnp.ndarray,
               bc_bottom_batch: jnp.ndarray, bc_top_batch: jnp.ndarray,
               building_batches: Dict[str, jnp.ndarray], # Can be empty if no building
               data_batch: jnp.ndarray, # Can be empty if no data loss
               weights_dict: Dict[str, float],
               optimiser: optax.GradientTransformation,
               config: FrozenDict
               ) -> Tuple[Any, Any, Dict[str, jnp.ndarray]]:
    """Perform a single training step for the PINN model."""
    has_building = "building" in config # Check if building is configured

    def loss_and_stats(p):
        terms = {}
        # Ensure batches are not empty before computing loss
        if pde_batch.shape[0] > 0:
            terms['pde'] = compute_pde_loss(model, p, pde_batch, config)
        if ic_batch.shape[0] > 0:
            terms['ic'] = compute_ic_loss(model, p, ic_batch)
        if (bc_left_batch.shape[0] > 0 or bc_right_batch.shape[0] > 0 or
            bc_bottom_batch.shape[0] > 0 or bc_top_batch.shape[0] > 0):
             terms['bc'] = compute_bc_loss(
                 model, p, bc_left_batch, bc_right_batch, bc_bottom_batch, bc_top_batch, config
             )

        # Compute building loss conditionally
        if has_building and building_batches and all(b.shape[0] > 0 for b in building_batches.values()):
             terms['building_bc'] = compute_building_bc_loss(
                 model, p,
                 building_batches.get('left', jnp.empty((0,3), dtype=DTYPE)),
                 building_batches.get('right', jnp.empty((0,3), dtype=DTYPE)),
                 building_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                 building_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
             )

        # Compute data loss if data_batch is provided and not empty
        if data_batch is not None and data_batch.shape[0] > 0:
             terms['data'] = compute_data_loss(model, p, data_batch, config)

        # Calculate total loss, handling potentially missing terms
        terms_with_defaults = {
            'pde': terms.get('pde', 0.0),
            'ic': terms.get('ic', 0.0),
            'bc': terms.get('bc', 0.0),
            'building_bc': terms.get('building_bc', 0.0),
            'data': terms.get('data', 0.0)
        }
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms # Return original terms dict for logging

    (loss_val, term_vals), grads = jax.value_and_grad(loss_and_stats, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, term_vals

# Jit the training step for performance
train_step_jitted = jax.jit(
    train_step,
    static_argnames=('model', 'optimiser', 'config')
)

def main(config_path: str):
    """Main training loop for the PINN (Handles both building and no-building scenarios)."""
    cfg_dict = load_config(config_path) # DTYPE is set globally here
    cfg = FrozenDict(cfg_dict)
    has_building = "building" in cfg # Determine if building is present

    # --- Scenario-specific warnings ---
    if has_building and "scenario" not in cfg:
        print("Warning: 'building' section found, but 'scenario' key is missing. Data paths might be incorrect.")
    if not has_building:
        print("Info: No 'building' section found in config. Running in no-building mode.")
        if cfg.get("loss_weights", {}).get("building_bc_weight", 0.0) > 0:
            print("Warning: 'building_bc_weight' > 0 but no 'building' section in config. Building BC loss will not be calculated.")
        # Removed warning about data_weight as it will now use training_dataset_sample.npy

    # --- Model Initialization ---
    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e

    key = random.PRNGKey(cfg["training"]["seed"])
    model, params = init_model(model_class, key, cfg)

    # --- Setup Directories and Tracking ---
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)

    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize Aim Run (with error handling)
    aim_repo = None
    aim_run = None
    run_hash = None
    try:
        aim_repo_path = "aim_repo"
        if not os.path.exists(aim_repo_path):
             os.makedirs(aim_repo_path, exist_ok=True)
        aim_repo = Repo(path=aim_repo_path, init=True)
        aim_run = Run(repo=aim_repo, experiment=trial_name)
        run_hash = aim_run.hash
        aim_run["hparams"] = cfg_dict # Log hyperparameters
        print(f"Aim tracking initialized for run: {trial_name} ({run_hash})")
    except Exception as e:
        print(f"Warning: Failed to initialize Aim tracking: {e}. Training will continue without Aim.")

    # --- Optimizer Setup ---
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=cfg["training"]["learning_rate"],
        boundaries_and_scales={15000: 0.1, 30000: 0.1}
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimiser.init(params)

    # Prepare loss weights dictionary
    weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    has_data_loss = 'data' in weights_dict and weights_dict['data'] > 0

    # --- Load Data (Separate Training and Validation) ---
    val_points, h_true_val = None, None # For validation metrics
    data_points_full = None # For training data loss term
    scenario_name = cfg.get('scenario', 'default_scenario') # Needed for data paths
    base_data_path = os.path.join("data", scenario_name)

    # --- Load Training Data (for data loss term) ---
    training_data_file = os.path.join(base_data_path, "training_dataset_sample.npy")
    if has_data_loss:
        if os.path.exists(training_data_file):
            try:
                print(f"Loading TRAINING data from: {training_data_file}")
                data_points_full = jnp.load(training_data_file).astype(DTYPE) # Shape (N, 6): [t, x, y, h, u, v]
                print(f"Using {data_points_full.shape[0]} points for data loss term (weight={weights_dict['data']:.2e}).")
            except Exception as e:
                print(f"Error loading training data file {training_data_file}: {e}")
                print("Disabling data loss term due to loading error.")
                data_points_full = None
                has_data_loss = False
        else:
            print(f"Warning: Training data file not found at {training_data_file}.")
            print("Data loss term cannot be computed and will be disabled.")
            has_data_loss = False

    # --- Load Validation Data (for NSE/RMSE metrics) ---
    validation_data_file = os.path.join(base_data_path, "validation_sample.npy")
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            loaded_val_data = jnp.load(validation_data_file).astype(DTYPE)

            if has_building:
                val_points_all = loaded_val_data[:, [1, 2, 0]] # Input points (x, y, t)
                h_true_val_all = loaded_val_data[:, 3]       # True water depth h
                print("Applying building mask to validation metrics points...")
                mask_val = mask_points_inside_building(val_points_all, cfg["building"])
                val_points = val_points_all[mask_val]
                h_true_val = h_true_val_all[mask_val]
                num_masked_val_points = val_points.shape[0]
                print(f"Masked validation metrics points remaining: {num_masked_val_points}.")
                if num_masked_val_points == 0:
                     print("Warning: No validation points remaining after masking. NSE/RMSE calculation will be skipped for building case.")
        except Exception as e:
            print(f"Error loading or processing validation data file {validation_data_file}: {e}")
            val_points, h_true_val = None, None
            if has_building:
                print("NSE/RMSE calculation for building scenario will be skipped due to validation data loading error.")
    else:
        print(f"Warning: Validation data file not found at {validation_data_file}.")
        if has_building:
            print("Validation metrics (NSE/RMSE) for building scenario will be skipped.")
        else:
            print("Validation metrics (NSE/RMSE) will use analytical solution (if PDE points exist).")
    # --- End Data Loading ---


    # --- Training Initialization ---
    print(f"\n--- Training Started: {trial_name} ---")
    print(f"Model: {cfg['model']['name']}, Epochs: {cfg['training']['epochs']}, Batch Size: {cfg['training']['batch_size']}")
    print(f"Scenario: {'Building' if has_building else 'No Building'}")
    print(f"Saving results to: {results_dir}")
    print(f"Saving model to: {model_dir}")

    best_nse: float = -jnp.inf
    best_epoch: int = 0
    best_params: Dict = None
    best_nse_time: float = 0.0
    start_time = time.time()

    # --- Main Training Loop ---
    # ... (Keep the training loop as it was in the previous response) ...
    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()

            # --- Dynamic Sampling ---
            key, pde_key, ic_key, bc_keys, bldg_keys, data_key = random.split(key, 6)
            l_key, r_key, b_key, t_key = random.split(bc_keys, 4)

            # Sample points for PDE, IC, Domain BCs
            pde_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["grid"]["nx"], cfg["grid"]["ny"], cfg["grid"]["nt"], pde_key)
            ic_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., 0., cfg["ic_bc_grid"]["nx_ic"], cfg["ic_bc_grid"]["ny_ic"], 1, ic_key)
            left_wall = sample_points(0., 0., 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_left"], cfg["ic_bc_grid"]["nt_bc_left"], l_key)
            right_wall = sample_points(cfg["domain"]["lx"], cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_right"], cfg["ic_bc_grid"]["nt_bc_right"], r_key)
            bottom_wall = sample_points(0., cfg["domain"]["lx"], 0., 0., 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_bottom"], 1, cfg["ic_bc_grid"]["nt_bc_other"], b_key)
            top_wall = sample_points(0., cfg["domain"]["lx"], cfg["domain"]["ly"], cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_top"], 1, cfg["ic_bc_grid"]["nt_bc_other"], t_key)

            # Sample points for Building BCs only if building exists
            building_points = {}
            if has_building:
                bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)
                b_cfg = cfg["building"]
                building_points['left'] = sample_points(b_cfg["x_min"], b_cfg["x_min"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_l_key)
                building_points['right'] = sample_points(b_cfg["x_max"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_r_key)
                building_points['bottom'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_min"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_b_key)
                building_points['top'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_max"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_t_key)

            # --- Create Batches ---
            batch_size = cfg["training"]["batch_size"]
            key, pde_b_key, ic_b_key, bc_b_keys, bldg_b_keys, data_b_key = random.split(key, 6)
            l_b_key, r_b_key, b_b_key, t_b_key = random.split(bc_b_keys, 4)

            pde_batches = get_batches(pde_b_key, pde_points, batch_size)
            ic_batches = get_batches(ic_b_key, ic_points, batch_size)
            left_batches = get_batches(l_b_key, left_wall, batch_size)
            right_batches = get_batches(r_b_key, right_wall, batch_size)
            bottom_batches = get_batches(b_b_key, bottom_wall, batch_size)
            top_batches = get_batches(t_b_key, top_wall, batch_size)

            # Get data batches if enabled and data is available
            data_batches = []
            if has_data_loss and data_points_full is not None and data_points_full.shape[0] > 0:
                 data_batches = get_batches(data_b_key, data_points_full, batch_size)

            # Create building batches only if building exists
            building_batches_dict = {}
            if has_building:
                bldg_l_b_key, bldg_r_b_key, bldg_b_b_key, bldg_t_b_key = random.split(bldg_b_keys, 4)
                building_b_keys_map = {'left': bldg_l_b_key, 'right': bldg_r_b_key, 'bottom': bldg_b_b_key, 'top': bldg_t_b_key}
                for wall, points in building_points.items():
                    if points.shape[0] > 0:
                        building_batches_dict[wall] = get_batches(building_b_keys_map[wall], points, batch_size)
                    else:
                        building_batches_dict[wall] = []

            # --- Batch Iterators ---
            ic_batch_iter = itertools.cycle(ic_batches) if ic_batches else iter(())
            left_batch_iter = itertools.cycle(left_batches) if left_batches else iter(())
            right_batch_iter = itertools.cycle(right_batches) if right_batches else iter(())
            bottom_batch_iter = itertools.cycle(bottom_batches) if bottom_batches else iter(())
            top_batch_iter = itertools.cycle(top_batches) if top_batches else iter(())
            data_batch_iter = itertools.cycle(data_batches) if data_batches else iter(())

            building_batch_iters = {}
            if has_building:
                for wall, batches in building_batches_dict.items():
                    building_batch_iters[wall] = itertools.cycle(batches) if batches else iter(())

            # --- Training Steps within Epoch ---
            num_batches = len(pde_batches)
            epoch_losses = {'pde': 0.0, 'ic': 0.0, 'bc': 0.0, 'building_bc': 0.0, 'data': 0.0}

            if num_batches == 0:
                print(f"Warning: Epoch {epoch+1} - No PDE batches generated. Skipping epoch.")
                continue

            for i in range(num_batches):
                # Get next batch, providing empty array if iterator is exhausted/empty
                ic_batch_data = next(ic_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                left_batch_data = next(left_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                right_batch_data = next(right_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                bottom_batch_data = next(bottom_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                top_batch_data = next(top_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                data_batch_data = next(data_batch_iter, jnp.empty((0, 6), dtype=DTYPE)) # Data batch has 6 columns

                current_building_batch_data = {}
                if has_building:
                    for wall, iterator in building_batch_iters.items():
                        current_building_batch_data[wall] = next(iterator, jnp.empty((0, 3), dtype=DTYPE))

                # Perform the training step (pass empty dict if no building)
                params, opt_state, batch_losses = train_step_jitted(
                    model, params, opt_state,
                    pde_batches[i],
                    ic_batch_data,
                    left_batch_data,
                    right_batch_data,
                    bottom_batch_data,
                    top_batch_data,
                    current_building_batch_data if has_building else {}, # Pass empty dict if no building
                    data_batch_data, # Pass the data batch
                    weights_dict, optimiser, cfg
                )

                # Accumulate losses
                for k in epoch_losses:
                    epoch_losses[k] += float(batch_losses.get(k, 0.0))

            # --- Epoch End Calculation & Validation ---
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            avg_total_loss = float(total_loss(avg_losses, weights_dict))

            nse_val, rmse_val = -jnp.inf, jnp.inf
            with jax.disable_jit():
                if has_building:
                    # Validate using masked validation data if available
                    if val_points is not None and h_true_val is not None and val_points.shape[0] > 0:
                         U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                         h_pred_val = U_pred_val[..., 0]
                         h_true_mean = jnp.mean(h_true_val)
                         denominator_nse = jnp.sum((h_true_val - h_true_mean)**2)
                         if denominator_nse > cfg.get("numerics", {}).get("eps", 1e-9):
                              numerator_nse = jnp.sum((h_true_val - h_pred_val)**2)
                              nse_val = float(1.0 - numerator_nse / denominator_nse)
                         rmse_val = float(rmse(h_pred_val, h_true_val))
                    # else: Validation skipped for building case if no valid data
                else:
                    # Validate against analytical solution for no-building case
                    if pde_points.shape[0] > 0: # Check if PDE points were sampled
                        U_pred_val_no_building = model.apply({'params': params['params']}, pde_points, train=False)
                        h_pred_val_no_building = U_pred_val_no_building[..., 0]
                        h_true_val_no_building = h_exact(pde_points[:, 0], pde_points[:, 2], cfg["physics"]["n_manning"], cfg["physics"]["u_const"])
                        h_true_mean = jnp.mean(h_true_val_no_building)
                        denominator_nse = jnp.sum((h_true_val_no_building - h_true_mean)**2)
                        if denominator_nse > cfg.get("numerics", {}).get("eps", 1e-9):
                            numerator_nse = jnp.sum((h_true_val_no_building - h_pred_val_no_building)**2)
                            nse_val = float(1.0 - numerator_nse / denominator_nse)
                        rmse_val = float(rmse(h_pred_val_no_building, h_true_val_no_building))
                    # else: Validation skipped if no PDE points sampled

            # --- Update Best Model ---
            if nse_val > best_nse:
                best_nse = nse_val
                best_epoch = epoch
                best_params = copy.deepcopy(params) # Store the best parameters
                best_nse_time = time.time() - start_time
                if nse_val > -jnp.inf:
                    print(f"    ---> New best NSE: {best_nse:.6f} at epoch {epoch+1}")

            # --- Logging and Reporting ---
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % 100 == 0:
                print_epoch_stats(
                    epoch, start_time, avg_total_loss,
                    avg_losses['pde'], avg_losses['ic'], avg_losses['bc'],
                    avg_losses.get('building_bc', 0.0),
                    avg_losses.get('data', 0.0),
                    nse_val, rmse_val, epoch_time
                )

            if aim_run:
                try:
                    log_metrics(aim_run, {
                        'total_loss': avg_total_loss, 'pde_loss': avg_losses['pde'],
                        'ic_loss': avg_losses['ic'], 'bc_loss': avg_losses['bc'],
                        'building_bc_loss': avg_losses.get('building_bc', 0.0),
                        'data_loss': avg_losses.get('data', 0.0),
                        'nse': nse_val, 'rmse': rmse_val, 'epoch_time': epoch_time
                    }, epoch)
                except Exception as e:
                    print(f"Warning: Failed to log metrics to Aim in epoch {epoch+1}: {e}")

            # --- Early Stopping Check ---
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))

            if epoch >= min_epochs and (epoch - best_epoch) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                print(f"Best NSE {best_nse:.6f} achieved at epoch {best_epoch+1}.")
                break

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\n--- An error occurred during training loop: {e} ---")
        import traceback
        traceback.print_exc()

    # --- Final Summary and Saving ---
    finally:
        if aim_run:
            try:
                aim_run.close()
                print("Aim run closed.")
            except Exception as e:
                 print(f"Warning: Error closing Aim run: {e}")

        total_time = time.time() - start_time
        print_final_summary(total_time, best_epoch, best_nse, best_nse_time)

        # Ask for confirmation to save results
        if ask_for_confirmation():
            if best_params is not None:
                try:
                    # Save the best model parameters
                    save_model(best_params, model_dir, trial_name)

                    # --- Conditional Plotting ---
                    print("Generating final plot...") # Changed message
                    plot_cfg = cfg.get("plotting", {})
                    eps_plot = cfg.get("numerics", {}).get("eps", 1e-6)
                    t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)

                    if has_building:
                        # --- Generate Meshgrid Data for Prediction Plot ---
                        print("  Generating meshgrid predictions...")
                        resolution = plot_cfg.get("plot_resolution", 100)
                        x_plot = jnp.linspace(0, cfg["domain"]["lx"], resolution, dtype=DTYPE)
                        y_plot = jnp.linspace(0, cfg["domain"]["ly"], resolution, dtype=DTYPE)
                        xx_plot, yy_plot = jnp.meshgrid(x_plot, y_plot)
                        t_plot = jnp.full_like(xx_plot, t_const_val_plot, dtype=DTYPE)
                        plot_points_mesh = jnp.stack([xx_plot.ravel(), yy_plot.ravel(), t_plot.ravel()], axis=-1)

                        U_plot_pred_mesh = model.apply({'params': best_params['params']}, plot_points_mesh, train=False)
                        h_plot_pred_mesh = U_plot_pred_mesh[..., 0].reshape(resolution, resolution)
                        h_plot_pred_mesh = jnp.where(h_plot_pred_mesh < eps_plot, 0.0, h_plot_pred_mesh)

                        # --- REMOVED: Original 2D Top View Plot Call ---
                        # plot_path_2d = os.path.join(results_dir, "final_2d_top_view.png")
                        # plot_h_2d_top_view(xx_plot, yy_plot, h_plot_pred_mesh, cfg_dict, plot_path_2d)

                        # --- Generate Stacked Comparison Plot using validation_plotting_t_XXXXs.npy ---
                        print("  Loading plotting data for comparison...")
                        plot_data_time = t_const_val_plot
                        plot_data_file = os.path.join(base_data_path, f"validation_plotting_t_{int(plot_data_time)}s.npy")
                        if os.path.exists(plot_data_file):
                            try:
                                plot_data = np.load(plot_data_file)
                                x_coords_plot = jnp.array(plot_data[:, 1], dtype=DTYPE) # x
                                y_coords_plot = jnp.array(plot_data[:, 2], dtype=DTYPE) # y
                                h_true_plot_data = jnp.array(plot_data[:, 3], dtype=DTYPE) # h

                                # Call the new stacked plotting function
                                plot_path_comp = os.path.join(results_dir, f"final_comparison_plot_t{int(plot_data_time)}s.png")
                                plot_h_prediction_vs_true_2d(
                                    xx_plot, yy_plot, h_plot_pred_mesh, # Predicted mesh data
                                    x_coords_plot, y_coords_plot, h_true_plot_data, # True scattered data
                                    cfg_dict, plot_path_comp
                                )
                            except Exception as e_plot:
                                print(f"  Error generating comparison plot: {e_plot}")
                        else:
                             print(f"  Warning: Plotting data file {plot_data_file} not found. Skipping comparison plot.")

                    else: # No building scenario
                        # Generate 1D Plot for no-building scenario
                        print("  Generating 1D validation plot...")
                        nx_val_plot = plot_cfg.get("nx_val", 101)
                        y_const_plot = plot_cfg.get("y_const_plot", 0.0)
                        x_val_plot = jnp.linspace(0.0, cfg["domain"]["lx"], nx_val_plot, dtype=DTYPE)
                        plot_points_1d = jnp.stack([x_val_plot, jnp.full_like(x_val_plot, y_const_plot, dtype=DTYPE), jnp.full_like(x_val_plot, t_const_val_plot, dtype=DTYPE)], axis=1)

                        U_plot_pred_1d = model.apply({'params': best_params['params']}, plot_points_1d, train=False)
                        h_plot_pred_1d = U_plot_pred_1d[..., 0]
                        h_plot_pred_1d = jnp.where(h_plot_pred_1d < eps_plot, 0.0, h_plot_pred_1d)

                        plot_path_1d = os.path.join(results_dir, "final_validation_plot.png")
                        plot_h_vs_x(x_val_plot, h_plot_pred_1d, t_const_val_plot, y_const_plot, cfg_dict, plot_path_1d) # Pass dict config

                    print(f"Model and comparison plot saved in {model_dir} and {results_dir}") # Updated message
                except Exception as e:
                     print(f"Error during saving/plotting: {e}")
                     import traceback
                     traceback.print_exc()
            else:
                print("Warning: No best model found (best_params is None). Skipping save and plot.")
        else:
            # User chose not to save, clean up artifacts
            print("Save aborted by user. Deleting artifacts...")
            try:
                if aim_run and run_hash and aim_repo:
                    aim_repo.delete_run(run_hash)
                    print("Aim run deleted.")
                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir)
                    print(f"Deleted results directory: {results_dir}")
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                    print(f"Deleted model directory: {model_dir}")
                print("Cleanup complete.")
            except Exception as e:
                print(f"Error during cleanup: {e}")

    # Return best NSE
    return best_nse if best_nse > -jnp.inf else -1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PINN model for SWE (Handles building/no-building).")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., experiments/one_building_config.yaml or experiments/fourier_pinn_config.yaml)")
    args = parser.parse_args()

    try:
        final_nse = main(args.config) # Call main
        print(f"\n--- Script Finished ---")
        if isinstance(final_nse, (float, int)) and final_nse > -float('inf'):
            print(f"Final best NSE reported: {final_nse:.6f}")
        else:
            print(f"Final best NSE value invalid or not achieved: {final_nse}")
        print(f"-----------------------")
    except FileNotFoundError as e:
         print(f"Error: {e}. Please check the config file path.")
    except ValueError as e:
         print(f"Configuration or Model Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()