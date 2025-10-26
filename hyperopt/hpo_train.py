# /workspaces/swe-pinn/hyperopt/hpo_train.py
import os
import time
import importlib
import itertools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.core import FrozenDict
from flax import linen as nn # <<<--- ADDED THIS IMPORT
import numpy as np # Import numpy for loading data

# --- Ensure src modules can be imported ---
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Use DTYPE from config ---
from src.config import load_config, DTYPE
from src.data import sample_points, get_batches
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss,
    compute_building_bc_loss, total_loss
    # Removed compute_data_loss
)
from src.utils import mask_points_inside_building, nse, rmse
from src.reporting import print_epoch_stats, log_metrics # Add reporting imports

# --- HPO Specific Settings ---
# These can be overridden by the config if needed, but provide defaults
HPO_EPOCHS_DEFAULT = 5000 # Fixed number of epochs per HPO trial

# --- JIT-Compiled Training Step for HPO ---
def train_step_hpo(model: nn.Module, params: Dict[str, Any], opt_state: Any,
                   pde_batch: jnp.ndarray, ic_batch: jnp.ndarray,
                   bc_left_batch: jnp.ndarray, bc_right_batch: jnp.ndarray,
                   bc_bottom_batch: jnp.ndarray, bc_top_batch: jnp.ndarray,
                   building_batches: Dict[str, jnp.ndarray],
                   weights_dict: Dict[str, float],
                   optimiser: optax.GradientTransformation,
                   config: FrozenDict
                   ) -> Tuple[Any, Any, Dict[str, jnp.ndarray]]:
    """
    Perform a single physics-loss-driven training step for HPO (JIT compiled).
    No data loss term included.
    """
    def loss_and_stats(p):
        terms = {}
        if pde_batch.shape[0] > 0:
            terms['pde'] = compute_pde_loss(model, p, pde_batch, config)
        else:
            terms['pde'] = jnp.array(0.0, dtype=DTYPE)
            
        if ic_batch.shape[0] > 0:
            terms['ic'] = compute_ic_loss(model, p, ic_batch)
        else:
            terms['ic'] = jnp.array(0.0, dtype=DTYPE)
            
        if (bc_left_batch.shape[0] > 0 or bc_right_batch.shape[0] > 0 or
            bc_bottom_batch.shape[0] > 0 or bc_top_batch.shape[0] > 0):
             terms['bc'] = compute_bc_loss(
                 model, p, bc_left_batch, bc_right_batch, bc_bottom_batch, bc_top_batch, config
             )
        else:
             terms['bc'] = jnp.array(0.0, dtype=DTYPE)
             
        if building_batches and any(b.shape[0] > 0 for b in building_batches.values()):
             terms['building_bc'] = compute_building_bc_loss(
                 model, p,
                 building_batches.get('left', jnp.empty((0,3), dtype=DTYPE)),
                 building_batches.get('right', jnp.empty((0,3), dtype=DTYPE)),
                 building_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                 building_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
             )
        else:
             terms['building_bc'] = jnp.array(0.0, dtype=DTYPE)

        # Ensure all terms have values for total_loss calculation
        terms_with_defaults = {
            'pde': terms.get('pde', jnp.array(0.0, dtype=DTYPE)),
            'ic': terms.get('ic', jnp.array(0.0, dtype=DTYPE)),
            'bc': terms.get('bc', jnp.array(0.0, dtype=DTYPE)),
            'building_bc': terms.get('building_bc', jnp.array(0.0, dtype=DTYPE)),
        }
        weights_no_data = {k: v for k, v in weights_dict.items() if k != 'data'}
        total = total_loss(terms_with_defaults, weights_no_data)
        return total, terms_with_defaults

    (loss_val, term_vals), grads = jax.value_and_grad(loss_and_stats, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, term_vals

# Wrap the function with jax.jit, marking non-array arguments as static
train_step_hpo_jitted = jax.jit(train_step_hpo, static_argnames=('model', 'optimiser', 'config'))
# --- End JIT Training Step ---

def run_hpo_trial_with_nse(config_dict: Dict) -> float:
    """
    Runs a physics-loss-driven training loop for one HPO trial for a fixed
    number of epochs, then evaluates and returns the NSE on validation data.

    Args:
        config_dict: Dictionary containing the hyperparameters for this trial.

    Returns:
        The final NSE calculated on the validation_sample data, or -inf on error/invalid NSE.
    """
    cfg = FrozenDict(config_dict)
    
    # --- Essential Config Checks ---
    if "building" not in cfg:
        print("ERROR: HPO script (with NSE) requires 'building' section in config.")
        return -float('inf')
    if "scenario" not in cfg:
        print("ERROR: HPO script (with NSE) requires 'scenario' for loading validation data.")
        return -float('inf')

    # --- 1. Load Validation Data (for NSE/RMSE calculation during training) ---
    scenario_name = cfg.get('scenario')
    validation_data_file = os.path.join("data", scenario_name, "validation_sample.npy")

    if not os.path.exists(validation_data_file):
        print(f"ERROR: Validation data file not found at {validation_data_file}.")
        return -float('inf')

    try:
        print(f"Loading VALIDATION data for NSE calculation from: {validation_data_file}")
        loaded_val_data_np = np.load(validation_data_file)
        loaded_val_data = jnp.array(loaded_val_data_np, dtype=DTYPE)

        if loaded_val_data.shape[1] != 6:
            print(f"Warning: Validation data expected 6 columns, got {loaded_val_data.shape[1]}.")

        val_points_all = loaded_val_data[:, [1, 2, 0]]  # x, y, t
        h_true_val_all = loaded_val_data[:, 3]

        print("Applying building mask to validation points...")
        mask_val = mask_points_inside_building(val_points_all, cfg["building"])
        
        # Store as numpy arrays first, convert to JAX when needed
        val_points_masked_np = np.array(val_points_all[mask_val])
        h_true_val_masked_np = np.array(h_true_val_all[mask_val])

        if val_points_masked_np.shape[0] == 0:
            print("Warning: No validation points remaining after masking.")
            return -float('inf')

        print(f"Using {val_points_masked_np.shape[0]} masked validation points for NSE.")
    except Exception as e:
        print(f"Error loading/processing validation data: {e}")
        return -float('inf')

    # --- 2. Model Initialization ---
    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        print(f"ERROR: Could not find model class '{cfg['model']['name']}' in src/models.py: {e}")
        return -float('inf')

    key = random.PRNGKey(cfg["training"]["seed"])
    model, params = init_model(model_class, key, cfg)

    # --- 3. Optimizer Setup ---
    optimiser = optax.adam(learning_rate=cfg["training"]["learning_rate"])
    opt_state = optimiser.init(params)

    # Prepare loss weights dictionary (excluding data weight)
    weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items() if k != 'data_weight'}

    # --- 4. Training Initialization ---
    print(f"--- HPO Trial Start (Target: Maximize NSE) ---")
    print(f"Model: {cfg['model']['name']}, LR: {cfg['training']['learning_rate']:.2e}, Batch: {cfg['training']['batch_size']}")
    print(f"Loss Weights (PDE/IC/BC/Bldg): {weights_dict.get('pde',0):.1e}/{weights_dict.get('ic',0):.1e}/{weights_dict.get('bc',0):.1e}/{weights_dict.get('building_bc',0):.1e}")

    hpo_epochs = cfg["training"].get("epochs", HPO_EPOCHS_DEFAULT)
    start_time = time.time()
    best_nse = -float('inf')
    best_epoch = 0

    # --- 5. HPO Training Loop ---
    try:
        for epoch in range(hpo_epochs):
            epoch_start_time = time.time()

            # --- Dynamic Sampling ---
            key, pde_key, ic_key, bc_keys, bldg_keys = random.split(key, 5)
            l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
            bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)

            # Sample points
            pde_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["grid"]["nx"], cfg["grid"]["ny"], cfg["grid"]["nt"], pde_key)
            ic_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., 0., cfg["ic_bc_grid"]["nx_ic"], cfg["ic_bc_grid"]["ny_ic"], 1, ic_key)
            left_wall = sample_points(0., 0., 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_left"], cfg["ic_bc_grid"]["nt_bc_left"], l_key)
            right_wall = sample_points(cfg["domain"]["lx"], cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_right"], cfg["ic_bc_grid"]["nt_bc_right"], r_key)
            bottom_wall = sample_points(0., cfg["domain"]["lx"], 0., 0., 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_bottom"], 1, cfg["ic_bc_grid"]["nt_bc_other"], b_key)
            top_wall = sample_points(0., cfg["domain"]["lx"], cfg["domain"]["ly"], cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_top"], 1, cfg["ic_bc_grid"]["nt_bc_other"], t_key)
            
            b_cfg = cfg["building"]
            building_points = {
                'left': sample_points(b_cfg["x_min"], b_cfg["x_min"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_l_key),
                'right': sample_points(b_cfg["x_max"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_r_key),
                'bottom': sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_min"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_b_key),
                'top': sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_max"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_t_key)
            }

            # --- Create Batches ---
            batch_size = cfg["training"]["batch_size"]
            key, pde_b_key, ic_b_key, bc_b_keys, bldg_b_keys = random.split(key, 5)
            l_b_key, r_b_key, b_b_key, t_b_key = random.split(bc_b_keys, 4)
            bldg_l_b_key, bldg_r_b_key, bldg_b_b_key, bldg_t_b_key = random.split(bldg_b_keys, 4)
            
            pde_batches = get_batches(pde_b_key, pde_points, batch_size)
            ic_batches = get_batches(ic_b_key, ic_points, batch_size)
            left_batches = get_batches(l_b_key, left_wall, batch_size)
            right_batches = get_batches(r_b_key, right_wall, batch_size)
            bottom_batches = get_batches(b_b_key, bottom_wall, batch_size)
            top_batches = get_batches(t_b_key, top_wall, batch_size)
            
            building_batches_dict = {}
            building_b_keys_map = {'left': bldg_l_b_key, 'right': bldg_r_b_key, 'bottom': bldg_b_b_key, 'top': bldg_t_b_key}
            for wall, points in building_points.items():
                building_batches_dict[wall] = get_batches(building_b_keys_map[wall], points, batch_size) if points.shape[0] > 0 else []

            # --- Batch Iterators ---
            ic_batch_iter = itertools.cycle(ic_batches) if ic_batches else iter(())
            left_batch_iter = itertools.cycle(left_batches) if left_batches else iter(())
            right_batch_iter = itertools.cycle(right_batches) if right_batches else iter(())
            bottom_batch_iter = itertools.cycle(bottom_batches) if bottom_batches else iter(())
            top_batch_iter = itertools.cycle(top_batches) if top_batches else iter(())
            building_batch_iters = {wall: itertools.cycle(batches) if batches else iter(()) for wall, batches in building_batches_dict.items()}

            # --- Training Steps within Epoch ---
            num_batches = len(pde_batches)
            epoch_losses = {'pde': 0.0, 'ic': 0.0, 'bc': 0.0, 'building_bc': 0.0}
            
            if num_batches == 0:
                print(f"Warning: Epoch {epoch+1} - No PDE batches generated. Skipping.")
                continue

            for i in range(num_batches):
                ic_batch_data = next(ic_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                left_batch_data = next(left_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                right_batch_data = next(right_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                bottom_batch_data = next(bottom_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                top_batch_data = next(top_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                current_building_batch_data = {wall: next(iterator, jnp.empty((0, 3), dtype=DTYPE)) for wall, iterator in building_batch_iters.items()}

                params, opt_state, batch_losses = train_step_hpo_jitted(
                    model, params, opt_state, pde_batches[i], ic_batch_data,
                    left_batch_data, right_batch_data, bottom_batch_data, top_batch_data,
                    current_building_batch_data, weights_dict, optimiser, cfg
                )
                
                for k in epoch_losses:
                    epoch_losses[k] += float(batch_losses.get(k, 0.0))

            # --- Epoch End Calculation ---
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            avg_total_loss = float(total_loss(avg_losses, weights_dict))

            if not np.isfinite(avg_total_loss):
                print(f"Epoch {epoch+1}: Loss became non-finite ({avg_total_loss}). Stopping trial.")
                return -float('inf')

            # --- Calculate NSE/RMSE during training (every 100 epochs) ---
            nse_val, rmse_val = -jnp.inf, jnp.inf
            if (epoch + 1) % 100 == 0 or epoch == hpo_epochs - 1:
                # Convert validation data to JAX arrays fresh each time
                val_points_masked = jnp.array(val_points_masked_np, dtype=DTYPE)
                h_true_val_masked = jnp.array(h_true_val_masked_np, dtype=DTYPE)
                
                with jax.disable_jit():
                    try:
                        # Get fresh predictions for this epoch's parameters
                        U_pred_val = model.apply({'params': params['params']}, val_points_masked, train=False)
                        h_pred_val = U_pred_val[..., 0]
                        
                        # Calculate NSE
                        h_true_mean = jnp.mean(h_true_val_masked)
                        denominator_nse = jnp.sum((h_true_val_masked - h_true_mean)**2)
                        eps = cfg.get("numerics", {}).get("eps", 1e-9)
                        
                        if denominator_nse > eps:
                            numerator_nse = jnp.sum((h_true_val_masked - h_pred_val)**2)
                            nse_val = float(1.0 - numerator_nse / denominator_nse)
                        else:
                            nse_val = -float('inf')
                        
                        # Calculate RMSE
                        mse = jnp.mean((h_pred_val - h_true_val_masked)**2)
                        rmse_val = float(jnp.sqrt(mse))
                        
                        # Update best NSE
                        if nse_val > best_nse:
                            best_nse = nse_val
                            best_epoch = epoch
                            
                    except Exception as e_val:
                        print(f"Warning: Error during validation at epoch {epoch+1}: {e_val}")
                        import traceback
                        traceback.print_exc()
                        nse_val, rmse_val = -jnp.inf, jnp.inf

                epoch_time = time.time() - epoch_start_time
                print_epoch_stats(
                    epoch, start_time, avg_total_loss,
                    avg_losses['pde'], avg_losses['ic'], avg_losses['bc'],
                    avg_losses.get('building_bc', 0.0), 0.0, # No data loss
                    nse_val, rmse_val, epoch_time
                )

    except KeyboardInterrupt:
        print("\n--- HPO Trial training interrupted by user ---")
        return -float('inf') if best_nse == -float('inf') else best_nse
    except Exception as e:
        print(f"\n--- An error occurred during HPO trial training: {e} ---")
        import traceback
        traceback.print_exc()
        return -float('inf')

    # --- 7. Final Summary ---
    total_time = time.time() - start_time
    print(f"--- HPO Trial Finished ---")
    print(f"Total time: {total_time:.2f} seconds for {hpo_epochs} epochs.")
    print(f"Final Average Physics Loss: {avg_total_loss:.6e}")
    print(f"Best NSE (on validation data): {best_nse:.6f} at epoch {best_epoch+1}")
    if np.isfinite(rmse_val): 
        print(f"Final RMSE (on validation data): {rmse_val:.6f}")
    print(f"--------------------------")

    return best_nse

# --- Example Usage (for testing this script directly) ---
if __name__ == "__main__":
    print("Testing hpo_train.py (with NSE objective, fixed epochs) directly...")
    # Make sure the base config path points to a config WITH a building section
    TEST_CONFIG_PATH = "../experiments/one_building_config.yaml"
    if os.path.exists(TEST_CONFIG_PATH):
        test_config = load_config(TEST_CONFIG_PATH)
        test_config = dict(test_config) # Convert to dict

        # --- Use shorter duration for testing ---
        test_config['training']['epochs'] = 10 # Override epochs for quick test
        # Remove early stopping keys if present, as they are ignored now
        if 'device' in test_config and 'early_stop_min_epochs' in test_config['device']:
            del test_config['device']['early_stop_min_epochs']
        if 'device' in test_config and 'early_stop_patience' in test_config['device']:
            del test_config['device']['early_stop_patience']
        # Ensure data_weight is removed if present
        if 'loss_weights' in test_config and 'data_weight' in test_config['loss_weights']:
            print("Removing data_weight for HPO test.")
            del test_config['loss_weights']['data_weight']

        # --- Run the trial ---
        final_trial_nse = run_hpo_trial_with_nse(test_config)
        print(f"\nDirect test finished with final NSE: {final_trial_nse:.6f}")
    else:
        print(f"ERROR: Test config file not found at {TEST_CONFIG_PATH}")