# /workspaces/swe-pinn/hyperopt/hpo_train.py
import os
import time
import copy
import importlib
import itertools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.core import FrozenDict
import numpy as np # Import numpy for loading data

# --- Ensure src modules can be imported ---
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DTYPE, load_config # Use DTYPE from config
from src.data import sample_points, get_batches
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss,
    compute_building_bc_loss, total_loss
)
from src.utils import mask_points_inside_building, nse, rmse # Added nse, rmse

# --- HPO Specific Settings ---
HPO_EPOCHS_DEFAULT = 5000
HPO_EARLY_STOP_MIN_EPOCHS_DEFAULT = 1000
HPO_EARLY_STOP_PATIENCE_DEFAULT = 1500
# ---

# (train_step_hpo remains the same as previous data-free version)
def train_step_hpo(model: Any, params: Dict[str, Any], opt_state: Any,
                   pde_batch: jnp.ndarray, ic_batch: jnp.ndarray,
                   bc_left_batch: jnp.ndarray, bc_right_batch: jnp.ndarray,
                   bc_bottom_batch: jnp.ndarray, bc_top_batch: jnp.ndarray,
                   building_batches: Dict[str, jnp.ndarray], # Must have building for this HPO
                   weights_dict: Dict[str, float],
                   optimiser: optax.GradientTransformation,
                   config: FrozenDict
                   ) -> Tuple[Any, Any, Dict[str, jnp.ndarray], jnp.ndarray]:
    """Perform a single data-free training step for HPO."""

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
        # Building BC Loss (required for this HPO version)
        if building_batches and all(b.shape[0] > 0 for b in building_batches.values()):
             terms['building_bc'] = compute_building_bc_loss(
                 model, p,
                 building_batches.get('left', jnp.empty((0,3), dtype=DTYPE)),
                 building_batches.get('right', jnp.empty((0,3), dtype=DTYPE)),
                 building_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                 building_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
             )
        else:
             terms['building_bc'] = 0.0

        # Calculate total loss (physics only)
        terms_with_defaults = {
            'pde': terms.get('pde', 0.0), 'ic': terms.get('ic', 0.0),
            'bc': terms.get('bc', 0.0), 'building_bc': terms.get('building_bc', 0.0),
        }
        weights_no_data = {k: v for k, v in weights_dict.items() if k != 'data'}
        total = total_loss(terms_with_defaults, weights_no_data)
        return total, terms

    (loss_val, term_vals), grads = jax.value_and_grad(loss_and_stats, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, term_vals, loss_val

train_step_hpo_jitted = jax.jit(
    train_step_hpo, static_argnames=('model', 'optimiser', 'config')
)


def run_hpo_trial_with_nse(config_dict: Dict) -> float:
    """
    Runs a physics-loss-driven training loop for one HPO trial,
    then evaluates and returns the NSE on validation data.

    Args:
        config_dict: Dictionary containing the hyperparameters for this trial.

    Returns:
        The final NSE calculated on the validation_sample data, or -inf on error/invalid NSE.
    """
    cfg = FrozenDict(config_dict)
    if "building" not in cfg:
        print("ERROR: HPO script (with NSE) requires 'building' section in config.")
        return -float('inf')
    if "scenario" not in cfg:
        print("ERROR: HPO script (with NSE) requires 'scenario' for loading validation data.")
        return -float('inf')

    # --- Load Validation Data ---
    val_points_masked, h_true_val_masked = None, None
    scenario_name = cfg.get('scenario')
    validation_data_file = os.path.join("data", scenario_name, "validation_sample.npy")
    validation_data_loaded = False
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data for NSE calculation from: {validation_data_file}")
            # Load with numpy first, then convert to JAX array
            loaded_val_data_np = np.load(validation_data_file)
            loaded_val_data = jnp.array(loaded_val_data_np, dtype=DTYPE)

            val_points_all = loaded_val_data[:, [1, 2, 0]] # Input points (x, y, t)
            h_true_val_all = loaded_val_data[:, 3]       # True water depth h

            print("Applying building mask to validation points...")
            mask_val = mask_points_inside_building(val_points_all, cfg["building"])
            val_points_masked = val_points_all[mask_val]
            h_true_val_masked = h_true_val_all[mask_val]
            num_masked_val_points = val_points_masked.shape[0]
            print(f"Masked validation points remaining: {num_masked_val_points}.")
            if num_masked_val_points > 0:
                validation_data_loaded = True
            else:
                 print("Warning: No validation points remaining after masking. NSE cannot be calculated.")
        except Exception as e:
            print(f"Error loading or processing validation data file {validation_data_file}: {e}")
            val_points_masked, h_true_val_masked = None, None
    else:
        print(f"ERROR: Validation data file not found at {validation_data_file}. NSE cannot be calculated.")
        return -float('inf') # Cannot proceed without validation data for NSE

    if not validation_data_loaded:
        print("ERROR: Failed to load valid validation data points. NSE cannot be calculated.")
        return -float('inf')

    # --- Model Initialization ---
    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        print(f"ERROR: Could not find model class '{cfg['model']['name']}' in src/models.py: {e}")
        return -float('inf')

    key = random.PRNGKey(cfg["training"]["seed"])
    model, params = init_model(model_class, key, cfg)

    # --- Optimizer Setup ---
    optimiser = optax.adam(learning_rate=cfg["training"]["learning_rate"])
    opt_state = optimiser.init(params)

    # Prepare loss weights dictionary (excluding data weight)
    weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items() if k != 'data_weight'}

    # --- Training Initialization ---
    print(f"--- HPO Trial Start (Target: Maximize NSE) ---")
    print(f"Model: {cfg['model']['name']}, LR: {cfg['training']['learning_rate']:.2e}, Batch: {cfg['training']['batch_size']}")
    print(f"Loss Weights (PDE/IC/BC/Bldg): {weights_dict.get('pde',0):.1e}/{weights_dict.get('ic',0):.1e}/{weights_dict.get('bc',0):.1e}/{weights_dict.get('building_bc',0):.1e}")

    best_phys_loss: float = float('inf')
    best_params: Dict = None # Store params corresponding to best physics loss
    best_epoch: int = 0
    start_time = time.time()

    hpo_epochs = cfg["training"].get("epochs", HPO_EPOCHS_DEFAULT)
    min_epochs = cfg["device"].get("early_stop_min_epochs", HPO_EARLY_STOP_MIN_EPOCHS_DEFAULT)
    patience = cfg["device"].get("early_stop_patience", HPO_EARLY_STOP_PATIENCE_DEFAULT)

    # --- HPO Training Loop (Minimize Physics Loss) ---
    final_nse = -float('inf') # Default NSE
    training_completed = False
    try:
        for epoch in range(hpo_epochs):
            epoch_start_time = time.time()

            # --- Dynamic Sampling ---
            key, pde_key, ic_key, bc_keys, bldg_keys = random.split(key, 5)
            l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
            bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)

            # Sample points (PDE, IC, BC, Building)
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
            epoch_total_phys_loss = 0.0
            if num_batches == 0: continue

            for i in range(num_batches):
                ic_batch_data = next(ic_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                left_batch_data = next(left_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                right_batch_data = next(right_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                bottom_batch_data = next(bottom_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                top_batch_data = next(top_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                current_building_batch_data = {wall: next(iterator, jnp.empty((0, 3), dtype=DTYPE)) for wall, iterator in building_batch_iters.items()}

                params, opt_state, _, batch_total_loss = train_step_hpo_jitted(
                    model, params, opt_state, pde_batches[i], ic_batch_data,
                    left_batch_data, right_batch_data, bottom_batch_data, top_batch_data,
                    current_building_batch_data, weights_dict, optimiser, cfg
                )
                epoch_total_phys_loss += float(batch_total_loss)

            # --- Epoch End Calculation ---
            avg_total_phys_loss = epoch_total_phys_loss / num_batches

            if not jnp.isfinite(avg_total_phys_loss):
                print(f"Epoch {epoch+1}: Physics Loss became non-finite ({avg_total_phys_loss}). Stopping trial.")
                return -float('inf') # Return -inf NSE for Optuna (failed trial)

            # --- Update Best Physics Loss Params ---
            if avg_total_phys_loss < best_phys_loss:
                best_phys_loss = avg_total_phys_loss
                best_params = copy.deepcopy(params) # Store the params achieving best physics loss
                best_epoch = epoch

            # --- Reporting (minimal) ---
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1:5d} | Phys Loss: {avg_total_phys_loss:.4e}")

            # --- Early Stopping Check (based on physics loss) ---
            if epoch >= min_epochs and (epoch - best_epoch) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} based on physics loss ---")
                print(f"Best Phys Loss {best_phys_loss:.6e} achieved at epoch {best_epoch+1}.")
                break
        
        training_completed = True # Mark that training finished or stopped early

    except KeyboardInterrupt:
        print("\n--- HPO Trial training interrupted by user ---")
        return -float('inf')
    except Exception as e:
        print(f"\n--- An error occurred during HPO trial training: {e} ---")
        import traceback
        traceback.print_exc()
        return -float('inf')

    # --- Final NSE Calculation (using best_params found during physics minimization) ---
    if training_completed and best_params is not None:
        print(f"\nCalculating final NSE using parameters from epoch {best_epoch+1} (Best Physics Loss)...")
        try:
            with jax.disable_jit(): # Ensure NSE calculation is done without JIT issues
                 # Ensure val_points_masked and h_true_val_masked are still valid JAX arrays
                 if isinstance(val_points_masked, np.ndarray):
                     val_points_masked_jax = jnp.array(val_points_masked, dtype=DTYPE)
                 else:
                     val_points_masked_jax = val_points_masked
                 if isinstance(h_true_val_masked, np.ndarray):
                     h_true_val_masked_jax = jnp.array(h_true_val_masked, dtype=DTYPE)
                 else:
                     h_true_val_masked_jax = h_true_val_masked

                 U_pred_val = model.apply({'params': best_params['params']}, val_points_masked_jax, train=False)
                 h_pred_val = U_pred_val[..., 0]

                 # Calculate NSE
                 final_nse = float(nse(h_pred_val, h_true_val_masked_jax))
                 final_rmse = float(rmse(h_pred_val, h_true_val_masked_jax))

                 if not jnp.isfinite(final_nse):
                     print(f"Warning: Calculated NSE is non-finite ({final_nse}). Returning -inf.")
                     final_nse = -float('inf')

        except Exception as e_nse:
            print(f"Error during final NSE calculation: {e_nse}")
            final_nse = -float('inf')
    else:
        print("Skipping NSE calculation as training did not complete successfully or no best params found.")
        final_nse = -float('inf')

    # --- Final Summary ---
    total_time = time.time() - start_time
    print(f"--- HPO Trial Finished ---")
    print(f"Total time: {total_time:.2f} seconds.")
    print(f"Best Physics Loss: {best_phys_loss:.6e} (Epoch {best_epoch+1})")
    print(f"Final NSE (on validation data): {final_nse:.6f}")
    if jnp.isfinite(final_rmse): print(f"Final RMSE (on validation data): {final_rmse:.6f}")
    print(f"--------------------------")

    # Return the final NSE for Optuna to maximize
    return final_nse

# Example usage (for testing this script directly)
if __name__ == "__main__":
    print("Testing hpo_train.py (with NSE objective) directly...")
    TEST_CONFIG_PATH = "../experiments/one_building_config.yaml"
    if os.path.exists(TEST_CONFIG_PATH):
        test_config = load_config(TEST_CONFIG_PATH)
        test_config = dict(test_config) # Convert to dict

        test_config['training']['epochs'] = 10
        test_config['device']['early_stop_min_epochs'] = 5
        test_config['device']['early_stop_patience'] = 5
        if 'loss_weights' in test_config and 'data_weight' in test_config['loss_weights']:
            del test_config['loss_weights']['data_weight']

        # Run the trial
        final_trial_nse = run_hpo_trial_with_nse(test_config)
        print(f"\nDirect test finished with final NSE: {final_trial_nse:.6f}")
    else:
        print(f"ERROR: Test config file not found at {TEST_CONFIG_PATH}")