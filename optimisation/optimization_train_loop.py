# optimisation/optimization_train_loop.py
"""
Contains the core training loop logic for a single Optuna trial,
adapted from src/train.py and src/train_gradnorm.py.
"""
import os
import jax
import jax.numpy as jnp
from jax import random, lax
import optax
from flax.core import FrozenDict
import numpy as np
import time
import itertools
import importlib
from typing import Any, Dict, Tuple
import optuna

# --- Imports from project src directory ---
# This assumes run_optimization.py adds the project root to sys.path
from src.config import DTYPE
from src.data import sample_domain, get_batches # <<<--- MODIFIED: Using sample_domain
from src.models import init_model
from src.losses import (
    compute_neg_h_loss, compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_building_bc_loss
)
# Note: get_initial_losses is specifically for GradNorm setup
from src.utils import nse, rmse, mask_points_inside_building
from src.physics import h_exact
# Do not import reporting here; objective function returns the value directly


# --- Define Training Step (Functionally JITted) ---
def train_step_trial(model: Any, params: FrozenDict, opt_state: Any,
                     all_batches: Dict[str, Any],
                     weights_dict: Dict[str, float],
                     optimiser: optax.GradientTransformation,
                     config: FrozenDict
                     ) -> Tuple[FrozenDict, Any, Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Perform a single training step for an optimization trial.
    Returns new params, new opt_state, INDIVIDUAL loss terms (unweighted), and TOTAL weighted loss.
    """
    has_building = "building" in config
    active_loss_keys_base = list(weights_dict.keys())

    def loss_and_individual_terms(p):
        terms = {}
        # Compute losses based on available non-empty batches and active weights
        pde_batch_data = all_batches.get('pde', jnp.empty((0,3), dtype=DTYPE))
        if 'pde' in active_loss_keys_base and pde_batch_data.shape[0] > 0:
            if has_building:
                pde_mask = mask_points_inside_building(pde_batch_data, config["building"])
                terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config, pde_mask)
            else:
                terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config)
            
            if 'neg_h' in active_loss_keys_base:
                # compute_neg_h_loss uses the pde_batch_data
                if has_building:
                    pde_mask = mask_points_inside_building(pde_batch_data, config["building"])
                    terms['neg_h'] = compute_neg_h_loss(model, p, pde_batch_data, pde_mask)
                else:
                    terms['neg_h'] = compute_neg_h_loss(model, p, pde_batch_data)

        ic_batch_data = all_batches.get('ic', jnp.empty((0,3), dtype=DTYPE))
        if 'ic' in active_loss_keys_base and ic_batch_data.shape[0] > 0:
            terms['ic'] = compute_ic_loss(model, p, ic_batch_data)

        bc_batches = all_batches.get('bc', {})
        bc_left_batch = bc_batches.get('left', jnp.empty((0,3), dtype=DTYPE))
        bc_right_batch = bc_batches.get('right', jnp.empty((0,3), dtype=DTYPE))
        bc_bottom_batch = bc_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE))
        bc_top_batch = bc_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
        if 'bc' in active_loss_keys_base and any(b.shape[0] > 0 for b in bc_batches.values() if hasattr(b, 'shape') and b.shape[0] > 0):
             terms['bc'] = compute_bc_loss(
                 model, p, bc_left_batch, bc_right_batch, bc_bottom_batch, bc_top_batch, config
             )

        if has_building and 'building_bc' in active_loss_keys_base:
            bldg_batches = all_batches.get('building_bc', {})
            bldg_left_batch = bldg_batches.get('left', jnp.empty((0,3), dtype=DTYPE))
            bldg_right_batch = bldg_batches.get('right', jnp.empty((0,3), dtype=DTYPE))
            bldg_bottom_batch = bldg_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE))
            bldg_top_batch = bldg_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
            if bldg_batches and any(b.shape[0] > 0 for b in bldg_batches.values() if hasattr(b, 'shape') and b.shape[0] > 0):
                terms['building_bc'] = compute_building_bc_loss(
                    model, p, bldg_left_batch, bldg_right_batch, bldg_bottom_batch, bldg_top_batch
                )

        # Calculate weighted total loss
        # Use only weights for terms that were actually computed in this step
        active_weights = {k: weights_dict.get(k, 0.0) for k in terms.keys()}
        # Ensure all keys from weights_dict are present for total_loss function, defaulting computed value to 0 if not present in `terms`
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}

        total = total_loss(terms_with_defaults, active_weights)
        return total, terms # Return only computed terms

    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, individual_terms_val, total_loss_val


# --- Main Training Function for a Single Trial ---
def run_training_trial(trial: optuna.trial.Trial, trial_cfg: FrozenDict) -> float:
    """
    Runs the training loop for a single Optuna trial.
    Args:
        trial: The Optuna trial object.
        trial_cfg: The configuration dictionary for this specific trial.
    Returns:
        The best NSE value achieved during the trial. Returns -1.0 if invalid.
    """
    has_building = "building" in trial_cfg

    print(f"--- Starting Trial {trial.number} ---")

    # --- 1. Setup Model, Optimizer, Keys ---
    key = random.PRNGKey(trial_cfg["training"]["seed"])
    
    # --- MODIFICATION: Split key for validation sampling ---
    model_key, train_key, val_key = random.split(key, 3)
    # --- END MODIFICATION ---

    model_name = trial_cfg["model"]["name"]

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, model_name)
        model, params = init_model(model_class, model_key, trial_cfg)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Trial {trial.number}: ERROR during model initialization: {e}")
        return -1.0 # Return poor value

    # Get the boundaries dict, which has string keys from the config
    raw_boundaries = trial_cfg["training"].get("lr_boundaries", {15000: 0.1, 30000: 0.1})
    
    # --- FIX: Convert string keys to int keys for Optax ---
    boundaries_and_scales_int_keys = {int(k): v for k, v in raw_boundaries.items()}
    # --- END FIX ---

    lr_schedule = optax.piecewise_constant_schedule(
        init_value=trial_cfg["training"]["learning_rate"],
        boundaries_and_scales=boundaries_and_scales_int_keys # Use the converted dict
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(trial_cfg["training"].get("clip_norm", 1.0)), # Use default if not in config
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimiser.init(params)

    # --- 2. Prepare Initial Weights ---
    current_weights_dict = {k.replace('_weight',''):v for k,v in trial_cfg["loss_weights"].items()}

    # Determine active loss keys based on weights > 0
    active_loss_term_keys = [
        k for k, v in current_weights_dict.items() if v > 0 and k != 'data'
    ]

    # --- 3. Load Validation Data ---
    val_points, h_true_val = None, None
    validation_data_loaded = False
    scenario_name_val = trial_cfg.get('scenario', 'default_scenario')
    base_data_path_val = os.path.join("data", scenario_name_val)
    validation_data_file = os.path.join(base_data_path_val, "validation_sample.npy")

    if os.path.exists(validation_data_file):
        # --- Case 1: validation_sample.npy file exists (standard case) ---
        try:
            print(f"Trial {trial.number}: Loading validation data from: {validation_data_file}")
            loaded_val_data = np.load(validation_data_file).astype(DTYPE)
            val_points_all = loaded_val_data[:, [1, 2, 0]] # (x, y, t)
            h_true_val_all = loaded_val_data[:, 3]       # (h)
            
            if has_building:
                # Apply building mask if building scenario
                print(f"Trial {trial.number}: Applying building mask to validation points.")
                mask_val = mask_points_inside_building(val_points_all, trial_cfg["building"])
                val_points = val_points_all[mask_val]
                h_true_val = h_true_val_all[mask_val]
            else:
                 # No building, use all loaded points
                 val_points = val_points_all
                 h_true_val = h_true_val_all

            if val_points is not None and val_points.shape[0] > 0:
                 validation_data_loaded = True
                 print(f"Trial {trial.number}: Loaded {val_points.shape[0]} validation points.")
            else:
                 print(f"Trial {trial.number}: WARNING - No validation points remain after loading/masking.")
        except Exception as e:
            print(f"Trial {trial.number}: WARNING - Error loading validation data {validation_data_file}: {e}.")
            
    # --- MODIFICATION: Use sample_domain for analytical validation ---
    elif not has_building and "validation_grid" in trial_cfg:
        # --- Case 2: No file, but NO building and validation_grid exists ---
        print(f"Trial {trial.number}: INFO - {validation_data_file} not found. Creating analytical validation set from 'validation_grid' config.")
        try:
            val_grid_cfg = trial_cfg["validation_grid"]
            domain_cfg = trial_cfg["domain"]
            
            # Check for n_points_val first, otherwise compute from nx/ny/nt
            if "n_points_val" in val_grid_cfg:
                n_val_points = val_grid_cfg["n_points_val"]
            else:
                n_val_points = val_grid_cfg.get("nx_val", 10) * val_grid_cfg.get("ny_val", 10) * val_grid_cfg.get("nt_val", 10)
            
            print(f"Trial {trial.number}: Sampling {n_val_points} analytical validation points...")
            # Sample points based on the validation grid using sample_domain
            val_points = sample_domain(
                val_key, # Use the dedicated key
                n_val_points,
                (0., domain_cfg["lx"]), 
                (0., domain_cfg["ly"]), 
                (0., domain_cfg["t_final"])
            )
            
            # Calculate the exact solution for these points
            h_true_val = h_exact(
                val_points[:, 0], # x
                val_points[:, 2], # t
                trial_cfg["physics"]["n_manning"],
                trial_cfg["physics"]["u_const"]
            )
            
            if val_points.shape[0] > 0:
                validation_data_loaded = True
                print(f"Trial {trial.number}: Created analytical validation set with {val_points.shape[0]} points.")
            else:
                print(f"Trial {trial.number}: WARNING - Analytical validation set is empty (check validation_grid config).")
                
        except Exception as e:
            print(f"Trial {trial.number}: WARNING - Error creating analytical validation set: {e}.")
            val_points, h_true_val = None, None
    # --- END MODIFICATION ---
            
    else:
        # --- Case 3: No file AND (it's a building scenario OR no validation_grid) ---
        print(f"Trial {trial.number}: WARNING - Validation file {validation_data_file} not found.")
        if has_building:
            print(f"Trial {trial.number}: WARNING - Building scenario, NSE/RMSE will not be calculated.")
        else:
            print(f"Trial {trial.number}: WARNING - No-building scenario, but 'validation_grid' not found in config. NSE/RMSE will not be calculated.")
        # In this case, validation_data_loaded remains False


    # --- 4. Training Loop ---
    best_nse_trial = -jnp.inf
    epochs = trial_cfg["training"]["epochs"]
    batch_size = trial_cfg["training"]["batch_size"]
    global_step = 0
    start_time_trial = time.time()

    # Helper to prepare batches for lax.scan
    def prepare_batches(batches, n_steps, shape_suffix):
        if not batches:
            return jnp.zeros((n_steps, 0) + shape_suffix, dtype=DTYPE)
        stacked = jnp.stack(batches)
        n_avail = stacked.shape[0]
        if n_avail == n_steps:
            return stacked
        # Repeat batches if fewer than n_steps
        indices = jnp.arange(n_steps) % n_avail
        return stacked[indices]

    # Define scan body function
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        new_params, new_opt_state, terms, total = train_step_trial(
            model, curr_params, curr_opt_state, batch_data, current_weights_dict, optimiser, trial_cfg
        )
        # We don't need to carry the history of losses for now to save memory
        return (new_params, new_opt_state), None

    for epoch in range(epochs):
        epoch_start_time = time.time()
        key = train_key # Use the training key for this epoch's sampling

        # --- MODIFICATION: Dynamic Sampling using sample_domain ---
        key, pde_key, ic_key, bc_keys, bldg_keys = random.split(key, 5)
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        domain_cfg = trial_cfg["domain"]
        sampling_cfg = trial_cfg["sampling"] # <-- Use new config section

        # Sample points only needed for active terms
        n_pde = sampling_cfg.get("n_points_pde", 1000)
        pde_points = sample_domain(pde_key, n_pde, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])) if 'pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        
        n_ic = sampling_cfg.get("n_points_ic", 100)
        ic_points = sample_domain(ic_key, n_ic, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.)) if 'ic' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)

        left_wall, right_wall, bottom_wall, top_wall = [jnp.empty((0,3), dtype=DTYPE)] * 4
        if 'bc' in active_loss_term_keys:
            n_bc = sampling_cfg.get("n_points_bc_domain", 100)
            n_bc_per_wall = max(5, n_bc // 4)
            left_wall = sample_domain(l_key, n_bc_per_wall, (0., 0.), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
            right_wall = sample_domain(r_key, n_bc_per_wall, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
            bottom_wall = sample_domain(b_key, n_bc_per_wall, (0., domain_cfg["lx"]), (0., 0.), (0., domain_cfg["t_final"]))
            top_wall = sample_domain(t_key, n_bc_per_wall, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]), (0., domain_cfg["t_final"]))

        building_points = {}
        if has_building and 'building_bc' in active_loss_term_keys:
            bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)
            b_cfg = trial_cfg["building"]
            n_bldg = sampling_cfg.get("n_points_bc_building", 100)
            n_bldg_per_wall = max(5, n_bldg // 4)
            building_points['left'] = sample_domain(bldg_l_key, n_bldg_per_wall, (b_cfg["x_min"], b_cfg["x_min"]), (b_cfg["y_min"], b_cfg["y_max"]), (0., domain_cfg["t_final"]))
            building_points['right'] = sample_domain(bldg_r_key, n_bldg_per_wall, (b_cfg["x_max"], b_cfg["x_max"]), (b_cfg["y_min"], b_cfg["y_max"]), (0., domain_cfg["t_final"]))
            building_points['bottom'] = sample_domain(bldg_b_key, n_bldg_per_wall, (b_cfg["x_min"], b_cfg["x_max"]), (b_cfg["y_min"], b_cfg["y_min"]), (0., domain_cfg["t_final"]))
            building_points['top'] = sample_domain(bldg_t_key, n_bldg_per_wall, (b_cfg["x_min"], b_cfg["x_max"]), (b_cfg["y_max"], b_cfg["y_max"]), (0., domain_cfg["t_final"]))
        # --- END MODIFICATION ---

        # --- Batch Creation ---
        key, pde_b_key, ic_b_key, bc_b_keys, bldg_b_keys = random.split(key, 5)
        l_b_key, r_b_key, b_b_key, t_b_key = random.split(bc_b_keys, 4)

        pde_batches = get_batches(pde_b_key, pde_points, batch_size) if pde_points.shape[0]>0 else []
        ic_batches = get_batches(ic_b_key, ic_points, batch_size) if ic_points.shape[0]>0 else []
        left_batches = get_batches(l_b_key, left_wall, batch_size) if left_wall.shape[0]>0 else []
        right_batches = get_batches(r_b_key, right_wall, batch_size) if right_wall.shape[0]>0 else []
        bottom_batches = get_batches(b_b_key, bottom_wall, batch_size) if bottom_wall.shape[0]>0 else []
        top_batches = get_batches(t_b_key, top_wall, batch_size) if top_wall.shape[0]>0 else []

        building_batches_dict = {}
        if has_building and 'building_bc' in active_loss_term_keys:
             bldg_l_b_key, bldg_r_b_key, bldg_b_b_key, bldg_t_b_key = random.split(bldg_b_keys, 4)
             building_b_keys_map = {'left': bldg_l_b_key, 'right': bldg_r_b_key, 'bottom': bldg_b_b_key, 'top': bldg_t_b_key}
             for wall, points in building_points.items():
                 building_batches_dict[wall] = get_batches(building_b_keys_map[wall], points, batch_size) if points.shape[0] > 0 else []

        # --- Determine Number of Batches ---
        # logic: Use the maximum batch count available to ensure no generated data is ignored.
        # prepare_batches will cycle smaller datasets to match this length.
        batch_counts = [
            len(pde_batches),
            len(ic_batches),
            len(left_batches),
            len(right_batches),
            len(bottom_batches),
            len(top_batches)
        ]
        
        if has_building and 'building_bc' in active_loss_term_keys:
            for b_batches in building_batches_dict.values():
                batch_counts.append(len(b_batches))

        num_batches = max(batch_counts)

        if num_batches == 0:
             # This should theoretically be unreachable given your constraint, 
             # but serves as a sane guardrail against sampling errors.
             print(f"Trial {trial.number}, Epoch {epoch+1}: Warning - Zero batches generated. Skipping.")
             continue

        # --- Prepare Data for lax.scan ---
        building_scan_data = {}
        if has_building and 'building_bc' in active_loss_term_keys:
            for wall, batches in building_batches_dict.items():
                building_scan_data[wall] = prepare_batches(batches, num_batches, (3,))

        scan_inputs = {
            'pde': prepare_batches(pde_batches, num_batches, (3,)),
            'ic': prepare_batches(ic_batches, num_batches, (3,)),
            'bc': {
                'left': prepare_batches(left_batches, num_batches, (3,)),
                'right': prepare_batches(right_batches, num_batches, (3,)),
                'bottom': prepare_batches(bottom_batches, num_batches, (3,)),
                'top': prepare_batches(top_batches, num_batches, (3,)),
            },
            'building_bc': building_scan_data,
            # 'data' term is not present in this simplified loop context or handled if needed
        }

        # --- Run Training Steps with lax.scan ---
        (params, opt_state), _ = lax.scan(scan_body, (params, opt_state), scan_inputs)
        
        global_step += num_batches

        # --- Epoch End Validation & Pruning ---
        validation_freq = trial_cfg.get("training", {}).get("validation_freq", 1) # Validate less frequently
        if (epoch + 1) % validation_freq == 0:
            current_nse = -jnp.inf # Default if validation fails/skipped

            # This single block now handles both loaded .npy data and pre-computed analytical data
            if validation_data_loaded: 
                try:
                    U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                    h_pred_val = U_pred_val[..., 0]
                    current_nse = float(nse(h_pred_val, h_true_val))
                except Exception as e_val:
                    print(f"Trial {trial.number}, Epoch {epoch+1}: Warning - NSE calculation error: {e_val}")
                    current_nse = -jnp.inf
            
            # Else (e.g., building case with no validation data), current_nse remains -inf

            if jnp.isnan(current_nse):
                 print(f"Trial {trial.number}, Epoch {epoch+1}: NaN NSE detected. Pruning.")
                 raise optuna.exceptions.TrialPruned()

            best_nse_trial = max(best_nse_trial, current_nse if current_nse > -jnp.inf else -1.0) # Keep track of best valid NSE

            # Optuna Pruning - Report the best NSE seen so far in this trial
            trial.report(best_nse_trial, epoch)
            if trial.should_prune():
                 print(f"Trial {trial.number}: Pruned at epoch {epoch+1}.")
                 raise optuna.exceptions.TrialPruned()

            # Optional: Log progress less frequently
            if (epoch + 1) % (validation_freq*200) == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"  Trial {trial.number}, Epoch {epoch+1}/{epochs}: "
                        f"NSE={current_nse:.6f}, "
                        f"Time={epoch_time:.2f}s, Current Best NSE={best_nse_trial:.6f}")


        # Update train_key for next epoch's sampling
        train_key = key


    # --- 5. Return Final Objective Value ---
    final_nse = best_nse_trial
    # Return a poor but valid float if NSE calculation failed or was skipped
    if final_nse <= -jnp.inf:
        print(f"Trial {trial.number}: Finished. No valid NSE achieved.")
        return -1.0

    print(f"Trial {trial.number}: Finished successfully. Best NSE = {final_nse:.6f}")
    return final_nse