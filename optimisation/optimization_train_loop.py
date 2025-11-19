# optimisation/optimization_train_loop.py
"""
Contains the core training loop logic for a single Optuna trial.
Refactored to exclude GradNorm and Data Loss (Physics-Only).
Includes robust batch calculation and safe loss weighting.
"""
import os
import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.core import FrozenDict
import numpy as np
import time
import itertools
import importlib
from typing import Any, Dict, Tuple
import optuna

# --- Imports from project src directory ---
from src.config import DTYPE
from src.data import sample_domain, get_batches
from src.models import init_model
from src.losses import (
    compute_neg_h_loss, compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_building_bc_loss
)
from src.utils import nse, rmse, mask_points_inside_building
from src.physics import h_exact


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
        # --- Compute Losses based on available data ---
        
        # 1. PDE Loss
        pde_batch_data = all_batches.get('pde', jnp.empty((0,3), dtype=DTYPE))
        if 'pde' in active_loss_keys_base and pde_batch_data.shape[0] > 0:
            if has_building:
                pde_mask = mask_points_inside_building(pde_batch_data, config["building"])
                terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config, pde_mask)
                if 'neg_h' in active_loss_keys_base:
                    terms['neg_h'] = compute_neg_h_loss(model, p, pde_batch_data, pde_mask)
            else:
                terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config)
                if 'neg_h' in active_loss_keys_base:
                    terms['neg_h'] = compute_neg_h_loss(model, p, pde_batch_data)

        # 2. IC Loss
        ic_batch_data = all_batches.get('ic', jnp.empty((0,3), dtype=DTYPE))
        if 'ic' in active_loss_keys_base and ic_batch_data.shape[0] > 0:
            terms['ic'] = compute_ic_loss(model, p, ic_batch_data)

        # 3. BC Loss (Domain Walls)
        bc_batches = all_batches.get('bc', {})
        if 'bc' in active_loss_keys_base:
             bc_left = bc_batches.get('left', jnp.empty((0,3), dtype=DTYPE))
             bc_right = bc_batches.get('right', jnp.empty((0,3), dtype=DTYPE))
             bc_bottom = bc_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE))
             bc_top = bc_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
             
             # Only compute if we actually have data in at least one wall
             if any(b.shape[0] > 0 for b in [bc_left, bc_right, bc_bottom, bc_top]):
                 terms['bc'] = compute_bc_loss(
                     model, p, bc_left, bc_right, bc_bottom, bc_top, config
                 )

        # 4. Building BC Loss
        if has_building and 'building_bc' in active_loss_keys_base:
            bldg_batches = all_batches.get('building_bc', {})
            b_left = bldg_batches.get('left', jnp.empty((0,3), dtype=DTYPE))
            b_right = bldg_batches.get('right', jnp.empty((0,3), dtype=DTYPE))
            b_bottom = bldg_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE))
            b_top = bldg_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
            
            if any(b.shape[0] > 0 for b in [b_left, b_right, b_bottom, b_top]):
                terms['building_bc'] = compute_building_bc_loss(
                    model, p, b_left, b_right, b_bottom, b_top
                )

        # --- Calculate Weighted Total Loss ---
        # We create a dictionary that contains a value for EVERY key in weights_dict.
        # If a term wasn't computed (e.g. batch empty), it defaults to 0.0.
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}

        # We pass the FULL weights_dict. 
        # Mathematical safety: sum(weight[k] * term[k]). If term[k] is 0.0, the contribution is 0.0.
        # This avoids KeyError if 'total_loss' iterates over weights_dict.
        total = total_loss(terms_with_defaults, weights_dict)
        
        return total, terms

    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, individual_terms_val, total_loss_val

# JIT the training step
train_step_trial_jitted = jax.jit(
    train_step_trial,
    static_argnames=('model', 'optimiser', 'config')
)

# --- Main Training Function for a Single Trial ---
def run_training_trial(trial: optuna.trial.Trial, trial_cfg: FrozenDict) -> float:
    """
    Runs the training loop for a single Optuna trial (Data-Free / Static Weights).
    Args:
        trial: The Optuna trial object.
        trial_cfg: The configuration dictionary for this specific trial.
    Returns:
        The best NSE value achieved during the trial. Returns -1.0 if invalid.
    """
    has_building = "building" in trial_cfg

    print(f"--- Starting Trial {trial.number} ---")
    print("Mode: Data-Free (Physics Only)")

    # --- 1. Setup Model, Optimizer, Keys ---
    key = random.PRNGKey(trial_cfg["training"]["seed"])
    
    model_key, init_key_main, train_key = random.split(key, 3)
    init_key, val_key = random.split(init_key_main, 2)

    model_name = trial_cfg["model"]["name"]

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, model_name)
        model, params = init_model(model_class, model_key, trial_cfg)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Trial {trial.number}: ERROR during model initialization: {e}")
        return -1.0

    raw_boundaries = trial_cfg["training"].get("lr_boundaries", {15000: 0.1, 30000: 0.1})
    boundaries_and_scales_int_keys = {int(k): v for k, v in raw_boundaries.items()}

    lr_schedule = optax.piecewise_constant_schedule(
        init_value=trial_cfg["training"]["learning_rate"],
        boundaries_and_scales=boundaries_and_scales_int_keys
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(trial_cfg["training"].get("clip_norm", 1.0)),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimiser.init(params)

    # --- 2. Prepare Static Weights ---
    current_weights_dict = {k.replace('_weight',''):v for k,v in trial_cfg["loss_weights"].items()}
    
    # Filter to active keys (weights > 0), explicitly excluding 'data'
    active_loss_term_keys = [
        k for k, v in current_weights_dict.items() if v > 0 and k != 'data'
    ]
    
    # --- 3. Load/Create Validation Data ---
    val_points, h_true_val = None, None
    validation_data_loaded = False
    scenario_name_val = trial_cfg.get('scenario', 'default_scenario')
    base_data_path_val = os.path.join("data", scenario_name_val)
    validation_data_file = os.path.join(base_data_path_val, "validation_sample.npy")

    if os.path.exists(validation_data_file):
        try:
            # Case 1: Validation file exists (Standard)
            loaded_val_data = np.load(validation_data_file).astype(DTYPE)
            val_points_all = loaded_val_data[:, [1, 2, 0]] # (x, y, t)
            h_true_val_all = loaded_val_data[:, 3]       # (h)
            
            if has_building:
                mask_val = mask_points_inside_building(val_points_all, trial_cfg["building"])
                val_points = val_points_all[mask_val]
                h_true_val = h_true_val_all[mask_val]
            else:
                 val_points = val_points_all
                 h_true_val = h_true_val_all

            if val_points is not None and val_points.shape[0] > 0:
                 validation_data_loaded = True
        except Exception as e:
            print(f"Trial {trial.number}: WARNING - Error loading validation data {validation_data_file}: {e}.")
            
    elif not has_building and "validation_grid" in trial_cfg:
        # Case 2: Analytical validation (No file, no building)
        try:
            val_grid_cfg = trial_cfg["validation_grid"]
            domain_cfg = trial_cfg["domain"]
            n_val_points = val_grid_cfg.get("n_points_val", 10000)
            
            val_points = sample_domain(
                val_key,
                n_val_points,
                (0., domain_cfg["lx"]), 
                (0., domain_cfg["ly"]), 
                (0., domain_cfg["t_final"])
            )
            
            h_true_val = h_exact(
                val_points[:, 0],
                val_points[:, 2],
                trial_cfg["physics"]["n_manning"],
                trial_cfg["physics"]["u_const"]
            )
            
            if val_points.shape[0] > 0:
                validation_data_loaded = True
        except Exception as e:
            print(f"Trial {trial.number}: WARNING - Error creating analytical validation set: {e}.")
            val_points, h_true_val = None, None

    # --- 4. Training Loop ---
    best_nse_trial = -jnp.inf
    epochs = trial_cfg["training"]["epochs"]
    batch_size = trial_cfg["training"]["batch_size"]
    start_time_trial = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        key = train_key # Use the training key for this epoch's sampling

        # --- Dynamic Sampling ---
        key, pde_key, ic_key, bc_keys, bldg_keys = random.split(key, 5)
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        domain_cfg = trial_cfg["domain"]
        sampling_cfg = trial_cfg["sampling"]

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

        # --- Determine Number of Batches (Robust) ---
        # Collect all valid batch lists to find the maximum number of iterations required
        all_active_batches = []
        if pde_batches: all_active_batches.append(pde_batches)
        if ic_batches: all_active_batches.append(ic_batches)
        
        # Domain BCs
        for b in [left_batches, right_batches, bottom_batches, top_batches]:
            if b: all_active_batches.append(b)
            
        # Building BCs
        for b in building_batches_dict.values():
            if b: all_active_batches.append(b)
            
        if not all_active_batches:
             # No data generated for this epoch (e.g. extremely small N)
             continue
             
        # Set num_batches to the length of the longest list (iterations per epoch)
        num_batches = max(len(b) for b in all_active_batches)

        # --- Batch Iterators ---
        pde_batch_iter = itertools.cycle(pde_batches) if pde_batches else iter(())
        ic_batch_iter = itertools.cycle(ic_batches) if ic_batches else iter(())
        left_batch_iter = itertools.cycle(left_batches) if left_batches else iter(())
        right_batch_iter = itertools.cycle(right_batches) if right_batches else iter(())
        bottom_batch_iter = itertools.cycle(bottom_batches) if bottom_batches else iter(())
        top_batch_iter = itertools.cycle(top_batches) if top_batches else iter(())

        building_batch_iters = {}
        if has_building and 'building_bc' in active_loss_term_keys:
             for wall, batches in building_batches_dict.items():
                 building_batch_iters[wall] = itertools.cycle(batches) if batches else iter(())

        # --- Training Steps within Epoch ---
        for i in range(num_batches):
            # Get batches
            pde_batch_data = next(pde_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            ic_batch_data = next(ic_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            left_batch_data = next(left_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            right_batch_data = next(right_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            bottom_batch_data = next(bottom_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            top_batch_data = next(top_batch_iter, jnp.empty((0, 3), dtype=DTYPE))

            current_building_batch_data = {}
            if has_building and 'building_bc' in active_loss_term_keys:
                for wall, iterator in building_batch_iters.items():
                    current_building_batch_data[wall] = next(iterator, jnp.empty((0, 3), dtype=DTYPE))

            # Aggregate batches
            current_all_batches = {
                'pde': pde_batch_data,
                'ic': ic_batch_data,
                'bc': {
                    'left': left_batch_data,
                    'right': right_batch_data,
                    'bottom': bottom_batch_data,
                    'top': top_batch_data,
                },
                'building_bc': {
                    'left': current_building_batch_data.get('left', jnp.empty((0, 3), dtype=DTYPE)),
                    'right': current_building_batch_data.get('right', jnp.empty((0, 3), dtype=DTYPE)),
                    'bottom': current_building_batch_data.get('bottom', jnp.empty((0, 3), dtype=DTYPE)),
                    'top': current_building_batch_data.get('top', jnp.empty((0, 3), dtype=DTYPE)),
                }
            }

            # --- Training Step ---
            params, opt_state, _, _ = train_step_trial_jitted(
                model, params, opt_state,
                current_all_batches,
                current_weights_dict,
                optimiser, trial_cfg
            )

        # --- Epoch End Validation & Pruning ---
        validation_freq = trial_cfg.get("training", {}).get("validation_freq", 1)
        if (epoch + 1) % validation_freq == 0:
            current_nse = -jnp.inf

            if validation_data_loaded: 
                try:
                    U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                    h_pred_val = U_pred_val[..., 0]
                    current_nse = float(nse(h_pred_val, h_true_val))
                except Exception as e_val:
                    print(f"Trial {trial.number}, Epoch {epoch+1}: Warning - NSE calculation error: {e_val}")
                    current_nse = -jnp.inf
            
            if jnp.isnan(current_nse):
                 print(f"Trial {trial.number}, Epoch {epoch+1}: NaN NSE detected. Pruning.")
                 raise optuna.exceptions.TrialPruned()

            best_nse_trial = max(best_nse_trial, current_nse if current_nse > -jnp.inf else -1.0)

            # --- Logging Block (Restored) ---
            if (epoch + 1) % (validation_freq * 10) == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"  Trial {trial.number}, Epoch {epoch+1}/{epochs}: "
                      f"NSE={current_nse:.6f}, "
                      f"Time={epoch_time:.2f}s, Current Best NSE={best_nse_trial:.6f}")

            trial.report(best_nse_trial, epoch)
            if trial.should_prune():
                 print(f"Trial {trial.number}: Pruned at epoch {epoch+1}.")
                 raise optuna.exceptions.TrialPruned()

        train_key = key

    # --- 5. Return Final Objective Value ---
    final_nse = best_nse_trial
    if final_nse <= -jnp.inf:
        print(f"Trial {trial.number}: Finished. No valid NSE achieved.")
        return -1.0

    print(f"Trial {trial.number}: Finished successfully. Best NSE = {final_nse:.6f}")
    return final_nse