# optimisation/optimization_train_loop.py
"""
Contains the core training loop logic for a single Optuna trial,
adapted from src/train.py and src/train_gradnorm.py. Includes periodic reporting.
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
from src.data import sample_points, get_batches
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_building_bc_loss, compute_data_loss
)
from src.train_gradnorm import get_initial_losses
from src.gradnorm import GradNormState, init_gradnorm, update_gradnorm_weights, LOSS_FN_MAP
from src.utils import nse, rmse, mask_points_inside_building
from src.physics import h_exact
# --- Import the reporting function ---
from src.reporting import print_epoch_stats # Import the specific function


# --- Define Training Step (Functionally JITted) ---
# [train_step_trial function remains the same]
def train_step_trial(model: Any, params: FrozenDict, opt_state: Any,
                     all_batches: Dict[str, Any],
                     weights_dict: Dict[str, float],
                     optimiser: optax.GradientTransformation,
                     config: FrozenDict,
                     data_free: bool = True
                     ) -> Tuple[FrozenDict, Any, Dict[str, jnp.ndarray], jnp.ndarray]:
    has_building = "building" in config
    active_loss_keys_base = list(weights_dict.keys())

    def loss_and_individual_terms(p):
        terms = {}
        pde_batch_data = all_batches.get('pde', jnp.empty((0,3), dtype=DTYPE))
        if 'pde' in active_loss_keys_base and pde_batch_data.shape[0] > 0:
            terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config)

        ic_batch_data = all_batches.get('ic', jnp.empty((0,3), dtype=DTYPE))
        if 'ic' in active_loss_keys_base and ic_batch_data.shape[0] > 0:
            terms['ic'] = compute_ic_loss(model, p, ic_batch_data)

        bc_batches = all_batches.get('bc', {})
        bc_left_batch = bc_batches.get('left', jnp.empty((0,3), dtype=DTYPE))
        bc_right_batch = bc_batches.get('right', jnp.empty((0,3), dtype=DTYPE))
        bc_bottom_batch = bc_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE))
        bc_top_batch = bc_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
        if 'bc' in active_loss_keys_base and any(isinstance(b, jnp.ndarray) and b.shape[0] > 0 for b in bc_batches.values()):
             terms['bc'] = compute_bc_loss(
                 model, p, bc_left_batch, bc_right_batch, bc_bottom_batch, bc_top_batch, config
             )

        if has_building and 'building_bc' in active_loss_keys_base:
            bldg_batches = all_batches.get('building_bc', {})
            bldg_left_batch = bldg_batches.get('left', jnp.empty((0,3), dtype=DTYPE))
            bldg_right_batch = bldg_batches.get('right', jnp.empty((0,3), dtype=DTYPE))
            bldg_bottom_batch = bldg_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE))
            bldg_top_batch = bldg_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
            if bldg_batches and any(isinstance(b, jnp.ndarray) and b.shape[0] > 0 for b in bldg_batches.values()):
                terms['building_bc'] = compute_building_bc_loss(
                    model, p, bldg_left_batch, bldg_right_batch, bldg_bottom_batch, bldg_top_batch
                )

        data_batch_data = all_batches.get('data', jnp.empty((0,6), dtype=DTYPE))
        if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0:
             terms['data'] = compute_data_loss(model, p, data_batch_data, config)

        active_weights = {k: weights_dict.get(k, 0.0) for k in terms.keys()}
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, active_weights)
        return total, terms

    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, individual_terms_val, total_loss_val

train_step_trial_jitted = jax.jit(
    train_step_trial,
    static_argnames=('model', 'optimiser', 'config', 'data_free')
)


# --- Main Training Function for a Single Trial ---
def run_training_trial(trial: optuna.trial.Trial, trial_cfg: FrozenDict, data_free: bool) -> float:
    has_building = "building" in trial_cfg
    enable_gradnorm = trial_cfg.get("gradnorm", {}).get("enable", False)

    print(f"--- Starting Trial {trial.number} ---")
    if data_free: print("Mode: Data-Free")
    else: print(f"Mode: With Data Loss (GradNorm: {enable_gradnorm})")

    # --- 1. Setup Model, Optimizer, Keys ---
    key = random.PRNGKey(trial_cfg["training"]["seed"])
    model_key, init_key, loop_key = random.split(key, 3) # Use loop_key for the training loop
    model_name = trial_cfg["model"]["name"]

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, model_name)
        model, params = init_model(model_class, model_key, trial_cfg)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Trial {trial.number}: ERROR during model initialization: {e}")
        return -1.0

    lr_schedule = optax.piecewise_constant_schedule(
        init_value=trial_cfg["training"]["learning_rate"],
        boundaries_and_scales=trial_cfg["training"].get("lr_boundaries", {15000: 0.1, 30000: 0.1})
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(trial_cfg["training"].get("clip_norm", 1.0)),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimiser.init(params)

    # --- 2. Prepare Initial Weights and GradNorm State (if enabled) ---
    current_weights_dict = {k.replace('_weight',''):v for k,v in trial_cfg["loss_weights"].items()}
    gradnorm_state = None
    gradnorm_alpha = trial_cfg.get("gradnorm", {}).get("alpha", 1.5)
    gradnorm_lr = trial_cfg.get("gradnorm", {}).get("learning_rate", 0.01)
    gradnorm_update_freq = trial_cfg.get("gradnorm", {}).get("update_freq", 100)

    active_loss_term_keys = [
        k for k, v in current_weights_dict.items() if v > 0 and (k != 'data' or not data_free)
    ]

    if enable_gradnorm:
        print(f"Trial {trial.number}: Initializing GradNorm...")
        # Use init_key here, don't consume loop_key
        init_sub_key, _ = random.split(init_key) # Keep init_key "pristine" if needed elsewhere
        key_for_init, pde_key, ic_key, bc_keys, bldg_keys, data_key_init = random.split(init_sub_key, 6)
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        batch_size_init = trial_cfg["training"]["batch_size"]

        domain_cfg = trial_cfg["domain"]
        grid_cfg = trial_cfg["grid"]
        ic_bc_grid_cfg = trial_cfg["ic_bc_grid"]

        pde_points_init = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"], pde_key) if 'pde' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        ic_points_init = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., 0., ic_bc_grid_cfg["nx_ic"], ic_bc_grid_cfg["ny_ic"], 1, ic_key) if 'ic' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        left_wall_init = sample_points(0., 0., 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_left"], ic_bc_grid_cfg["nt_bc_left"], l_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        right_wall_init = sample_points(domain_cfg["lx"], domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_right"], ic_bc_grid_cfg["nt_bc_right"], r_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        bottom_wall_init = sample_points(0., domain_cfg["lx"], 0., 0., 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_bottom"], 1, ic_bc_grid_cfg["nt_bc_other"], b_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        top_wall_init = sample_points(0., domain_cfg["lx"], domain_cfg["ly"], domain_cfg["ly"], 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_top"], 1, ic_bc_grid_cfg["nt_bc_other"], t_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)

        building_points_init = {}
        if has_building and 'building_bc' in active_loss_term_keys:
            bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)
            b_cfg = trial_cfg["building"]
            building_points_init['left'] = sample_points(b_cfg["x_min"], b_cfg["x_min"], b_cfg["y_min"], b_cfg["y_max"], 0., domain_cfg["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_l_key)
            building_points_init['right'] = sample_points(b_cfg["x_max"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_max"], 0., domain_cfg["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_r_key)
            building_points_init['bottom'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_min"], 0., domain_cfg["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_b_key)
            building_points_init['top'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_max"], b_cfg["y_max"], 0., domain_cfg["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_t_key)

        data_points_init = None
        if not data_free and 'data' in active_loss_term_keys:
             try:
                scenario_name_init = trial_cfg.get('scenario', 'default_scenario')
                base_data_path_init = os.path.join("data", scenario_name_init)
                training_data_file_init = os.path.join(base_data_path_init, "training_dataset_sample.npy")
                loaded_train_data_init = jnp.load(training_data_file_init).astype(DTYPE)
                data_points_init = loaded_train_data_init[:batch_size_init]
             except Exception as e:
                print(f"Trial {trial.number}: WARN - Failed to load training data for GradNorm init: {e}. Skipping data term in init.")
                active_loss_term_keys = [k for k in active_loss_term_keys if k != 'data']

        key_for_init, pde_b_key, ic_b_key, bc_b_keys, bldg_b_keys, data_b_key_init = random.split(key_for_init, 6) # Use the derived key
        l_b_key, r_b_key, b_b_key, t_b_key = random.split(bc_b_keys, 4)
        init_batches = {}
        if 'pde' in active_loss_term_keys and pde_points_init.shape[0] > 0: init_batches['pde'] = get_batches(pde_b_key, pde_points_init, batch_size_init)[0]
        if 'ic' in active_loss_term_keys and ic_points_init.shape[0] > 0: init_batches['ic'] = get_batches(ic_b_key, ic_points_init, batch_size_init)[0]
        if 'bc' in active_loss_term_keys: init_batches['bc'] = { 'left': ..., 'right': ..., 'bottom': ..., 'top': ... } # Fill as before
        if has_building and 'building_bc' in active_loss_term_keys: init_batches['building_bc'] = { 'left': ..., 'right': ..., 'bottom': ..., 'top': ... } # Fill as before
        if 'data' in active_loss_term_keys and data_points_init is not None and data_points_init.shape[0] > 0: init_batches['data'] = get_batches(data_b_key_init, data_points_init, batch_size_init)[0]

        with jax.disable_jit():
             relevant_init_batches = {k: init_batches.get(k) for k in active_loss_term_keys if init_batches.get(k) is not None}
             initial_losses = get_initial_losses(model, params, relevant_init_batches, trial_cfg)

        active_initial_losses = {k: initial_losses.get(k, 1e-8) for k in active_loss_term_keys}

        gradnorm_state = init_gradnorm(
            loss_keys=list(active_initial_losses.keys()),
            initial_losses=active_initial_losses,
            gradnorm_lr=gradnorm_lr
        )
        current_weights_dict = {key: float(w) for key, w in zip(active_initial_losses.keys(), gradnorm_state.weights)}
        print(f"Trial {trial.number}: GradNorm Initial Weights: {current_weights_dict}")

    # --- 3. Load Validation Data ---
    # [Validation data loading logic remains the same]
    val_points, h_true_val = None, None
    validation_data_loaded = False
    scenario_name_val = trial_cfg.get('scenario', 'default_scenario')
    base_data_path_val = os.path.join("data", scenario_name_val)
    validation_data_file = os.path.join(base_data_path_val, "validation_sample.npy")

    if os.path.exists(validation_data_file):
        try:
            loaded_val_data = jnp.load(validation_data_file).astype(DTYPE)
            val_points_all = loaded_val_data[:, [1, 2, 0]]
            h_true_val_all = loaded_val_data[:, 3]
            if has_building:
                mask_val = mask_points_inside_building(val_points_all, trial_cfg["building"])
                val_points = val_points_all[mask_val]
                h_true_val = h_true_val_all[mask_val]
            else:
                 val_points = val_points_all
                 h_true_val = h_true_val_all

            if val_points is not None and val_points.shape[0] > 0:
                 validation_data_loaded = True
                 print(f"Trial {trial.number}: Loaded {val_points.shape[0]} validation points.")
            else:
                 print(f"Trial {trial.number}: WARNING - No validation points remain after masking.")
        except Exception as e:
            print(f"Trial {trial.number}: WARNING - Error loading validation data {validation_data_file}: {e}.")
    else:
        print(f"Trial {trial.number}: WARNING - Validation file {validation_data_file} not found.")

    if not validation_data_loaded and not has_building:
         print(f"Trial {trial.number}: INFO - No-building scenario without validation file. NSE will use analytical solution on PDE points.")

    # --- Load Training Data (if not data_free) ---
    data_points_full = None
    if not data_free:
        training_data_file = os.path.join(base_data_path_val, "training_dataset_sample.npy")
        if os.path.exists(training_data_file):
             try:
                  data_points_full = jnp.load(training_data_file).astype(DTYPE)
                  if data_points_full.shape[0] == 0:
                       print(f"Trial {trial.number}: ERROR - Training data file is empty. Stopping trial.")
                       return -1.0
             except Exception as e:
                  print(f"Trial {trial.number}: ERROR - Could not load training data {training_data_file}: {e}. Stopping trial.")
                  return -1.0
        else:
             print(f"Trial {trial.number}: ERROR - Training data file {training_data_file} not found for data loss term. Stopping trial.")
             return -1.0

    # --- 4. Training Loop ---
    best_nse_trial = -jnp.inf
    epochs = trial_cfg["training"]["epochs"]
    batch_size = trial_cfg["training"]["batch_size"]
    global_step = 0
    start_time_trial = time.time()
    key = loop_key # *** Use the dedicated loop key ***

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # --- Generate all keys needed for THIS epoch ---
        # *** START KEY CORRECTION ***
        key, epoch_subkey = random.split(key) # Generate a subkey for this epoch

        # Determine how many keys are needed for sampling and batching
        num_sampling_splits = 2 # pde, ic
        num_batching_splits = 2 # pde, ic
        if 'bc' in active_loss_term_keys:
            num_sampling_splits += 4 # left, right, bottom, top
            num_batching_splits += 4
        if has_building and 'building_bc' in active_loss_term_keys:
            num_sampling_splits += 4 # left, right, bottom, top
            num_batching_splits += 4
        if not data_free and 'data' in active_loss_term_keys:
            num_sampling_splits += 1 # No spatial sampling for data usually
            num_batching_splits += 1

        # Split the epoch subkey
        all_epoch_keys = random.split(epoch_subkey, num_sampling_splits + num_batching_splits)
        sampling_keys = all_epoch_keys[:num_sampling_splits]
        batching_keys = all_epoch_keys[num_sampling_splits:]

        # Assign keys carefully based on active terms
        key_idx_s, key_idx_b = 0, 0
        pde_key = sampling_keys[key_idx_s]; key_idx_s += 1
        pde_b_key = batching_keys[key_idx_b]; key_idx_b += 1
        ic_key = sampling_keys[key_idx_s]; key_idx_s += 1
        ic_b_key = batching_keys[key_idx_b]; key_idx_b += 1

        if 'bc' in active_loss_term_keys:
            l_key, r_key, b_key, t_key = sampling_keys[key_idx_s:key_idx_s+4]; key_idx_s += 4
            l_b_key, r_b_key, b_b_key, t_b_key = batching_keys[key_idx_b:key_idx_b+4]; key_idx_b += 4
        else: # Provide dummy keys if not active, though they won't be used
            l_key, r_key, b_key, t_key = [None]*4
            l_b_key, r_b_key, b_b_key, t_b_key = [None]*4

        if has_building and 'building_bc' in active_loss_term_keys:
            bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = sampling_keys[key_idx_s:key_idx_s+4]; key_idx_s += 4
            bldg_l_b_key, bldg_r_b_key, bldg_b_b_key, bldg_t_b_key = batching_keys[key_idx_b:key_idx_b+4]; key_idx_b += 4
        else:
            bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = [None]*4
            bldg_l_b_key, bldg_r_b_key, bldg_b_b_key, bldg_t_b_key = [None]*4

        if not data_free and 'data' in active_loss_term_keys:
            # data_key_epoch = sampling_keys[key_idx_s]; key_idx_s += 1 # Usually no sampling key needed
            data_b_key_epoch = batching_keys[key_idx_b]; key_idx_b += 1
        else:
             data_b_key_epoch = None
        # *** END KEY CORRECTION ***


        # --- Dynamic Sampling ---
        domain_cfg = trial_cfg["domain"]
        grid_cfg = trial_cfg["grid"]
        ic_bc_grid_cfg = trial_cfg["ic_bc_grid"]

        pde_points = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"], pde_key) if 'pde' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        ic_points = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., 0., ic_bc_grid_cfg["nx_ic"], ic_bc_grid_cfg["ny_ic"], 1, ic_key) if 'ic' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        left_wall = sample_points(0., 0., 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_left"], ic_bc_grid_cfg["nt_bc_left"], l_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        right_wall = sample_points(domain_cfg["lx"], domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_right"], ic_bc_grid_cfg["nt_bc_right"], r_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        bottom_wall = sample_points(0., domain_cfg["lx"], 0., 0., 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_bottom"], 1, ic_bc_grid_cfg["nt_bc_other"], b_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
        top_wall = sample_points(0., domain_cfg["lx"], domain_cfg["ly"], domain_cfg["ly"], 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_top"], 1, ic_bc_grid_cfg["nt_bc_other"], t_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)

        building_points = {}
        if has_building and 'building_bc' in active_loss_term_keys:
            b_cfg = trial_cfg["building"]
            building_points['left'] = sample_points(b_cfg["x_min"], b_cfg["x_min"], b_cfg["y_min"], b_cfg["y_max"], 0., domain_cfg["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_l_key)
            building_points['right'] = sample_points(b_cfg["x_max"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_max"], 0., domain_cfg["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_r_key)
            building_points['bottom'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_min"], 0., domain_cfg["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_b_key)
            building_points['top'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_max"], b_cfg["y_max"], 0., domain_cfg["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_t_key)

        # --- Batch Creation ---
        pde_batches = get_batches(pde_b_key, pde_points, batch_size) if pde_points.shape[0]>0 else []
        ic_batches = get_batches(ic_b_key, ic_points, batch_size) if ic_points.shape[0]>0 else []
        left_batches = get_batches(l_b_key, left_wall, batch_size) if left_wall.shape[0]>0 else []
        right_batches = get_batches(r_b_key, right_wall, batch_size) if right_wall.shape[0]>0 else []
        bottom_batches = get_batches(b_b_key, bottom_wall, batch_size) if bottom_wall.shape[0]>0 else []
        top_batches = get_batches(t_b_key, top_wall, batch_size) if top_wall.shape[0]>0 else []

        data_batches = []
        if not data_free and data_points_full is not None:
             data_batches = get_batches(data_b_key_epoch, data_points_full, batch_size)

        building_batches_dict = {}
        if has_building and 'building_bc' in active_loss_term_keys:
             building_b_keys_map = {'left': bldg_l_b_key, 'right': bldg_r_b_key, 'bottom': bldg_b_b_key, 'top': bldg_t_b_key}
             for wall, points in building_points.items():
                 if wall in building_b_keys_map: # Check if key exists
                     building_batches_dict[wall] = get_batches(building_b_keys_map[wall], points, batch_size) if points.shape[0] > 0 else []

        # --- Determine Number of Batches ---
        num_batches = 0
        if 'pde' in active_loss_term_keys and pde_batches: num_batches = len(pde_batches)
        elif 'data' in active_loss_term_keys and data_batches: num_batches = len(data_batches)
        elif 'ic' in active_loss_term_keys and ic_batches: num_batches = len(ic_batches)
        elif any(k in active_loss_term_keys for k in ['bc', 'building_bc']):
            active_batch_lists = [left_batches, right_batches, bottom_batches, top_batches]
            if has_building and 'building_bc' in active_loss_term_keys:
                 active_batch_lists.extend(building_batches_dict.values())
            num_batches = max((len(b) for b in active_batch_lists if b), default=0)

        if num_batches == 0:
             print(f"Trial {trial.number}, Epoch {epoch+1}: Warning - No batches generated. Skipping.")
             continue

        # --- Batch Iterators ---
        pde_batch_iter = itertools.cycle(pde_batches) if pde_batches and num_batches != len(pde_batches) else iter(pde_batches)
        ic_batch_iter = itertools.cycle(ic_batches) if ic_batches else iter(())
        left_batch_iter = itertools.cycle(left_batches) if left_batches else iter(())
        right_batch_iter = itertools.cycle(right_batches) if right_batches else iter(())
        bottom_batch_iter = itertools.cycle(bottom_batches) if bottom_batches else iter(())
        top_batch_iter = itertools.cycle(top_batches) if top_batches else iter(())
        data_batch_iter = itertools.cycle(data_batches) if data_batches and num_batches != len(data_batches) else iter(data_batches)
        building_batch_iters = {}
        if has_building and 'building_bc' in active_loss_term_keys:
             for wall, batches in building_batches_dict.items():
                 building_batch_iters[wall] = itertools.cycle(batches) if batches else iter(())

        # --- Training Steps within Epoch ---
        epoch_total_losses = []
        epoch_individual_losses = {key: [] for key in active_loss_term_keys} # Only track active terms

        for i in range(num_batches):
            global_step += 1

            # --- Get Batches ---
            pde_batch_data = next(pde_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            ic_batch_data = next(ic_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            left_batch_data = next(left_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            right_batch_data = next(right_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            bottom_batch_data = next(bottom_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            top_batch_data = next(top_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
            data_batch_data = next(data_batch_iter, jnp.empty((0, 6), dtype=DTYPE))

            current_building_batch_data = {}
            if has_building and 'building_bc' in active_loss_term_keys:
                for wall, iterator in building_batch_iters.items():
                    current_building_batch_data[wall] = next(iterator, jnp.empty((0, 3), dtype=DTYPE))

            # Aggregate batches relevant to this step
            current_all_batches = {}
            if pde_batch_data.shape[0] > 0: current_all_batches['pde'] = pde_batch_data
            if ic_batch_data.shape[0] > 0: current_all_batches['ic'] = ic_batch_data
            bc_batch_list_step = [left_batch_data, right_batch_data, bottom_batch_data, top_batch_data]
            if any(b.shape[0]>0 for b in bc_batch_list_step):
                 current_all_batches['bc'] = {'left': left_batch_data, 'right': right_batch_data, 'bottom': bottom_batch_data, 'top': top_batch_data}
            if current_building_batch_data and any(b.shape[0]>0 for b in current_building_batch_data.values()):
                 current_all_batches['building_bc'] = current_building_batch_data
            if data_batch_data.shape[0] > 0 and not data_free:
                 current_all_batches['data'] = data_batch_data

            # --- GradNorm Update ---
            if enable_gradnorm and global_step % gradnorm_update_freq == 0:
                 active_batch_keys_gn = [LOSS_FN_MAP[k]['batch_key'] for k in gradnorm_state.initial_losses.keys() if k in LOSS_FN_MAP]
                 gradnorm_update_batches = {k: current_all_batches[k] for k in active_batch_keys_gn if k in current_all_batches}
                 with jax.disable_jit():
                      gradnorm_state, current_weights_dict = update_gradnorm_weights(
                           gradnorm_state, params, model, gradnorm_update_batches,
                           trial_cfg, gradnorm_alpha, gradnorm_lr
                      )

            # --- Training Step ---
            params, opt_state, individual_losses, total_weighted_loss = train_step_trial_jitted(
                model, params, opt_state,
                current_all_batches,
                current_weights_dict,
                optimiser, trial_cfg, data_free
            )

            # Track losses for epoch averaging
            epoch_total_losses.append(float(total_weighted_loss))
            for key in epoch_individual_losses.keys(): # Only iterate active keys
                if key in individual_losses: # Check if loss was computed
                    epoch_individual_losses[key].append(float(individual_losses[key]))

        # Calculate epoch averages
        avg_epoch_total_weighted_loss = np.mean(epoch_total_losses) if epoch_total_losses else 0.0
        avg_epoch_unweighted_losses = {}
        for key, values in epoch_individual_losses.items():
            avg_epoch_unweighted_losses[key] = np.mean(values) if values else 0.0


        # --- Epoch End Validation & Pruning ---
        validation_freq = trial_cfg.get("training", {}).get("validation_freq", 100)
        current_nse = -jnp.inf # Default NSE
        current_rmse = jnp.inf # Default RMSE

        if (epoch + 1) % validation_freq == 0:
            if validation_data_loaded:
                 with jax.disable_jit():
                     try:
                         U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                         h_pred_val = U_pred_val[..., 0]
                         current_nse = float(nse(h_pred_val, h_true_val))
                         current_rmse = float(rmse(h_pred_val, h_true_val))
                     except Exception as e_val:
                          print(f"Trial {trial.number}, Epoch {epoch+1}: Warning - Val NSE/RMSE calculation error: {e_val}")
                          current_nse, current_rmse = -jnp.inf, jnp.inf
            elif not has_building and pde_points.shape[0] > 0: # Analytical for no-building
                 with jax.disable_jit():
                      try:
                           U_pred_val_no_building = model.apply({'params': params['params']}, pde_points, train=False)
                           h_pred_val_no_building = U_pred_val_no_building[..., 0]
                           h_true_val_no_building = h_exact(pde_points[:, 0], pde_points[:, 2], trial_cfg["physics"]["n_manning"], trial_cfg["physics"]["u_const"])
                           current_nse = float(nse(h_pred_val_no_building, h_true_val_no_building))
                           current_rmse = float(rmse(h_pred_val_no_building, h_true_val_no_building))
                      except Exception as e_val:
                           print(f"Trial {trial.number}, Epoch {epoch+1}: Warning - Analytical NSE/RMSE calculation error: {e_val}")
                           current_nse, current_rmse = -jnp.inf, jnp.inf
            else: # No validation possible
                current_nse, current_rmse = -jnp.inf, jnp.inf

            if np.isnan(current_nse) or np.isnan(current_rmse):
                 print(f"Trial {trial.number}, Epoch {epoch+1}: NaN Metric detected (NSE={current_nse}, RMSE={current_rmse}). Pruning.")
                 raise optuna.exceptions.TrialPruned()

            best_nse_trial = max(best_nse_trial, current_nse if current_nse > -jnp.inf else -1.0)

            # Optuna Pruning
            trial.report(best_nse_trial, epoch)
            if trial.should_prune():
                 print(f"Trial {trial.number}: Pruned at epoch {epoch+1}.")
                 raise optuna.exceptions.TrialPruned()

        # --- Reporting Every 100 Epochs ---
        # *** Import print_epoch_stats inside the function or globally ***
        from src.reporting import print_epoch_stats # Ensure import is correct

        if (epoch + 1) % 100 == 0:
            epoch_time = time.time() - epoch_start_time
            report_nse = current_nse if (epoch + 1) % validation_freq == 0 else best_nse_trial # Report best NSE seen so far if not validating this epoch
            report_rmse = current_rmse if (epoch + 1) % validation_freq == 0 else jnp.inf # Report inf RMSE if not validating this epoch

            print(f"Trial {trial.number} ", end="")
            print_epoch_stats(
                epoch=epoch,
                start_time=start_time_trial,
                total_loss=float(avg_epoch_total_weighted_loss), # Convert JAX array if needed
                pde_loss=float(avg_epoch_unweighted_losses.get('pde', 0.0)),
                ic_loss=float(avg_epoch_unweighted_losses.get('ic', 0.0)),
                bc_loss=float(avg_epoch_unweighted_losses.get('bc', 0.0)),
                building_bc_loss=float(avg_epoch_unweighted_losses.get('building_bc', 0.0)),
                data_loss=float(avg_epoch_unweighted_losses.get('data', 0.0)),
                nse=float(report_nse),
                rmse=float(report_rmse),
                epoch_time=epoch_time
            )
            if enable_gradnorm:
                 print(f"      Current Weights: { {k: f'{v:.2e}' for k, v in current_weights_dict.items()} }")
        # --- End Reporting ---

        # *** NO NEED TO REASSIGN key at the end of the loop with the new method ***

    # --- 5. Return Final Objective Value ---
    final_nse = best_nse_trial
    if final_nse <= -jnp.inf:
        print(f"Trial {trial.number}: Finished. No valid NSE achieved.")
        return -1.0

    print(f"Trial {trial.number}: Finished successfully. Best NSE = {final_nse:.6f}")
    return final_nse