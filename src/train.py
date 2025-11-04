# src/train_unified.py
# Unified training script handling scenarios with and without buildings,
# with optional GradNorm for dynamic loss weighting.
# Based on the robust logic from optimisation/optimization_train_loop.py

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
import numpy as np 

# --- Use DTYPE from config ---
from src.config import load_config, DTYPE
from src.data import sample_points, get_batches
from src.models import init_model
# --- Updated losses import ---
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_building_bc_loss, compute_data_loss
)
# --- Import GradNorm components ---
from src.gradnorm import (
    GradNormState, init_gradnorm, update_gradnorm_weights, LOSS_FN_MAP,
    get_initial_losses
)
from src.utils import ( 
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation,
    mask_points_inside_building, plot_h_vs_x,
    plot_h_prediction_vs_true_2d
)
from src.physics import h_exact
from src.reporting import print_epoch_stats, log_metrics, print_final_summary


# --- Define Training Step (based on optimization_train_loop.py) ---
def train_step(model: Any, params: FrozenDict, opt_state: Any,
               all_batches: Dict[str, Any],
               weights_dict: Dict[str, float],
               optimiser: optax.GradientTransformation,
               config: FrozenDict,
               data_free: bool = True
               ) -> Tuple[FrozenDict, Any, Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Perform a single training step.
    Returns new params, new opt_state, INDIVIDUAL loss terms (unweighted), and TOTAL weighted loss.
    """
    has_building = "building" in config
    active_loss_keys_base = list(weights_dict.keys())

    def loss_and_individual_terms(p):
        terms = {}
        # Compute losses based on available non-empty batches and active weights
        pde_batch_data = all_batches.get('pde', jnp.empty((0,3), dtype=DTYPE))
        if 'pde' in active_loss_keys_base and pde_batch_data.shape[0] > 0:
            terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config)

        ic_batch_data = all_batches.get('ic', jnp.empty((0,3), dtype=DTYPE))
        if 'ic' in active_loss_keys_base and ic_batch_data.shape[0] > 0:
            terms['ic'] = compute_ic_loss(model, p, ic_batch_data)

        bc_batches = all_batches.get('bc', {})
        if 'bc' in active_loss_keys_base and any(b.shape[0] > 0 for b in bc_batches.values() if hasattr(b, 'shape') and b.shape[0] > 0):
             terms['bc'] = compute_bc_loss(
                 model, p, 
                 bc_batches.get('left', jnp.empty((0,3), dtype=DTYPE)),
                 bc_batches.get('right', jnp.empty((0,3), dtype=DTYPE)),
                 bc_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                 bc_batches.get('top', jnp.empty((0,3), dtype=DTYPE)),
                 config
             )

        if has_building and 'building_bc' in active_loss_keys_base:
            bldg_batches = all_batches.get('building_bc', {})
            if bldg_batches and any(b.shape[0] > 0 for b in bldg_batches.values() if hasattr(b, 'shape') and b.shape[0] > 0):
                terms['building_bc'] = compute_building_bc_loss(
                    model, p, 
                    bldg_batches.get('left', jnp.empty((0,3), dtype=DTYPE)),
                    bldg_batches.get('right', jnp.empty((0,3), dtype=DTYPE)),
                    bldg_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                    bldg_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
                )

        data_batch_data = all_batches.get('data', jnp.empty((0,6), dtype=DTYPE))
        if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0:
             terms['data'] = compute_data_loss(model, p, data_batch_data, config)

        # Calculate weighted total loss
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms # Return only computed terms

    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, individual_terms_val, total_loss_val

# JIT the training step
train_step_jitted = jax.jit(
    train_step,
    static_argnames=('model', 'optimiser', 'config', 'data_free')
)


def main(config_path: str):
    """Main training loop for the PINN (Handles building/no-building, data-driven/data-free, and optional GradNorm)."""
    cfg_dict = load_config(config_path) # DTYPE is set globally here
    cfg = FrozenDict(cfg_dict)
    has_building = "building" in cfg

    # --- Scenario-specific warnings ---
    if has_building and "scenario" not in cfg:
        print("Warning: 'building' section found, but 'scenario' key is missing. Data paths might be incorrect.")
    if not has_building:
        print("Info: No 'building' section found in config. Running in no-building mode.")
        if cfg.get("loss_weights", {}).get("building_bc_weight", 0.0) > 0:
            print("Warning: 'building_bc_weight' > 0 but no 'building' section in config. Building BC loss will not be calculated.")

    # --- Model Initialization ---
    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e

    key = random.PRNGKey(cfg["training"]["seed"])
    model_key, init_key, train_key = random.split(key, 3)
    model, params = init_model(model_class, model_key, cfg)

    # --- Setup Directories and Tracking ---
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)

    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- Optimizer Setup ---
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=cfg["training"]["learning_rate"],
        boundaries_and_scales=cfg.get("training", {}).get("lr_boundaries", {15000: 0.1, 30000: 0.1})
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(cfg.get("training", {}).get("clip_norm", 1.0)),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimiser.init(params)

    # --- Prepare Initial Weights and GradNorm Config ---
    static_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    
    gradnorm_cfg = cfg.get("gradnorm", {})
    enable_gradnorm = gradnorm_cfg.get("enable", False) 
    gradnorm_alpha = gradnorm_cfg.get("alpha", 1.5)
    gradnorm_lr = gradnorm_cfg.get("learning_rate", 0.01) # LR for weight updates
    gradnorm_update_freq = gradnorm_cfg.get("update_freq", 100)
    
    gradnorm_state = None

    # --- Load Data (Separate Training and Validation) ---
    val_points, h_true_val = None, None # For validation metrics
    data_points_full = None # For training data loss term
    scenario_name = cfg.get('scenario', 'default_scenario') 
    base_data_path = os.path.join("data", scenario_name)

    # --- NEW: Check for explicit data_free flag in config (like optimization loop) ---
    data_free_flag = cfg.get("data_free", None) # Look for top-level flag
    
    if data_free_flag is True:
        print("Info: 'data_free: true' found in config. Data loss term will be disabled.")
        has_data_loss = False
        data_free = True
    elif data_free_flag is False:
        print("Info: 'data_free: false' found in config. Attempting to load data for data loss term.")
        has_data_loss = True # Will attempt to load file
        data_free = False
    else: # data_free_flag is None (not present)
        print("Info: 'data_free' flag not found in config. Inferring from 'data_weight'.")
        # Use original logic based on weight
        has_data_loss = 'data' in static_weights_dict and static_weights_dict['data'] > 0
        data_free = not has_data_loss

    # --- Load Training Data (for data loss term) ---
    # This block now runs if has_data_loss is True (from flag or weight)
    training_data_file = os.path.join(base_data_path, "training_dataset_sample.npy")
    if has_data_loss: # Only attempt load if data loss is intended
        if os.path.exists(training_data_file):
            try:
                print(f"Loading TRAINING data from: {training_data_file}")
                data_points_full = jnp.load(training_data_file).astype(DTYPE) 
                if data_points_full.shape[0] == 0:
                     print("Warning: Training data file is empty. Disabling data loss.")
                     data_points_full = None
                     has_data_loss = False # <-- Update flag
                else:
                     print(f"Using {data_points_full.shape[0]} points for data loss term (weight={static_weights_dict.get('data', 0.0):.2e}).")
            except Exception as e:
                print(f"Error loading training data file {training_data_file}: {e}")
                print("Disabling data loss term due to loading error.")
                data_points_full = None
                has_data_loss = False # <-- Update flag
        else:
            print(f"Warning: Training data file not found at {training_data_file}.")
            print("Data loss term cannot be computed and will be disabled.")
            has_data_loss = False # <-- Update flag
    
    # This is the final, definitive flag, accounting for file existence
    data_free = not has_data_loss

    # --- Load Validation Data (for NSE/RMSE metrics) ---
    validation_data_file = os.path.join(base_data_path, "validation_sample.npy")
    validation_data_loaded = False
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
                if num_masked_val_points > 0:
                    validation_data_loaded = True
                else:
                     print("Warning: No validation points remaining after masking. NSE/RMSE calculation will be skipped for building case.")
            else: # No building
                 val_points = loaded_val_data[:, [1, 2, 0]]
                 h_true_val = loaded_val_data[:, 3]
                 print(f"Using all {val_points.shape[0]} validation points for NSE/RMSE (no building).")
                 if val_points.shape[0] > 0:
                    validation_data_loaded = True

        except Exception as e:
            print(f"Error loading or processing validation data file {validation_data_file}: {e}")
            val_points, h_true_val = None, None
            print("NSE/RMSE calculation using loaded data will be skipped.")
    else:
        print(f"Warning: Validation data file not found at {validation_data_file}.")
        if has_building:
            print("Validation metrics (NSE/RMSE) for building scenario will be skipped.")
        else:
            print("Validation metrics (NSE/RMSE) will use analytical solution (if PDE points exist).")
    # --- End Data Loading ---

    # --- Initialize Aim Run (with error handling) ---
    # *** BUG FIX: Initialize variables to None before try block ***
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
        # Log hparams (create dict for logging)
        hparams_to_log = {**cfg_dict} # Copy config
        hparams_to_log["data_free_config_flag"] = data_free_flag # Log the flag itself (True, False, or None)
        hparams_to_log["data_loss_active_final"] = has_data_loss # Log the *actual* data loss status
        aim_run["hparams"] = hparams_to_log
        print(f"Aim tracking initialized for run: {trial_name} ({run_hash})")
    except Exception as e:
        print(f"Warning: Failed to initialize Aim tracking: {e}. Training will continue without Aim.")


    # --- Determine Active Loss Keys ---
    active_loss_term_keys = []
    for k, v in static_weights_dict.items():
        if v > 0:
            if k == 'data' and data_free: # Check the *final* data_free flag
                continue 
            active_loss_term_keys.append(k)
    
    # This dict holds the weights, either static or dynamic
    current_weights_dict = {k: static_weights_dict[k] for k in active_loss_term_keys}

    # --- GradNorm Initialization (if enabled) ---
    if enable_gradnorm:
        print("GradNorm enabled. Initializing dynamic weights...")
        key, pde_key, ic_key, bc_keys, bldg_keys, data_key_init = random.split(init_key, 6)
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        batch_size_init = cfg["training"]["batch_size"]

        domain_cfg = cfg["domain"]; grid_cfg = cfg["grid"]; ic_bc_grid_cfg = cfg["ic_bc_grid"]
        init_batches = {} # This dict will hold one batch for each active loss

        if 'pde' in active_loss_term_keys:
            pde_points_init = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"], pde_key)
            if pde_points_init.shape[0] > 0: init_batches['pde'] = get_batches(pde_key, pde_points_init, batch_size_init)[0]
        
        if 'ic' in active_loss_term_keys:
            ic_points_init = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., 0., ic_bc_grid_cfg["nx_ic"], ic_bc_grid_cfg["ny_ic"], 1, ic_key)
            if ic_points_init.shape[0] > 0: init_batches['ic'] = get_batches(ic_key, ic_points_init, batch_size_init)[0]

        if 'bc' in active_loss_term_keys:
            bc_batches_init = {}
            bc_batches_init['left'] = get_batches(l_key, sample_points(0., 0., 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_left"], ic_bc_grid_cfg["nt_bc_left"], l_key), batch_size_init)[0]
            bc_batches_init['right'] = get_batches(r_key, sample_points(domain_cfg["lx"], domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_right"], ic_bc_grid_cfg["nt_bc_right"], r_key), batch_size_init)[0]
            bc_batches_init['bottom'] = get_batches(b_key, sample_points(0., domain_cfg["lx"], 0., 0., 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_bottom"], 1, ic_bc_grid_cfg["nt_bc_other"], b_key), batch_size_init)[0]
            bc_batches_init['top'] = get_batches(t_key, sample_points(0., domain_cfg["lx"], domain_cfg["ly"], domain_cfg["ly"], 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_top"], 1, ic_bc_grid_cfg["nt_bc_other"], t_key), batch_size_init)[0]
            init_batches['bc'] = {k: (v if v.shape[0] > 0 else jnp.empty((0,3), dtype=DTYPE)) for k, v in bc_batches_init.items() if v.shape[0] > 0}

        if has_building and 'building_bc' in active_loss_term_keys:
            bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)
            b_cfg = cfg["building"]
            bldg_batches_init = {}
            bldg_batches_init['left'] = get_batches(bldg_l_key, sample_points(b_cfg["x_min"], b_cfg["x_min"], b_cfg["y_min"], b_cfg["y_max"], 0., domain_cfg["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_l_key), batch_size_init)[0]
            bldg_batches_init['right'] = get_batches(bldg_r_key, sample_points(b_cfg["x_max"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_max"], 0., domain_cfg["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_r_key), batch_size_init)[0]
            bldg_batches_init['bottom'] = get_batches(bldg_b_key, sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_min"], 0., domain_cfg["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_b_key), batch_size_init)[0]
            bldg_batches_init['top'] = get_batches(bldg_t_key, sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_max"], b_cfg["y_max"], 0., domain_cfg["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_t_key), batch_size_init)[0]
            init_batches['building_bc'] = {k: (v if v.shape[0] > 0 else jnp.empty((0,3), dtype=DTYPE)) for k, v in bldg_batches_init.items() if v.shape[0] > 0}


        if not data_free and 'data' in active_loss_term_keys: 
             if data_points_full is not None and data_points_full.shape[0] > 0:
                 init_batches['data'] = get_batches(data_key_init, data_points_full, batch_size_init)[0]
        
        # Filter init_batches to only include non-empty, active batches
        relevant_init_batches = {
            k: init_batches[k] for k in active_loss_term_keys if k in init_batches and 
            (isinstance(init_batches[k], jnp.ndarray) and init_batches[k].shape[0] > 0) or
            (isinstance(init_batches[k], dict) and any(b.shape[0] > 0 for b in init_batches[k].values() if isinstance(b, jnp.ndarray)))
        }
        active_loss_term_keys = list(relevant_init_batches.keys()) # Update keys to only those with batches
        print(f"GradNorm active keys for init: {active_loss_term_keys}")

        # Calculate L_i(0) - run outside JIT
        with jax.disable_jit():
            initial_losses = get_initial_losses(model, params, relevant_init_batches, cfg)

        # Initialize GradNorm state using only the active keys
        gradnorm_state = init_gradnorm(
            loss_keys=list(initial_losses.keys()), # Use keys from computed initial losses
            initial_losses=initial_losses,
            gradnorm_lr=gradnorm_lr
        )
        # Update current_weights_dict with initial GradNorm weights
        current_weights_dict = {key: float(w) for key, w in zip(initial_losses.keys(), gradnorm_state.weights)}
        # Ensure all active keys are present, even if init failed (fallback to 1.0)
        for k in active_loss_term_keys:
            if k not in current_weights_dict:
                current_weights_dict[k] = 1.0
        
        print(f"GradNorm initialized. Initial Weights: {current_weights_dict}")
    else:
         print(f"GradNorm disabled. Using Static Weights: {current_weights_dict}")
    
    # --- Aim Run hparams update (moved after data loading) ---
    if aim_run:
        try:
            # Log the final active loss terms
            aim_run["hparams"]["active_loss_terms_final"] = active_loss_term_keys
        except Exception as e:
            print(f"Warning: Could not update Aim hparams: {e}")

    # --- Training Initialization ---
    print(f"\n--- Training Started: {trial_name} ---")
    print(f"Model: {cfg['model']['name']}, Epochs: {cfg['training']['epochs']}, Batch Size: {cfg['training']['batch_size']}")
    print(f"Scenario: {'Building' if has_building else 'No Building'}")
    print(f"Saving results to: {results_dir}")
    print(f"Saving model to: {model_dir}")
    print(f"GradNorm Enabled: {enable_gradnorm}")
    print(f"Data Loss Active: {has_data_loss} (Final Data-Free: {data_free})")
    print(f"Active Loss Terms: {active_loss_term_keys}")
    print(f"Initial Weights: {current_weights_dict}")

    best_nse: float = -jnp.inf
    best_epoch: int = 0
    best_params: Dict = None
    best_nse_time: float = 0.0
    global_step = 0 
    start_time = time.time()

    # --- Main Training Loop ---
    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()

            # --- Dynamic Sampling ---
            key, pde_key, ic_key, bc_keys, bldg_keys, data_key_epoch = random.split(key, 6)
            l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
            domain_cfg = cfg["domain"]; grid_cfg = cfg["grid"]; ic_bc_grid_cfg = cfg["ic_bc_grid"]

            pde_points = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], grid_cfg["nx"], grid_cfg["ny"], grid_cfg["nt"], pde_key) if 'pde' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            ic_points = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., 0., ic_bc_grid_cfg["nx_ic"], ic_bc_grid_cfg["ny_ic"], 1, ic_key) if 'ic' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            left_wall = sample_points(0., 0., 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_left"], ic_bc_grid_cfg["nt_bc_left"], l_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            right_wall = sample_points(domain_cfg["lx"], domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_right"], ic_bc_grid_cfg["nt_bc_right"], r_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            bottom_wall = sample_points(0., domain_cfg["lx"], 0., 0., 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_bottom"], 1, ic_bc_grid_cfg["nt_bc_other"], b_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            top_wall = sample_points(0., domain_cfg["lx"], domain_cfg["ly"], domain_cfg["ly"], 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_top"], 1, ic_bc_grid_cfg["nt_bc_other"], t_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)

            building_points = {}
            if has_building and 'building_bc' in active_loss_term_keys:
                bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)
                b_cfg = cfg["building"]
                building_points['left'] = sample_points(b_cfg["x_min"], b_cfg["x_min"], b_cfg["y_min"], b_cfg["y_max"], 0., domain_cfg["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_l_key)
                building_points['right'] = sample_points(b_cfg["x_max"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_max"], 0., domain_cfg["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_r_key)
                building_points['bottom'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_min"], 0., domain_cfg["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_b_key)
                building_points['top'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_max"], b_cfg["y_max"], 0., domain_cfg["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_t_key)

            # --- Create Batches ---
            batch_size = cfg["training"]["batch_size"]
            key, pde_b_key, ic_b_key, bc_b_keys, bldg_b_keys, data_b_key_epoch = random.split(key, 6)
            l_b_key, r_b_key, b_b_key, t_b_key = random.split(bc_b_keys, 4)

            pde_batches = get_batches(pde_b_key, pde_points, batch_size) if pde_points.shape[0] > 0 else []
            ic_batches = get_batches(ic_b_key, ic_points, batch_size) if ic_points.shape[0] > 0 else []
            left_batches = get_batches(l_b_key, left_wall, batch_size) if left_wall.shape[0] > 0 else []
            right_batches = get_batches(r_b_key, right_wall, batch_size) if right_wall.shape[0] > 0 else []
            bottom_batches = get_batches(b_b_key, bottom_wall, batch_size) if bottom_wall.shape[0] > 0 else []
            top_batches = get_batches(t_b_key, top_wall, batch_size) if top_wall.shape[0] > 0 else []

            data_batches = []
            if not data_free and data_points_full is not None:
                 data_batches = get_batches(data_b_key_epoch, data_points_full, batch_size)

            building_batches_dict = {}
            if has_building and 'building_bc' in active_loss_term_keys:
                 bldg_l_b_key, bldg_r_b_key, bldg_b_b_key, bldg_t_b_key = random.split(bldg_b_keys, 4)
                 building_b_keys_map = {'left': bldg_l_b_key, 'right': bldg_r_b_key, 'bottom': bldg_b_b_key, 'top': bldg_t_b_key}
                 for wall, points in building_points.items():
                     building_batches_dict[wall] = get_batches(building_b_keys_map[wall], points, batch_size) if points.shape[0] > 0 else []

            # --- Determine Number of Batches (driving loop) ---
            all_batch_lists = [pde_batches, ic_batches, left_batches, right_batches, bottom_batches, top_batches, data_batches]
            all_batch_lists.extend(building_batches_dict.values())
            num_batches = max([len(b_list) for b_list in all_batch_lists if b_list], default=0)

            if num_batches == 0:
                 print(f"Warning: Epoch {epoch+1} - No batches generated for active terms. Skipping epoch.")
                 continue

            # --- Batch Iterators (use cycle for all) ---
            pde_batch_iter = itertools.cycle(pde_batches) if pde_batches else iter(())
            ic_batch_iter = itertools.cycle(ic_batches) if ic_batches else iter(())
            left_batch_iter = itertools.cycle(left_batches) if left_batches else iter(())
            right_batch_iter = itertools.cycle(right_batches) if right_batches else iter(())
            bottom_batch_iter = itertools.cycle(bottom_batches) if bottom_batches else iter(())
            top_batch_iter = itertools.cycle(top_batches) if top_batches else iter(())
            data_batch_iter = itertools.cycle(data_batches) if data_batches else iter(())

            building_batch_iters = {}
            if has_building and 'building_bc' in active_loss_term_keys:
                 for wall, batches in building_batches_dict.items():
                     building_batch_iters[wall] = itertools.cycle(batches) if batches else iter(())

            # --- Training Steps within Epoch ---
            epoch_losses_unweighted = {k: 0.0 for k in active_loss_term_keys} # Track unweighted
            epoch_total_weighted_loss = 0.0

            for i in range(num_batches):
                global_step += 1

                # Get batches, ensuring empty tensors if not active or iterator empty
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

                # Aggregate batches into the dictionary structure expected by train_step
                # *** RENAMED for consistency ***
                current_all_batches = {
                    'pde': pde_batch_data,
                    'ic': ic_batch_data,
                    'bc': {'left': left_batch_data, 'right': right_batch_data, 'bottom': bottom_batch_data, 'top': top_batch_data},
                    'building_bc': current_building_batch_data,
                    'data': data_batch_data,
                }
                
                # --- GradNorm Weight Update (Periodically, Outside JIT) ---
                if enable_gradnorm and global_step % gradnorm_update_freq == 0:
                    # Filter batches to only those GradNorm is tracking
                    active_batches_for_gradnorm = {
                        k: current_all_batches[LOSS_FN_MAP[k]['batch_key']] 
                        for k in active_loss_term_keys if k in LOSS_FN_MAP
                    }
                    
                    with jax.disable_jit(): # GradNorm update itself should be outside JIT
                         gradnorm_state, current_weights_dict = update_gradnorm_weights(
                              gradnorm_state, params, model, active_batches_for_gradnorm,
                              cfg, gradnorm_alpha, gradnorm_lr
                         )

                # --- Training Step ---
                # *** BUG FIX: Pass the dictionary, not individual arrays ***
                params, opt_state, batch_losses_unweighted, batch_total_weighted_loss = train_step_jitted(
                    model, params, opt_state,
                    current_all_batches,  # Pass the dictionary
                    current_weights_dict, # Pass (potentially dynamic) weights
                    optimiser, cfg, data_free # Pass data_free flag
                )

                # Accumulate unweighted losses for epoch average
                for k in active_loss_term_keys:
                    epoch_losses_unweighted[k] += float(batch_losses_unweighted.get(k, 0.0))
                epoch_total_weighted_loss += float(batch_total_weighted_loss)

            # --- Epoch End Calculation & Validation ---
            avg_losses_unweighted = {k: v / num_batches for k, v in epoch_losses_unweighted.items()}
            avg_total_weighted_loss = epoch_total_weighted_loss / num_batches

            nse_val, rmse_val = -jnp.inf, jnp.inf
            
            # *** FIX: Removed 'with jax.disable_jit():' wrapper ***
            # This logic now runs in the main Python loop (eager mode),
            # which is consistent with optimization_train_loop.py.
            if validation_data_loaded: # Use loaded validation data
                U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                h_pred_val = U_pred_val[..., 0]
                h_true_mean = jnp.mean(h_true_val)
                denominator_nse = jnp.sum((h_true_val - h_true_mean)**2)
                if denominator_nse > cfg.get("numerics", {}).get("eps", 1e-9):
                     numerator_nse = jnp.sum((h_true_val - h_pred_val)**2)
                     nse_val = float(1.0 - numerator_nse / denominator_nse)
                rmse_val = float(rmse(h_pred_val, h_true_val))
            
            elif not has_building and pde_points.shape[0] > 0: # Fallback: No building, no validation file -> use analytical
                if (epoch + 1) % 100 == 0: # Only log this message periodically
                    print(f"Info: No validation data. Using analytical solution on PDE points for NSE/RMSE.")
                U_pred_val_no_building = model.apply({'params': params['params']}, pde_points, train=False)
                h_pred_val_no_building = U_pred_val_no_building[..., 0]
                h_true_val_no_building = h_exact(pde_points[:, 0], pde_points[:, 2], cfg["physics"]["n_manning"], cfg["physics"]["u_const"])
                h_true_mean = jnp.mean(h_true_val_no_building)
                denominator_nse = jnp.sum((h_true_val_no_building - h_true_mean)**2)
                if denominator_nse > cfg.get("numerics", {}).get("eps", 1e-9):
                    numerator_nse = jnp.sum((h_true_val_no_building - h_pred_val_no_building)**2)
                    nse_val = float(1.0 - numerator_nse / denominator_nse)
                rmse_val = float(rmse(h_pred_val_no_building, h_true_val_no_building))
            # --- End of validation block ---

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
                    epoch, start_time, avg_total_weighted_loss,
                    avg_losses_unweighted.get('pde', 0.0), 
                    avg_losses_unweighted.get('ic', 0.0), 
                    avg_losses_unweighted.get('bc', 0.0),
                    avg_losses_unweighted.get('building_bc', 0.0),
                    avg_losses_unweighted.get('data', 0.0), # Pass unweighted data loss
                    nse_val, rmse_val, epoch_time
                )
                if enable_gradnorm:
                     print(f"      Current Weights: { {k: f'{v:.2e}' for k, v in current_weights_dict.items()} }")


            if aim_run:
                try:
                    metrics_to_log = {
                        'total_loss': avg_total_weighted_loss, # Log weighted total
                        'pde_loss': avg_losses_unweighted.get('pde', 0.0),
                        'ic_loss': avg_losses_unweighted.get('ic', 0.0),
                        'bc_loss': avg_losses_unweighted.get('bc', 0.0),
                        'building_bc_loss': avg_losses_unweighted.get('building_bc', 0.0),
                        'data_loss': avg_losses_unweighted.get('data', 0.0), # Log unweighted
                        'nse': nse_val, 'rmse': rmse_val, 'epoch_time': epoch_time
                    }
                    if enable_gradnorm:
                        for k, w in current_weights_dict.items():
                            metrics_to_log[f'weight_{k}'] = float(w)
                    
                    log_metrics(aim_run, metrics_to_log, epoch)
                except Exception as e:
                    print(f"Warning: Failed to log metrics to Aim in epoch {epoch+1}: {e}")

            # --- Early Stopping Check ---
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))

            if epoch >= min_epochs and (epoch - best_epoch) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                print(f"Best NSE {best_nse:.6f} achieved at epoch {best_epoch+1}.")
                break

            # Update train_key for next epoch's sampling
            train_key = key

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
                    print("Generating final plot...")
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

                    print(f"Model and comparison plot saved in {model_dir} and {results_dir}")
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
                    # Note: You must have imported aim_repo and run_hash from the try block
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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., experiments/one_building_config.yaml)")
    args = parser.parse_args()

    try:
        final_nse = main(args.config) # Call main
        print(f"\n--- Script Finished ---")
        if isinstance(final_nse, (float, int)) and final_nse > -jnp.inf:
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