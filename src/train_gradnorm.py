# src/train_gradnorm.py
import os
import jax
import jax.numpy as jnp
from jax import random
import optax
from aim import Repo, Run
from flax.core import FrozenDict
import numpy as np
import time
import copy
import itertools
import shutil
import argparse
import importlib
from typing import Any, Dict, Tuple

# --- Use DTYPE from config ---
from src.config import load_config, DTYPE
from src.data import sample_points, get_batches
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_building_bc_loss#, compute_data_loss # Removed data loss import
)
# --- Import GradNorm ---
from src.gradnorm import GradNormState, init_gradnorm, update_gradnorm_weights, LOSS_FN_MAP

from src.utils import (
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation,
    mask_points_inside_building, plot_h_vs_x,
    plot_h_prediction_vs_true_2d
)
from src.physics import h_exact
from src.reporting import print_epoch_stats, log_metrics, print_final_summary

# --- Modified train_step ---
# Now accepts weights_dict directly
def train_step(model: Any, params: FrozenDict, opt_state: Any,
               all_batches: Dict[str, Any], # Pass all batches needed
               weights_dict: Dict[str, float], # Current dynamic weights
               optimiser: optax.GradientTransformation,
               config: FrozenDict
               ) -> Tuple[FrozenDict, Any, Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Perform a single training step using the provided dynamic weights.
    Returns new params, new opt_state, INDIVIDUAL loss terms (unweighted), and TOTAL weighted loss.
    Data loss term is explicitly excluded.
    """
    has_building = "building" in config
    # Get keys from config *excluding* data_weight
    loss_keys_in_config = [k for k in config["loss_weights"].keys() if k != 'data_weight']

    def loss_and_individual_terms(p):
        terms = {}
        # Compute individual losses based on available batches and config weights (excluding data)
        # PDE
        if 'pde_weight' in loss_keys_in_config and all_batches['pde'].shape[0] > 0:
            terms['pde'] = compute_pde_loss(model, p, all_batches['pde'], config)
        # IC
        if 'ic_weight' in loss_keys_in_config and all_batches['ic'].shape[0] > 0:
            terms['ic'] = compute_ic_loss(model, p, all_batches['ic'])
        # BC (Domain)
        bc_batches = all_batches['bc']
        if 'bc_weight' in loss_keys_in_config and any(b.shape[0] > 0 for b in bc_batches.values()):
            terms['bc'] = compute_bc_loss(
                 model, p, bc_batches['left'], bc_batches['right'], bc_batches['bottom'], bc_batches['top'], config
             )
        # BC (Building)
        if has_building and 'building_bc_weight' in loss_keys_in_config:
            bldg_batches = all_batches['building_bc']
            if bldg_batches and any(b.shape[0] > 0 for b in bldg_batches.values()):
                terms['building_bc'] = compute_building_bc_loss(
                    model, p,
                    bldg_batches.get('left', jnp.empty((0,3), dtype=DTYPE)),
                    bldg_batches.get('right', jnp.empty((0,3), dtype=DTYPE)),
                    bldg_batches.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                    bldg_batches.get('top', jnp.empty((0,3), dtype=DTYPE))
                )

        # Calculate weighted total loss using CURRENT dynamic weights_dict (excluding data)
        active_loss_keys = list(terms.keys())
        active_weights = {k: weights_dict.get(k, 0.0) for k in active_loss_keys if k != 'data'}
        # Provide default 0.0 for terms not computed in this step (excluding data)
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys() if k != 'data'}

        total = total_loss(terms_with_defaults, active_weights) # Use active weights here
        return total, terms # Return terms dict containing only computed losses

    # Compute total loss and gradients w.r.t model parameters (params)
    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)

    # Update model parameters
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Return individual (unweighted) losses and the total weighted loss for logging
    return new_params, new_opt_state, individual_terms_val, total_loss_val

# Apply JIT functionally
train_step_jitted = jax.jit(
    train_step,
    static_argnames=('model', 'optimiser', 'config')
)

# --- Function to get initial losses (run once before training, NO JIT) ---
def get_initial_losses(model: Any, params: FrozenDict, all_batches: Dict[str, Any], config: FrozenDict) -> Dict[str, float]:
    """Computes the initial value for each loss term (excluding data loss)."""
    initial_losses = {}
    # Use keys from config *excluding* data_weight
    loss_keys = [k for k in config["loss_weights"].keys() if k != 'data_weight']
    has_building = "building" in config

    print("Calculating initial losses (L_i(0))...")
    for key in loss_keys:
        loss_key_base = key.replace('_weight', '') # e.g., 'pde_weight' -> 'pde'
        if loss_key_base == 'data': # Explicitly skip data loss
            continue
        if loss_key_base not in LOSS_FN_MAP:
            print(f"Warning: Loss key '{loss_key_base}' derived from config not in LOSS_FN_MAP. Skipping initial loss calculation.")
            continue

        loss_info = LOSS_FN_MAP[loss_key_base]
        loss_func = loss_info['func']
        batch_key = loss_info['batch_key']
        batch_data = all_batches.get(batch_key)

        # Skip if batch is empty/missing
        is_empty_batch = False
        if batch_data is None: is_empty_batch = True
        elif isinstance(batch_data, jnp.ndarray) and batch_data.shape[0] == 0: is_empty_batch = True
        elif isinstance(batch_data, dict) and not any(b.shape[0] > 0 for b in batch_data.values() if isinstance(b, jnp.ndarray)): is_empty_batch = True

        if is_empty_batch:
            initial_losses[loss_key_base] = 0.0
            print(f"  Initial loss for {loss_key_base}: 0.0 (empty batch)")
            continue

        # Compute the loss value
        try:
            # Need to call the specific loss functions correctly
            p_frozen = FrozenDict({'params': params['params']}) # Wrap params correctly
            if loss_key_base == 'pde':
                 loss_val = compute_pde_loss(model, p_frozen, batch_data, config)
            elif loss_key_base == 'ic':
                 loss_val = compute_ic_loss(model, p_frozen, batch_data)
            elif loss_key_base == 'bc':
                 loss_val = compute_bc_loss(model, p_frozen, batch_data['left'], batch_data['right'], batch_data['bottom'], batch_data['top'], config)
            elif loss_key_base == 'building_bc':
                 loss_val = compute_building_bc_loss(model, p_frozen,
                                                      batch_data.get('left', jnp.empty((0,3), dtype=DTYPE)),
                                                      batch_data.get('right', jnp.empty((0,3), dtype=DTYPE)),
                                                      batch_data.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                                                      batch_data.get('top', jnp.empty((0,3), dtype=DTYPE)))
            else:
                 # Fallback for potentially other losses, though unlikely needed now
                 loss_val = loss_func(p_frozen, model, batch_data, config)

            initial_losses[loss_key_base] = max(float(loss_val), 1e-8) # Store as float, ensure minimum value
            print(f"  Initial loss for {loss_key_base}: {initial_losses[loss_key_base]:.4e}")
        except Exception as e:
            print(f"  Error calculating initial loss for {loss_key_base}: {e}. Setting to 1e-8.")
            initial_losses[loss_key_base] = 1e-8 # Default small value on error

    # Ensure all relevant keys (excluding data) are present
    final_initial_losses = {}
    for cfg_key in config['loss_weights']:
        base_key = cfg_key.replace('_weight', '')
        if base_key != 'data': # Exclude data key
            final_initial_losses[base_key] = initial_losses.get(base_key, 1e-8) # Use default if calculation failed or skipped

    print(f"Final Initial Losses (excluding data): {final_initial_losses}")
    return final_initial_losses


def main(config_path: str):
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)
    has_building = "building" in cfg

    # --- GradNorm Configuration ---
    gradnorm_cfg = cfg.get("gradnorm", {})
    enable_gradnorm = gradnorm_cfg.get("enable", False)
    gradnorm_alpha = gradnorm_cfg.get("alpha", 1.5)
    gradnorm_lr = gradnorm_cfg.get("learning_rate", 0.025) # LR for weight updates
    gradnorm_update_freq = gradnorm_cfg.get("update_freq", 100) # Update weights every N steps

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
    os.makedirs(results_dir, exist_ok=True); os.makedirs(model_dir, exist_ok=True)
    # --- Aim Init ---
    aim_repo = None; aim_run = None; run_hash = None
    try:
        aim_repo_path = "aim_repo"
        if not os.path.exists(aim_repo_path): os.makedirs(aim_repo_path, exist_ok=True)
        aim_repo = Repo(path=aim_repo_path, init=True)
        aim_run = Run(repo=aim_repo, experiment=trial_name)
        run_hash = aim_run.hash
        hparams_to_log = copy.deepcopy(cfg_dict)
        hparams_to_log["data_free_training"] = True # Indicate data-free
        if enable_gradnorm:
             hparams_to_log["gradnorm_enabled"] = True
             hparams_to_log["gradnorm_alpha"] = gradnorm_alpha
             hparams_to_log["gradnorm_lr"] = gradnorm_lr
             hparams_to_log["gradnorm_update_freq"] = gradnorm_update_freq
        else:
             hparams_to_log["gradnorm_enabled"] = False
        # Remove data_weight from logged hparams if present
        hparams_to_log.get('loss_weights', {}).pop('data_weight', None)
        aim_run["hparams"] = hparams_to_log
        print(f"Aim tracking initialized for run: {trial_name} ({run_hash})")
    except Exception as e: print(f"Warning: Failed to initialize Aim tracking: {e}.")


    # --- Optimizer Setup (for model parameters) ---
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=cfg["training"]["learning_rate"],
        boundaries_and_scales={15000: 0.1, 30000: 0.1}
    )
    optimiser = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule))
    opt_state = optimiser.init(params)

    # --- Prepare Initial Loss Weights ---
    # Get relevant keys excluding data_weight
    loss_keys_from_config = [k for k in cfg["loss_weights"].keys() if k != 'data_weight']
    active_loss_term_keys = [k.replace('_weight', '') for k in loss_keys_from_config]

    if enable_gradnorm:
        print("GradNorm enabled. Initializing dynamic weights (excluding data).")
        # Initialize weights to 1.0; they will be updated by GradNormState init
        current_weights_dict = {key: 1.0 for key in active_loss_term_keys}
    else:
        # Use static weights from config (excluding data weight)
        print("Using static weights from config (excluding data).")
        current_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items() if k != 'data_weight'}

    # Data loss is explicitly disabled
    has_data_loss = False

    # --- Load Validation Data (ONLY for NSE/RMSE) ---
    val_points, h_true_val = None, None
    scenario_name = cfg.get('scenario', 'default_scenario')
    base_data_path = os.path.join("data", scenario_name)
    validation_data_file = os.path.join(base_data_path, "validation_sample.npy")

    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file} (for NSE/RMSE only)")
            loaded_val_data = jnp.load(validation_data_file).astype(DTYPE)

            if loaded_val_data.shape[1] != 6:
                print(f"Warning: Validation data expected 6 columns [t,x,y,h,u,v], found {loaded_val_data.shape[1]}. NSE/RMSE might be incorrect if columns are wrong.")

            val_points_all = loaded_val_data[:, [1, 2, 0]] # Input points (x, y, t)
            h_true_val_all = loaded_val_data[:, 3]       # True water depth h

            if has_building:
                print("Applying building mask to validation points for NSE/RMSE...")
                mask_val = mask_points_inside_building(val_points_all, cfg["building"])
                val_points = val_points_all[mask_val]
                h_true_val = h_true_val_all[mask_val]
                print(f"Masked validation points remaining: {val_points.shape[0]}.")
                if val_points.shape[0] == 0:
                     print("Warning: No validation points remaining after masking. NSE/RMSE calculation will be skipped for building case.")
            else:
                 # For no-building case, use all loaded validation points
                 val_points = val_points_all
                 h_true_val = h_true_val_all
                 print(f"Using all {val_points.shape[0]} validation points for NSE/RMSE (no building mask).")

        except Exception as e:
            print(f"Error loading or processing validation data file {validation_data_file}: {e}")
            val_points, h_true_val = None, None
            print("NSE/RMSE calculation will be skipped due to validation data loading error.")
    else:
        print(f"Warning: Validation data file not found at {validation_data_file}.")
        print("NSE/RMSE calculation will be skipped.")

    # --- GradNorm Initialization (if enabled) ---
    gradnorm_state = None
    if enable_gradnorm:
        print("Performing initial step to get L_i(0) for GradNorm...")
        # Sample one set of points/batches for initialization (excluding data)
        key, pde_key, ic_key, bc_keys, bldg_keys = random.split(init_key, 5) # No data_key needed
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        pde_points_init = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["grid"]["nx"], cfg["grid"]["ny"], cfg["grid"]["nt"], pde_key)
        ic_points_init = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., 0., cfg["ic_bc_grid"]["nx_ic"], cfg["ic_bc_grid"]["ny_ic"], 1, ic_key)
        left_wall_init = sample_points(0., 0., 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_left"], cfg["ic_bc_grid"]["nt_bc_left"], l_key)
        right_wall_init = sample_points(cfg["domain"]["lx"], cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_right"], cfg["ic_bc_grid"]["nt_bc_right"], r_key)
        bottom_wall_init = sample_points(0., cfg["domain"]["lx"], 0., 0., 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_bottom"], 1, cfg["ic_bc_grid"]["nt_bc_other"], b_key)
        top_wall_init = sample_points(0., cfg["domain"]["lx"], cfg["domain"]["ly"], cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_top"], 1, cfg["ic_bc_grid"]["nt_bc_other"], t_key)
        building_points_init = {}
        if has_building:
            bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)
            b_cfg = cfg["building"]
            building_points_init['left'] = sample_points(b_cfg["x_min"], b_cfg["x_min"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_l_key)
            building_points_init['right'] = sample_points(b_cfg["x_max"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_r_key)
            building_points_init['bottom'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_min"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_b_key)
            building_points_init['top'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_max"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_t_key)

        # Create *one* batch for initialization
        batch_size_init = cfg["training"]["batch_size"]
        key, pde_b_key, ic_b_key, bc_b_keys, bldg_b_keys = random.split(key, 5) # No data_b_key
        l_b_key, r_b_key, b_b_key, t_b_key = random.split(bc_b_keys, 4)

        init_batches = {}
        init_batches['pde'] = get_batches(pde_b_key, pde_points_init, batch_size_init)[0] if pde_points_init.shape[0] > 0 else jnp.empty((0, 3), dtype=DTYPE)
        init_batches['ic'] = get_batches(ic_b_key, ic_points_init, batch_size_init)[0] if ic_points_init.shape[0] > 0 else jnp.empty((0, 3), dtype=DTYPE)
        init_batches['bc'] = {
            'left': get_batches(l_b_key, left_wall_init, batch_size_init)[0] if left_wall_init.shape[0] > 0 else jnp.empty((0, 3), dtype=DTYPE),
            'right': get_batches(r_b_key, right_wall_init, batch_size_init)[0] if right_wall_init.shape[0] > 0 else jnp.empty((0, 3), dtype=DTYPE),
            'bottom': get_batches(b_b_key, bottom_wall_init, batch_size_init)[0] if bottom_wall_init.shape[0] > 0 else jnp.empty((0, 3), dtype=DTYPE),
            'top': get_batches(t_b_key, top_wall_init, batch_size_init)[0] if top_wall_init.shape[0] > 0 else jnp.empty((0, 3), dtype=DTYPE),
        }
        init_batches['building_bc'] = {}
        if has_building:
             bldg_l_b_key, bldg_r_b_key, bldg_b_b_key, bldg_t_b_key = random.split(bldg_b_keys, 4)
             building_b_keys_map = {'left': bldg_l_b_key, 'right': bldg_r_b_key, 'bottom': bldg_b_b_key, 'top': bldg_t_b_key}
             for wall, points in building_points_init.items():
                 init_batches['building_bc'][wall] = get_batches(building_b_keys_map[wall], points, batch_size_init)[0] if points.shape[0] > 0 else jnp.empty((0, 3), dtype=DTYPE)

        # Filter init_batches to only include batches relevant to active loss terms
        relevant_init_batches = {k: init_batches[k] for k in active_loss_term_keys if k in init_batches}

        # Calculate L_i(0) - run outside JIT
        with jax.disable_jit():
            initial_losses = get_initial_losses(model, params, relevant_init_batches, cfg)

        # Initialize GradNorm state
        gradnorm_state = init_gradnorm(
            loss_keys=list(initial_losses.keys()), # Use keys from computed initial losses
            initial_losses=initial_losses,
            gradnorm_lr=gradnorm_lr
        )
        # Update current_weights_dict with initial GradNorm weights
        current_weights_dict = {key: float(w) for key, w in zip(initial_losses.keys(), gradnorm_state.weights)}
        print(f"GradNorm initialized. Initial Weights: {current_weights_dict}")

    # --- Training Initialization ---
    print(f"\n--- Training Started: {trial_name} ---")
    print(f"Model: {cfg['model']['name']}, Epochs: {cfg['training']['epochs']}, Batch Size: {cfg['training']['batch_size']}")
    print(f"Scenario: {'Building' if has_building else 'No Building'}")
    print(f"GradNorm Enabled: {enable_gradnorm}")
    if enable_gradnorm: print(f"GradNorm Alpha: {gradnorm_alpha}, LR: {gradnorm_lr}, Update Freq: {gradnorm_update_freq}")
    print(f"Initial Weights: {current_weights_dict}")
    print("Training is DATA-FREE (using only PDE, IC, BC losses).")
    print(f"Saving results to: {results_dir}, Model to: {model_dir}")

    best_nse = -jnp.inf; best_epoch = 0; best_params = None; best_nse_time = 0.0
    start_time = time.time()
    global_step = 0 # Track total training steps for GradNorm update frequency

    # --- Main Training Loop ---
    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()
            key = train_key # Use the dedicated training key

            # --- Dynamic Sampling (excluding data) ---
            key, pde_key, ic_key, bc_keys, bldg_keys = random.split(key, 5) # No data_key
            l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
            pde_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["grid"]["nx"], cfg["grid"]["ny"], cfg["grid"]["nt"], pde_key)
            ic_points = sample_points(0., cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., 0., cfg["ic_bc_grid"]["nx_ic"], cfg["ic_bc_grid"]["ny_ic"], 1, ic_key)
            left_wall = sample_points(0., 0., 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_left"], cfg["ic_bc_grid"]["nt_bc_left"], l_key)
            right_wall = sample_points(cfg["domain"]["lx"], cfg["domain"]["lx"], 0., cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], 1, cfg["ic_bc_grid"]["ny_bc_right"], cfg["ic_bc_grid"]["nt_bc_right"], r_key)
            bottom_wall = sample_points(0., cfg["domain"]["lx"], 0., 0., 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_bottom"], 1, cfg["ic_bc_grid"]["nt_bc_other"], b_key)
            top_wall = sample_points(0., cfg["domain"]["lx"], cfg["domain"]["ly"], cfg["domain"]["ly"], 0., cfg["domain"]["t_final"], cfg["ic_bc_grid"]["nx_bc_top"], 1, cfg["ic_bc_grid"]["nt_bc_other"], t_key)
            building_points = {}
            if has_building:
                bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)
                b_cfg = cfg["building"]
                building_points['left'] = sample_points(b_cfg["x_min"], b_cfg["x_min"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_l_key)
                building_points['right'] = sample_points(b_cfg["x_max"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], 1, b_cfg["ny"], b_cfg["nt"], bldg_r_key)
                building_points['bottom'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_min"], b_cfg["y_min"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_b_key)
                building_points['top'] = sample_points(b_cfg["x_min"], b_cfg["x_max"], b_cfg["y_max"], b_cfg["y_max"], 0., cfg["domain"]["t_final"], b_cfg["nx"], 1, b_cfg["nt"], bldg_t_key)

            # --- Create Batches (excluding data) ---
            batch_size = cfg["training"]["batch_size"]
            key, pde_b_key, ic_b_key, bc_b_keys, bldg_b_keys = random.split(key, 5) # No data_b_key
            l_b_key, r_b_key, b_b_key, t_b_key = random.split(bc_b_keys, 4)
            pde_batches = get_batches(pde_b_key, pde_points, batch_size)
            ic_batches = get_batches(ic_b_key, ic_points, batch_size)
            left_batches = get_batches(l_b_key, left_wall, batch_size)
            right_batches = get_batches(r_b_key, right_wall, batch_size)
            bottom_batches = get_batches(b_b_key, bottom_wall, batch_size)
            top_batches = get_batches(t_b_key, top_wall, batch_size)
            building_batches_dict = {}
            if has_building:
                bldg_l_b_key, bldg_r_b_key, bldg_b_b_key, bldg_t_b_key = random.split(bldg_b_keys, 4)
                building_b_keys_map = {'left': bldg_l_b_key, 'right': bldg_r_b_key, 'bottom': bldg_b_b_key, 'top': bldg_t_b_key}
                for wall, points in building_points.items():
                    building_batches_dict[wall] = get_batches(building_b_keys_map[wall], points, batch_size) if points.shape[0] > 0 else []

            # --- Batch Iterators (excluding data) ---
            ic_batch_iter = itertools.cycle(ic_batches) if ic_batches else iter(())
            left_batch_iter = itertools.cycle(left_batches) if left_batches else iter(())
            right_batch_iter = itertools.cycle(right_batches) if right_batches else iter(())
            bottom_batch_iter = itertools.cycle(bottom_batches) if bottom_batches else iter(())
            top_batch_iter = itertools.cycle(top_batches) if top_batches else iter(())
            building_batch_iters = {}
            if has_building:
                for wall, batches in building_batches_dict.items():
                    building_batch_iters[wall] = itertools.cycle(batches) if batches else iter(())

            # --- Training Steps within Epoch ---
            num_batches = len(pde_batches)
            epoch_losses_unweighted = {k: 0.0 for k in active_loss_term_keys} # Store unweighted losses
            epoch_total_weighted_loss = 0.0

            if num_batches == 0:
                print(f"Warning: Epoch {epoch+1} - No PDE batches generated. Skipping.")
                continue

            for i in range(num_batches):
                global_step += 1 # Increment step counter

                # Get batch data
                pde_batch_data = pde_batches[i]
                ic_batch_data = next(ic_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                left_batch_data = next(left_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                right_batch_data = next(right_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                bottom_batch_data = next(bottom_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                top_batch_data = next(top_batch_iter, jnp.empty((0, 3), dtype=DTYPE))
                current_building_batch_data = {}
                if has_building:
                    for wall, iterator in building_batch_iters.items():
                        current_building_batch_data[wall] = next(iterator, jnp.empty((0, 3), dtype=DTYPE))

                # Aggregate batches into a dictionary (excluding data)
                current_all_batches = {
                    'pde': pde_batch_data,
                    'ic': ic_batch_data,
                    'bc': {'left': left_batch_data, 'right': right_batch_data, 'bottom': bottom_batch_data, 'top': top_batch_data},
                    'building_bc': current_building_batch_data if has_building else {},
                }
                # Filter empty dicts if building_bc wasn't populated
                current_all_batches = {k: v for k, v in current_all_batches.items() if not (isinstance(v, dict) and not v)}


                # --- GradNorm Weight Update (Periodically, Outside JIT) ---
                if enable_gradnorm and global_step % gradnorm_update_freq == 0:
                    # Removed print statement from here
                    with jax.disable_jit(): # Ensure GradNorm update runs without JIT
                        gradnorm_state, current_weights_dict = update_gradnorm_weights(
                            gradnorm_state=gradnorm_state,
                            model_params=params,
                            model=model,
                            all_batches=current_all_batches, # Use batches from this step
                            config=cfg,
                            alpha=gradnorm_alpha,
                            gradnorm_lr=gradnorm_lr
                        )
                    # Log weights to Aim if enabled
                    if aim_run:
                        try:
                            for k, w in current_weights_dict.items():
                                aim_run.track(float(w), name=f'weight_{k}', step=global_step, context={'subset': 'gradnorm'})
                        except Exception as e: print(f"Warning: Failed to log weights to Aim: {e}")


                # --- Perform Model Parameter Update Step (Jitted) ---
                params, opt_state, batch_losses_unweighted, batch_total_weighted_loss = train_step_jitted(
                    model, params, opt_state,
                    current_all_batches,
                    current_weights_dict, # Pass current weights
                    optimiser, cfg
                )

                # Accumulate unweighted losses for epoch average
                for k in epoch_losses_unweighted:
                    epoch_losses_unweighted[k] += float(batch_losses_unweighted.get(k, 0.0))
                epoch_total_weighted_loss += float(batch_total_weighted_loss)

            # --- Epoch End Calculation & Validation ---
            avg_losses_unweighted = {k: v / num_batches for k, v in epoch_losses_unweighted.items()}
            # Calculate avg total weighted loss using the final weights of the epoch
            avg_total_weighted_loss = float(total_loss(avg_losses_unweighted, current_weights_dict))

            # --- Validation (NSE/RMSE calculation using loaded validation_sample data) ---
            nse_val, rmse_val = -jnp.inf, jnp.inf
            with jax.disable_jit():
                if val_points is not None and h_true_val is not None and val_points.shape[0] > 0:
                    try:
                        U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                        h_pred_val = U_pred_val[..., 0]
                        h_true_mean = jnp.mean(h_true_val)
                        denominator_nse = jnp.sum((h_true_val - h_true_mean)**2)
                        eps = cfg.get("numerics", {}).get("eps", 1e-9) # Use eps from config

                        if denominator_nse > eps:
                            numerator_nse = jnp.sum((h_true_val - h_pred_val)**2)
                            nse_val = float(1.0 - numerator_nse / denominator_nse)
                        # else: nse_val remains -inf

                        rmse_val = float(rmse(h_pred_val, h_true_val))
                    except Exception as e_val:
                         print(f"Warning: Error during NSE/RMSE calculation at epoch {epoch+1}: {e_val}")
                         nse_val, rmse_val = -jnp.inf, jnp.inf # Reset on error
                # else: Validation skipped if no valid validation data loaded

            # --- Update Best Model ---
            if nse_val > best_nse:
                best_nse = nse_val; best_epoch = epoch; best_params = copy.deepcopy(params); best_nse_time = time.time() - start_time
                if nse_val > -jnp.inf: print(f"    ---> New best NSE: {best_nse:.6f} at epoch {epoch+1}")

            # --- Logging and Reporting ---
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % 100 == 0:
                # Use avg_total_weighted_loss and avg_losses_unweighted here
                # Pass data_loss=0.0 explicitly as it's not computed
                print_epoch_stats(
                    epoch, start_time, avg_total_weighted_loss, # Log weighted total
                    avg_losses_unweighted.get('pde', 0.0), avg_losses_unweighted.get('ic', 0.0), avg_losses_unweighted.get('bc', 0.0),
                    avg_losses_unweighted.get('building_bc', 0.0), 0.0, # data_loss is 0
                    nse_val, rmse_val, epoch_time
                )
                # Print current weights if using GradNorm
                if enable_gradnorm:
                     print(f"      Current Weights: { {k: f'{v:.2e}' for k, v in current_weights_dict.items()} }")


            # Log metrics to Aim (pass unweighted individual losses, data_loss=0)
            if aim_run:
                metrics_to_log = {
                        'total_loss': avg_total_weighted_loss, # Log weighted total
                        'pde_loss': avg_losses_unweighted.get('pde', 0.0),
                        'ic_loss': avg_losses_unweighted.get('ic', 0.0),
                        'bc_loss': avg_losses_unweighted.get('bc', 0.0),
                        'building_bc_loss': avg_losses_unweighted.get('building_bc', 0.0),
                        'data_loss': 0.0, # Log 0 for data loss
                        'nse': nse_val, 'rmse': rmse_val, 'epoch_time': epoch_time
                    }
                # Add current weights if GradNorm enabled
                if enable_gradnorm:
                    for k, w in current_weights_dict.items():
                        metrics_to_log[f'weight_{k}'] = float(w)
                try:
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

        # Need to assign the key back for the next epoch if loop continues
        train_key = key

    except KeyboardInterrupt: print("\n--- Training interrupted by user ---")
    except Exception as e: print(f"\n--- An error occurred during training: {e} ---"); import traceback; traceback.print_exc()

    # --- Final Summary and Saving ---
    finally:
        if aim_run:
            try: aim_run.close(); print("Aim run closed.")
            except Exception as e: print(f"Warning: Error closing Aim run: {e}")
        total_time = time.time() - start_time
        print_final_summary(total_time, best_epoch, best_nse, best_nse_time)
        if ask_for_confirmation():
            if best_params is not None:
                try:
                    save_model(best_params, model_dir, trial_name)
                    # --- Conditional Plotting (remains the same) ---
                    print("Generating final plot...")
                    plot_cfg = cfg.get("plotting", {}); eps_plot = cfg.get("numerics", {}).get("eps", 1e-6); t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
                    if has_building:
                        print("  Generating meshgrid predictions...")
                        resolution = plot_cfg.get("plot_resolution", 150); x_plot = jnp.linspace(0, cfg["domain"]["lx"], resolution, dtype=DTYPE); y_plot = jnp.linspace(0, cfg["domain"]["ly"], resolution, dtype=DTYPE); xx_plot, yy_plot = jnp.meshgrid(x_plot, y_plot); t_plot = jnp.full_like(xx_plot, t_const_val_plot, dtype=DTYPE); plot_points_mesh = jnp.stack([xx_plot.ravel(), yy_plot.ravel(), t_plot.ravel()], axis=-1)
                        U_plot_pred_mesh = model.apply({'params': best_params['params']}, plot_points_mesh, train=False); h_plot_pred_mesh = U_plot_pred_mesh[..., 0].reshape(resolution, resolution); h_plot_pred_mesh = jnp.where(h_plot_pred_mesh < eps_plot, 0.0, h_plot_pred_mesh)
                        print("  Loading plotting data for comparison...")
                        plot_data_time = t_const_val_plot; plot_data_file = os.path.join(base_data_path, f"validation_plotting_t_{int(plot_data_time)}s.npy")
                        if os.path.exists(plot_data_file):
                            try:
                                plot_data = np.load(plot_data_file); x_coords_plot = jnp.array(plot_data[:, 1], dtype=DTYPE); y_coords_plot = jnp.array(plot_data[:, 2], dtype=DTYPE); h_true_plot_data = jnp.array(plot_data[:, 3], dtype=DTYPE)
                                plot_path_comp = os.path.join(results_dir, f"final_comparison_plot_t{int(plot_data_time)}s.png")
                                plot_h_prediction_vs_true_2d(xx_plot, yy_plot, h_plot_pred_mesh, x_coords_plot, y_coords_plot, h_true_plot_data, cfg_dict, plot_path_comp)
                            except Exception as e_plot: print(f"  Error generating comparison plot: {e_plot}")
                        else: print(f"  Warning: Plotting data file {plot_data_file} not found. Skipping comparison plot.")
                    else:
                        print("  Generating 1D validation plot (using analytical solution)...")
                        nx_val_plot = plot_cfg.get("nx_val", 101); y_const_plot = plot_cfg.get("y_const_plot", 0.0); x_val_plot = jnp.linspace(0.0, cfg["domain"]["lx"], nx_val_plot, dtype=DTYPE); plot_points_1d = jnp.stack([x_val_plot, jnp.full_like(x_val_plot, y_const_plot, dtype=DTYPE), jnp.full_like(x_val_plot, t_const_val_plot, dtype=DTYPE)], axis=1)
                        U_plot_pred_1d = model.apply({'params': best_params['params']}, plot_points_1d, train=False); h_plot_pred_1d = U_plot_pred_1d[..., 0]; h_plot_pred_1d = jnp.where(h_plot_pred_1d < eps_plot, 0.0, h_plot_pred_1d)
                        plot_path_1d = os.path.join(results_dir, "final_validation_plot.png")
                        # Note: plot_h_vs_x uses the analytical solution for comparison
                        plot_h_vs_x(x_val_plot, h_plot_pred_1d, t_const_val_plot, y_const_plot, cfg_dict, plot_path_1d)
                    print(f"Model and plot saved in {model_dir} and {results_dir}")
                except Exception as e: print(f"Error during saving/plotting: {e}"); import traceback; traceback.print_exc()
            else: print("Warning: No best model found. Skipping save/plot.")
        else:
            print("Save aborted by user. Deleting artifacts...")
            try:
                if aim_run and run_hash and aim_repo: aim_repo.delete_run(run_hash); print("Aim run deleted.")
                if os.path.exists(results_dir): shutil.rmtree(results_dir); print(f"Deleted results directory: {results_dir}")
                if os.path.exists(model_dir): shutil.rmtree(model_dir); print(f"Deleted model directory: {model_dir}")
                print("Cleanup complete.")
            except Exception as e: print(f"Error during cleanup: {e}")

    return best_nse if best_nse > -jnp.inf else -1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SWE PINN (Data-free, handles building/no-building, optional GradNorm).")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    try:
        final_nse = main(args.config)
        print(f"\n--- Script Finished ---")
        if isinstance(final_nse, (float, int)) and final_nse > -float('inf'): print(f"Final best NSE reported: {final_nse:.6f}")
        else: print(f"Final best NSE value invalid or not achieved: {final_nse}")
        print(f"-----------------------")
    except FileNotFoundError as e: print(f"Error: {e}. Check config path.")
    except ValueError as e: print(f"Config/Model Error: {e}")
    except Exception as e: print(f"Unexpected error: {e}"); import traceback; traceback.print_exc()