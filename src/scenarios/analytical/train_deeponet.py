"""
Training script for the DeepONet model on the "analytical" (no-building) scenario.

This script trains a Physics-Informed DeepONet (PI-DeepONet).
It supports both data-free (physics-only) and data-driven (physics + data)
training modes, controlled by the 'data_free' flag in the config.
"""

import os
import sys
import time
import copy
import argparse
import importlib
import itertools
from typing import Any, Dict, Tuple
import shutil
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random
import optax
from aim import Repo, Run, Image
from flax.core import FrozenDict
import numpy as np 

# --- Local application imports ---
# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config, DTYPE
from src.data import sample_points, sample_parameters, get_batches, create_deeponet_dataset
from src.models import init_deeponet_model # <-- Using DeepONet specific init
from src.operator_learning.losses_op import ( # <-- Using _op.py file
    compute_pde_loss_op, compute_ic_loss_op, compute_bc_loss_op, 
    total_loss, compute_neg_h_loss_op, compute_data_loss_op
)
from src.operator_learning.physics_op import h_exact # <-- Using _op.py file
from src.gradnorm import (
    init_gradnorm, 
    get_initial_losses_operator,
    update_gradnorm_weights_operator,
    OPERATOR_LOSS_FN_MAP # <-- Using DeepONet specific map
)
from src.utils import ( 
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation
)
from src.reporting import (
    print_epoch_stats, log_metrics, print_final_summary
)
# --- End Imports ---

def train_step_deeponet(model: Any, params: FrozenDict, opt_state: Any,
                        all_batches: Dict[str, Any],
                        weights_dict: Dict[str, float],
                        optimiser: optax.GradientTransformation,
                        config: FrozenDict,
                        data_free: bool = True
                        ) -> Tuple[FrozenDict, Any, Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Performs a single training step for the PI-DeepONet.
    """
    active_loss_keys_base = list(weights_dict.keys())

    def loss_and_individual_terms(p):
        terms = {}
        
        # --- PDE Loss ---
        pde_batch = all_batches.get('pde', {})
        if 'pde' in active_loss_keys_base and pde_batch.get('trunk', jnp.empty((0,3))).shape[0] > 0:
            terms['pde'] = compute_pde_loss_op(
                model, p, pde_batch['branch'], pde_batch['trunk'], config
            )
            if 'neg_h' in active_loss_keys_base:
                terms['neg_h'] = compute_neg_h_loss_op(
                    model, p, pde_batch['branch'], pde_batch['trunk'], config
                )

        # --- IC Loss ---
        ic_batch = all_batches.get('ic', {})
        if 'ic' in active_loss_keys_base and ic_batch.get('trunk', jnp.empty((0,3))).shape[0] > 0:
            terms['ic'] = compute_ic_loss_op(
                model, p, ic_batch['branch'], ic_batch['trunk'], config
            )

        # --- BC Loss ---
        bc_batch = all_batches.get('bc', {})
        if 'bc' in active_loss_keys_base and bc_batch.get('trunk_left', jnp.empty((0,3))).shape[0] > 0:
             terms['bc'] = compute_bc_loss_op(model, p, bc_batch, config)
        
        # --- Data Loss (if not data_free) ---
        data_batch = all_batches.get('data', {})
        if not data_free and 'data' in active_loss_keys_base and data_batch.get('trunk', jnp.empty((0,6))).shape[0] > 0:
             terms['data'] = compute_data_loss_op(
                 model, p, data_batch['branch'], data_batch['trunk'], config
             )
             
        # --- Total Weighted Loss ---
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms

    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, individual_terms_val, total_loss_val

# JIT compile the training step
train_step_deeponet_jitted = jax.jit(
    train_step_deeponet,
    static_argnames=('model', 'optimiser', 'config', 'data_free')
)

def main(config_path: str):
    """
    Main training function for the DeepONet analytical scenario.
    """
    # --- 1. Load Config and Initialize Model ---
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)
    
    if "param_bounds" not in cfg["physics"]:
        raise ValueError("Config must contain 'physics.param_bounds' for DeepONet training.")
    if cfg["model"]["name"] != "DeepONet":
        raise ValueError(f"This script requires model.name = 'DeepONet', but got {cfg['model']['name']}")
        
    print("Info: Running Physics-Informed DeepONet training for analytical scenario.")

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e

    key = random.PRNGKey(cfg["training"]["seed"])
    # Split key for validation and training data sampling
    model_key, init_key, train_key, val_key, data_key = random.split(key, 5)
    
    model, params = init_deeponet_model(model_class, model_key, cfg)
    param_names = tuple(cfg["physics"]["param_bounds"].keys())
    n_params = len(param_names)

    # --- 2. Setup Directories for Results and Models ---
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- 3. Setup Optimizer ---
    raw_boundaries = cfg.get("training", {}).get("lr_boundaries", {15000: 0.1, 30000: 0.1})
    boundaries_and_scales_int_keys = {int(k): v for k, v in raw_boundaries.items()}

    lr_schedule = optax.piecewise_constant_schedule(
        init_value=cfg["training"]["learning_rate"],
        boundaries_and_scales=boundaries_and_scales_int_keys
    )
    optimiser = optax.adam(learning_rate=lr_schedule)
    opt_state = optimiser.init(params)

    # --- 4. Prepare Loss Weights and GradNorm ---
    static_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    gradnorm_cfg = cfg.get("gradnorm", {})
    enable_gradnorm = gradnorm_cfg.get("enable", False) 
    gradnorm_alpha = gradnorm_cfg.get("alpha", 1.5)
    gradnorm_lr = gradnorm_cfg.get("learning_rate", 0.01)
    gradnorm_update_freq = gradnorm_cfg.get("update_freq", 100)
    gradnorm_state = None

    # --- 5. Create Validation & Training Data ---
    
    # Create Analytical Validation Data (Parameterized)
    print("Creating fixed validation dataset...")
    val_dataset = create_deeponet_dataset(
        val_key, cfg, 
        cfg["validation_sampling"]["n_val_functions"],
        cfg["validation_sampling"]["n_val_points_per_function"],
        h_exact,
        param_names
    )
    print(f"Validation dataset created with {val_dataset[0].shape[0]} pairs.")
    validation_data_loaded = True
    
    # Determine Data-Free Mode
    data_points_full_branch = None
    data_points_full_trunk_data = None # This will be the (N, 6) array [t,x,y,h,u,v]
    
    data_free_flag = cfg.get("data_free")
    
    if data_free_flag is False:
        print("Info: 'data_free: false' found in config. Activating data-driven mode.")
        has_data_loss = True
        data_free = False
    else:
        if data_free_flag is None:
            print("Warning: 'data_free' flag not specified in config. Defaulting to 'data_free: true'.")
        else:
            print("Info: 'data_free: true' found in config. Data loss term will be disabled.")
        has_data_loss = False
        data_free = True

    # Create Analytical Training Data (if data_free is False)
    if not data_free:
        try:
            train_data_cfg = cfg["train_data_sampling"]
            print(f"Creating analytical training dataset from 'train_data_sampling' config...")
            
            # Use data_key to ensure it's different data
            train_dataset = create_deeponet_dataset(
                data_key, cfg,
                train_data_cfg["n_train_functions"],
                train_data_cfg["n_train_points_per_function"],
                h_exact,
                param_names
            )
            
            data_points_full_branch = train_dataset[0]
            trunk_coords = train_dataset[1] # (N, 3) [x, y, t]
            h_true_train = train_dataset[2] # (N, 1) [h]
            
            # Get u, v from the branch parameters
            param_map = {name: i for i, name in enumerate(param_names)}
            u_const_idx = param_map.get('u_const')
            u_const_default = cfg["physics"]["u_const"]
            u_true_train = data_points_full_branch[..., u_const_idx] if u_const_idx is not None else jnp.full(data_points_full_branch.shape[0], u_const_default, dtype=DTYPE)
            v_true_train = jnp.zeros_like(u_true_train)

            data_points_full_trunk_data = jnp.hstack([
                trunk_coords[:, 2:3], # t
                trunk_coords[:, 0:1], # x
                trunk_coords[:, 1:2], # y
                h_true_train,           # h
                u_true_train[..., None],# u
                v_true_train[..., None] # v
            ]).astype(DTYPE)

            if data_points_full_branch.shape[0] == 0:
                 print("Warning: Analytical training data is empty. Disabling data loss.")
                 data_points_full_branch, data_points_full_trunk_data = None, None
                 has_data_loss = False
            else:
                 print(f"Created {data_points_full_branch.shape[0]} points for data loss term (weight={static_weights_dict.get('data', 0.0):.2e}).")

        except KeyError:
            print("Error: 'data_free: false' but 'train_data_sampling' not found in config. Disabling data loss.")
            has_data_loss = False
            data_free = True
        except Exception as e:
            print(f"Error creating analytical training data: {e}. Disabling data loss.")
            has_data_loss = False
            data_free = True
    
    data_free = not has_data_loss # Final determination

    # --- 6. Initialize Aim Run ---
    aim_repo, aim_run, run_hash = None, None, None
    try:
        aim_repo_path = "aim_repo"
        os.makedirs(aim_repo_path, exist_ok=True)
        aim_repo = Repo(path=aim_repo_path, init=True)
        aim_run = Run(repo=aim_repo, experiment=trial_name)
        run_hash = aim_run.hash

        artifact_storage_path = os.path.join(aim_repo_path, "aim_artifacts")
        os.makedirs(artifact_storage_path, exist_ok=True)
        abs_artifact_path = os.path.abspath(artifact_storage_path)
        aim_run.set_artifacts_uri(f"file://{abs_artifact_path}")
        
        hparams_to_log = copy.deepcopy(cfg_dict)
        aim_run["hparams"] = hparams_to_log
        
        aim_run['flags'] = {
            "scenario_type": "deeponet_analytical",
            "data_free_config_flag": data_free_flag,
            "data_loss_active_final": has_data_loss,
            "gradnorm_enabled": enable_gradnorm,
            "has_building": False
        }
        
        aim_run.log_artifact(config_path, name='run_config.yaml')
        print(f"Aim tracking initialized for run: {trial_name} ({run_hash})")
        
    except Exception as e:
        print(f"Warning: Failed to initialize Aim tracking: {e}.")

    # --- 7. Determine Active Loss Terms ---
    active_loss_term_keys = []
    for k, v in static_weights_dict.items():
        if v > 0:
            if k == 'data' and data_free: continue 
            if k == 'building_bc': continue
            active_loss_term_keys.append(k)
    
    current_weights_dict = {k: static_weights_dict[k] for k in active_loss_term_keys}

    # --- 8. Initialize GradNorm if Enabled ---
    if enable_gradnorm:
        print("GradNorm enabled. Initializing dynamic weights...")
        key, pde_key, ic_key, bc_keys, data_key_init = random.split(init_key, 5)
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        batch_size_init = cfg["training"]["batch_size"]
        
        n_init_funcs = max(1, batch_size_init // 100)
        n_init_funcs = min(n_init_funcs, batch_size_init)
        
        domain_cfg = cfg["domain"]
        phys_sampling_cfg = cfg["grid"] # Use new config section
        ic_bc_grid_cfg = cfg["ic_bc_grid"]
        param_bounds = cfg["physics"]["param_bounds"]
        
        key, init_param_key = random.split(key)
        init_params, _ = sample_parameters(init_param_key, param_bounds, n_init_funcs)

        relevant_init_batches = {}

        if 'pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys:
            pde_points_init = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 
                                            phys_sampling_cfg["nx"], phys_sampling_cfg["ny"], phys_sampling_cfg["nt"], pde_key)
            if pde_points_init.shape[0] > 0:
                pde_batch_data = get_batches(pde_key, pde_points_init, batch_size_init)[0]
                pde_params_batch = jnp.repeat(init_params, -(-pde_batch_data.shape[0] // n_init_funcs), axis=0)[:pde_batch_data.shape[0]]
                batch_dict = {'branch': pde_params_batch, 'trunk': pde_batch_data}
                relevant_init_batches['pde'] = batch_dict
                if 'neg_h' in active_loss_term_keys:
                    relevant_init_batches['neg_h'] = batch_dict
        
        if 'ic' in active_loss_term_keys:
            ic_points_init = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., 0., ic_bc_grid_cfg["nx_ic"], ic_bc_grid_cfg["ny_ic"], 1, ic_key)
            if ic_points_init.shape[0] > 0: 
                ic_batch_data = get_batches(ic_key, ic_points_init, batch_size_init)[0]
                ic_params_batch = jnp.repeat(init_params, -(-ic_batch_data.shape[0] // n_init_funcs), axis=0)[:ic_batch_data.shape[0]]
                relevant_init_batches['ic'] = {'branch': ic_params_batch, 'trunk': ic_batch_data}

        if 'bc' in active_loss_term_keys:
            bc_batches_init = {}
            bc_batches_init['trunk_left'] = get_batches(l_key, sample_points(0., 0., 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_left"], ic_bc_grid_cfg["nt_bc_left"], l_key), batch_size_init)[0]
            bc_batches_init['trunk_right'] = get_batches(r_key, sample_points(domain_cfg["lx"], domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_right"], ic_bc_grid_cfg["nt_bc_right"], r_key), batch_size_init)[0]
            bc_batches_init['trunk_bottom'] = get_batches(b_key, sample_points(0., domain_cfg["lx"], 0., 0., 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_bottom"], 1, ic_bc_grid_cfg["nt_bc_other"], b_key), batch_size_init)[0]
            bc_batches_init['trunk_top'] = get_batches(t_key, sample_points(0., domain_cfg["lx"], domain_cfg["ly"], domain_cfg["ly"], 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_top"], 1, ic_bc_grid_cfg["nt_bc_other"], t_key), batch_size_init)[0]
            
            for k_wall in ['left', 'right', 'bottom', 'top']:
                trunk_batch = bc_batches_init[f'trunk_{k_wall}']
                if trunk_batch.shape[0] > 0:
                    bc_batches_init[f'branch_{k_wall}'] = jnp.repeat(init_params, -(-trunk_batch.shape[0] // n_init_funcs), axis=0)[:trunk_batch.shape[0]]
                else:
                    bc_batches_init[f'branch_{k_wall}'] = jnp.empty((0, n_params), dtype=DTYPE)
            relevant_init_batches['bc'] = bc_batches_init

        if not data_free and 'data' in active_loss_term_keys: 
             if data_points_full_trunk_data is not None and data_points_full_trunk_data.shape[0] > 0:
                 indices_init = jax.random.permutation(data_key_init, data_points_full_trunk_data.shape[0])[:batch_size_init]
                 init_batch_trunk_data = data_points_full_trunk_data[indices_init]
                 init_batch_branch = data_points_full_branch[indices_init]
                 relevant_init_batches['data'] = {'branch': init_batch_branch, 'trunk': init_batch_trunk_data}
        
        active_loss_term_keys = list(relevant_init_batches.keys())
        print(f"GradNorm active keys for init: {active_loss_term_keys}")

        with jax.disable_jit():
            initial_losses = get_initial_losses_operator(model, params, relevant_init_batches, cfg)

        gradnorm_state = init_gradnorm(
            loss_keys=list(initial_losses.keys()),
            initial_losses=initial_losses,
            gradnorm_lr=gradnorm_lr
        )
        current_weights_dict = {key: float(w) for key, w in zip(initial_losses.keys(), gradnorm_state.weights)}
        for k in active_loss_term_keys:
            if k not in current_weights_dict:
                current_weights_dict[k] = 1.0
        
        print(f"GradNorm initialized. Initial Weights: {current_weights_dict}")
    else:
         print(f"GradNorm disabled. Using Static Weights: {current_weights_dict}")
    
    # --- 9. Pre-Training Summary ---
    print(f"\n--- Training Started: {trial_name} ---")
    print(f"Model: {cfg['model']['name']}, Epochs: {cfg['training']['epochs']}, Batch Size: {cfg['training']['batch_size']}")
    print(f"Scenario: PI-DeepONet (Analytical)")
    print(f"Learning Parameters: {param_names}")
    print(f"GradNorm Enabled: {enable_gradnorm}")
    print(f"Data Loss Active: {has_data_loss} (Final Data-Free: {data_free})")
    print(f"Active Loss Terms: {active_loss_term_keys}")
    print(f"Initial Weights: {current_weights_dict}")

    best_nse_stats = {
        'nse': -jnp.inf, 'rmse': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {}
    }
    best_params_nse: Dict = None
    best_loss_stats = {
        'total_weighted_loss': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'nse': -jnp.inf, 'rmse': jnp.inf, 'unweighted_losses': {}
    }
    log_freq_steps = cfg.get("training", {}).get("log_freq_steps", 100)
    global_step = 0 
    start_time = time.time()

    # --- 10. Main Training Loop ---
    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()

            # --- Dynamic Point and Parameter Sampling ---
            key, pde_key, ic_key, bc_keys, data_key_epoch = random.split(key, 5)
            l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
            key, pde_param_key, ic_param_key, bc_param_key = random.split(key, 4)
            
            domain_cfg = cfg["domain"]
            phys_sampling_cfg = cfg["grid"] # Physics points
            ic_bc_grid_cfg = cfg["ic_bc_grid"]     # IC/BC points
            param_bounds = cfg["physics"]["param_bounds"]
            batch_size = cfg["training"]["batch_size"]

            # --- Trunks ---
            pde_points = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 
                                        phys_sampling_cfg["nx"], phys_sampling_cfg["ny"], phys_sampling_cfg["nt"], pde_key) if 'pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            ic_points = sample_points(0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., 0., ic_bc_grid_cfg["nx_ic"], ic_bc_grid_cfg["ny_ic"], 1, ic_key) if 'ic' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            left_wall = sample_points(0., 0., 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_left"], ic_bc_grid_cfg["nt_bc_left"], l_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            right_wall = sample_points(domain_cfg["lx"], domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"], 1, ic_bc_grid_cfg["ny_bc_right"], ic_bc_grid_cfg["nt_bc_right"], r_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            bottom_wall = sample_points(0., domain_cfg["lx"], 0., 0., 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_bottom"], 1, ic_bc_grid_cfg["nt_bc_other"], b_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            top_wall = sample_points(0., domain_cfg["lx"], domain_cfg["ly"], domain_cfg["ly"], 0., domain_cfg["t_final"], ic_bc_grid_cfg["nx_bc_top"], 1, ic_bc_grid_cfg["nt_bc_other"], t_key) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            
            # --- Branches (Sample *per point* for physics losses) ---
            n_pde_funcs = phys_sampling_cfg.get("n_pde_functions", pde_points.shape[0])
            n_ic_funcs = phys_sampling_cfg.get("n_ic_functions", ic_points.shape[0])
            n_bc_funcs = phys_sampling_cfg.get("n_bc_functions", left_wall.shape[0] + right_wall.shape[0] + bottom_wall.shape[0] + top_wall.shape[0])

            pde_params, _ = sample_parameters(pde_param_key, param_bounds, n_pde_funcs) if pde_points.shape[0] > 0 else (jnp.empty((0,n_params), dtype=DTYPE), [])
            ic_params, _ = sample_parameters(ic_param_key, param_bounds, n_ic_funcs) if ic_points.shape[0] > 0 else (jnp.empty((0,n_params), dtype=DTYPE), [])
            
            # Repeat sampled funcs to match number of points
            pde_params = jnp.repeat(pde_params, -(-pde_points.shape[0] // n_pde_funcs), axis=0)[:pde_points.shape[0]] if pde_points.shape[0] > 0 else pde_params
            ic_params = jnp.repeat(ic_params, -(-ic_points.shape[0] // n_ic_funcs), axis=0)[:ic_points.shape[0]] if ic_points.shape[0] > 0 else ic_params

            # For BCs, sample a set of functions and repeat them across all walls
            bc_params_set, _ = sample_parameters(bc_param_key, param_bounds, n_bc_funcs) if n_bc_funcs > 0 else (jnp.empty((0,n_params), dtype=DTYPE), [])
            bc_params = {}
            bc_params['left'] = jnp.repeat(bc_params_set, -(-left_wall.shape[0] // n_bc_funcs), axis=0)[:left_wall.shape[0]] if left_wall.shape[0] > 0 else jnp.empty((0,n_params), dtype=DTYPE)
            bc_params['right'] = jnp.repeat(bc_params_set, -(-right_wall.shape[0] // n_bc_funcs), axis=0)[:right_wall.shape[0]] if right_wall.shape[0] > 0 else jnp.empty((0,n_params), dtype=DTYPE)
            bc_params['bottom'] = jnp.repeat(bc_params_set, -(-bottom_wall.shape[0] // n_bc_funcs), axis=0)[:bottom_wall.shape[0]] if bottom_wall.shape[0] > 0 else jnp.empty((0,n_params), dtype=DTYPE)
            bc_params['top'] = jnp.repeat(bc_params_set, -(-top_wall.shape[0] // n_bc_funcs), axis=0)[:top_wall.shape[0]] if top_wall.shape[0] > 0 else jnp.empty((0,n_params), dtype=DTYPE)

            # --- Create Batches ---
            key, pde_b_key, ic_b_key, bc_b_keys, data_b_key_epoch = random.split(key, 5)
            l_b_key, r_b_key, b_b_key, t_b_key = random.split(bc_b_keys, 4)

            pde_trunk_batches = get_batches(pde_b_key, pde_points, batch_size) if pde_points.shape[0] > 0 else []
            pde_branch_batches = get_batches(pde_b_key, pde_params, batch_size) if pde_params.shape[0] > 0 else []
            
            ic_trunk_batches = get_batches(ic_b_key, ic_points, batch_size) if ic_points.shape[0] > 0 else []
            ic_branch_batches = get_batches(ic_b_key, ic_params, batch_size) if ic_params.shape[0] > 0 else []

            left_trunk_batches = get_batches(l_b_key, left_wall, batch_size) if left_wall.shape[0] > 0 else []
            left_branch_batches = get_batches(l_b_key, bc_params['left'], batch_size) if bc_params['left'].shape[0] > 0 else []
            
            right_trunk_batches = get_batches(r_b_key, right_wall, batch_size) if right_wall.shape[0] > 0 else []
            right_branch_batches = get_batches(r_b_key, bc_params['right'], batch_size) if bc_params['right'].shape[0] > 0 else []

            bottom_trunk_batches = get_batches(b_b_key, bottom_wall, batch_size) if bottom_wall.shape[0] > 0 else []
            bottom_branch_batches = get_batches(b_b_key, bc_params['bottom'], batch_size) if bc_params['bottom'].shape[0] > 0 else []

            top_trunk_batches = get_batches(t_b_key, top_wall, batch_size) if top_wall.shape[0] > 0 else []
            top_branch_batches = get_batches(t_b_key, bc_params['top'], batch_size) if bc_params['top'].shape[0] > 0 else []
            
            data_trunk_batches = []
            data_branch_batches = []
            if not data_free and data_points_full_trunk_data is not None:
                 indices = jax.random.permutation(data_b_key_epoch, data_points_full_trunk_data.shape[0])
                 shuffled_trunks = data_points_full_trunk_data[indices]
                 shuffled_branches = data_points_full_branch[indices]
                 data_trunk_batches = [shuffled_trunks[i:i + batch_size] for i in range(0, shuffled_trunks.shape[0], batch_size)]
                 data_branch_batches = [shuffled_branches[i:i + batch_size] for i in range(0, shuffled_branches.shape[0], batch_size)]

            num_batches = len(pde_trunk_batches) if pde_trunk_batches else (len(data_trunk_batches) if data_trunk_batches else 0)
            if num_batches == 0:
                 print(f"Warning: Epoch {epoch+1} - No driving batches (PDE or Data) generated. Skipping epoch.")
                 continue

            # --- Batch Iterators ---
            pde_trunk_iter = iter(pde_trunk_batches) if num_batches == len(pde_trunk_batches) else itertools.cycle(pde_trunk_batches)
            pde_branch_iter = iter(pde_branch_batches) if num_batches == len(pde_branch_batches) else itertools.cycle(pde_branch_batches)
            ic_trunk_iter = itertools.cycle(ic_trunk_batches) if ic_trunk_batches else iter(())
            ic_branch_iter = itertools.cycle(ic_branch_batches) if ic_branch_batches else iter(())
            data_trunk_iter = iter(data_trunk_batches) if num_batches == len(data_trunk_batches) else itertools.cycle(data_trunk_batches)
            data_branch_iter = iter(data_branch_batches) if num_batches == len(data_branch_batches) else itertools.cycle(data_branch_batches)
            
            left_trunk_iter = itertools.cycle(left_trunk_batches) if left_trunk_batches else iter(())
            left_branch_iter = itertools.cycle(left_branch_batches) if left_branch_batches else iter(())
            right_trunk_iter = itertools.cycle(right_trunk_batches) if right_trunk_batches else iter(())
            right_branch_iter = itertools.cycle(right_branch_batches) if right_branch_batches else iter(())
            bottom_trunk_iter = itertools.cycle(bottom_trunk_batches) if bottom_trunk_batches else iter(())
            bottom_branch_iter = itertools.cycle(bottom_branch_batches) if bottom_branch_batches else iter(())
            top_trunk_iter = itertools.cycle(top_trunk_batches) if top_trunk_batches else iter(())
            top_branch_iter = itertools.cycle(top_branch_batches) if top_branch_batches else iter(())

            epoch_losses_unweighted_sum = {k: 0.0 for k in active_loss_term_keys}
            epoch_total_weighted_loss_sum = 0.0

            for i in range(num_batches):
                global_step += 1
                
                pde_batch = {'branch': next(pde_branch_iter, jnp.empty((0,n_params), dtype=DTYPE)), 
                             'trunk': next(pde_trunk_iter, jnp.empty((0,3), dtype=DTYPE))}
                ic_batch = {'branch': next(ic_branch_iter, jnp.empty((0,n_params), dtype=DTYPE)), 
                            'trunk': next(ic_trunk_iter, jnp.empty((0,3), dtype=DTYPE))}
                data_batch = {'branch': next(data_branch_iter, jnp.empty((0,n_params), dtype=DTYPE)),
                              'trunk': next(data_trunk_iter, jnp.empty((0,6), dtype=DTYPE))}
                bc_batch = {
                    'branch_left': next(left_branch_iter), 'trunk_left': next(left_trunk_iter),
                    'branch_right': next(right_branch_iter), 'trunk_right': next(right_trunk_iter),
                    'branch_bottom': next(bottom_branch_iter), 'trunk_bottom': next(bottom_trunk_iter),
                    'branch_top': next(top_branch_iter), 'trunk_top': next(top_trunk_iter),
                }

                current_all_batches = {'pde': pde_batch, 'ic': ic_batch, 'bc': bc_batch, 'data': data_batch}
                
                # --- GradNorm Update ---
                if enable_gradnorm and global_step % gradnorm_update_freq == 0:
                    active_batches_for_gradnorm = {
                        k: current_all_batches[OPERATOR_LOSS_FN_MAP[k]['batch_key']] 
                        for k in active_loss_term_keys if k in OPERATOR_LOSS_FN_MAP
                    }
                    with jax.disable_jit():
                         gradnorm_state, current_weights_dict = update_gradnorm_weights_operator(
                              gradnorm_state, params, model, 
                              active_batches_for_gradnorm,
                              cfg, gradnorm_alpha, gradnorm_lr
                         )

                # --- Training Step ---
                params, opt_state, batch_losses_unweighted, batch_total_weighted_loss = train_step_deeponet_jitted(
                    model, params, opt_state,
                    current_all_batches,
                    current_weights_dict,
                    optimiser, cfg, data_free
                )

                for k in active_loss_term_keys:
                    epoch_losses_unweighted_sum[k] += float(batch_losses_unweighted.get(k, 0.0))
                epoch_total_weighted_loss_sum += float(batch_total_weighted_loss)
            
            # --- End of Epoch ---
            avg_losses_unweighted = {k: v / num_batches for k, v in epoch_losses_unweighted_sum.items()}
            avg_total_weighted_loss = epoch_total_weighted_loss_sum / num_batches

            # --- Validation ---
            nse_val, rmse_val = -jnp.inf, jnp.inf
            if validation_data_loaded:
                try:
                    h_pred_val = model.apply(
                        {'params': params['params']}, 
                        val_dataset[0], # branch_inputs_paired
                        val_dataset[1], # trunk_inputs_paired
                        train=False
                    )
                    h_pred_val_h = h_pred_val[..., 0:1]
                    nse_val = float(nse(h_pred_val_h, val_dataset[2]))
                    rmse_val = float(rmse(h_pred_val_h, val_dataset[2]))
                except Exception as e:
                    print(f"Warning: Epoch {epoch+1} - NSE/RMSE calculation failed: {e}")

            # --- Update Best Model Statistics ---
            if nse_val > best_nse_stats['nse']:
                best_nse_stats.update({
                    'nse': nse_val, 'rmse': rmse_val, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'total_weighted_loss': avg_total_weighted_loss,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()}
                })
                best_params_nse = copy.deepcopy(params)
                if nse_val > -jnp.inf:
                    print(f"    ---> New best NSE: {best_nse_stats['nse']:.6f} at epoch {epoch+1}")

            if avg_total_weighted_loss < best_loss_stats['total_weighted_loss']:
                best_loss_stats.update({
                    'total_weighted_loss': avg_total_weighted_loss, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'nse': nse_val, 'rmse': rmse_val,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()}
                })

            # --- Per-Epoch Logging ---
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % 100 == 0:
                print_epoch_stats(
                    epoch, global_step, start_time, avg_total_weighted_loss,
                    avg_losses_unweighted.get('pde', 0.0), 
                    avg_losses_unweighted.get('ic', 0.0), 
                    avg_losses_unweighted.get('bc', 0.0),
                    0.0, # No building
                    avg_losses_unweighted.get('data', 0.0), # Data loss
                    avg_losses_unweighted.get('neg_h', 0.0),
                    nse_val, rmse_val, epoch_time
                )
                if enable_gradnorm:
                     print(f"      Current Weights: { {k: f'{v:.2e}' for k, v in current_weights_dict.items()} }")

            if aim_run:
                epoch_metrics_to_log = {
                    'validation_metrics': {'nse': nse_val, 'rmse': rmse_val},
                    'epoch_avg_losses': avg_losses_unweighted,
                    'epoch_avg_total_weighted_loss': avg_total_weighted_loss,
                    'system_metrics': {'epoch_time': epoch_time},
                    'training_metrics': {'learning_rate': float(lr_schedule(global_step))}
                }
                if enable_gradnorm:
                    for k, w in current_weights_dict.items():
                        epoch_metrics_to_log[f'gradnorm/weight_{k}'] = float(w)
                log_metrics(aim_run, step=global_step, epoch=epoch, metrics=epoch_metrics_to_log)

            # --- Early Stopping Check ---
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))

            if epoch >= min_epochs and (epoch - best_nse_stats['epoch']) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                break

            train_key = key

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\n--- An error occurred during training loop: {e} ---")
        import traceback
        traceback.print_exc()

    # --- 11. Final Summary and Artifact Saving ---
    finally:
        total_time = time.time() - start_time
        print_final_summary(total_time, best_nse_stats, best_loss_stats)

        if aim_run:
            try:
                summary_best_nse = best_nse_stats.copy(); summary_best_nse['epoch'] += 1
                summary_best_loss = best_loss_stats.copy(); summary_best_loss['epoch'] += 1
                aim_run['summary'] = {
                    'best_validation_model': summary_best_nse,
                    'best_loss_model': summary_best_loss,
                    'final_system': { 
                        'total_training_time_seconds': total_time,
                        'total_epochs_run': (epoch + 1) if 'epoch' in locals() else 0,
                        'total_steps_run': global_step
                    }
                }
                print("Summary metrics logged to Aim.")
            except Exception as e:
                 print(f"Warning: Error logging summary metrics to Aim: {e}")

        if ask_for_confirmation():
            if best_params_nse is not None:
                try:
                    model_save_path = save_model(best_params_nse, model_dir, trial_name)
                    print(f"Best model (by NSE) saved to: {model_save_path}")

                    if aim_run:
                        aim_run.log_artifact(model_save_path, name='model_weights.pkl')
                        print(f"  Logged model weights .pkl file to Aim run: {run_hash}")

                    # --- Generate 1D Plot for one example function ---
                    print("  Generating 1D validation plot for one function...")
                    plot_cfg = cfg.get("plotting", {})
                    t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
                    
                    example_func_params = val_dataset[0][0:1]
                    param_str = ", ".join(f"{name}={val[0]:.3f}" for name, val in zip(val_param_names, example_func_params))
                    
                    nx_val_plot = plot_cfg.get("nx_val", 101)
                    y_const_plot = plot_cfg.get("y_const_plot", 0.0)
                    
                    x_val_plot = jnp.linspace(0.0, cfg["domain"]["lx"], nx_val_plot, dtype=DTYPE)
                    plot_points_1d_trunk = jnp.stack([
                        x_val_plot, 
                        jnp.full_like(x_val_plot, y_const_plot, dtype=DTYPE), 
                        jnp.full_like(x_val_plot, t_const_val_plot, dtype=DTYPE)
                    ], axis=1)
                    plot_points_1d_branch = jnp.repeat(example_func_params, nx_val_plot, axis=0)
                    
                    U_plot_pred_1d = model.apply(
                        {'params': best_params_nse['params']}, 
                        plot_points_1d_branch,
                        plot_points_1d_trunk,
                        train=False
                    )
                    h_plot_pred_1d = U_plot_pred_1d[..., 0]
                    
                    param_map = {name: i for i, name in enumerate(val_param_names)}
                    n_manning_ex = example_func_params[0, param_map['n_manning']] if 'n_manning' in param_map else cfg["physics"]["n_manning"]
                    u_const_ex = example_func_params[0, param_map['u_const']] if 'u_const' in param_map else cfg["physics"]["u_const"]
                    h_plot_true_1d = h_exact(x_val_plot, t_const_val_plot, n_manning_ex, u_const_ex)
                    
                    plot_path_1d = os.path.join(results_dir, f"final_validation_plot_{param_str.replace(', ', '_').replace('=', '')}.png")
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(x_val_plot, h_plot_true_1d, 'b-', label="Exact $h$", linewidth=2.5)
                    plt.plot(x_val_plot, h_plot_pred_1d, 'r--', label="DeepONet $h$", linewidth=2)
                    plt.xlabel("x (m)", fontsize=12); plt.ylabel("Depth $h$ (m)", fontsize=12)
                    plt.title(f"h vs x at y={y_const_plot:.2f}, t={t_const_val_plot:.2f} (Params: {param_str})", fontsize=14)
                    plt.legend(fontsize=11); plt.grid(True, linestyle='--', alpha=0.6)
                    plt.ylim(bottom=0); plt.tight_layout()
                    plt.savefig(plot_path_1d)
                    plt.close()
                    print(f"1D plot saved as {plot_path_1d}")

                    if aim_run:
                        aim_run.track(Image(plot_path_1d), name=f'validation_plot_example_func', epoch=best_nse_stats['epoch'],
                                      caption=f"Example function: {param_str}")
                        print(f"  Logged 1D plot to Aim run: {run_hash}")
                except Exception as e:
                     print(f"Error during saving/plotting: {e}")
                     import traceback
                     traceback.print_exc()
            else:
                print("Warning: No best model found (best_params_nse is None).")
        else:
            print("Save aborted by user. Deleting artifacts...")
            try:
                if aim_run and run_hash and aim_repo: aim_repo.delete_run(run_hash)
                if os.path.exists(results_dir): shutil.rmtree(results_dir)
                if os.path.exists(model_dir): shutil.rmtree(model_dir)
                if run_hash:
                    run_artifact_dir = os.path.join("aim_repo", "aim_artifacts", run_hash)
                    if os.path.exists(run_artifact_dir): shutil.rmtree(run_artifact_dir)
                print("Cleanup complete.")
            except Exception as e:
                print(f"Error during cleanup: {e}")

        if aim_run:
            try:
                aim_run.close()
                print("Aim run closed.")
            except Exception as e:
                 print(f"Warning: Error closing Aim run: {e}")

    return best_nse_stats['nse'] if best_nse_stats['nse'] > -jnp.inf else -1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Physics-Informed DeepONet model for SWE (Analytical Scenario).")
    parser.add_argument("--config", type=str, required=True, help="Path to the DeepONet configuration file (e.g., experiments/config_deeponet_analytical.yaml)")
    
    args = parser.parse_args()
    
    try:
        final_nse = main(args.config)
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