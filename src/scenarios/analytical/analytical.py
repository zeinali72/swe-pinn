"""
Training script for the "analytical" (no-building) scenario for the
Shallow Water Equation (SWE) PINN model.

This script handles both data-free and data-driven (analytical) training modes.
It supports dynamic loss weighting using GradNorm and provides
logging and result visualization through Aim.
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

import jax
import jax.numpy as jnp
from jax import random, lax
import optax
from aim import Repo, Run, Image
from flax.core import FrozenDict
import numpy as np 

# Local application imports
# (Assuming this file is at src/scenarios/analytical.py, adjust paths if needed)
from src.config import load_config, DTYPE
from src.data import sample_domain, get_batches
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_data_loss, compute_neg_h_loss
)
# from src.gradnorm import (
#     init_gradnorm, update_gradnorm_weights, LOSS_FN_MAP,
#     get_initial_losses
# )
from src.utils import ( 
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation,
    plot_h_vs_x
)
from src.physics import h_exact
from src.reporting import (
    print_epoch_stats, log_metrics, print_final_summary
)

# To enable 64-bit precision, uncomment the following line:
# jax.config.update('jax_enable_x64', True)


def train_step(model: Any, params: FrozenDict, opt_state: Any,
               all_batches: Dict[str, Any],
               weights_dict: Dict[str, float],
               optimiser: optax.GradientTransformation,
               config: FrozenDict,
               data_free: bool = True
               ) -> Tuple[FrozenDict, Any, Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Performs a single training step for the analytical scenario.
    """
    active_loss_keys_base = list(weights_dict.keys())

    def loss_and_individual_terms(p):
        terms = {}
        # --- PDE Residual Loss (and optional Negative Height penalty) ---
        pde_batch_data = all_batches.get('pde', jnp.empty((0,3), dtype=DTYPE))
        if 'pde' in active_loss_keys_base and pde_batch_data.shape[0] > 0:
            terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config)
            if 'neg_h' in active_loss_keys_base:
                terms['neg_h'] = compute_neg_h_loss(model, p, pde_batch_data)

        # --- Initial Condition Loss ---
        ic_batch_data = all_batches.get('ic', jnp.empty((0,3), dtype=DTYPE))
        if 'ic' in active_loss_keys_base and ic_batch_data.shape[0] > 0:
            terms['ic'] = compute_ic_loss(model, p, ic_batch_data)

        # --- Boundary Condition Loss ---
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

        # --- Data-Driven Loss (from analytical data) ---
        data_batch_data = all_batches.get('data', jnp.empty((0,6), dtype=DTYPE))
        if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0:
             terms['data'] = compute_data_loss(model, p, data_batch_data, config)

        # --- Total Weighted Loss ---
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms

    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, individual_terms_val, total_loss_val

# JIT compile the training step
train_step_jitted = jax.jit(
    train_step,
    static_argnames=('model', 'optimiser', 'config', 'data_free')
)


def main(config_path: str):
    """
    Main training function for the analytical scenario.
    """
    # --- 1. Load Config and Initialize Model ---
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)

    print("Info: Running in analytical (no-building) mode.")

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e

    key = random.PRNGKey(cfg["training"]["seed"])
    # Split key for model, gradnorm init, training loop, and validation data generation
    model_key, init_key, train_key, val_key = random.split(key, 4)
    model, params = init_model(model_class, model_key, cfg)

    # --- 2. Setup Directories for Results and Models ---
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- 3. Setup Optimizer ---
    # Get the boundaries dict, which might have string keys from the config
    raw_boundaries = cfg.get("training", {}).get("lr_boundaries", {15000: 0.1, 30000: 0.1})

    # --- FIX: Convert string keys to int keys for Optax ---
    boundaries_and_scales_int_keys = {int(k): v for k, v in raw_boundaries.items()}
    # --- END FIX ---

    lr_schedule = optax.piecewise_constant_schedule(
        init_value=cfg["training"]["learning_rate"],
        boundaries_and_scales=boundaries_and_scales_int_keys # Use the converted dict
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(cfg.get("training", {}).get("clip_norm", 1.0)),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimiser.init(params)

    # --- 4. Prepare Loss Weights and GradNorm ---
    static_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    # gradnorm_cfg = cfg.get("gradnorm", {})
    # enable_gradnorm = gradnorm_cfg.get("enable", False) 
    enable_gradnorm = False
    # gradnorm_alpha = gradnorm_cfg.get("alpha", 1.5)
    # gradnorm_lr = gradnorm_cfg.get("learning_rate", 0.01)
    # gradnorm_update_freq = gradnorm_cfg.get("update_freq", 100)
    gradnorm_state = None

    # --- 5. Create Validation and Training Data (Analytical) ---
    
    # Create Analytical Validation Data
    val_points, h_true_val = None, None
    validation_data_loaded = False
    try:
        val_grid_cfg = cfg["validation_grid"]
        domain_cfg = cfg["domain"]
        print(f"Creating analytical validation set from 'validation_grid' config...")
        
        val_points = sample_domain(
            val_key,
            val_grid_cfg["n_points_val"],
            (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
        )
        h_true_val = h_exact(
            val_points[:, 0], # x
            val_points[:, 2], # t
            cfg["physics"]["n_manning"],
            cfg["physics"]["u_const"]
        )
        
        if val_points.shape[0] > 0:
            validation_data_loaded = True
            print(f"Created analytical validation set with {val_points.shape[0]} points.")
        else:
            print("Warning: Analytical validation set is empty.")
            
    except KeyError:
        print("Warning: 'validation_grid' not found in config. Skipping NSE/RMSE calculation.")
    except Exception as e:
        print(f"Warning: Error creating analytical validation set: {e}. Skipping NSE/RMSE.")
        
    # Determine Data-Free Mode
    data_points_full = None
    data_free_flag = cfg.get("data_free") # Get the flag (might be None)
    
    if data_free_flag is False:
        print("Info: 'data_free: false' found in config. Activating data-driven mode.")
        has_data_loss = True
        data_free = False
    else:
        if data_free_flag is None:
            print("Warning: 'data_free' flag not specified in config. Defaulting to 'data_free: true'.")
        else:
            # This catches 'data_free: true'
            print("Info: 'data_free: true' found in config. Data loss term will be disabled.")
        has_data_loss = False
        data_free = True

    # Create Analytical Training Data (if data_free is False)
    if not data_free:
        try:
            train_grid_cfg = cfg["train_grid"]
            domain_cfg = cfg["domain"]
            print(f"Creating analytical training dataset from 'train_grid' config...")
            
            # 1. Sample points (x, y, t)
            data_points_coords = sample_domain(
                train_key,
                train_grid_cfg["n_points_train"],
                (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
            )
            
            # 2. Calculate true values (h, u, v)
            h_true_train = h_exact(
                data_points_coords[:, 0], # x
                data_points_coords[:, 2], # t
                cfg["physics"]["n_manning"],
                cfg["physics"]["u_const"]
            )
            u_true_train = jnp.full_like(h_true_train, cfg["physics"]["u_const"])
            v_true_train = jnp.zeros_like(h_true_train)
            
            # 3. Stack into (N, 6) format: [t, x, y, h, u, v]
            data_points_full = jnp.stack([
                data_points_coords[:, 2], # t
                data_points_coords[:, 0], # x
                data_points_coords[:, 1], # y
                h_true_train,
                u_true_train,
                v_true_train
            ], axis=1).astype(DTYPE)

            if data_points_full.shape[0] == 0:
                 print("Warning: Analytical training data is empty. Disabling data loss.")
                 data_points_full = None
                 has_data_loss = False
            else:
                 print(f"Created {data_points_full.shape[0]} points for data loss term (weight={static_weights_dict.get('data', 0.0):.2e}).")

        except KeyError:
            print("Error: 'data_free: false' but 'train_grid' not found in config. Disabling data loss.")
            has_data_loss = False
            data_free = True # Revert to data-free
        except Exception as e:
            print(f"Error creating analytical training data: {e}. Disabling data loss.")
            has_data_loss = False
            data_free = True # Revert to data-free
    
    # --- 6. Initialize Aim Run ---
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

        artifact_storage_path = os.path.join(aim_repo_path, "aim_artifacts")
        os.makedirs(artifact_storage_path, exist_ok=True)
        abs_artifact_path = os.path.abspath(artifact_storage_path)
        aim_run.set_artifacts_uri(f"file://{abs_artifact_path}")
        print(f"Set Aim artifact storage to: {abs_artifact_path}")
        
        hparams_to_log = copy.deepcopy(cfg_dict)
        aim_run["hparams"] = hparams_to_log
        
        # Updated flags
        aim_run['flags'] = {
            "scenario_type": "analytical",
            "data_free_config_flag": data_free_flag,
            "data_loss_active_final": has_data_loss,
            "gradnorm_enabled": enable_gradnorm
        }
        
        try:
            aim_run.log_artifact(config_path, name='run_config.yaml')
            print("Logged config file to Aim.")
        except Exception as e_aim:
            print(f"  Warning: Failed to log config artifact to Aim: {e_aim}")
            
        print(f"Aim tracking initialized for run: {trial_name} ({run_hash})")
        
    except Exception as e:
        print(f"Warning: Failed to initialize Aim tracking: {e}. Training will continue without Aim.")

    # --- 7. Determine Active Loss Terms ---
    active_loss_term_keys = []
    for k, v in static_weights_dict.items():
        if v > 0:
            if k == 'data' and data_free:
                continue 
            if k == 'building_bc': # Explicitly skip building loss
                continue
            active_loss_term_keys.append(k)
    
    current_weights_dict = {k: static_weights_dict[k] for k in active_loss_term_keys}

    # --- 8. Initialize GradNorm if Enabled ---
    # if enable_gradnorm:
    #     print("GradNorm enabled. Initializing dynamic weights...")
    #     key, pde_key, ic_key, bc_keys, data_key_init = random.split(init_key, 5)
    #     l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
    #     batch_size_init = cfg["training"]["batch_size"]
    #
    #     domain_cfg = cfg["domain"]
    #     sampling_cfg = cfg.get("sampling", {}) # <-- Use new config section
    #     init_batches = {} 
    #
    #     # --- 1. PDE Init Batch ---
    #     if 'pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys:
    #         n_pde_init = sampling_cfg.get("n_points_pde", 1000)
    #         pde_points_init = sample_domain(pde_key, n_pde_init,
    #                                         (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
    #         if pde_points_init.shape[0] > 0: 
    #             init_batches['pde'] = get_batches(pde_key, pde_points_init, batch_size_init)[0]
    #             if 'neg_h' in active_loss_term_keys:
    #                 init_batches['neg_h'] = init_batches['pde']
    #     
    #     # --- 2. IC Init Batch ---
    #     if 'ic' in active_loss_term_keys:
    #         n_ic_init = sampling_cfg.get("n_points_ic", 100)
    #         ic_points_init = sample_domain(ic_key, n_ic_init,
    #                                        (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))
    #         if ic_points_init.shape[0] > 0: 
    #             init_batches['ic'] = get_batches(ic_key, ic_points_init, batch_size_init)[0]
    #
    #     # --- 3. Domain BC Init Batch ---
    #     if 'bc' in active_loss_term_keys:
    #         n_bc_init = sampling_cfg.get("n_points_bc_domain", 100)
    #         n_bc_per_wall_init = max(5, n_bc_init // 4)
    #         bc_batches_init = {}
    #         bc_batches_init['left'] = get_batches(l_key, sample_domain(l_key, n_bc_per_wall_init, (0., 0.), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])), batch_size_init)[0]
    #         bc_batches_init['right'] = get_batches(r_key, sample_domain(r_key, n_bc_per_wall_init, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])), batch_size_init)[0]
    #         bc_batches_init['bottom'] = get_batches(b_key, sample_domain(b_key, n_bc_per_wall_init, (0., domain_cfg["lx"]), (0., 0.), (0., domain_cfg["t_final"])), batch_size_init)[0]
    #         bc_batches_init['top'] = get_batches(t_key, sample_domain(t_key, n_bc_per_wall_init, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]), (0., domain_cfg["t_final"])), batch_size_init)[0]
    #         init_batches['bc'] = {k: (v if v.shape[0] > 0 else jnp.empty((0,3), dtype=DTYPE)) for k, v in bc_batches_init.items() if v.shape[0] > 0}
    #     if not data_free and 'data' in active_loss_term_keys: 
    #          if data_points_full is not None and data_points_full.shape[0] > 0:
    #              init_data_sample = data_points_full[np.random.choice(data_points_full.shape[0], batch_size_init, replace=False)]
    #              init_batches['data'] = get_batches(data_key_init, init_data_sample, batch_size_init)[0]
    #     
    #     relevant_init_batches = {}
    #     for k in active_loss_term_keys:
    #         if k not in LOSS_FN_MAP: continue
    #         batch_key = LOSS_FN_MAP[k]['batch_key']
    #         if batch_key in init_batches:
    #             batch = init_batches[batch_key]
    #             is_valid = (isinstance(batch, jnp.ndarray) and batch.shape[0] > 0) or \
    #                       (isinstance(batch, dict) and any(b.shape[0] > 0 for b in batch.values() if isinstance(b, jnp.ndarray)))
    #             if is_valid:
    #                 relevant_init_batches[k] = batch
    #     
    #     active_loss_term_keys = list(relevant_init_batches.keys())
    #     print(f"GradNorm active keys for init: {active_loss_term_keys}")
    #
    #     with jax.disable_jit():
    #         initial_losses = get_initial_losses(model, params, relevant_init_batches, cfg)
    #
    #     gradnorm_state = init_gradnorm(
    #         loss_keys=list(initial_losses.keys()),
    #         initial_losses=initial_losses,
    #         gradnorm_lr=gradnorm_lr
    #     )
    #     current_weights_dict = {key: float(w) for key, w in zip(initial_losses.keys(), gradnorm_state.weights)}
    #     for k in active_loss_term_keys:
    #         if k not in current_weights_dict:
    #             current_weights_dict[k] = 1.0
    #     
    #     print(f"GradNorm initialized. Initial Weights: {current_weights_dict}")
    # else:
    print(f"GradNorm disabled. Using Static Weights: {current_weights_dict}")
    
    # --- 9. Pre-Training Summary ---
    print(f"\n--- Training Started (Jax-Scan): {trial_name} ---")
    print(f"Model: {cfg['model']['name']}, Epochs: {cfg['training']['epochs']}, Batch Size: {cfg['training']['batch_size']}")
    print(f"Scenario: analytical (No Building)")
    print(f"Saving results to: {results_dir}")
    print(f"Saving model to: {model_dir}")
    print(f"GradNorm Enabled: {enable_gradnorm}")
    print(f"Data Loss Active: {has_data_loss} (Final Data-Free: {data_free})")
    print(f"Active Loss Terms: {active_loss_term_keys}")
    print(f"Initial Weights: {current_weights_dict}")

    # --- Metrics Tracking Dictionaries ---
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
        # Note: GradNorm updates are disabled inside scan for performance/JIT compatibility.
        # Weights are constant during the epoch.
        new_params, new_opt_state, terms, total = train_step_jitted(
            model, curr_params, curr_opt_state, batch_data, current_weights_dict, optimiser, cfg, data_free
        )
        return (new_params, new_opt_state), (terms, total)

    # --- 10. Main Training Loop ---
    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()

            # --- Dynamic Point Sampling ---
            key, pde_key, ic_key, bc_keys, data_key_epoch = random.split(key, 5)
            l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
            domain_cfg = cfg["domain"]
            sampling_cfg = cfg["sampling"] # <-- Use new config section

            # 1. PDE Points
            n_pde = sampling_cfg.get("n_points_pde", 1000)
            pde_points = sample_domain(pde_key, n_pde, 
                                    (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])) \
                                    if 'pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)

            # 2. IC Points
            n_ic = sampling_cfg.get("n_points_ic", 100)
            ic_points = sample_domain(ic_key, n_ic, 
                                    (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.)) \
                                    if 'ic' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)

            # 3. Domain BC Points
            n_bc = sampling_cfg.get("n_points_bc_domain", 100)
            n_bc_per_wall = max(5, n_bc // 4)
            left_wall = sample_domain(l_key, n_bc_per_wall, (0., 0.), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            right_wall = sample_domain(r_key, n_bc_per_wall, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            bottom_wall = sample_domain(b_key, n_bc_per_wall, (0., domain_cfg["lx"]), (0., 0.), (0., domain_cfg["t_final"])) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            top_wall = sample_domain(t_key, n_bc_per_wall, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]), (0., domain_cfg["t_final"])) if 'bc' in active_loss_term_keys else jnp.empty((0,3), dtype=DTYPE)
            # --- Create Batches ---
            batch_size = cfg["training"]["batch_size"]
            key, pde_b_key, ic_b_key, bc_b_keys, data_b_key_epoch = random.split(key, 5)
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

            all_batch_lists = [pde_batches, ic_batches, left_batches, right_batches, bottom_batches, top_batches, data_batches]
            num_batches = max([len(b_list) for b_list in all_batch_lists if b_list], default=0)

            if num_batches == 0:
                 print(f"Warning: Epoch {epoch+1} - No batches generated for active terms. Skipping epoch.")
                 continue

            # --- Prepare Data for lax.scan ---
            scan_inputs = {
                'pde': prepare_batches(pde_batches, num_batches, (3,)),
                'ic': prepare_batches(ic_batches, num_batches, (3,)),
                'bc': {
                    'left': prepare_batches(left_batches, num_batches, (3,)),
                    'right': prepare_batches(right_batches, num_batches, (3,)),
                    'bottom': prepare_batches(bottom_batches, num_batches, (3,)),
                    'top': prepare_batches(top_batches, num_batches, (3,)),
                },
                'data': prepare_batches(data_batches, num_batches, (6,)) if not data_free else jnp.zeros((num_batches, 0, 6), dtype=DTYPE),
                'building_bc': {} # Empty dict for analytical
            }

            # --- Run Training Steps with lax.scan ---
            (params, opt_state), (batch_losses_unweighted, batch_total_weighted_loss) = lax.scan(
                scan_body, (params, opt_state), scan_inputs
            )
            
            global_step += num_batches

            # --- Aggregate Losses ---
            # batch_losses_unweighted is a dict of arrays (num_batches, ...)
            # batch_total_weighted_loss is an array (num_batches,)
            avg_losses_unweighted = {k: float(jnp.mean(v)) for k, v in batch_losses_unweighted.items()}
            avg_total_weighted_loss = float(jnp.mean(batch_total_weighted_loss))

            # --- GradNorm Update (Per Epoch) ---
            # Since we can't easily do this inside scan without JIT issues, we do it once per epoch if enabled.
            # if enable_gradnorm:
                # Use the last batch of the epoch for update
                # last_batches = {
                #     k: v[-1] for k, v in scan_inputs.items() if isinstance(v, jnp.ndarray) and v.shape[1] > 0
                # }
                # Handle nested BCs
                # if 'bc' in scan_inputs:
                #     for wall, v in scan_inputs['bc'].items():
                #         if v.shape[1] > 0:
                #             # We need to reconstruct the nested structure expected by update_gradnorm_weights
                #             # But update_gradnorm_weights expects 'bc' key to map to a dict of batches
                #             # Here we just need to ensure we pass compatible data.
                #             # Simplified: We skip complex reconstruction here for brevity, 
                #             # assuming GradNorm is less critical or user accepts per-epoch updates.
                #             pass

                # For now, we skip the explicit GradNorm update call here to keep the loop clean 
                # and because constructing the exact batch dict from scan inputs is verbose.
                # If strict GradNorm adherence is needed, it should be done here.
                # pass

            # --- Validation ---
            nse_val, rmse_val = -jnp.inf, jnp.inf
            if validation_data_loaded:
                try:
                    U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                    h_pred_val = U_pred_val[..., 0]
                    nse_val = float(nse(h_pred_val, h_true_val))
                    rmse_val = float(rmse(h_pred_val, h_true_val))
                except Exception as e:
                    print(f"Warning: Epoch {epoch+1} - NSE/RMSE calculation failed: {e}")
            elif (epoch + 1) % 100 == 0: # Only log this periodically
                print(f"Warning: Epoch {epoch+1} - No validation data available. Skipping NSE/RMSE calculation.")

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
                    0.0, # building_bc_loss
                    avg_losses_unweighted.get('data', 0.0),
                    avg_losses_unweighted.get('neg_h', 0.0),
                    nse_val, rmse_val, epoch_time
                )
                # if enable_gradnorm:
                #      print(f"      Current Weights: { {k: f'{v:.2e}' for k, v in current_weights_dict.items()} }")

            if aim_run:
                epoch_metrics_to_log = {
                    'validation_metrics': {'nse': nse_val, 'rmse': rmse_val},
                    'epoch_avg_losses': avg_losses_unweighted,
                    'epoch_avg_total_weighted_loss': avg_total_weighted_loss,
                    'system_metrics': {'epoch_time': epoch_time},
                    'training_metrics': {'learning_rate': float(lr_schedule(global_step))}
                }
                log_metrics(aim_run, step=global_step, epoch=epoch, metrics=epoch_metrics_to_log)

            # --- Early Stopping Check ---
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))

            if epoch >= min_epochs and (epoch - best_nse_stats['epoch']) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                print(f"Best NSE {best_nse_stats['nse']:.6f} achieved at epoch {best_nse_stats['epoch']+1}.")
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
                summary_best_nse = best_nse_stats.copy()
                summary_best_nse['epoch'] = best_nse_stats.get('epoch', 0) + 1
                summary_best_loss = best_loss_stats.copy()
                summary_best_loss['epoch'] = best_loss_stats.get('epoch', 0) + 1
                
                summary_metrics = {
                    'best_validation_model': summary_best_nse,
                    'best_loss_model': summary_best_loss,
                    'final_system': {
                        'total_training_time_seconds': total_time,
                        'total_epochs_run': (epoch + 1) if 'epoch' in locals() else 0,
                        'total_steps_run': global_step
                    }
                }
                aim_run['summary'] = summary_metrics
                print("Summary metrics logged to Aim.")
            except Exception as e:
                 print(f"Warning: Error logging summary metrics to Aim: {e}")

        # --- Save Model and Generate Final Plots ---
        if ask_for_confirmation():
            if best_params_nse is not None:
                try:
                    model_save_path = save_model(best_params_nse, model_dir, trial_name)
                    print(f"Best model (by NSE) saved to: {model_save_path}")

                    if aim_run:
                        try:
                            aim_run.log_artifact(model_save_path, name='model_weights.pkl')
                            print(f"  Logged model weights .pkl file to Aim run: {run_hash}")
                        except Exception as e_aim:
                            print(f"  Warning: Failed to log model .pkl to Aim: {e_aim}")

                    # --- Generate 1D Plot ---
                    print("  Generating 1D validation plot...")
                    plot_cfg = cfg.get("plotting", {})
                    eps_plot = cfg.get("numerics", {}).get("eps", 1e-6)
                    t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
                    
                    nx_val_plot = plot_cfg.get("nx_val", 101)
                    y_const_plot = plot_cfg.get("y_const_plot", 0.0)
                    x_val_plot = jnp.linspace(0.0, cfg["domain"]["lx"], nx_val_plot, dtype=DTYPE)
                    plot_points_1d = jnp.stack([
                        x_val_plot, 
                        jnp.full_like(x_val_plot, y_const_plot, dtype=DTYPE), 
                        jnp.full_like(x_val_plot, t_const_val_plot, dtype=DTYPE)
                    ], axis=1)

                    U_plot_pred_1d = model.apply({'params': best_params_nse['params']}, plot_points_1d, train=False)
                    h_plot_pred_1d = U_plot_pred_1d[..., 0]
                    h_plot_pred_1d = jnp.where(h_plot_pred_1d < eps_plot, 0.0, h_plot_pred_1d)

                    plot_path_1d = os.path.join(results_dir, "final_validation_plot.png")
                    plot_h_vs_x(x_val_plot, h_plot_pred_1d, t_const_val_plot, y_const_plot, cfg_dict, plot_path_1d)

                    if aim_run:
                        try:
                            aim_run.track(Image(plot_path_1d), name='validation_plot_1D', epoch=best_nse_stats['epoch'])
                            print(f"  Logged 1D plot to Aim run: {run_hash}")
                        except Exception as e_aim:
                            print(f"  Warning: Failed to log 1D plot to Aim: {e_aim}")

                    print(f"Model and plot saved in {model_dir} and {results_dir} (and logged to Aim)")
                except Exception as e:
                     print(f"Error during saving/plotting: {e}")
                     import traceback
                     traceback.print_exc()
            else:
                print("Warning: No best model found (best_params_nse is None). Skipping save and plot.")
        else:
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

                if run_hash:
                    run_artifact_dir = os.path.join("aim_repo", "aim_artifacts", run_hash)
                    if os.path.exists(run_artifact_dir):
                        shutil.rmtree(run_artifact_dir)
                        print(f"Deleted run artifact directory: {run_artifact_dir}")

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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Analytical Scenario).")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., experiments/fourier_pinn_config.yaml)")
    args = parser.parse_args()

    # This allows the script to be run directly, assuming it's in src/scenarios/
    # and the CWD is the project root.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to path: {project_root}")

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