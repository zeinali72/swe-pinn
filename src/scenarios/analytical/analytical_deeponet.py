"""
Training script for the "analytical" scenario using DeepONet (Operator Learning).
This script trains a network to learn the solution operator G: (n, u_const) -> h(x, t).
"""

import os
import sys
import time
import copy
import argparse
import importlib
import itertools
from typing import Any, Dict, Tuple, List
import shutil

import jax
import jax.numpy as jnp
from jax import random, lax
import optax
from aim import Repo, Run, Image
from flax.core import FrozenDict
import numpy as np 

# Local application imports
# Adjust paths if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config, DTYPE
from src.data import sample_domain, get_batches, get_batches_tensor, get_sample_count, DeepONetParametricSampler
from src.models import init_deeponet_model
from src.losses import (
    compute_operator_pde_loss, compute_operator_ic_loss, compute_operator_bc_loss,
    compute_operator_neg_h_loss, compute_operator_data_loss, total_loss
)
from src.utils import ( 
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation,
    plot_h_vs_x
)
from src.physics import h_exact
from src.reporting import (
    print_epoch_stats, log_metrics, print_final_summary
)

def train_step(model: Any, params: FrozenDict, opt_state: Any,
               all_batches: Dict[str, Any],
               weights_dict: Dict[str, float],
               optimiser: optax.GradientTransformation,
               config: FrozenDict
               ) -> Tuple[FrozenDict, Any, Dict[str, jnp.ndarray], jnp.ndarray]:
    """
    Performs a single training step for the DeepONet analytical scenario.
    """
    active_loss_keys_base = list(weights_dict.keys())

    def loss_and_individual_terms(p):
        terms = {}
        # --- PDE Residual Loss ---
        if 'pde' in active_loss_keys_base:
            pde_data = all_batches.get('pde', {})
            if pde_data.get('branch') is not None:
                terms['pde'] = compute_operator_pde_loss(
                    model, p, pde_data['branch'], pde_data['trunk'], config
                )
                if 'neg_h' in active_loss_keys_base:
                     terms['neg_h'] = compute_operator_neg_h_loss(
                         model, p, pde_data['branch'], pde_data['trunk'], config
                     )

        # --- Initial Condition Loss ---
        if 'ic' in active_loss_keys_base:
             ic_data = all_batches.get('ic', {})
             if ic_data.get('branch') is not None:
                 terms['ic'] = compute_operator_ic_loss(
                     model, p, ic_data['branch'], ic_data['trunk'], config
                 )

        # --- Boundary Condition Loss ---
        if 'bc' in active_loss_keys_base:
            bc_batches = all_batches.get('bc', {})
            # Check if we have data for walls
            if bc_batches.get('left', {}).get('branch') is not None:
                terms['bc'] = compute_operator_bc_loss(
                    model, p, bc_batches, config
                )

        # --- Data Loss ---
        if 'data' in active_loss_keys_base:
            data_batch = all_batches.get('data', {})
            # Check if we have data
            if data_batch.get('branch') is not None and data_batch.get('trunk') is not None:
                terms['data'] = compute_operator_data_loss(
                     model, p, data_batch['branch'], data_batch['trunk'], data_batch['h_true'], config
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
train_step_jitted = jax.jit(
    train_step,
    static_argnames=('model', 'optimiser', 'config')
)

def main(config_path: str):
    """
    Main training function for the DeepONet analytical scenario.
    """
    # --- 1. Load Config and Initialize Model ---
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)

    print("Info: Running in DeepONet Analytical mode.")
    
    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e

    key = random.PRNGKey(cfg["training"]["seed"])
    model_key, train_key, val_key = random.split(key, 3)
    
    # Use init_deeponet_model
    try:
        model, params = init_deeponet_model(model_class, model_key, cfg_dict)
    except Exception as e:
         raise RuntimeError(f"Failed to initialize DeepONet model: {e}")

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
    optimiser = optax.chain(
        optax.clip_by_global_norm(cfg.get("training", {}).get("clip_norm", 1.0)),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimiser.init(params)

    # --- 4. Prepare Loss Weights ---
    static_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    
    # --- 5. Initialize Sampler ---
    sampler = DeepONetParametricSampler(cfg)

    # --- 5.1 Initialize Data (for Data-Driven Mode) ---
    data_points_full = None # Dictionary to hold branch/trunk/h_true
    data_free_flag = cfg.get("data_free")
    has_data_loss = False
    
    if data_free_flag is False:
        print("Info: 'data_free: false' found in config. Activating data-driven mode.")
        has_data_loss = True
    else:
        if data_free_flag is None:
             print("Warning: 'data_free' flag not specified in config. Defaulting to 'data_free: true'.")
        else:
             print(f"Info: 'data_free: {data_free_flag}' found. Data loss disabled.")
        has_data_loss = False

    if has_data_loss:
        try:
            train_data_cfg = cfg.get("train_data_sampling")
            if not train_data_cfg:
                 raise KeyError("'train_data_sampling' missing in config.")

            print(f"Creating analytical training data...")
            n_funcs = train_data_cfg["n_train_functions"]
            n_pts_per_func = train_data_cfg["n_train_points_per_function"]
            
            # 1. Sample Parameters (Branch)
            # Use data_key for distinct randomness
            data_key, branch_key, trunk_key = random.split(train_key, 3) 
            # Reset train_key to avoid correlation if needed, or just branch off it
            
            # Sample parameters for N functions
            func_params = sampler.sample_parameters(branch_key, n_funcs) # (n_funcs, n_params)
            
            # 2. Sample Points (Trunk) per function
            # We want different points for each function to cover domain well, 
            # or same points? Standard DeepONet usually has same points or random. 
            # Let's do random points for each function to maximize coverage.
            
            # Efficient way: Sample total points and reshape? 
            # Or just repeat params? 
            # Let's assume we want (n_funcs * n_pts_per_func) total pairs.
            
            # Expand params:
            branch_batch_full = jnp.repeat(func_params, n_pts_per_func, axis=0) # (Total, n_params)
            
            # Sample coords:
            trunk_batch_full = sample_domain(
                trunk_key, n_funcs * n_pts_per_func,
                (0., cfg["domain"]["lx"]), (0., cfg["domain"]["ly"]), (0., cfg["domain"]["t_final"])
            )
            
            # 3. Calculate Ground Truth
            # We need to extract params to pass to h_exact
            # Param order is handled by sampler (sorted keys)
            param_names = sampler.param_names
            pm = {name: i for i, name in enumerate(param_names)}
            
            nm_idx = pm.get('n_manning')
            uc_idx = pm.get('u_const')
            
            n_manning_vals = branch_batch_full[:, nm_idx]
            u_const_vals = branch_batch_full[:, uc_idx]
            
            h_true_full = h_exact(
                trunk_batch_full[:, 0], # x
                trunk_batch_full[:, 2], # t
                n_manning_vals,
                u_const_vals
            )
            
            data_points_full = {
                'branch': branch_batch_full,
                'trunk': trunk_batch_full,
                'h_true': h_true_full
            }
            print(f"Created {branch_batch_full.shape[0]} data pairs.")
            
        except Exception as e:
            print(f"Error creating training data: {e}. Disabling data loss.")
            has_data_loss = False

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
        
        hparams_to_log = copy.deepcopy(cfg_dict)
        aim_run["hparams"] = hparams_to_log
        
        aim_run['flags'] = {
            "scenario_type": "analytical_deeponet",
            "data_free": not has_data_loss,
            "data_loss_active": has_data_loss
        }
        print(f"Aim tracking initialized for run: {trial_name} ({run_hash})")
        
    except Exception as e:
        print(f"Warning: Failed to initialize Aim tracking: {e}. Training will continue without Aim.")

    # --- 7. Determine Active Loss Terms ---
    # --- 7. Determine Active Loss Terms ---
    active_loss_term_keys = []
    for k, v in static_weights_dict.items():
        if v > 0:
            if k == 'data' and not has_data_loss:
                continue
            active_loss_term_keys.append(k)
    current_weights_dict = {k: static_weights_dict[k] for k in active_loss_term_keys}

    # --- 9. Pre-Training Summary ---
    print(f"\n--- Training Started: {trial_name} ---")
    print(f"Model: {cfg['model']['name']}")
    print(f"Scenario: DeepONet Analytical")
    print(f"Active Loss Terms: {active_loss_term_keys}")

    # Fix: Initialize all keys required by reporting.py
    best_nse_stats = {
        'nse': -jnp.inf, 'rmse': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {}
    }
    best_loss_stats = {
        'total_weighted_loss': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'nse': -jnp.inf, 'rmse': jnp.inf, 'unweighted_losses': {}
    }
    best_params_nse: Dict = None

    global_step = 0 
    start_time = time.time()
    
    # --- Batch Config ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    
    # Calculate required samples
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000)
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 500)
    n_bc_domain = get_sample_count(sampling_cfg, "n_points_bc_domain", 500)
    n_bc_per_wall = max(5, n_bc_domain // 4) if n_bc_domain > 0 else 0
    
    # Calculate batches
    bc_counts = [n_pde // batch_size, n_ic // batch_size, n_bc_per_wall // batch_size]
    if data_points_full is not None:
         bc_counts.append(data_points_full['branch'].shape[0] // batch_size)
    num_batches = max(bc_counts) if bc_counts else 0
    
    if num_batches == 0:
        print("Error: Batch size too large.")
        return -1.0

    # --- JIT Data Generator ---
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys = random.split(key, 4)
        
        # Helper to get batched tensor pairs
        def get_op_batches(rng, n, mode, x_bounds=None, y_bounds=None, t_bounds=None):
            # Optimisation: The user suggested 'using if might cause overhead'
            # Here n and batch_size are static config values.
            # JAX tracing handles this fine (static branch).
            if n // batch_size > 0:
                b, t = sampler.sample_batch(rng, n, mode, x_bounds, y_bounds, t_bounds)
                b_batch = get_batches_tensor(rng, b, batch_size, num_batches)
                t_batch = get_batches_tensor(rng, t, batch_size, num_batches)
                return {'branch': b_batch, 'trunk': t_batch}
            # Return dummy empty arrays (but correct shape/size 0) if needed or handle in train_step
            # However, train_step checks for None, so let's keep returning a dict with Nones/empty
            return {'branch': None, 'trunk': None}

        # PDE
        pde_data = get_op_batches(pde_key, n_pde, 'pde')

        # IC
        ic_data = get_op_batches(ic_key, n_ic, 'ic')
        
        # BCs
        bc_data = {}
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        domain_cfg = cfg["domain"]
        lx, ly, tf = domain_cfg["lx"], domain_cfg["ly"], domain_cfg["t_final"]
        
        # Left (x=0)
        bc_data['left'] = get_op_batches(l_key, n_bc_per_wall, 'bc', x_bounds=(0.,0.))
        # Right (x=lx)
        bc_data['right'] = get_op_batches(r_key, n_bc_per_wall, 'bc', x_bounds=(lx,lx))
        # Bottom (y=0)
        bc_data['bottom'] = get_op_batches(b_key, n_bc_per_wall, 'bc', y_bounds=(0.,0.))
        # Top (y=ly)
        bc_data['top'] = get_op_batches(t_key, n_bc_per_wall, 'bc', y_bounds=(ly,ly))

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc': bc_data,
            'data': {}
        }
        
        if data_points_full is not None:
             # Batch data points
             b_full = data_points_full['branch']
             t_full = data_points_full['trunk']
             h_full = data_points_full['h_true']
             
             # Shuffle and reshuffle logic handled by get_batches_tensor?? 
             # usage: get_batches_tensor(key, data, batch_size, total_batches)
             # It does random permutation inside.
             
             b_data = get_batches_tensor(data_key_epoch, b_full, batch_size, num_batches)
             # We need to use SAME permutation for t and h
             # get_batches_tensor shuffles internally with key. 
             # We need a synchronised shuffle helper or manually shuffle here.
             
             # Manual shuffle here to keep correspondence:
             n_d = b_full.shape[0]
             inds = random.permutation(data_key_epoch, n_d)
             
             # Trim to multiple of batch size (optional, or handled by reshaping)
             # Let's use simple logic: Just take needed batches for this epoch
             # But num_batches is fixed. 
             
             # We can't easily rely on get_batches_tensor for synchronized arrays unless we concat them.
             # Let's concat branch(N,P) + trunk(N,3) + h(N,1) -> (N, P+4)
             
             comb = jnp.hstack([b_full, t_full, h_full[:, None]])
             comb_batched = get_batches_tensor(data_key_epoch, comb, batch_size, num_batches)
             
             # Split back
             n_p = b_full.shape[1]
             
             ret_b = comb_batched[..., :n_p]
             ret_t = comb_batched[..., n_p:n_p+3]
             ret_h = comb_batched[..., n_p+3]
             
             ret_dict['data'] = {'branch': ret_b, 'trunk': ret_t, 'h_true': ret_h}

        return ret_dict

    generate_epoch_data_jit = jax.jit(generate_epoch_data)

    # --- Scan Body ---
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        
        new_params, new_opt_state, terms, total = train_step(
            model, curr_params, curr_opt_state,
            batch_data,
            current_weights_dict,
            optimiser, cfg
        )
        return (new_params, new_opt_state), (terms, total)

    # --- Pre-Generate Static Validation Data (Before Training Loop) ---
    print("\n--- Generating Static Validation Data ---")
    val_params_list = cfg.get("validation_params", [])
    validation_data_static = []  # List of dicts: {'branch', 'trunk', 'h_true', 'params'}
    
    if val_params_list:
        p_names = sorted(cfg['physics']['param_bounds'].keys())
        n_val_points = cfg["validation_grid"]["n_points_val"]
        
        for i, val_p in enumerate(val_params_list):
            # Construct branch input for this validation scenario
            b_input = []
            for name in p_names:
                b_input.append(val_p.get(name, cfg['physics'][name]))
            b_tensor = jnp.array([b_input], dtype=DTYPE)  # (1, n_params)
            
            # Generate validation points (trunk) - use deterministic key per scenario
            val_scenario_key = random.fold_in(val_key, i)
            v_points = sample_domain(
                val_scenario_key, n_val_points,
                (0., cfg["domain"]["lx"]), 
                (0., cfg["domain"]["ly"]), 
                (0., cfg["domain"]["t_final"])
            )
            
            # Repeat branch input for all points
            b_inputs = jnp.repeat(b_tensor, v_points.shape[0], axis=0)
            
            # Pre-compute ground truth
            h_true = h_exact(
                v_points[:, 0],  # x
                v_points[:, 2],  # t
                val_p['n_manning'], 
                val_p['u_const']
            )
            
            validation_data_static.append({
                'branch': b_inputs,
                'trunk': v_points,
                'h_true': h_true,
                'params': val_p
            })
            print(f"  Scenario {i+1}/{len(val_params_list)}: n={val_p['n_manning']:.4f}, u={val_p['u_const']:.4f}, points={n_val_points}")
    
    print(f"Generated {len(validation_data_static)} static validation scenarios.\n")

    # --- 10. Main Loop ---
    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()
            train_key, epoch_key = random.split(train_key)
            scan_inputs = generate_epoch_data_jit(epoch_key)

            (params, opt_state), (batch_losses_unweighted_stacked, batch_total_weighted_loss_stacked) = lax.scan(
                scan_body, (params, opt_state), scan_inputs
            )
            global_step += num_batches

            # Aggregate
            epoch_losses_unweighted_sum = {k: jnp.sum(v) for k, v in batch_losses_unweighted_stacked.items()}
            epoch_total_weighted_loss_sum = jnp.sum(batch_total_weighted_loss_stacked)
            avg_losses_unweighted = {k: float(v) / num_batches for k, v in epoch_losses_unweighted_sum.items()}
            avg_total_weighted_loss = float(epoch_total_weighted_loss_sum) / num_batches

            # --- Validation Using Static Pre-Generated Data ---
            nse_vals = []
            rmse_vals = []
            
            if validation_data_static and (epoch + 1) % 10 == 0:  # Check frequency
                for val_scenario in validation_data_static:
                    # Use pre-computed static data
                    b_inputs = val_scenario['branch']
                    v_points = val_scenario['trunk']
                    h_true = val_scenario['h_true']
                    
                    # Predict using current model parameters
                    U_pred = model.apply({'params': params['params']}, b_inputs, v_points, train=False)
                    h_pred = U_pred[..., 0]
                    
                    # Calculate NSE and RMSE for THIS SCENARIO (not globally)
                    scenario_nse = float(nse(h_pred, h_true))
                    scenario_rmse = float(rmse(h_pred, h_true))
                    
                    nse_vals.append(scenario_nse)
                    rmse_vals.append(scenario_rmse)
                
                # Average across scenarios (scenario-wise averaging, not global)
                avg_nse = np.mean(nse_vals) if nse_vals else -jnp.inf
                avg_rmse = np.mean(rmse_vals) if rmse_vals else jnp.inf
            else:
                avg_nse = -jnp.inf
                avg_rmse = jnp.inf

            # Update best model based on HIGHEST NSE (maximization)
            if avg_nse > best_nse_stats['nse']:
                best_nse_stats.update({
                    'nse': avg_nse, 'rmse': avg_rmse, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'total_weighted_loss': avg_total_weighted_loss,
                    'unweighted_losses': avg_losses_unweighted
                })
                best_params_nse = copy.deepcopy(params)

            if avg_total_weighted_loss < best_loss_stats['total_weighted_loss']:
                best_loss_update = {
                    'total_weighted_loss': avg_total_weighted_loss, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'unweighted_losses': avg_losses_unweighted
                }
                # Only update validation metrics if they were computed this epoch
                if avg_nse > -jnp.inf:
                     best_loss_update.update({'nse': avg_nse, 'rmse': avg_rmse})
                
                best_loss_stats.update(best_loss_update)

            if (epoch + 1) % cfg.get("training", {}).get("log_freq_steps", 100) == 0:
                epoch_time = time.time() - epoch_start_time
                print_epoch_stats(
                    epoch, global_step, start_time, avg_total_weighted_loss,
                    avg_losses_unweighted.get('pde', 0.0), 
                    avg_losses_unweighted.get('ic', 0.0), 
                    avg_losses_unweighted.get('bc', 0.0),
                    0.0, # building_bc_loss
                    avg_losses_unweighted.get('data', 0.0),
                    avg_losses_unweighted.get('neg_h', 0.0),
                    avg_nse, avg_rmse, epoch_time
                )
                
                if aim_run:
                     log_metrics(aim_run, global_step, epoch, {
                         'epoch_avg_total_weighted_loss': avg_total_weighted_loss,
                         'validation_metrics': {'avg_nse': avg_nse, 'avg_rmse': avg_rmse}
                     })

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        print_final_summary(time.time() - start_time, best_nse_stats, best_loss_stats)
        
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
                        'total_training_time_seconds': time.time() - start_time,
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
                    
                    # 1D plot only (simple)
                    # We need to choose ONE parameter set to plot, e.g. the first one in validation list (baseline)
                    if validation_data_static:
                         plot_p = validation_data_static[0]['params']
                         print(f"  Generating 1D validation plot for params: {plot_p}")
                         
                         p_names = sorted(cfg['physics']['param_bounds'].keys())
                         b_input = []
                         for name in p_names:
                             b_input.append(plot_p.get(name, cfg['physics'][name]))
                         b_tensor = jnp.array([b_input], dtype=DTYPE) 

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
                         
                         # Repeat branch input
                         b_inputs = jnp.repeat(b_tensor, plot_points_1d.shape[0], axis=0)

                         U_plot_pred_1d = model.apply({'params': best_params_nse['params']}, b_inputs, plot_points_1d, train=False)
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
                 if os.path.exists(results_dir):
                     shutil.rmtree(results_dir)
                 if os.path.exists(model_dir):
                     shutil.rmtree(model_dir)
             except Exception:
                 pass

        if aim_run: aim_run.close()
    
    return best_nse_stats['nse']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    main(args.config)
