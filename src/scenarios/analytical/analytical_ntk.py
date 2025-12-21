"""
Updated Training script with NTK-based weighting for the analytical scenario.
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
from src.config import load_config, DTYPE
from src.data import sample_domain, get_batches, get_batches_tensor, get_sample_count
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_data_loss, compute_neg_h_loss
)
from src.ntk import compute_ntk_traces, update_ntk_weights # <-- New NTK imports
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
               config: FrozenDict,
               data_free: bool = True
               ) -> Tuple[FrozenDict, Any, Dict[str, jnp.ndarray], jnp.ndarray]:
    
    active_loss_keys_base = list(weights_dict.keys())

    def loss_and_individual_terms(p):
        terms = {}
        pde_batch_data = all_batches.get('pde', jnp.empty((0,3), dtype=DTYPE))
        if 'pde' in active_loss_keys_base and pde_batch_data.shape[0] > 0:
            terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config)
            if 'neg_h' in active_loss_keys_base:
                terms['neg_h'] = compute_neg_h_loss(model, p, pde_batch_data)

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

        data_batch_data = all_batches.get('data', jnp.empty((0,6), dtype=DTYPE))
        if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0:
             terms['data'] = compute_data_loss(model, p, data_batch_data, config)

        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms

    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, individual_terms_val, total_loss_val

def main(config_path: str):
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e

    key = random.PRNGKey(cfg["training"]["seed"])
    model_key, train_key, val_key = random.split(key, 3)
    model, params = init_model(model_class, model_key, cfg)

    # Setup directories
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    results_dir, model_dir = os.path.join("results", trial_name), os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True); os.makedirs(model_dir, exist_ok=True)

    # Setup Optimizer
    raw_boundaries = cfg.get("training", {}).get("lr_boundaries", {15000: 0.1, 30000: 0.1})
    boundaries_and_scales_int_keys = {int(k): v for k, v in raw_boundaries.items()}
    lr_schedule = optax.piecewise_constant_schedule(init_value=cfg["training"]["learning_rate"], boundaries_and_scales=boundaries_and_scales_int_keys)
    optimiser = optax.chain(optax.clip_by_global_norm(cfg.get("training", {}).get("clip_norm", 1.0)), optax.adam(learning_rate=lr_schedule))
    opt_state = optimiser.init(params)

    # NTK Configuration
    ntk_cfg = cfg.get("ntk", {})
    enable_ntk = ntk_cfg.get("enable", True) # Enabled by default
    ntk_update_freq = ntk_cfg.get("update_freq", 100) # Update every 100 steps
    ntk_ema = ntk_cfg.get("ema_alpha", 0.1)

    # Determine Active Loss Terms
    static_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    data_free = cfg.get("data_free", True)
    active_loss_term_keys = [k for k, v in static_weights_dict.items() if v > 0 and k != 'building_bc']
    if data_free and 'data' in active_loss_term_keys: active_loss_term_keys.remove('data')

    # Initialize weights to 1.0 (they will be scaled by NTK immediately)
    current_weights_dict = {k: jnp.array(1.0) for k in active_loss_term_keys}

    # Setup Aim
    aim_run = None
    try:
        aim_repo = Repo(path="aim_repo", init=True)
        aim_run = Run(repo=aim_repo, experiment=trial_name)
        aim_run["hparams"] = cfg_dict
    except Exception as e: print(f"Aim disabled: {e}")

    # Data Sampling configuration
    sampling_cfg = cfg["sampling"]; batch_size = cfg["training"]["batch_size"]; domain_cfg = cfg["domain"]
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000)
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100)
    n_bc_domain = get_sample_count(sampling_cfg, "n_points_bc_domain", 100)
    n_bc_per_wall = max(5, n_bc_domain // 4)
    num_batches = max(n_pde // batch_size, 1)

    @jax.jit
    def generate_epoch_data_jit(key):
        k1, k2, k3, k4 = random.split(key, 4)
        pde_pts = sample_domain(k1, n_pde, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
        ic_pts = sample_domain(k2, n_ic, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))
        bc_keys = random.split(k3, 4)
        left = sample_domain(bc_keys[0], n_bc_per_wall, (0., 0.), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
        right = sample_domain(bc_keys[1], n_bc_per_wall, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
        bottom = sample_domain(bc_keys[2], n_bc_per_wall, (0., domain_cfg["lx"]), (0., 0.), (0., domain_cfg["t_final"]))
        top = sample_domain(bc_keys[3], n_bc_per_wall, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]), (0., domain_cfg["t_final"]))
        
        return {
            'pde': get_batches_tensor(k1, pde_pts, batch_size, num_batches),
            'ic': get_batches_tensor(k2, ic_pts, batch_size, num_batches),
            'bc': {
                'left': get_batches_tensor(bc_keys[0], left, batch_size, num_batches),
                'right': get_batches_tensor(bc_keys[1], right, batch_size, num_batches),
                'bottom': get_batches_tensor(bc_keys[2], bottom, batch_size, num_batches),
                'top': get_batches_tensor(bc_keys[3], top, batch_size, num_batches)
            },
            'data': jnp.zeros((num_batches, 0, 6)),
            'building_bc': {}
        }

    # --- THE SCAN BODY (With GPU NTK Logic) ---
    def scan_body(carry, input_pack):
        curr_params, curr_opt_state, curr_weights, step_counter = carry
        batch_data = input_pack

        # 1. Periodic NTK Update inside the Scan (Fully GPU)
        def perform_ntk_update(_):
            traces = compute_ntk_traces(model, curr_params, batch_data, cfg, active_loss_term_keys)
            return update_ntk_weights(traces, curr_weights, ntk_ema)

        # Only update if (step % frequency == 0) and NTK is enabled
        new_weights = lax.cond(
            jnp.logical_and(enable_ntk, (step_counter % ntk_update_freq == 0)),
            perform_ntk_update,
            lambda _: curr_weights,
            operand=None
        )

        # 2. Standard Training Step
        new_params, new_opt_state, terms, total = train_step(
            model, curr_params, curr_opt_state, batch_data, new_weights, optimiser, cfg, data_free
        )
        
        return (new_params, new_opt_state, new_weights, step_counter + 1), (terms, total, new_weights)

    # --- Training Loop ---
    best_nse_stats = {'nse': -jnp.inf}; global_step = 0; start_time = time.time()
    
    # Validation data prep (as per your original script)
    val_points = sample_domain(val_key, 1000, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
    h_true_val = h_exact(val_points[:, 0], val_points[:, 2], cfg["physics"]["n_manning"], cfg["physics"]["u_const"])

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()
            train_key, epoch_key = random.split(train_key)
            scan_inputs = generate_epoch_data_jit(epoch_key)

            # --- RUN SCAN ---
            # Carry includes weights so they persist across steps and epochs
            (params, opt_state, current_weights_dict, global_step), (batch_losses, batch_totals, weight_history) = lax.scan(
                scan_body, (params, opt_state, current_weights_dict, global_step), scan_inputs
            )
            
            # Epoch Metrics
            avg_weighted_loss = jnp.mean(batch_totals)
            U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
            nse_val = float(nse(U_pred_val[..., 0], h_true_val))

            if nse_val > best_nse_stats['nse']:
                best_nse_stats.update({'nse': nse_val, 'epoch': epoch})
                best_params_nse = copy.deepcopy(params)
                print(f"Epoch {epoch+1}: New best NSE {nse_val:.6f}")
                # Log the massive weights to see the NTK scale
                print(f"   Current Weights: {{k: f'{float(v):.2e}' for k, v in current_weights_dict.items()}}")

            if aim_run and (epoch % 10 == 0):
                log_metrics(aim_run, step=int(global_step), epoch=epoch, metrics={
                    'validation_metrics': {'nse': nse_val},
                    'ntk_weights': {k: float(v) for k, v in current_weights_dict.items()}
                })

    except KeyboardInterrupt: print("Interrupted.")
    finally:
        print_final_summary(time.time() - start_time, best_nse_stats, {})
        if ask_for_confirmation() and 'best_params_nse' in locals():
            save_model(best_params_nse, model_dir, trial_name)

    return best_nse_stats['nse']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)