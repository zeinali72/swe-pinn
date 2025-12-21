# src/scenarios/analytical/analytical_ntk.py
import os
import sys
import time
import copy
import argparse
import importlib
from typing import Any, Dict, Tuple
import shutil

import jax
import jax.numpy as jnp
from jax import random, lax
import optax
from aim import Repo, Run, Image
from flax.core import FrozenDict
import numpy as np 

from src.config import load_config, DTYPE
from src.data import sample_domain, get_batches_tensor, get_sample_count
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_data_loss, compute_neg_h_loss
)
from src.ntk import compute_ntk_traces, update_ntk_weights
from src.utils import ( 
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation,
    plot_h_vs_x
)
from src.physics import h_exact
from src.reporting import (
    print_epoch_stats, log_metrics, print_final_summary
)

def train_step(model: Any, params: FrozenDict, opt_state: Any,
               all_batches: Dict[str, Any], weights_dict: Dict[str, float],
               optimiser: optax.GradientTransformation, config: FrozenDict,
               data_free: bool = True):
    active_loss_keys_base = list(weights_dict.keys())
    def loss_and_individual_terms(p):
        terms = {}
        pde_batch_data = all_batches.get('pde', jnp.empty((0,3), dtype=DTYPE))
        if 'pde' in active_loss_keys_base:
            terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config)
            if 'neg_h' in active_loss_keys_base:
                terms['neg_h'] = compute_neg_h_loss(model, p, pde_batch_data)
        if 'ic' in active_loss_keys_base:
            terms['ic'] = compute_ic_loss(model, p, all_batches.get('ic', jnp.empty((0,3), dtype=DTYPE)))
        bc_batches = all_batches.get('bc', {})
        if 'bc' in active_loss_keys_base:
             terms['bc'] = compute_bc_loss(model, p, bc_batches.get('left', jnp.empty((0,3))), bc_batches.get('right', jnp.empty((0,3))), bc_batches.get('bottom', jnp.empty((0,3))), bc_batches.get('top', jnp.empty((0,3))), config)
        if not data_free and 'data' in active_loss_keys_base:
             terms['data'] = compute_data_loss(model, p, all_batches.get('data', jnp.empty((0,6))), config)
        total = total_loss({k: terms.get(k, 0.0) for k in weights_dict.keys()}, weights_dict)
        return total, terms
    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt_state, individual_terms_val, total_loss_val

def main(config_path: str):
    cfg_dict = load_config(config_path); cfg = FrozenDict(cfg_dict)
    models_module = importlib.import_module("src.models")
    model_class = getattr(models_module, cfg["model"]["name"])
    key = random.PRNGKey(cfg["training"]["seed"]); model_key, train_key, val_key = random.split(key, 3)
    model, params = init_model(model_class, model_key, cfg)
    
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    results_dir, model_dir = os.path.join("results", trial_name), os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True); os.makedirs(model_dir, exist_ok=True)

    lr_schedule = optax.piecewise_constant_schedule(cfg["training"]["learning_rate"], {int(k): v for k, v in cfg.get("training", {}).get("lr_boundaries", {}).items()})
    optimiser = optax.chain(optax.clip_by_global_norm(cfg.get("training", {}).get("clip_norm", 1.0)), optax.adam(learning_rate=lr_schedule))
    opt_state = optimiser.init(params)

    ntk_cfg = cfg.get("ntk", {}); enable_ntk = ntk_cfg.get("enable", True); ntk_update_freq = ntk_cfg.get("update_freq", 100); ntk_ema = ntk_cfg.get("ema_alpha", 0.1)
    static_weights = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    data_free = cfg.get("data_free", True); active_keys = [k for k, v in static_weights.items() if v > 0 and k != 'building_bc']
    if data_free and 'data' in active_keys: active_keys.remove('data')
    current_weights = {k: jnp.array(1.0) for k in active_keys}

    aim_run = None
    try:
        aim_run = Run(repo=Repo(path="aim_repo", init=True), experiment=trial_name)
        aim_run["hparams"] = cfg_dict
    except: pass

    sampling_cfg = cfg["sampling"]; batch_size = cfg["training"]["batch_size"]; domain_cfg = cfg["domain"]
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000); n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100); n_bc = get_sample_count(sampling_cfg, "n_points_bc_domain", 100)
    num_batches = max(n_pde // batch_size, 1)

    @jax.jit
    def generate_epoch_data_jit(key):
        k1, k2, k3 = random.split(key, 3); bc_keys = random.split(k3, 4)
        pde_pts = sample_domain(k1, n_pde, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
        ic_pts = sample_domain(k2, n_ic, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))
        return {
            'pde': get_batches_tensor(k1, pde_pts, batch_size, num_batches),
            'ic': get_batches_tensor(k2, ic_pts, batch_size, num_batches),
            'bc': {wall: get_batches_tensor(bc_keys[i], sample_domain(bc_keys[i], n_bc//4, x, y, (0., domain_cfg["t_final"])), batch_size, num_batches) for i, (wall, x, y) in enumerate([('left', (0.,0.), (0., domain_cfg["ly"])), ('right', (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"])), ('bottom', (0., domain_cfg["lx"]), (0., 0.)), ('top', (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]))])},
            'data': jnp.zeros((num_batches, 0, 6)), 'building_bc': {}
        }

    def scan_body(carry, batch_data):
        curr_p, curr_os, curr_w, step = carry
        new_w = lax.cond(jnp.logical_and(enable_ntk, (step % ntk_update_freq == 0)), lambda _: update_ntk_weights(compute_ntk_traces(model, curr_p, batch_data, cfg, active_keys), curr_w, ntk_ema), lambda _: curr_w, None)
        new_p, new_os, terms, total = train_step(model, curr_p, curr_os, batch_data, new_w, optimiser, cfg, data_free)
        return (new_p, new_os, new_w, step + 1), (terms, total)

    best_nse_stats = {'nse': -jnp.inf, 'rmse': jnp.inf, 'epoch': 0, 'global_step': 0, 'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {}}
    global_step = 0; start_time = time.time()
    val_pts = sample_domain(val_key, cfg["validation_grid"]["n_points_val"], (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
    h_true_val = h_exact(val_pts[:, 0], val_pts[:, 2], cfg["physics"]["n_manning"], cfg["physics"]["u_const"])

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start = time.time(); train_key, epoch_key = random.split(train_key)
            (params, opt_state, current_weights, global_step), (batch_losses, batch_totals) = lax.scan(scan_body, (params, opt_state, current_weights, global_step), generate_epoch_data_jit(epoch_key))
            
            avg_unweighted = {k: float(jnp.mean(v)) for k, v in batch_losses.items()}
            avg_weighted = float(jnp.mean(batch_totals))
            h_pred_val = model.apply({'params': params['params']}, val_pts, train=False)[..., 0]
            nse_val, rmse_val = float(nse(h_pred_val, h_true_val)), float(rmse(h_pred_val, h_true_val))

            if nse_val > best_nse_stats['nse']:
                best_nse_stats.update({'nse': nse_val, 'rmse': rmse_val, 'epoch': epoch, 'global_step': int(global_step), 'time_elapsed_seconds': time.time() - start_time, 'total_weighted_loss': avg_weighted, 'unweighted_losses': avg_unweighted})
                best_params_nse = copy.deepcopy(params)
                print(f"Epoch {epoch+1}: New best NSE {nse_val:.6f}")
                weights_out = {k: f"{float(v):.2e}" for k, v in current_weights.items()}
                print(f"   Current Weights: {weights_out}")

            if (epoch + 1) % 100 == 0:
                print_epoch_stats(epoch, int(global_step), start_time, avg_weighted, avg_unweighted.get('pde', 0.0), avg_unweighted.get('ic', 0.0), avg_unweighted.get('bc', 0.0), 0.0, avg_unweighted.get('data', 0.0), avg_unweighted.get('neg_h', 0.0), nse_val, rmse_val, time.time()-epoch_start)

    except KeyboardInterrupt: pass
    finally:
        print_final_summary(time.time() - start_time, best_nse_stats, {})
        if ask_for_confirmation() and 'best_params_nse' in locals():
            save_model(best_params_nse, model_dir, trial_name)
    return best_nse_stats['nse']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=str, required=True)
    main(parser.parse_args().config)