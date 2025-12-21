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
from aim import Repo, Run
from flax.core import FrozenDict
import numpy as np 

from src.config import load_config, DTYPE
from src.data import sample_domain, get_batches_tensor, get_sample_count
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_data_loss, compute_neg_h_loss
)
from src.ntk import compute_ntk_traces_original, update_ntk_weights_stable
from src.utils import nse, rmse, generate_trial_name, save_model, ask_for_confirmation
from src.physics import h_exact
from src.reporting import print_epoch_stats, log_metrics, print_final_summary

def train_step(model, params, opt_state, all_batches, weights_dict, optimiser, config, data_free):
    def loss_fn(p):
        terms = {}
        if 'pde' in weights_dict:
            terms['pde'] = compute_pde_loss(model, p, all_batches['pde'], config)
            if 'neg_h' in weights_dict: terms['neg_h'] = compute_neg_h_loss(model, p, all_batches['pde'])
        if 'ic' in weights_dict: terms['ic'] = compute_ic_loss(model, p, all_batches['ic'])
        if 'bc' in weights_dict:
            bc = all_batches['bc']
            terms['bc'] = compute_bc_loss(model, p, bc['left'], bc['right'], bc['bottom'], bc['top'], config)
        total = total_loss({k: terms.get(k, 0.0) for k in weights_dict.keys()}, weights_dict)
        return total, terms

    (val, terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt_state, terms, val

def main(config_path: str):
    cfg = FrozenDict(load_config(config_path))
    model_class = getattr(importlib.import_module("src.models"), cfg["model"]["name"])
    
    # FIX: Correctly unpack train_key to avoid UnboundLocalError
    key = random.PRNGKey(cfg["training"]["seed"])
    m_key, train_key, v_key = random.split(key, 3)
    model, params = init_model(model_class, m_key, cfg)

    # 1. Functional JIT Wrappers
    train_step_jit = jax.jit(train_step, static_argnames=('model', 'optimiser', 'config', 'data_free'))
    ntk_jit = jax.jit(compute_ntk_traces_original, static_argnames=('model', 'config', 'active_keys'))
    weights_jit = jax.jit(update_ntk_weights_stable)

    # 2. Optimizer Setup
    lr_sch = optax.piecewise_constant_schedule(cfg["training"]["learning_rate"], {int(k): v for k, v in cfg.get("training", {}).get("lr_boundaries", {}).items()})
    opt = optax.chain(optax.clip_by_global_norm(cfg.get("training", {}).get("clip_norm", 1.0)), optax.adam(learning_rate=lr_sch))
    opt_state = opt.init(params)
    
    # 3. NTK State
    ntk_freq = cfg.get("ntk", {}).get("update_freq", 100)
    ema = cfg.get("ntk", {}).get("ema_alpha", 0.1)
    static_w = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    active_keys = [k for k, v in static_w.items() if v > 0 and k != 'building_bc']
    current_weights = {k: jnp.array(1.0) for k in active_keys}

    # 4. Data Sampling configuration
    samp = cfg["sampling"]; b_size = cfg["training"]["batch_size"]; dom = cfg["domain"]
    n_pde = get_sample_count(samp, "n_points_pde", 1000); n_ic = get_sample_count(samp, "n_points_ic", 100); n_bc = get_sample_count(samp, "n_points_bc_domain", 100)
    num_b = max(n_pde // b_size, 1)

    @jax.jit
    def get_data(k):
        k1, k2, k3 = random.split(k, 3); bc_k = random.split(k3, 4)
        pde_pts = sample_domain(k1, n_pde, (0., dom["lx"]), (0., dom["ly"]), (0., dom["t_final"]))
        ic_pts = sample_domain(k2, n_ic, (0., dom["lx"]), (0., dom["ly"]), (0., 0.))
        bc_struct = {
            'left': get_batches_tensor(bc_k[0], sample_domain(bc_k[0], n_bc//4, (0.,0.), (0., dom["ly"]), (0., dom["t_final"])), b_size, num_b),
            'right': get_batches_tensor(bc_k[1], sample_domain(bc_k[1], n_bc//4, (dom["lx"],dom["lx"]), (0., dom["ly"]), (0., dom["t_final"])), b_size, num_b),
            'bottom': get_batches_tensor(bc_k[2], sample_domain(bc_k[2], n_bc//4, (0., dom["lx"]), (0.,0.), (0., dom["t_final"])), b_size, num_b),
            'top': get_batches_tensor(bc_k[3], sample_domain(bc_k[3], n_bc//4, (0., dom["lx"]), (dom["ly"], dom["ly"]), (0., dom["t_final"])), b_size, num_b)
        }
        return {'pde': get_batches_tensor(k1, pde_pts, b_size, num_b), 'ic': get_batches_tensor(k2, ic_pts, b_size, num_b), 'bc': bc_struct, 'data': jnp.zeros((num_b, 0, 6))}

    # 5. Scan Body (GPU execution)
    def body(carry, batch):
        p, os, w, step = carry
        w = lax.cond(jnp.logical_and(cfg.get("ntk", {}).get("enable", True), (step % ntk_freq == 0)), 
                     lambda _: weights_jit(ntk_jit(model, p, batch, cfg, active_keys), w, ema), 
                     lambda _: w, None)
        p, os, terms, total = train_step_jit(model, p, os, batch, w, opt, cfg, cfg.get("data_free", True))
        return (p, os, w, step + 1), (terms, total)

    # 6. Training Initialization
    stats = {'nse': -jnp.inf, 'rmse': jnp.inf, 'epoch': 0, 'global_step': 0, 'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {}}
    g_step = 0; start = time.time()
    
    val_pts = sample_domain(v_key, cfg["validation_grid"]["n_points_val"], (0., dom["lx"]), (0., dom["ly"]), (0., dom["t_final"]))
    h_true = h_exact(val_pts[:, 0], val_pts[:, 2], cfg["physics"]["n_manning"], cfg["physics"]["u_const"])

    # Trial directories
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    results_dir, model_dir = os.path.join("results", trial_name), os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True); os.makedirs(model_dir, exist_ok=True)

    # 7. Training Loop
    try:
        for ep in range(cfg["training"]["epochs"]):
            ep_start = time.time()
            train_key, epoch_key = random.split(train_key) # train_key is now defined
            (params, opt_state, current_weights, g_step), (b_losses, b_totals) = lax.scan(body, (params, opt_state, current_weights, g_step), get_data(epoch_key))
            
            avg_uw = {k: float(jnp.mean(v)) for k, v in b_losses.items()}
            h_pred = model.apply({'params': params['params']}, val_pts, train=False)[..., 0]
            cur_nse = float(nse(h_pred, h_true))
            
            if cur_nse > stats['nse']:
                stats.update({'nse': cur_nse, 'rmse': float(rmse(h_pred, h_true)), 'epoch': ep, 'global_step': int(g_step), 'time_elapsed_seconds': time.time() - start, 'total_weighted_loss': float(jnp.mean(b_totals)), 'unweighted_losses': avg_uw})
                best_params_nse = copy.deepcopy(params)
                print(f"Epoch {ep+1}: New best NSE {cur_nse:.6f}")
                # FIX: Fixed dictionary comprehension syntax
                w_print = {k: f"{float(v):.2e}" for k, v in current_weights.items()}
                print(f"   Current Weights: {w_print}")

            if (ep + 1) % 100 == 0:
                print_epoch_stats(ep, int(g_step), start, float(jnp.mean(b_totals)), avg_uw.get('pde', 0.0), avg_uw.get('ic', 0.0), avg_uw.get('bc', 0.0), 0.0, 0.0, avg_uw.get('neg_h', 0.0), cur_nse, float(rmse(h_pred, h_true)), time.time()-ep_start)

    finally:
        print_final_summary(time.time() - start, stats, {})
        if 'best_params_nse' in locals() and ask_for_confirmation():
            save_model(best_params_nse, model_dir, trial_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=str, required=True); main(parser.parse_args().config)