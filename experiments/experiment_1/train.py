"""Experiment 1 — Analytical dam-break on flat domain (Phase 1).

Verifies the PINN framework against an analytical dam-break solution.
Requires: configs/experiment_1_*.yaml
Builds on: None (baseline verification).

This script handles both data-free and data-driven (analytical) training modes.
It supports static loss weighting and provides
logging and result visualization through Aim.
"""

import os
import sys
import time
import copy
import argparse
import itertools
from typing import Any, Dict, Tuple
import shutil

import jax
import jax.numpy as jnp
from jax import random, lax
import optax
try:
    from aim import Image
except ImportError:
    Image = None
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
from src.utils import (
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation,
    plot_h_vs_x
)
from src.physics import h_exact
from src.monitoring import ConsoleLogger, AimTracker, compute_negative_depth_diagnostics
from src.monitoring.diagnostics import compute_grad_norm
from src.metrics.accuracy import compute_validation_metrics
from src.checkpointing import CheckpointManager
from src.predict import Predictor
from src.training import (
    create_optimizer,
    extract_loss_weights,
    post_training_save,
    resolve_data_mode,
    run_training_loop,
    setup_experiment,
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
    
    # MODIFIED: Pass 'value' (loss) to update for reduce_on_plateau
    updates, new_opt_state = optimiser.update(grads, opt_state, params, value=total_loss_val)
    
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
    setup = setup_experiment(config_path, "experiment_1")
    cfg_dict = setup["cfg_dict"]
    cfg = setup["cfg"]
    model = setup["model"]
    params = setup["params"]
    train_key = setup["train_key"]
    val_key = setup["val_key"]
    trial_name = setup["trial_name"]
    results_dir = setup["results_dir"]
    model_dir = setup["model_dir"]

    print("Info: Running in analytical (no-building) mode.")

    # --- 3. Setup Optimizer ---
    optimiser = create_optimizer(cfg)
    opt_state = optimiser.init(params)

    # --- 4. Prepare Loss Weights ---
    static_weights_dict, _ = extract_loss_weights(cfg)
    
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
    data_free, has_data_loss = resolve_data_mode(cfg)

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
    
    # --- 6. Determine Active Loss Terms ---
    active_loss_term_keys = []
    for k, v in static_weights_dict.items():
        if v > 0:
            if k == 'data' and data_free:
                continue 
            if k == 'building_bc': # Explicitly skip building loss
                continue
            active_loss_term_keys.append(k)
    
    current_weights_dict = {k: static_weights_dict[k] for k in active_loss_term_keys}

    # --- 7. Pre-calculate Batch Counts and Total Batches (for jax.lax.scan) ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]

    # Calculate expected points
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000) if ('pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys) else 0
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100) if 'ic' in active_loss_term_keys else 0
    n_bc_domain = get_sample_count(sampling_cfg, "n_points_bc_domain", 100) if 'bc' in active_loss_term_keys else 0
    n_bc_per_wall = max(5, n_bc_domain // 4) if n_bc_domain > 0 else 0
    
    # Calculate available batches per term
    bc_counts = [
        n_pde // batch_size,
        n_ic // batch_size,
        n_bc_per_wall // batch_size, # left
        n_bc_per_wall // batch_size, # right
        n_bc_per_wall // batch_size, # bottom
        n_bc_per_wall // batch_size, # top
    ]
    if not data_free and data_points_full is not None:
         bc_counts.append(data_points_full.shape[0] // batch_size)

    num_batches = max(bc_counts) if bc_counts else 0
    
    if num_batches == 0:
        print(f"Error: Batch size {batch_size} is too large for configured sample counts or data. No training will occur.")
        return -1.0

    # --- Define JIT Data Generator ---
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, data_key = random.split(key, 5)
        
        # PDE
        if n_pde // batch_size > 0:
            pde_points = sample_domain(pde_key, n_pde, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
            pde_data = get_batches_tensor(pde_key, pde_points, batch_size, num_batches)
        else:
            pde_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        # IC
        if n_ic // batch_size > 0:
            ic_points = sample_domain(ic_key, n_ic, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))
            ic_data = get_batches_tensor(ic_key, ic_points, batch_size, num_batches)
        else:
            ic_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)
            
        # BCs
        bc_data = {}
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        
        # Helper for walls
        def get_wall_data(k, n, x_rng, y_rng, t_rng):
            if n // batch_size > 0:
                pts = sample_domain(k, n, x_rng, y_rng, t_rng)
                return get_batches_tensor(k, pts, batch_size, num_batches)
            return jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        bc_data['left'] = get_wall_data(l_key, n_bc_per_wall, (0., 0.), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
        bc_data['right'] = get_wall_data(r_key, n_bc_per_wall, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
        bc_data['bottom'] = get_wall_data(b_key, n_bc_per_wall, (0., domain_cfg["lx"]), (0., 0.), (0., domain_cfg["t_final"]))
        bc_data['top'] = get_wall_data(t_key, n_bc_per_wall, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]), (0., domain_cfg["t_final"]))

        # Data
        data_data = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)
        if not data_free and data_points_full is not None:
             # get_batches_tensor reshapes/repeats data_points_full to match num_batches
             data_data = get_batches_tensor(data_key, data_points_full, batch_size, num_batches)

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc': bc_data,
            'data': data_data,
            'building_bc': {} # Empty for analytical
        }

    generate_epoch_data_jit = jax.jit(generate_epoch_data)

    # --- Define Scan Body Function ---
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        
        # Reconstruct hierarchical dict expected by train_step
        current_all_batches = {
            'pde': batch_data['pde'],
            'ic': batch_data['ic'],
            'bc': batch_data['bc'],
            'data': batch_data['data'],
            'building_bc': batch_data['building_bc']
        }

        new_params, new_opt_state, terms, total = train_step(
            model, curr_params, curr_opt_state,
            current_all_batches,
            current_weights_dict,
            optimiser, cfg, data_free
        )
        return (new_params, new_opt_state), (terms, total)

    def validation_fn(model, params):
        nse_val, rmse_val = -jnp.inf, jnp.inf
        if validation_data_loaded:
            try:
                U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                h_pred_val = U_pred_val[..., 0]
                nse_val = float(nse(h_pred_val, h_true_val))
                rmse_val = float(rmse(h_pred_val, h_true_val))
            except Exception as exc:
                print(f"Warning: Validation calculation failed: {exc}")
        return {'nse_h': float(nse_val), 'rmse_h': float(rmse_val)}

    loop_result = run_training_loop(
        cfg=cfg,
        cfg_dict=cfg_dict,
        model=model,
        params=params,
        opt_state=opt_state,
        train_key=train_key,
        optimiser=optimiser,
        generate_epoch_data_jit=generate_epoch_data_jit,
        scan_body=scan_body,
        num_batches=num_batches,
        experiment_name="experiment_1",
        trial_name=trial_name,
        results_dir=results_dir,
        model_dir=model_dir,
        config_path=config_path,
        validation_fn=validation_fn,
        source_script_path=__file__,
    )

    def plot_fn(final_params):
        print("  Generating 1D validation plot...")
        aim_tracker = loop_result["aim_tracker"]
        final_epoch = loop_result["epoch"]
        plot_cfg = cfg.get("plotting", {})
        eps_plot = cfg.get("numerics", {}).get("eps", 1e-6)
        t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
        nx_val_plot = plot_cfg.get("nx_val", 101)
        y_const_plot = plot_cfg.get("y_const_plot", 0.0)
        x_val_plot = jnp.linspace(0.0, cfg["domain"]["lx"], nx_val_plot, dtype=DTYPE)
        plot_points_1d = jnp.stack([
            x_val_plot,
            jnp.full_like(x_val_plot, y_const_plot, dtype=DTYPE),
            jnp.full_like(x_val_plot, t_const_val_plot, dtype=DTYPE),
        ], axis=1)
        U_plot_pred_1d = model.apply({'params': final_params['params']}, plot_points_1d, train=False)
        h_plot_pred_1d = jnp.where(U_plot_pred_1d[..., 0] < eps_plot, 0.0, U_plot_pred_1d[..., 0])
        plot_path_1d = os.path.join(results_dir, "final_validation_plot.png")
        plot_h_vs_x(x_val_plot, h_plot_pred_1d, t_const_val_plot, y_const_plot, cfg_dict, plot_path_1d)
        aim_tracker.log_image(plot_path_1d, 'validation_plot_1D', final_epoch)
        print(f"Model and plot saved in {model_dir} and {results_dir} (and logged to Aim)")

    post_training_save(
        loop_result=loop_result,
        model=model,
        model_dir=model_dir,
        results_dir=results_dir,
        trial_name=trial_name,
        plot_fn=plot_fn,
    )

    return loop_result["best_nse_stats"]["nse"] if loop_result["best_nse_stats"]["nse"] > -jnp.inf else -1.0


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