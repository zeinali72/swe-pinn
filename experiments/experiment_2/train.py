"""Experiment 2 — Dam-break with building obstacle (Phase 1).

Introduces a building obstacle; motivates Fourier-MLP and DGM adoption.
Requires: configs/experiment_2_*.yaml, data/experiment_2/
Builds on: Experiment 1.

This script handles training for scenarios with building
structures. It supports static loss weighting and provides
comprehensive logging and result visualization through Aim.

This is derived from the unified 'src/train.py'.
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
# (Assuming this file is at src/scenarios/building/building.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to path: {project_root}")

from src.config import load_config, DTYPE
from src.data import sample_domain, get_batches, get_batches_tensor, get_sample_count, load_validation_data
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_building_bc_loss, compute_data_loss, compute_neg_h_loss
)
from src.utils import ( 
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation,
    mask_points_inside_building,
    plot_comparison_scatter_2d
)
# Note: h_exact and plot_h_vs_x are omitted as they are for analytical scenario
from src.monitoring import ConsoleLogger, AimTracker, compute_negative_depth_diagnostics
from src.metrics.accuracy import compute_validation_metrics
from src.checkpointing import CheckpointManager
from src.training import (
    create_optimizer,
    extract_loss_weights,
    load_training_data,
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
    Performs a single training step, including loss calculation and parameter updates.
    (This function is identical to the one in src/train.py)
    """
    has_building = "building" in config # This will always be true for this script
    active_loss_keys_base = list(weights_dict.keys())

    def loss_and_individual_terms(p):
        terms = {}
        pde_batch_data = all_batches.get('pde', jnp.empty((0,3), dtype=DTYPE))
        if 'pde' in active_loss_keys_base and pde_batch_data.shape[0] > 0:
            pde_mask = mask_points_inside_building(pde_batch_data, config["building"])
            terms['pde'] = compute_pde_loss(model, p, pde_batch_data, config, pde_mask)
            if 'neg_h' in active_loss_keys_base:
                terms['neg_h'] = compute_neg_h_loss(model, p, pde_batch_data, pde_mask)

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

        # This block is the key part for the building scenario
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

        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms

    (total_loss_val, individual_terms_val), grads = jax.value_and_grad(loss_and_individual_terms, has_aux=True)(params)
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
    Main training function for the BUILDING scenario.
    """
    # --- 1. Load Config and Initialize Model ---
    setup = setup_experiment(config_path, "experiment_2")
    cfg_dict = setup["cfg_dict"]
    cfg = setup["cfg"]
    model = setup["model"]
    params = setup["params"]
    train_key = setup["train_key"]
    trial_name = setup["trial_name"]
    results_dir = setup["results_dir"]
    model_dir = setup["model_dir"]
    
    # --- BUILDING SCRIPT ASSERTION ---
    has_building = "building" in cfg
    if not has_building:
        print(f"Error: This script ('{__file__}') is for 'building' scenarios only.")
        print(f"Config '{config_path}' is missing the 'building' section.")
        print("Please use 'src/scenarios/analytical/analytical.py' for this config.")
        sys.exit(1)
    print("Info: Running in building mode.")
    # --- END ASSERTION ---

    # --- 3. Setup Optimizer ---
    optimiser = create_optimizer(cfg)
    opt_state = optimiser.init(params)


    # --- 4. Prepare Loss Weights ---
    static_weights_dict, _ = extract_loss_weights(cfg)

    # --- 5. Load Validation and Training Data ---
    val_points, h_true_val = None, None
    data_points_full = None
    
    scenario_name = cfg.get('scenario')
    if not scenario_name:
         print(f"Error: 'scenario' key must be set in config '{config_path}' for building mode.")
         sys.exit(1)
         
    base_data_path = os.path.join("data", scenario_name)

    # === START MODIFIED BLOCK ===
    # This logic now mirrors analytical.py
    data_free, has_data_loss = resolve_data_mode(cfg)
    data_points_full, has_data_loss, data_free = load_training_data(
        base_data_path,
        has_data_loss,
        static_weights_dict,
    )

    validation_data_file = os.path.join(base_data_path, "validation_sample.npy")
    validation_data_loaded = False
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            _, val_points_all, val_targets_all = load_validation_data(validation_data_file, dtype=DTYPE)
            h_true_val_all = val_targets_all[:, 0]
            print("Applying building mask to validation metrics points...")
            mask_val = mask_points_inside_building(val_points_all, cfg["building"])
            val_points = val_points_all[mask_val]
            h_true_val = h_true_val_all[mask_val]
            num_masked_val_points = val_points.shape[0]
            print(f"Masked validation metrics points remaining: {num_masked_val_points}.")
            if num_masked_val_points > 0:
                validation_data_loaded = True
            else:
                 print("Warning: No validation points remaining after masking. NSE/RMSE calculation will be skipped.")
        except Exception as e:
            print(f"Error loading or processing validation data file {validation_data_file}: {e}")
            val_points, h_true_val = None, None
            print("NSE/RMSE calculation using loaded data will be skipped.")
    else:
        print(f"Warning: Validation data file not found at {validation_data_file}.")
        print("Validation metrics (NSE/RMSE) for building scenario will be skipped.")

    # --- 6. Determine Active Loss Terms for the Run ---
    active_loss_term_keys = []
    for k, v in static_weights_dict.items():
        if v > 0:
            if k == 'data' and data_free: # Use the final data_free flag
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
    
    # Building BC points
    n_bldg_per_wall = 0
    if has_building and 'building_bc' in active_loss_term_keys:
        n_bldg = get_sample_count(sampling_cfg, "n_points_bc_building", 100)
        n_bldg_per_wall = max(5, n_bldg // 4)

    # Calculate available batches per term
    bc_counts = [
        n_pde // batch_size,
        n_ic // batch_size,
        n_bc_per_wall // batch_size, # left
        n_bc_per_wall // batch_size, # right
        n_bc_per_wall // batch_size, # bottom
        n_bc_per_wall // batch_size, # top
    ]
    if has_building and 'building_bc' in active_loss_term_keys:
        bc_counts.extend([
            n_bldg_per_wall // batch_size, # left
            n_bldg_per_wall // batch_size, # right
            n_bldg_per_wall // batch_size, # bottom
            n_bldg_per_wall // batch_size, # top
        ])

    if not data_free and data_points_full is not None:
         bc_counts.append(data_points_full.shape[0] // batch_size)

    num_batches = max(bc_counts) if bc_counts else 0
    
    if num_batches == 0:
        print(f"Error: Batch size {batch_size} is too large for configured sample counts or data. No training will occur.")
        return -1.0
    print(f"Calculated number of batches per epoch: {num_batches}")


    # --- Define JIT Data Generator ---
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, bldg_keys, data_key = random.split(key, 6)
        
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
            
        # Domain BCs
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

        # Building BCs
        building_bc_data = {}
        if has_building and 'building_bc' in active_loss_term_keys:
             bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)
             b_cfg = cfg["building"]
             
             building_bc_data['left'] = get_wall_data(bldg_l_key, n_bldg_per_wall, (b_cfg["x_min"], b_cfg["x_min"]), (b_cfg["y_min"], b_cfg["y_max"]), (0., domain_cfg["t_final"]))
             building_bc_data['right'] = get_wall_data(bldg_r_key, n_bldg_per_wall, (b_cfg["x_max"], b_cfg["x_max"]), (b_cfg["y_min"], b_cfg["y_max"]), (0., domain_cfg["t_final"]))
             building_bc_data['bottom'] = get_wall_data(bldg_b_key, n_bldg_per_wall, (b_cfg["x_min"], b_cfg["x_max"]), (b_cfg["y_min"], b_cfg["y_min"]), (0., domain_cfg["t_final"]))
             building_bc_data['top'] = get_wall_data(bldg_t_key, n_bldg_per_wall, (b_cfg["x_min"], b_cfg["x_max"]), (b_cfg["y_max"], b_cfg["y_max"]), (0., domain_cfg["t_final"]))

        # Data
        data_data = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)
        if not data_free and data_points_full is not None:
             data_data = get_batches_tensor(data_key, data_points_full, batch_size, num_batches)

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc': bc_data,
            'data': data_data,
            'building_bc': building_bc_data
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
        experiment_name="experiment_2",
        trial_name=trial_name,
        results_dir=results_dir,
        model_dir=model_dir,
        config_path=config_path,
        validation_fn=validation_fn,
        source_script_path=__file__,
    )

    def plot_fn(final_params):
        print("  Generating 2D comparison plots...")
        aim_tracker = loop_result["aim_tracker"]
        final_epoch = loop_result["epoch"]
        plot_cfg = cfg.get("plotting", {})
        eps_plot = cfg.get("numerics", {}).get("eps", 1e-6)
        t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
        plot_data_time = t_const_val_plot
        plot_data_file = os.path.join(base_data_path, f"validation_plotting_t_{int(plot_data_time)}s.npy")
        if not os.path.exists(plot_data_file):
            print(f"  Warning: Plotting data file {plot_data_file} not found. Skipping comparison plot.")
            return

        plot_data = np.load(plot_data_file)
        plot_points_scatter = jnp.array(plot_data[:, [1, 2, 0]], dtype=DTYPE)
        x_coords_plot = jnp.array(plot_data[:, 1], dtype=DTYPE)
        y_coords_plot = jnp.array(plot_data[:, 2], dtype=DTYPE)
        h_true_plot = jnp.array(plot_data[:, 3], dtype=DTYPE)
        u_true_plot = jnp.array(plot_data[:, 4], dtype=DTYPE)
        v_true_plot = jnp.array(plot_data[:, 5], dtype=DTYPE)
        h_true_safe = jnp.maximum(h_true_plot, eps_plot)
        hu_true_plot = h_true_safe * u_true_plot
        hv_true_plot = h_true_safe * v_true_plot
        U_plot_pred_scatter = model.apply({'params': final_params['params']}, plot_points_scatter, train=False)
        h_pred_plot = U_plot_pred_scatter[..., 0]
        hu_pred_plot = U_plot_pred_scatter[..., 1]
        hv_pred_plot = U_plot_pred_scatter[..., 2]

        plot_path_h = os.path.join(results_dir, f"final_comparison_plot_t{int(plot_data_time)}s_h.png")
        plot_path_hu = os.path.join(results_dir, f"final_comparison_plot_t{int(plot_data_time)}s_hu.png")
        plot_path_hv = os.path.join(results_dir, f"final_comparison_plot_t{int(plot_data_time)}s_hv.png")
        plot_comparison_scatter_2d(x_coords_plot, y_coords_plot, h_pred_plot, h_true_plot, 'h', cfg_dict, plot_path_h)
        plot_comparison_scatter_2d(x_coords_plot, y_coords_plot, hu_pred_plot, hu_true_plot, 'hu', cfg_dict, plot_path_hu)
        plot_comparison_scatter_2d(x_coords_plot, y_coords_plot, hv_pred_plot, hv_true_plot, 'hv', cfg_dict, plot_path_hv)
        aim_tracker.log_image(plot_path_h, 'validation_plot_h', final_epoch)
        aim_tracker.log_image(plot_path_hu, 'validation_plot_hu', final_epoch)
        aim_tracker.log_image(plot_path_hv, 'validation_plot_hv', final_epoch)
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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Building Scenario).")
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