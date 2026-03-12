"""Experiment 3 — Terrain slope in x-direction (Phase 2).

Introduces terrain via bi-linear DEM interpolation; establishes data
sampling ratio methodology when physics-only training is insufficient.
Requires: configs/experiment_3.yaml, data/experiment_3/
Builds on: Experiment 2.
"""
import os
import sys
import time
import copy
import argparse
import itertools
from typing import Any, Dict, Tuple
import shutil

from jaxtyping import config

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
import matplotlib.pyplot as plt 

# Local application imports
from src.config import load_config, DTYPE
from src.data import (
    sample_domain, 
    get_batches_tensor,
    get_sample_count,
    bathymetry_fn,
    load_boundary_condition,
    load_bathymetry,
    sample_lhs,
    load_validation_data,
    resolve_scenario_asset_path,
)
from src.models import init_model
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet_h,
    loss_boundary_wall_horizontal,
    loss_boundary_wall_vertical,
    compute_neg_h_loss,
    compute_data_loss,
    total_loss
)
from src.utils import ( 
   nse, rmse, generate_trial_name, save_model, ask_for_confirmation
)

from src.monitoring import ConsoleLogger, AimTracker, compute_negative_depth_diagnostics
from src.metrics.accuracy import compute_validation_metrics
from src.checkpointing import CheckpointManager
from src.training import (
    create_optimizer,
    extract_loss_weights,
    load_training_data,
    load_validation_from_file,
    post_training_save,
    resolve_data_mode,
    run_training_loop,
    setup_experiment,
)


def train_step(
        model: Any, 
        optimiser: optax.GradientTransformation, 
        params: FrozenDict, 
        opt_state: optax.OptState, 
        batch: Dict[str, jnp.ndarray], 
        config: Dict[str, Any],
        data_free: bool,
        bc_fn_static: Any,
        weights_dict: FrozenDict # Type hint updated
        ) -> Tuple[FrozenDict, optax.OptState, Dict[str, float], float]:
    """
    Performs one step of gradient descent.
    """
    
    # weights_dict is now a FrozenDict (hashable), so .keys() works fine
    active_loss_keys_base = list(weights_dict.keys())

    def loss_fn(params):
        
        terms = {}
        # --- 1. PDE Loss (Physics + Bathymetry) ---
        loss_pde = compute_pde_loss(model, params, batch['pde'], config)
        loss_neg_h = compute_neg_h_loss(model, params, batch['pde'])
        
        # --- 2. Initial Condition Loss (t=0, h=9.7) ---
        U_ic = model.apply(params, batch['ic'], train=True)        
        h_ic_pred = U_ic[..., 0]
        hu_ic_pred = U_ic[..., 1]
        hv_ic_pred = U_ic[..., 2]

        # Get bathymetry at IC points
        z_ic, _, _ = bathymetry_fn(batch['ic'][..., 0], batch['ic'][..., 1])
        
        # Calculate target depth based on absolute water level 9.7m
        # h_target = max(0, 9.7 - z)
        h_target_ic = jnp.maximum(0.0, 9.7 - z_ic)

        loss_ic_h = jnp.mean((h_ic_pred - h_target_ic)**2)
        # Enforce zero velocity at t=0
        loss_ic_vel = jnp.mean(hu_ic_pred**2 + hv_ic_pred**2) 
        loss_ic = loss_ic_h + loss_ic_vel

        # --- 3. Boundary Conditions ---
        
        # A. Left Boundary (x=0): Time-Varying Water Level
        t_left = batch['bc_left'][..., 2]
        bc_level_abs = bc_fn_static(t_left) # Interpolate target Absolute Level
        
        # Get Z at boundary to calculate depth h = Level - Z
        z_left, _, _ = bathymetry_fn(batch['bc_left'][..., 0], batch['bc_left'][..., 1])
        h_target_left = jnp.maximum(0.0, bc_level_abs - z_left)
        
        loss_bc_left = loss_boundary_dirichlet_h(model, params, batch['bc_left'], h_target_left)
        
        # B. Right Boundary (x=700): Slip Walls (No flux x)
        loss_bc_right = loss_boundary_wall_vertical(model, params, batch['bc_right'])
        
        # C. Top & Bottom Boundaries (y=0, y=100): Slip Walls (No flux y)
        loss_bc_top = loss_boundary_wall_horizontal(model, params, batch['bc_top'])
        loss_bc_bottom = loss_boundary_wall_horizontal(model, params, batch['bc_bottom'])
        
        total_bc = loss_bc_left + loss_bc_right + loss_bc_top + loss_bc_bottom

        data_batch_data = batch.get('data', jnp.empty((0,6), dtype=DTYPE))
        if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0:
             loss_data = compute_data_loss(model, params, data_batch_data, config)

        terms = {
            'pde': loss_pde,
            'neg_h': loss_neg_h,
            'ic': loss_ic,
            'bc': total_bc, # Renamed to 'bc' to match weights keys if needed, or 'total_bc'
            'data': loss_data if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0 else 0.0
        }

        # --- 4. Weighted Sum ---
        # Helper to safely get term or 0.0
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        
        return total, terms

    # Calculate Gradients
    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Update Parameters
    updates, new_opt_state = optimiser.update(grads, opt_state, params, value=loss_val)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, metrics, loss_val

# JIT Compile
train_step_jitted = jax.jit(train_step, static_argnames=['model', 'optimiser', 'config', 'bc_fn_static', 'weights_dict', 'data_free'])

def main(config_path: str):
    """
    Main training loop for Experiment 3 scenario.
    """
    
    #--- 1. LOAD CONFIGURATION ---
    setup = setup_experiment(config_path, "experiment_3")
    cfg_dict = setup["cfg_dict"]
    cfg = setup["cfg"]
    model = setup["model"]
    params = setup["params"]
    train_key = setup["train_key"]
    val_key = setup["val_key"]
    trial_name = setup["trial_name"]
    results_dir = setup["results_dir"]
    model_dir = setup["model_dir"]

    print("Info: Running Experiment 3 Scenario model training...")

    # --- 4. Prepare Loss Weights (Moved Up) ---
    static_weights_dict, current_weights_dict = extract_loss_weights(cfg)

    # --- 5. Load Data Assets ---
    scenario_name = cfg.get('scenario')
    if not scenario_name:
         print(f"Error: 'scenario' key must be set in config '{config_path}'.")
         sys.exit(1)
         
    base_data_path = os.path.join("data", scenario_name)
    
    # A. Load Bathymetry (REQUIRED)
    try:
        dem_path = resolve_scenario_asset_path(base_data_path, scenario_name, "dem")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)

    # B. Load Boundary Condition Function
    try:
        bc_csv_path = resolve_scenario_asset_path(base_data_path, scenario_name, "boundary_condition")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    bc_fn_static = load_boundary_condition(bc_csv_path)

    # --- 5b. Load Validation and Training Data ---
    data_points_full = None
    data_free, has_data_loss = resolve_data_mode(cfg)
    data_points_full, has_data_loss, data_free = load_training_data(
        base_data_path,
        has_data_loss,
        static_weights_dict,
    )

    # C. Load Validation Data (Optional)
    validation = load_validation_from_file(base_data_path, "validation_gauges.npy")
    validation_data_loaded = validation["loaded"]
    full_val_data = validation["full_val_data"]
    val_points_all = validation["val_points"]
    h_true_val_all = validation["h_true_val"]

    # --- 6. Initialize Aim & Log Source Code ---
    aim_enabled = cfg_dict.get('aim', {}).get('enable', True)
    aim_tracker = AimTracker(cfg_dict, trial_name, enable=aim_enabled)
    aim_tracker.log_flags({"scenario_type": "experiment_3"})
    if aim_enabled:
        try:
            aim_tracker.log_artifact(config_path, 'run_config.yaml')
            aim_tracker.log_artifact(os.path.abspath(__file__), 'source_script.py')
        except Exception:
            pass

    # --- 7. Data Generation Setup ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]
    
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000)
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100)
    n_bc_domain = get_sample_count(sampling_cfg, "n_points_bc_domain", 100)
    n_bc_per_wall = max(5, n_bc_domain // 4)

    # Check batch size viability
    bc_counts = [n_pde//batch_size, n_ic//batch_size, n_bc_per_wall//batch_size]

    if not data_free and data_points_full is not None:
        bc_counts.append(data_points_full.shape[0] // batch_size)

    num_batches = max(bc_counts) if bc_counts else 0
    
    if num_batches == 0:
        print(f"Error: Batch size {batch_size} is too large for sample counts.")
        return -1.0
    print(f"Batches per epoch: {num_batches}")

        # --- 3. Setup Optimizer ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # JIT Data Generator
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, data_key = random.split(key, 5)
        
        # PDE
        if n_pde // batch_size > 0:
            pde_pts = sample_lhs(pde_key, n_pde, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
            pde_data = get_batches_tensor(pde_key, pde_pts, batch_size, num_batches)
        else:
            pde_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        # IC
        if n_ic // batch_size > 0:
            ic_pts = sample_lhs(ic_key, n_ic, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))
            ic_data = get_batches_tensor(ic_key, ic_pts, batch_size, num_batches)
        else:
            ic_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)
            
        # BCs
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        def get_wall(k, n, x_b, y_b):
            if n // batch_size > 0:
                pts = sample_lhs(k, n, x_b, y_b, (0., domain_cfg["t_final"]))
                return get_batches_tensor(k, pts, batch_size, num_batches)
            return jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        bc_left = get_wall(l_key, n_bc_per_wall, (0., 0.), (0., domain_cfg["ly"]))
        bc_right = get_wall(r_key, n_bc_per_wall, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]))
        bc_bot = get_wall(b_key, n_bc_per_wall, (0., domain_cfg["lx"]), (0., 0.))
        bc_top = get_wall(t_key, n_bc_per_wall, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]))

        # Data
        data_data = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)
        if not data_free and data_points_full is not None:
             data_data = get_batches_tensor(data_key, data_points_full, batch_size, num_batches)

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc': {'left': bc_left, 'right': bc_right, 'bottom': bc_bot, 'top': bc_top}
            ,'data': data_data
        }
    
    generate_epoch_data_jitted = jax.jit(generate_epoch_data)

    # Scan Body
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        
        current_all_batches = {
            'pde': batch_data['pde'],
            'ic': batch_data['ic'],
            'bc_left': batch_data['bc']['left'],
            'bc_right': batch_data['bc']['right'],
            'bc_bottom': batch_data['bc']['bottom'],
            'bc_top': batch_data['bc']['top'],
            'data': batch_data['data'] # May be None
        }

        # current_weights_dict is now FrozenDict, so it's hashable
        new_params, new_opt_state, terms, total = train_step_jitted(
            model, optimiser, curr_params, curr_opt_state,
            current_all_batches, cfg, data_free, bc_fn_static, current_weights_dict
        )
        return (new_params, new_opt_state), (terms, total)

    loop_result = run_training_loop(
        cfg=cfg,
        cfg_dict=cfg_dict,
        model=model,
        params=params,
        opt_state=opt_state,
        train_key=train_key,
        optimiser=optimiser,
        generate_epoch_data_jit=generate_epoch_data_jitted,
        scan_body=scan_body,
        num_batches=num_batches,
        experiment_name="experiment_3",
        trial_name=trial_name,
        results_dir=results_dir,
        model_dir=model_dir,
        config_path=config_path,
        validation_data_loaded=validation_data_loaded,
        val_points_all=val_points_all,
        h_true_val_all=h_true_val_all,
        source_script_path=__file__,
    )

    def plot_fn(final_params):
        print("Generating Experiment 3 plots...")
        t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=DTYPE)
        aim_tracker = loop_result["aim_tracker"]
        final_epoch = loop_result["epoch"]

        def plot_gauge(x, y, name, color, filename):
            pts = jnp.stack([jnp.full_like(t_plot, x), jnp.full_like(t_plot, y), t_plot], axis=-1)
            U = model.apply(final_params, pts, train=False)
            h_pred = U[..., 0]

            plt.figure(figsize=(10, 6))
            if full_val_data is not None:
                val_np = np.array(full_val_data)
                mask = np.isclose(val_np[:, 1], x) & np.isclose(val_np[:, 2], y)
                gauge_data = val_np[mask]
                if gauge_data.shape[0] > 0:
                    gauge_data = gauge_data[gauge_data[:, 0].argsort()]
                    plt.plot(gauge_data[:, 0], gauge_data[:, 3], 'k--', linewidth=1.5, alpha=0.7, label=f'Baseline {name}')

            plt.plot(t_plot, h_pred, label=f'Predicted h @ ({x},{y})', color=color)
            plt.xlabel('Time (s)')
            plt.ylabel('Water Level h (m)')
            plt.title(f'{name} - Water Level vs Time')
            plt.legend()
            plt.grid(True)
            path = os.path.join(results_dir, filename)
            plt.savefig(path)
            plt.close()
            aim_tracker.log_image(path, filename, final_epoch)

        plot_gauge(3.9587225e+02, 4.9646515e+01, "Point 1", "blue", "P1_timeseries.png")
        plot_gauge(6.0435474e+02, 5.0565735e+01, "Point 2", "red", "P2_timeseries.png")
        print(f"Plots saved to {results_dir}")

    post_training_save(
        loop_result=loop_result,
        model=model,
        model_dir=model_dir,
        results_dir=results_dir,
        trial_name=trial_name,
        plot_fn=plot_fn,
    )

    return loop_result["best_nse_stats"]["nse"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Experiment 3).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    main(args.config)
