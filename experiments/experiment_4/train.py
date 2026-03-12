"""Experiment 4 — Terrain slope in x+y directions (Phase 2).

Extends terrain slope to both x and y directions.
Requires: configs/experiment_4.yaml, data/experiment_4/
Builds on: Experiment 3.
"""
import os
import sys
import time
import copy
import argparse
import itertools
from typing import Any, Dict, Tuple
import shutil
import pandas as pd # Added for reading output CSV

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
    loss_boundary_dirichlet_hu,
    loss_boundary_dirichlet_hv,
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
    calculate_num_batches,
    extract_loss_weights,
    get_data_filename,
    get_sampling_count_from_config,
    get_boundary_segment_count,
    load_training_data,
    load_validation_from_file,
    post_training_save,
    resolve_configured_asset_path,
    resolve_experiment_paths,
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
        weights_dict: FrozenDict 
        ) -> Tuple[FrozenDict, optax.OptState, Dict[str, float], float]:
    """
    Performs one step of gradient descent for Experiment 4.
    """
    
    active_loss_keys_base = list(weights_dict.keys())

    def loss_fn(params):
        
        terms = {}
        # --- 1. PDE Loss (Physics + Bathymetry) ---
        loss_pde = compute_pde_loss(model, params, batch['pde'], config)
        loss_neg_h = compute_neg_h_loss(model, params, batch['pde'])
        
        # --- 2. Initial Condition Loss (Dry Bed: h=0, u=0, v=0) ---
        # Test 2 Spec: "Initial condition: Dry bed."
        U_ic = model.apply(params, batch['ic'], train=True)        
        h_ic_pred = U_ic[..., 0]
        hu_ic_pred = U_ic[..., 1]
        hv_ic_pred = U_ic[..., 2]

        # Target is strictly zero
        loss_ic_h = jnp.mean(h_ic_pred**2)
        loss_ic_vel = jnp.mean(hu_ic_pred**2 + hv_ic_pred**2) 
        loss_ic = loss_ic_h + loss_ic_vel

        # --- 3. Boundary Conditions ---
        
        # A. Inflow Boundary (Part of Left Edge): Flux BC
        # Test 2 Spec: Inflow hydrograph along 100m line from NW corner.
        # We model this as imposing flux q = Q(t) / Width. Width = 100m.
        
        # Get Time and interpolated Flow Q(t)
        t_inflow = batch['bc_inflow'][..., 2]
        Q_target_x = bc_fn_static(t_inflow) # Returns m^3/s
        inflow_width = config["boundary_conditions"]["inflow_discharge_width"]
        flux_target_x = Q_target_x / inflow_width

        loss_inflow_x=loss_boundary_dirichlet_hu(model, params, batch['bc_inflow'], flux_target_x)
        loss_inflow_y=loss_boundary_dirichlet_hv(model, params, batch['bc_inflow'], jnp.zeros_like(flux_target_x))
        
        loss_bc_inflow = loss_inflow_x + loss_inflow_y

        # B. Left Wall (Remaining part of Left Edge): Slip Wall
        loss_bc_left_wall = loss_boundary_wall_vertical(model, params, batch['bc_left_wall'])
        
        # C. Right Boundary: Slip Walls (No flux x)
        loss_bc_right = loss_boundary_wall_vertical(model, params, batch['bc_right'])
        
        # D. Top & Bottom Boundaries: Slip Walls (No flux y)
        loss_bc_top = loss_boundary_wall_horizontal(model, params, batch['bc_top'])
        loss_bc_bottom = loss_boundary_wall_horizontal(model, params, batch['bc_bottom'])
        
        # Aggregate BC losses
        total_bc = loss_bc_inflow + loss_bc_left_wall + loss_bc_right + loss_bc_top + loss_bc_bottom

        data_batch_data = batch.get('data', jnp.empty((0,6), dtype=DTYPE))
        if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0:
             loss_data = compute_data_loss(model, params, data_batch_data, config)
        else:
             loss_data = 0.0

        terms = {
            'pde': loss_pde,
            'neg_h': loss_neg_h,
            'ic': loss_ic,
            'bc': total_bc, 
            'data': loss_data if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0 else 0.0
        }

        # --- 4. Weighted Sum ---
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
    Main training loop for Experiment 4 Scenario.
    """
    
    #--- 1. LOAD CONFIGURATION ---
    setup = setup_experiment(config_path)
    cfg_dict = setup["cfg_dict"]
    cfg = setup["cfg"]
    experiment_name = setup["experiment_name"]
    model = setup["model"]
    params = setup["params"]
    train_key = setup["train_key"]
    trial_name = setup["trial_name"]
    results_dir = setup["results_dir"]
    model_dir = setup["model_dir"]

    print("Info: Running Experiment 4 Scenario model training...")

    # --- 4. Prepare Loss Weights ---
    static_weights_dict, current_weights_dict = extract_loss_weights(cfg)

    # --- 5. Load Data Assets ---
    try:
        experiment_paths = resolve_experiment_paths(cfg, experiment_name, require_scenario=True)
    except ValueError as exc:
        print(f"Error: {exc} in config '{config_path}'.")
        sys.exit(1)

    scenario_name = experiment_paths["scenario_name"]
    base_data_path = experiment_paths["base_data_path"]
    
    # A. Load Bathymetry (Test 2 DEM)
    try:
        dem_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "dem")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)

    # B. Load Boundary Condition Function
    try:
        bc_csv_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "boundary_condition")
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
        filename=get_data_filename(cfg, "training_file", "training_dataset_sample.npy"),
    )

    # C. Load Validation Data (Optional)
    validation = load_validation_from_file(
        base_data_path,
        get_data_filename(cfg, "validation_file", "validation_gauges_ground_truth.npy"),
    )
    validation_data_loaded = validation["loaded"]
    full_val_data = validation["full_val_data"]
    val_points_all = validation["val_points"]
    h_true_val_all = validation["h_true_val"]

    # --- 6. Data Generation Setup ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]
    
    n_pde = get_sampling_count_from_config(cfg, "n_points_pde")
    n_ic = get_sampling_count_from_config(cfg, "n_points_ic")
    n_bc_domain_wall = get_sampling_count_from_config(cfg, "n_points_bc_domain")
    n_bc_inflow = get_sampling_count_from_config(cfg, "n_points_bc_inflow")
    n_bc_per_wall = get_boundary_segment_count(cfg, n_bc_domain_wall)

    num_batches = calculate_num_batches(
        batch_size,
        [n_pde, n_ic, n_bc_per_wall, n_bc_inflow],
        data_points_full,
        data_free=data_free,
    )
    if num_batches == 0:
        print(f"Error: Batch size {batch_size} is too large for sample counts.")
        return -1.0
    print(f"Batches per epoch: {num_batches}")

    # --- Optimizer ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # JIT Data Generator
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, data_key = random.split(key, 5)
        
        # PDE: Domain is 2000x2000, t_final from config (usually 48h)
        if n_pde // batch_size > 0:
            pde_pts = sample_lhs(pde_key, n_pde, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
            pde_data = get_batches_tensor(pde_key, pde_pts, batch_size, num_batches)
        else:
            pde_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        # IC: Dry Bed everywhere
        if n_ic // batch_size > 0:
            ic_pts = sample_lhs(ic_key, n_ic, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))
            ic_data = get_batches_tensor(ic_key, ic_pts, batch_size, num_batches)
        else:
            ic_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)
            
        # BCs
        # Test 2 Logic: 
        # Left Boundary (x=0) is split:
        #   - Inflow: y in [1900, 2000] (North-West 100m)
        #   - Wall: y in [0, 1900]
        # Right, Top, Bottom are simple walls.
        
        l_in_key, l_wall_key, r_key, b_key, t_key = random.split(bc_keys, 5)
        
        def get_bc_batch(k, n, x_range, y_range):
            if n // batch_size > 0:
                pts = sample_lhs(k, n, x_range, y_range, (0., domain_cfg["t_final"]))
                return get_batches_tensor(k, pts, batch_size, num_batches)
            return jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        # Split Left Boundary
        inflow_segment = cfg["boundary_conditions"]["left_inflow_segment"]
        y_inflow_start = inflow_segment["y_start"]
        y_inflow_end = inflow_segment["y_end"]
        bc_inflow = get_bc_batch(l_in_key, n_bc_inflow, (0., 0.), (y_inflow_start, y_inflow_end))
        bc_left_wall = get_bc_batch(l_wall_key, n_bc_per_wall, (0., 0.), (0., y_inflow_start))

        # Other Boundaries
        bc_right = get_bc_batch(r_key, n_bc_per_wall, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]))
        bc_bot = get_bc_batch(b_key, n_bc_per_wall, (0., domain_cfg["lx"]), (0., 0.))
        bc_top = get_bc_batch(t_key, n_bc_per_wall, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]))

        # Data
        data_data = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)
        if not data_free and data_points_full is not None:
             data_data = get_batches_tensor(data_key, data_points_full, batch_size, num_batches)

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc': {
                'inflow': bc_inflow, 
                'left_wall': bc_left_wall, 
                'right': bc_right, 
                'bottom': bc_bot, 
                'top': bc_top
            },
            'data': data_data
        }
    
    generate_epoch_data_jitted = jax.jit(generate_epoch_data)

    # Scan Body
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        
        current_all_batches = {
            'pde': batch_data['pde'],
            'ic': batch_data['ic'],
            'bc_inflow': batch_data['bc']['inflow'],
            'bc_left_wall': batch_data['bc']['left_wall'],
            'bc_right': batch_data['bc']['right'],
            'bc_bottom': batch_data['bc']['bottom'],
            'bc_top': batch_data['bc']['top'],
            'data': batch_data['data']
        }

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
        experiment_name=experiment_name,
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
        print("Generating Experiment 4 plots...")
        t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=DTYPE)
        aim_tracker = loop_result["aim_tracker"]
        final_epoch = loop_result["epoch"]
        output_csv_path = resolve_configured_asset_path(
            cfg, base_data_path, scenario_name, "output_reference", required=False
        )
        output_points = []
        if os.path.exists(output_csv_path):
            try:
                df_out = pd.read_csv(output_csv_path)
                if 'X' in df_out.columns and 'Y' in df_out.columns:
                    for idx, row in df_out.iterrows():
                        output_points.append((row['X'], row['Y'], f"Point_{idx+1}"))
                else:
                    arr_out = df_out.values
                    for i in range(arr_out.shape[0]):
                        output_points.append((arr_out[i, 0], arr_out[i, 1], f"Point_{i+1}"))
                print(f"Loaded {len(output_points)} output points from CSV.")
            except Exception as e:
                print(f"Warning: Could not read output reference CSV: {e}")

        if not output_points:
            print("Using default representative points (Depression centers).")
            output_points = [
                (250.0, 250.0, "Depression_1_1"),
                (1250.0, 1250.0, "Depression_3_3"),
                (1750.0, 1750.0, "Depression_4_4"),
            ]

        def plot_gauge(x, y, name, filename):
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

            plt.plot(t_plot, h_pred, label=f'Predicted h @ ({x},{y})')
            plt.xlabel('Time (s)')
            plt.ylabel('Water Level h (m)')
            plt.title(f'{name} - Water Level vs Time')
            plt.legend()
            plt.grid(True)
            path = os.path.join(results_dir, filename)
            plt.savefig(path)
            plt.close()
            aim_tracker.log_image(path, filename, final_epoch)

        for px, py, pname in output_points:
            plot_gauge(px, py, pname, f"{pname}_timeseries.png")

        print(f"Plots saved to {results_dir}")

    post_training_save(
        loop_result=loop_result,
        model=model,
        model_dir=model_dir,
        results_dir=results_dir,
        trial_name=trial_name,
        prefer_loss_model=True,
        plot_fn=plot_fn,
    )

    return loop_result["best_nse_stats"]["nse"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Experiment 4).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    main(args.config)
