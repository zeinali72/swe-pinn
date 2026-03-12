"""Experiment 8 — Real urban domain, Eastbourne UK (Phase 3).

Applies the framework to a real urban subcatchment (Blue Heart Project).
Buildings excluded from mesh by construction, treated as wall boundaries.
Requires: configs/experiment_8.yaml, data/experiment_8/
Builds on: Experiment 7.
"""
import os
import sys
import time
import copy
import argparse
import itertools
from typing import Any, Dict, Tuple
import shutil
import pandas as pd 

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
import matplotlib
matplotlib.use('Agg') # Ensure headless plotting
import matplotlib.pyplot as plt 

# Local application imports
from src.config import load_config, DTYPE
from src.data import (
    get_batches_tensor,
    get_sample_count,
    load_boundary_condition,
    IrregularDomainSampler,
    load_bathymetry,
    load_validation_data,
    resolve_scenario_asset_path,
)
from src.models import init_model
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet_hu,
    loss_boundary_dirichlet_hv,
    loss_slip_wall_generalized,
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
    create_output_dirs,
    extract_loss_weights,
    init_model_from_config,
    load_training_data,
    post_training_save,
    resolve_data_mode,
    run_training_loop,
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
    
    active_loss_keys_base = list(weights_dict.keys())

    def loss_fn(params):
        
        # --- 1. PDE Loss ---
        loss_pde = compute_pde_loss(model, params, batch['pde'], config)
        loss_neg_h = compute_neg_h_loss(model, params, batch['pde'])
        
        # --- 2. Initial Condition Loss ---
        U_ic = model.apply(params, batch['ic'], train=True)        
        loss_ic = jnp.mean(U_ic[..., 0]**2) + jnp.mean(U_ic[..., 1]**2 + U_ic[..., 2]**2)

        # --- 3. Boundary Conditions ---
        
        # A. Upstream Boundary (Flux prescribed)
        # CHANGED: Using 'bc_upstream' to match preprocess output
        if batch['bc_upstream'].shape[0] > 0:
            t_inflow = batch['bc_upstream'][..., 2]
            Q_target = bc_fn_static(t_inflow) # m^3/s
            
            # Assuming 100m width for Experiment 8 upstream boundary
            flux_target_x = Q_target / 372.92  # m^2/s 
            
            loss_inflow_x = loss_boundary_dirichlet_hu(model, params, batch['bc_upstream'], flux_target_x)
            loss_inflow_y = loss_boundary_dirichlet_hv(model, params, batch['bc_upstream'], jnp.zeros_like(flux_target_x))
            loss_bc_inflow = loss_inflow_x + loss_inflow_y
        else:
            loss_bc_inflow = 0.0

        # B. Wall Boundaries (Generalized Slip) - Outer Domain
        loss_bc_wall = loss_slip_wall_generalized(model, params, batch['bc_wall'])
        
        # C. Building Boundaries (Generalized Slip) - Obstacles
        loss_bldg = loss_slip_wall_generalized(model, params, batch['bc_building'])
        
        total_bc = loss_bc_inflow + loss_bc_wall + loss_bldg

        # Data Loss
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
            'data': loss_data,
            # 'building' is now part of 'bc' sum, but can be tracked separately if needed in reporting
            'building': loss_bldg 
        }

        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params, value=loss_val)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics, loss_val

# JIT Compile
train_step_jitted = jax.jit(train_step, static_argnames=['model', 'optimiser', 'config', 'bc_fn_static', 'weights_dict', 'data_free'])

def main(config_path: str):
    """
    Main training loop for Experiment 8 Scenario.
    """
    #--- 1. LOAD CONFIGURATION (MUTABLE) ---
    cfg_dict = load_config(config_path)
    
    print("Info: Running Experiment 8 training...")

    # --- 2. SETUP DATA & COMPUTE DOMAIN EXTENT ---
    scenario_name = cfg_dict.get('scenario', 'experiment_8')
    base_data_path = os.path.join("data", scenario_name)

    # A. Init Irregular Domain Sampler
    try:
        artifacts_path = resolve_scenario_asset_path(base_data_path, scenario_name, "domain_artifacts")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    
    print(f"Loading domain geometry from: {artifacts_path}")
    domain_sampler = IrregularDomainSampler(artifacts_path)

    # --- CALCULATE DOMAIN EXTENT ---
    # tri_coords shape: (N_tri, 3, 2)
    all_coords = domain_sampler.tri_coords.reshape(-1, 2)
    min_vals = jnp.min(all_coords, axis=0)
    max_vals = jnp.max(all_coords, axis=0)
    
    x_min, y_min = float(min_vals[0]), float(min_vals[1])
    x_max, y_max = float(max_vals[0]), float(max_vals[1])
    
    calc_lx = x_max - x_min
    calc_ly = y_max - y_min
    
    print(f"Computed Domain Extent:")
    print(f"  X Range: [{x_min:.4f}, {x_max:.4f}]")
    print(f"  Y Range: [{y_min:.4f}, {y_max:.4f}]")
    print(f"  Calculated Dimensions: lx = {calc_lx:.4f}, ly = {calc_ly:.4f}")
    
    # Update Config with calculated values
    if 'domain' not in cfg_dict: cfg_dict['domain'] = {}
    cfg_dict['domain']['lx'] = calc_lx
    cfg_dict['domain']['ly'] = calc_ly
    cfg_dict['domain']['x_min'] = x_min
    cfg_dict['domain']['x_max'] = x_max
    cfg_dict['domain']['y_min'] = y_min
    cfg_dict['domain']['y_max'] = y_max

    # Standard scaling for this experiment
    h_scale = 1.0  
    hu_scale = 1.0 
    hv_scale = 1.0

    if 'model' not in cfg_dict: cfg_dict['model'] = {}
    cfg_dict['model']['output_scales'] = (h_scale, hu_scale, hv_scale)
    
    print(f"Active Output Scaling: {cfg_dict['model']['output_scales']}")

    # --- 3. FINALIZE CONFIG & INIT MODEL ---
    cfg = FrozenDict(cfg_dict)
    model, params, train_key, val_key = init_model_from_config(cfg)
    trial_name, results_dir, model_dir = create_output_dirs(cfg, "experiment_8")

    # --- 5. Prepare Loss Weights ---
    static_weights_dict, current_weights_dict = extract_loss_weights(cfg)

    # --- 6. Load Remaining Assets ---

    # B. Load Bathymetry (REQUIRED)
    try:
        dem_path = resolve_scenario_asset_path(base_data_path, scenario_name, "dem")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)
    
    # C. Load Boundary Condition Function
    try:
        bc_csv_path = resolve_scenario_asset_path(base_data_path, scenario_name, "boundary_condition")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    bc_fn_static = load_boundary_condition(bc_csv_path)

    # D. Load Validation and Training Data
    val_points, h_true_val = None, None
    data_points_full = None
    
    data_free, has_data_loss = resolve_data_mode(cfg)
    data_points_full, has_data_loss, data_free = load_training_data(
        base_data_path,
        has_data_loss,
        static_weights_dict,
    )

    # E. Load Validation Data (Ground Truth)
    validation_data_file = os.path.join(base_data_path, "validation_gauges_ground_truth.npy")
    validation_data_loaded = False
    
    # Validation arrays
    val_pts_batch = None
    val_h_true = None
    val_hu_true = None
    val_hv_true = None
    
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            _, val_pts_batch, val_targets = load_validation_data(validation_data_file, dtype=DTYPE)
            val_h_true = val_targets[:, 0]
            u_temp = val_targets[:, 1]
            v_temp = val_targets[:, 2]
            
            val_hu_true = val_h_true * u_temp
            val_hv_true = val_h_true * v_temp
            
            if val_pts_batch.shape[0] > 0:
                validation_data_loaded = True
                print(f"Validation loaded: {val_pts_batch.shape[0]} points. Targets: h, hu, hv prepared.")
            else:
                 print("Warning: Validation data file is empty.")
        except Exception as e:
            print(f"Error loading validation data: {e}")
            val_pts_batch = None
    else:
        print(f"Warning: Validation data not found at {validation_data_file}.")

    # --- 7. Data Generation Setup ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]
    
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000)
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100)
    # Using 'n_points_bc_inflow' for upstream sampling
    n_bc_upstream = get_sample_count(sampling_cfg, "n_points_bc_inflow", 100) 
    n_bc_wall = get_sample_count(sampling_cfg, "n_points_bc_domain", 100)
    n_building = get_sample_count(sampling_cfg, "n_points_bc_building", 100)

    bc_counts = [n_pde//batch_size, n_ic//batch_size, n_bc_wall//batch_size, n_bc_upstream//batch_size, n_building//batch_size]
    if not data_free and data_points_full is not None:
        bc_counts.append(data_points_full.shape[0] // batch_size)

    num_batches = max(bc_counts) if bc_counts else 0
    if num_batches == 0:
        print(f"Error: Batch size {batch_size} is too large for sample counts.")
        return -1.0
    print(f"Batches per epoch: {num_batches}")

    # --- Optimizer ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    def generate_epoch_data(key):
        k1, k2, k3, k4, k5 = random.split(key, 5)
        
        # Interior
        pde_pts = domain_sampler.sample_interior(k1, n_pde, (0., domain_cfg["t_final"]))
        pde_data = get_batches_tensor(k1, pde_pts, batch_size, num_batches)
        
        # IC
        ic_pts = domain_sampler.sample_interior(k2, n_ic, (0., 0.))
        ic_data = get_batches_tensor(k2, ic_pts, batch_size, num_batches)
        
        # BCs
        # CHANGED: Sampling 'upstream' boundary instead of generic 'inflow'
        bc_upstream_pts = domain_sampler.sample_boundary(k3, n_bc_upstream, (0., domain_cfg["t_final"]), 'upstream')
        bc_upstream = get_batches_tensor(k3, bc_upstream_pts, batch_size, num_batches)

        bc_wall_pts = domain_sampler.sample_boundary(k4, n_bc_wall, (0., domain_cfg["t_final"]), 'wall')
        bc_wall = get_batches_tensor(k4, bc_wall_pts, batch_size, num_batches)
        
        bc_building_pts = domain_sampler.sample_boundary(k5, n_building, (0., domain_cfg["t_final"]), 'building')
        bc_building = get_batches_tensor(k5, bc_building_pts, batch_size, num_batches)
        
        # Data
        if not data_free and data_points_full is not None:
             data_d = get_batches_tensor(k5, data_points_full, batch_size, num_batches)
        else:
             data_d = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)

        return {
            'pde': pde_data, 'ic': ic_data, 
            'bc_upstream': bc_upstream,
            'bc_wall': bc_wall, 
            'bc_building': bc_building,
            'data': data_d
        }
    
    generate_epoch_data_jitted = jax.jit(generate_epoch_data)

    # Scan Body
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        current_all_batches = {
            'pde': batch_data['pde'],
            'ic': batch_data['ic'],
            'bc_upstream': batch_data['bc_upstream'],
            'bc_wall': batch_data['bc_wall'], 
            'bc_building': batch_data['bc_building'],
            'data': batch_data['data']
        }
        new_params, new_opt_state, terms, total = train_step_jitted(
            model, optimiser, curr_params, curr_opt_state,
            current_all_batches, cfg, data_free, bc_fn_static, current_weights_dict
        )
        return (new_params, new_opt_state), (terms, total)

    def validation_fn(model, params):
        combined_nse_val = -float('inf')
        nse_h_val, nse_hu_val, nse_hv_val = -float('inf'), -float('inf'), -float('inf')
        rmse_val = float('inf')
        if validation_data_loaded:
            try:
                U_pred = model.apply(params, val_pts_batch, train=False)
                h_pred = U_pred[..., 0]
                hu_pred = U_pred[..., 1]
                hv_pred = U_pred[..., 2]
                nse_h_val = float(nse(h_pred, val_h_true))
                nse_hu_val = float(nse(hu_pred, val_hu_true))
                nse_hv_val = float(nse(hv_pred, val_hv_true))
                combined_nse_val = (nse_h_val + nse_hu_val + nse_hv_val) / 3.0
                rmse_val = float(rmse(h_pred, val_h_true))
            except Exception as exc:
                print(f"Validation Error: {exc}")

        return {
            'selection_metric': float(combined_nse_val),
            'nse_h': float(nse_h_val),
            'nse_hu': float(nse_hu_val),
            'nse_hv': float(nse_hv_val),
            'combined_nse': float(combined_nse_val),
            'rmse_h': float(rmse_val),
        }

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
        experiment_name="experiment_8",
        trial_name=trial_name,
        results_dir=results_dir,
        model_dir=model_dir,
        config_path=config_path,
        pde_key_for_diag="pde",
        validation_fn=validation_fn,
        selection_metric_key="selection_metric",
        source_script_path=__file__,
    )

    def plot_fn(final_params):
        print("Generating Experiment 8 plots...")
        t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=DTYPE)
        aim_tracker = loop_result["aim_tracker"]
        final_epoch = loop_result["epoch"]
        output_csv_path = resolve_scenario_asset_path(
            base_data_path, scenario_name, "output_reference", required=False
        )
        output_points = []
        if os.path.exists(output_csv_path):
            try:
                df_out = pd.read_csv(output_csv_path)
                df_out.columns = [c.strip() for c in df_out.columns]
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
            cx, cy = (x_max + x_min) / 2, (y_max + y_min) / 2
            output_points = [(cx, cy, "Center_Point")]

        def plot_gauge(x, y, name, filename):
            pts = jnp.stack([jnp.full_like(t_plot, x), jnp.full_like(t_plot, y), t_plot], axis=-1)
            U = model.apply(final_params, pts, train=False)
            h_pred = U[..., 0]
            plt.figure(figsize=(10, 6))
            plt.plot(np.array(t_plot), np.array(h_pred), label=f'Predicted h @ ({x:.1f},{y:.1f})')
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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Experiment 8 - Irregular).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    main(args.config)
