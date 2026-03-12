"""Experiment 3 — Terrain slope in x-direction (Phase 2).

Introduces terrain via bi-linear DEM interpolation; establishes data
sampling ratio methodology when physics-only training is insufficient.
Requires: configs/experiment_3.yaml, data/experiment_3/
Builds on: Experiment 2.
"""
import os
import sys
import argparse
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.core import FrozenDict
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from src.config import DTYPE
from src.data import (
    get_batches_tensor,
    bathymetry_fn,
    load_boundary_condition,
    load_bathymetry,
    sample_lhs,
)
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet_h,
    loss_boundary_wall_horizontal,
    loss_boundary_wall_vertical,
    compute_neg_h_loss,
    compute_data_loss,
    total_loss
)
from src.utils import nse, rmse
from src.training import (
    create_optimizer,
    calculate_num_batches,
    extract_loss_weights,
    get_data_filename,
    get_sampling_count_from_config,
    get_boundary_segment_count,
    load_training_data,
    load_validation_from_file,
    make_scan_body,
    sample_and_batch,
    maybe_batch_data,
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
        initial_water_level = config["initial_condition"]["absolute_water_level"]
        h_target_ic = jnp.maximum(0.0, initial_water_level - z_ic)

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
    setup = setup_experiment(config_path)
    cfg_dict = setup["cfg_dict"]
    cfg = setup["cfg"]
    experiment_name = setup["experiment_name"]
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
    try:
        experiment_paths = resolve_experiment_paths(cfg, experiment_name, require_scenario=True)
    except ValueError as exc:
        print(f"Error: {exc} in config '{config_path}'.")
        sys.exit(1)

    scenario_name = experiment_paths["scenario_name"]
    base_data_path = experiment_paths["base_data_path"]
    
    # A. Load Bathymetry (REQUIRED)
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
        get_data_filename(cfg, "validation_file", "validation_gauges.npy"),
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
    n_bc_domain = get_sampling_count_from_config(cfg, "n_points_bc_domain")
    n_bc_per_wall = get_boundary_segment_count(cfg, n_bc_domain)

    # Check batch size viability
    num_batches = calculate_num_batches(
        batch_size,
        [n_pde, n_ic, n_bc_per_wall],
        data_points_full,
        data_free=data_free,
    )
    
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
        x_range = (0., domain_cfg["lx"])
        y_range = (0., domain_cfg["ly"])
        t_range = (0., domain_cfg["t_final"])

        pde_data = sample_and_batch(pde_key, sample_lhs, n_pde, batch_size, num_batches, x_range, y_range, t_range)
        ic_data = sample_and_batch(ic_key, sample_lhs, n_ic, batch_size, num_batches, x_range, y_range, (0., 0.))

        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        bc_left = sample_and_batch(l_key, sample_lhs, n_bc_per_wall, batch_size, num_batches, (0., 0.), y_range, t_range)
        bc_right = sample_and_batch(r_key, sample_lhs, n_bc_per_wall, batch_size, num_batches, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range)
        bc_bot = sample_and_batch(b_key, sample_lhs, n_bc_per_wall, batch_size, num_batches, x_range, (0., 0.), t_range)
        bc_top = sample_and_batch(t_key, sample_lhs, n_bc_per_wall, batch_size, num_batches, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range)

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc_left': bc_left,
            'bc_right': bc_right,
            'bc_bottom': bc_bot,
            'bc_top': bc_top,
            'data': maybe_batch_data(data_key, data_points_full, batch_size, num_batches, data_free),
        }

    generate_epoch_data_jitted = jax.jit(generate_epoch_data)

    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights_dict,
        cfg, data_free, extra_static_args=(bc_fn_static,),
    )

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
