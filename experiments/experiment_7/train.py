"""Experiment 7 — Irregular boundaries with mesh-based sampling (Phase 3).

Tackles non-rectangular domains using triangulated mesh sampling,
automated boundary detection, and computed wall normals for slip BCs.
Requires: configs/experiment_7.yaml, data/experiment_7/
Builds on: Experiment 5.
"""
import os
import sys
import argparse
from typing import Any, Dict, Tuple
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.core import FrozenDict
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from src.config import load_config, DTYPE
from src.data import (
    get_batches_tensor,
    load_boundary_condition,
    IrregularDomainSampler,
    load_bathymetry,
)
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet_hu,
    loss_boundary_dirichlet_hv,
    loss_slip_wall_generalized,
    compute_neg_h_loss,
    compute_data_loss,
    total_loss
)
from src.utils import nse, rmse
from src.training import (
    apply_irregular_domain_bounds,
    apply_output_scales,
    calculate_num_batches,
    create_optimizer,
    create_output_dirs,
    extract_loss_weights,
    get_data_filename,
    get_experiment_name,
    get_sampling_count_from_config,
    init_model_from_config,
    load_training_data,
    load_validation_from_file,
    make_scan_body,
    maybe_batch_data,
    post_training_save,
    resolve_configured_asset_path,
    resolve_experiment_paths,
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
        
        # A. Inflow Boundary (Flux prescribed)
        t_inflow = batch['bc_inflow'][..., 2]
        Q_target = bc_fn_static(t_inflow) # m^3/s
        inflow_width = config["boundary_conditions"]["inflow_discharge_width"]
        flux_target_x = Q_target / inflow_width

        loss_inflow_x = loss_boundary_dirichlet_hu(model, params, batch['bc_inflow'], flux_target_x)
        loss_inflow_y = loss_boundary_dirichlet_hv(model, params, batch['bc_inflow'], jnp.zeros_like(flux_target_x))
        loss_bc_inflow = loss_inflow_x + loss_inflow_y

        # B. Wall Boundaries (Generalized Slip)
        loss_bc_wall = loss_slip_wall_generalized(model, params, batch['bc_wall'])
        
        total_bc = loss_bc_inflow + loss_bc_wall

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
            'data': loss_data
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
    Main training loop for Experiment 7.
    """
    #--- 1. LOAD CONFIGURATION (MUTABLE) ---
    cfg_dict = load_config(config_path)
    experiment_name = get_experiment_name(cfg_dict, "experiment_7")
    
    print("Info: Running Experiment 7 model training...")

    # --- 2. SETUP DATA & COMPUTE DOMAIN EXTENT ---
    try:
        experiment_paths = resolve_experiment_paths(cfg_dict, experiment_name, require_scenario=True)
    except ValueError as exc:
        print(f"Error: {exc} in config '{config_path}'.")
        sys.exit(1)

    scenario_name = experiment_paths["scenario_name"]
    base_data_path = experiment_paths["base_data_path"]

    # A. Init Irregular Domain Sampler & Calculate lx, ly
    try:
        artifacts_path = resolve_configured_asset_path(cfg_dict, base_data_path, scenario_name, "domain_artifacts")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    
    print(f"Loading domain geometry from: {artifacts_path}")
    domain_sampler = IrregularDomainSampler(artifacts_path)

    domain_extent = apply_irregular_domain_bounds(cfg_dict, domain_sampler)
    
    print(f"Computed Domain Extent:")
    print(f"  X Range: [{domain_extent['x_min']:.4f}, {domain_extent['x_max']:.4f}]")
    print(f"  Y Range: [{domain_extent['y_min']:.4f}, {domain_extent['y_max']:.4f}]")
    print(f"  Calculated Dimensions: lx = {domain_extent['lx']:.4f}, ly = {domain_extent['ly']:.4f}")
    output_scales = apply_output_scales(cfg_dict, (1.0, 1.0, 1.0))
    print(f"Active Output Scaling: {output_scales}")

    # Derive inflow discharge width from the domain geometry
    bc_cfg = cfg_dict.setdefault("boundary_conditions", {})
    if 'inflow' in domain_sampler.boundaries:
        computed_width = domain_sampler.boundary_length('inflow')
        bc_cfg["inflow_discharge_width"] = computed_width
        print(f"Inflow discharge width derived from shapefile: {computed_width:.4f} m")

    # --- 3. FINALIZE CONFIG & INIT MODEL ---
    cfg = FrozenDict(cfg_dict)
    model, params, train_key, val_key = init_model_from_config(cfg)
    trial_name, results_dir, model_dir = create_output_dirs(cfg, experiment_name)

    # --- 5. Prepare Loss Weights ---
    static_weights_dict, current_weights_dict = extract_loss_weights(cfg)

    # --- 6. Load Remaining Assets ---

    # B. Load Bathymetry (REQUIRED)
    try:
        dem_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "dem")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)
    
    # C. Load Boundary Condition Function
    try:
        bc_csv_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "boundary_condition")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    bc_fn_static = load_boundary_condition(bc_csv_path)

    # D. Load Validation and Training Data
    data_points_full = None
    data_free, has_data_loss = resolve_data_mode(cfg)
    data_points_full, has_data_loss, data_free = load_training_data(
        base_data_path,
        has_data_loss,
        static_weights_dict,
        filename=get_data_filename(cfg, "training_file", "training_dataset_sample.npy"),
    )

    # E. Load Validation Data (Optional)
    validation = load_validation_from_file(
        base_data_path,
        get_data_filename(cfg, "validation_file", "validation_gauges_ground_truth.npy"),
    )
    validation_data_loaded = validation["loaded"]
    full_val_data = validation["full_val_data"]
    val_points_all = validation["val_points"]
    h_true_val_all = validation["h_true_val"]

    # --- 7. Data Generation Setup ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]
    
    n_pde = get_sampling_count_from_config(cfg, "n_points_pde")
    n_ic = get_sampling_count_from_config(cfg, "n_points_ic")
    n_bc_inflow = get_sampling_count_from_config(cfg, "n_points_bc_inflow")
    n_bc_wall = get_sampling_count_from_config(cfg, "n_points_bc_domain")

    num_batches = calculate_num_batches(
        batch_size,
        [n_pde, n_ic, n_bc_wall, n_bc_inflow],
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

    def generate_epoch_data(key):
        k1, k2, k3, k4, k5 = random.split(key, 5)
        t_range = (0., domain_cfg["t_final"])

        pde_pts = domain_sampler.sample_interior(k1, n_pde, t_range)
        pde_data = get_batches_tensor(k1, pde_pts, batch_size, num_batches)

        ic_pts = domain_sampler.sample_interior(k2, n_ic, (0., 0.))
        ic_data = get_batches_tensor(k2, ic_pts, batch_size, num_batches)

        bc_inflow_pts = domain_sampler.sample_boundary(k3, n_bc_inflow, t_range, 'inflow')
        bc_inflow = get_batches_tensor(k3, bc_inflow_pts, batch_size, num_batches)

        bc_wall_pts = domain_sampler.sample_boundary(k4, n_bc_wall, t_range, 'wall')
        bc_wall = get_batches_tensor(k4, bc_wall_pts, batch_size, num_batches)

        return {
            'pde': pde_data, 'ic': ic_data,
            'bc_inflow': bc_inflow, 'bc_wall': bc_wall,
            'data': maybe_batch_data(k5, data_points_full, batch_size, num_batches, data_free),
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
        print("Generating Experiment 7 plots...")
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
            print("Using default representative points (Center).")
            cx = (cfg['domain']['x_max'] + cfg['domain']['x_min']) / 2
            cy = (cfg['domain']['y_max'] + cfg['domain']['y_min']) / 2
            output_points = [(cx, cy, "Center_Point")]

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

            plt.plot(t_plot, h_pred, label=f'Predicted h @ ({x:.1f},{y:.1f})')
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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Experiment 7 - Irregular).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    main(args.config)
