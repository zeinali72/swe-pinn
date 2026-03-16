"""Experiment 7 — Irregular boundaries with mesh-based sampling (Phase 3).

Tackles non-rectangular domains using triangulated mesh sampling,
automated boundary detection, and computed wall normals for slip BCs.
Requires: configs/experiment_7.yaml, data/experiment_7/
Builds on: Experiment 5.
"""
import os
import sys
import argparse
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict
import numpy as np

# Local application imports
from src.config import load_config, get_dtype
from src.data import (
    get_batches_tensor,
    IrregularDomainSampler,
)
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet,
    loss_slip_wall_generalized,
    compute_neg_h_loss,
    compute_data_loss,
)
from src.utils import nse, rmse
from src.utils.plotting import plot_gauge_timeseries
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
    load_terrain_assets,
    load_training_data,
    load_validation_from_file,
    train_step_jitted,
    make_scan_body,
    maybe_batch_data,
    post_training_save,
    resolve_experiment_paths,
    resolve_data_mode,
    run_training_loop,
)

def make_compute_losses(bc_fn_static):
    """Return a compute_losses closure for Experiment 7 (irregular boundaries)."""

    def compute_losses(model, params, batch, config, data_free):
        terms = {}
        terms['pde'] = compute_pde_loss(model, params, batch['pde'], config)
        terms['neg_h'] = compute_neg_h_loss(model, params, batch['pde'])

        # IC: dry bed
        U_ic = model.apply(params, batch['ic'], train=False)
        terms['ic'] = jnp.mean(U_ic[..., 0] ** 2) + jnp.mean(U_ic[..., 1] ** 2 + U_ic[..., 2] ** 2)

        # BC: inflow flux + generalized slip wall
        t_inflow = batch['bc_inflow'][..., 2]
        Q_target = bc_fn_static(t_inflow)
        inflow_width = config["boundary_conditions"]["inflow_discharge_width"]
        flux_target_x = Q_target / inflow_width
        loss_bc_inflow = (
            loss_boundary_dirichlet(model, params, batch['bc_inflow'], flux_target_x, var_idx=1)
            + loss_boundary_dirichlet(model, params, batch['bc_inflow'], jnp.zeros_like(flux_target_x), var_idx=2)
        )
        loss_bc_wall = loss_slip_wall_generalized(model, params, batch['bc_wall'])
        terms['bc'] = loss_bc_inflow + loss_bc_wall

        data_batch_data = batch.get('data', jnp.empty((0, 6), dtype=get_dtype()))
        if not data_free and data_batch_data.shape[0] > 0:
            terms['data'] = compute_data_loss(model, params, data_batch_data, config)

        return terms

    return compute_losses

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
    terrain = load_terrain_assets(cfg, base_data_path, scenario_name)
    bc_fn_static = terrain["bc_fn"]

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
    val_targets_all = validation["val_targets"]

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

    compute_losses_fn = make_compute_losses(bc_fn_static)
    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights_dict,
        cfg, data_free, compute_losses_fn=compute_losses_fn,
    )

    # --- Evaluate All Physics Losses (including zero-weight terms) ---
    n_eval = 200
    def compute_all_losses_fn(model, params):
        eval_key = random.PRNGKey(0)
        keys = random.split(eval_key, 5)
        t_range = (0., domain_cfg["t_final"])
        batch = {
            'pde': domain_sampler.sample_interior(keys[0], n_eval, t_range),
            'ic': domain_sampler.sample_interior(keys[1], n_eval, (0., 0.)),
            'bc_inflow': domain_sampler.sample_boundary(keys[2], n_eval, t_range, 'inflow'),
            'bc_wall': domain_sampler.sample_boundary(keys[3], n_eval, t_range, 'wall'),
            'data': jnp.empty((0, 6), dtype=get_dtype()),
        }
        return compute_losses_fn(model, params, batch, cfg, data_free=True)

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
        val_targets_all=val_targets_all,
        source_script_path=__file__,
        compute_all_losses_fn=compute_all_losses_fn,
    )

    def plot_fn(final_params):
        print("Generating Experiment 7 plots...")
        t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=get_dtype())
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

        gauge_kwargs = dict(
            model=model, params=final_params, t_plot=t_plot, cfg=cfg,
            results_dir=results_dir, aim_tracker=aim_tracker, epoch=final_epoch,
            full_val_data=full_val_data,
        )
        for px, py, pname in output_points:
            plot_gauge_timeseries(px, py, pname, f"{pname}_timeseries.png", **gauge_kwargs)

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
