"""Experiment 3 — Terrain slope in x-direction (Phase 2).

Introduces terrain via bi-linear DEM interpolation; establishes data
sampling ratio methodology when physics-only training is insufficient.
Requires: configs/experiment_3.yaml, data/experiment_3/
Builds on: Experiment 2.
"""
import os
import sys
import argparse
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from src.config import get_dtype
from src.data import (
    get_batches_tensor,
    bathymetry_fn,
    sample_lhs,
)
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet,
    loss_boundary_wall_horizontal,
    loss_boundary_wall_vertical,
    compute_neg_h_loss,
    compute_data_loss,
)
from src.utils import nse, rmse
from src.utils.plotting import plot_gauge_timeseries
from src.training import (
    create_optimizer,
    calculate_num_batches,
    extract_loss_weights,
    get_data_filename,
    get_sampling_count_from_config,
    get_boundary_segment_count,
    load_terrain_assets,
    load_training_data,
    load_validation_from_file,
    train_step_jitted,
    make_scan_body,
    sample_and_batch,
    maybe_batch_data,
    post_training_save,
    resolve_experiment_paths,
    resolve_data_mode,
    run_training_loop,
    setup_experiment,
)


def make_compute_losses(bc_fn_static):
    """Return a compute_losses closure for Experiment 3 (x-direction slope)."""

    def compute_losses(model, params, batch, config, data_free):
        terms = {}
        terms['pde'] = compute_pde_loss(model, params, batch['pde'], config)
        terms['neg_h'] = compute_neg_h_loss(model, params, batch['pde'])

        # IC: target depth = max(0, water_level - z)
        U_ic = model.apply(params, batch['ic'], train=False)
        z_ic, _, _ = bathymetry_fn(batch['ic'][..., 0], batch['ic'][..., 1])
        initial_water_level = config["initial_condition"]["absolute_water_level"]
        h_target_ic = jnp.maximum(0.0, initial_water_level - z_ic)
        loss_ic_h = jnp.mean((U_ic[..., 0] - h_target_ic) ** 2)
        loss_ic_vel = jnp.mean(U_ic[..., 1] ** 2 + U_ic[..., 2] ** 2)
        terms['ic'] = loss_ic_h + loss_ic_vel

        # BC: left = time-varying water level, others = slip walls
        t_left = batch['bc_left'][..., 2]
        bc_level_abs = bc_fn_static(t_left)
        z_left, _, _ = bathymetry_fn(batch['bc_left'][..., 0], batch['bc_left'][..., 1])
        h_target_left = jnp.maximum(0.0, bc_level_abs - z_left)
        loss_bc_left = loss_boundary_dirichlet(model, params, batch['bc_left'], h_target_left, var_idx=0)
        loss_bc_right = loss_boundary_wall_vertical(model, params, batch['bc_right'])
        loss_bc_top = loss_boundary_wall_horizontal(model, params, batch['bc_top'])
        loss_bc_bottom = loss_boundary_wall_horizontal(model, params, batch['bc_bottom'])
        terms['bc'] = loss_bc_left + loss_bc_right + loss_bc_top + loss_bc_bottom

        data_batch_data = batch.get('data', jnp.empty((0, 6), dtype=get_dtype()))
        if not data_free and data_batch_data.shape[0] > 0:
            terms['data'] = compute_data_loss(model, params, data_batch_data, config)

        return terms

    return compute_losses

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

    # A. Load Bathymetry + Boundary Condition
    terrain = load_terrain_assets(cfg, base_data_path, scenario_name)
    bc_fn_static = terrain["bc_fn"]

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
    val_targets_all = validation["val_targets"]

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
        x_range = (0., domain_cfg["lx"])
        y_range = (0., domain_cfg["ly"])
        t_range = (0., domain_cfg["t_final"])
        batch = {
            'pde': sample_lhs(keys[0], n_eval, x_range, y_range, t_range),
            'ic': sample_lhs(keys[1], n_eval, x_range, y_range, (0., 0.)),
            'bc_left': sample_lhs(keys[2], n_eval, (0., 0.), y_range, t_range),
            'bc_right': sample_lhs(keys[2], n_eval, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range),
            'bc_bottom': sample_lhs(keys[3], n_eval, x_range, (0., 0.), t_range),
            'bc_top': sample_lhs(keys[3], n_eval, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range),
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
        print("Generating Experiment 3 plots...")
        t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=get_dtype())
        aim_tracker = loop_result["aim_tracker"]
        final_epoch = loop_result["epoch"]

        gauge_kwargs = dict(
            model=model, params=final_params, t_plot=t_plot, cfg=cfg,
            results_dir=results_dir, aim_tracker=aim_tracker, epoch=final_epoch,
            full_val_data=full_val_data,
        )
        plot_gauge_timeseries(3.9587225e+02, 4.9646515e+01, "Point 1", "P1_timeseries.png", color="blue", **gauge_kwargs)
        plot_gauge_timeseries(6.0435474e+02, 5.0565735e+01, "Point 2", "P2_timeseries.png", color="red", **gauge_kwargs)
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
