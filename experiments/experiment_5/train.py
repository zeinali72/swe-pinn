"""Experiment 5 — Synthetic complexity stage 1 (Phase 2).

Validates robustness on increasingly complex synthetic domains.
Requires: configs/experiment_5.yaml, data/experiment_5/
Builds on: Experiment 4.
"""
import os
import sys
import argparse
import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from src.config import load_config, get_dtype
from src.data import (
    get_batches_tensor,
    bathymetry_fn,
    load_boundary_condition,
    load_bathymetry,
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
from src.training import (
    create_optimizer,
    calculate_num_batches,
    extract_loss_weights,
    get_data_filename,
    get_experiment_name,
    get_sampling_count_from_config,
    get_boundary_segment_count,
    init_model_from_config,
    load_training_data,
    load_validation_from_file,
    train_step_jitted,
    make_scan_body,
    sample_and_batch,
    maybe_batch_data,
    post_training_save,
    resolve_configured_asset_path,
    resolve_experiment_paths,
    resolve_data_mode,
    run_training_loop,
    create_output_dirs,
)


def make_compute_losses(bc_fn_static):
    """Return a compute_losses closure for Experiment 5 (single left inflow)."""

    def compute_losses(model, params, batch, config, data_free):
        terms = {}
        terms['pde'] = compute_pde_loss(model, params, batch['pde'], config)
        terms['neg_h'] = compute_neg_h_loss(model, params, batch['pde'])

        # IC: dry bed
        U_ic = model.apply(params, batch['ic'], train=False)
        terms['ic'] = jnp.mean(U_ic[..., 0] ** 2) + jnp.mean(U_ic[..., 1] ** 2 + U_ic[..., 2] ** 2)

        # BC: left inflow hu + slip walls
        t_inflow = batch['bc_left'][..., 2]
        Q_target_x = bc_fn_static(t_inflow)
        inflow_width = config["boundary_conditions"]["inflow_discharge_width"]
        flux_target_x = Q_target_x / inflow_width
        loss_bc_left = loss_boundary_dirichlet(model, params, batch['bc_left'], flux_target_x, var_idx=1)
        loss_bc_right = loss_boundary_wall_vertical(model, params, batch['bc_right'])
        loss_bc_top = loss_boundary_wall_horizontal(model, params, batch['bc_top'])
        loss_bc_bottom = loss_boundary_wall_horizontal(model, params, batch['bc_bottom'])
        terms['bc'] = loss_bc_left + loss_bc_right + loss_bc_top + loss_bc_bottom

        data_batch_data = batch.get('data', jnp.empty((0, 6), dtype=get_dtype()))
        if not data_free and data_batch_data.shape[0] > 0:
            terms['data'] = compute_data_loss(model, params, data_batch_data, config)

        return terms

    return compute_losses


def setup_trial(cfg_dict: dict, hpo_mode: bool = False) -> dict:
    """Set up all training components for Experiment 5 from a config dict.

    Args:
        cfg_dict: Mutable configuration dictionary (not a file path). This is the
            interface used by HPO to pass trial-specific configs directly.

    Returns:
        Dictionary containing all objects needed to call run_training_loop, plus
        production metadata fields (experiment_name, validation_data_loaded, etc.).
    """
    cfg = FrozenDict(cfg_dict)
    experiment_name = get_experiment_name(cfg_dict, "experiment_5")

    model, params, train_key, val_key = init_model_from_config(cfg)

    print("Info: Running Experiment 5 Scenario model training...")

    # --- Prepare Loss Weights ---
    static_weights_dict, current_weights_dict = extract_loss_weights(cfg)

    # --- Load Data Assets ---
    try:
        experiment_paths = resolve_experiment_paths(cfg, experiment_name, require_scenario=True)
    except ValueError as exc:
        raise ValueError(f"Experiment path resolution failed: {exc}") from exc

    scenario_name = experiment_paths["scenario_name"]
    base_data_path = experiment_paths["base_data_path"]

    # A. Load Bathymetry (REQUIRED)
    try:
        dem_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "dem")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"DEM asset not found: {exc}") from exc
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)

    # B. Load Boundary Condition Function
    try:
        bc_csv_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "boundary_condition")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Boundary condition asset not found: {exc}") from exc
    bc_fn_static = load_boundary_condition(bc_csv_path)

    # --- Load Validation and Training Data ---
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
    val_targets_all = validation["val_targets"]

    # --- Data Generation Setup ---
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]

    n_pde = get_sampling_count_from_config(cfg, "n_points_pde")
    n_ic = get_sampling_count_from_config(cfg, "n_points_ic")
    n_bc_domain = get_sampling_count_from_config(cfg, "n_points_bc_domain")
    n_bc_per_wall = get_boundary_segment_count(cfg, n_bc_domain)

    num_batches = calculate_num_batches(
        batch_size,
        [n_pde, n_ic, n_bc_per_wall],
        data_points_full,
        data_free=data_free,
    )

    if num_batches == 0:
        raise ValueError(
            f"Batch size {batch_size} is too large for configured sample counts or data."
        )
    print(f"Batches per epoch: {num_batches}")

    # --- Setup Optimizer ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # --- JIT Data Generator ---
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

    generate_epoch_data_jit = jax.jit(generate_epoch_data)

    compute_losses_fn = make_compute_losses(bc_fn_static)
    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights_dict,
        cfg, data_free, compute_losses_fn=compute_losses_fn,
    )

    # --- Validation Function ---
    def validation_fn(model, params):
        metrics = {}
        if validation_data_loaded and val_points_all is not None:
            U_pred = model.apply(params, val_points_all, train=False)
            h_pred = U_pred[..., 0]
            metrics['nse_h'] = float(nse(h_pred, h_true_val_all))
            metrics['rmse_h'] = float(rmse(h_pred, h_true_val_all))
        if not metrics:
            metrics = {'nse_h': float(-jnp.inf), 'rmse_h': float(jnp.inf)}
        return metrics

    # --- Evaluate All Physics Losses ---
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

    return {
        "cfg": cfg,
        "cfg_dict": cfg_dict,
        "model": model,
        "params": params,
        "train_key": train_key,
        "optimiser": optimiser,
        "opt_state": opt_state,
        "generate_epoch_data_jit": generate_epoch_data_jit,
        "scan_body": scan_body,
        "num_batches": num_batches,
        "validation_fn": validation_fn,
        "data_free": data_free,
        "compute_all_losses_fn": compute_all_losses_fn,
        # Production extras
        "experiment_name": experiment_name,
        "validation_data_loaded": validation_data_loaded,
        "val_points_all": val_points_all,
        "h_true_val_all": h_true_val_all,
        "val_targets_all": val_targets_all,
        # For plot_fn in main()
        "full_val_data": full_val_data,
        "base_data_path": base_data_path,
        "scenario_name": scenario_name,
    }


def main(config_path: str):
    """Main training loop for Experiment 5 scenario."""
    cfg_dict = load_config(config_path)
    ctx = setup_trial(cfg_dict)

    experiment_name = ctx["experiment_name"]
    trial_name, results_dir, model_dir = create_output_dirs(ctx["cfg"], experiment_name)

    model = ctx["model"]
    cfg = ctx["cfg"]
    full_val_data = ctx["full_val_data"]

    loop_result = run_training_loop(
        cfg=cfg,
        cfg_dict=ctx["cfg_dict"],
        model=model,
        params=ctx["params"],
        opt_state=ctx["opt_state"],
        train_key=ctx["train_key"],
        optimiser=ctx["optimiser"],
        generate_epoch_data_jit=ctx["generate_epoch_data_jit"],
        scan_body=ctx["scan_body"],
        num_batches=ctx["num_batches"],
        experiment_name=experiment_name,
        trial_name=trial_name,
        results_dir=results_dir,
        model_dir=model_dir,
        config_path=config_path,
        validation_data_loaded=ctx.get("validation_data_loaded", False),
        val_points_all=ctx.get("val_points_all"),
        h_true_val_all=ctx.get("h_true_val_all"),
        val_targets_all=ctx.get("val_targets_all"),
        compute_all_losses_fn=ctx["compute_all_losses_fn"],
    )

    def plot_fn(final_params):
        print("Generating Experiment 5 plots...")
        t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=get_dtype())
        tracker = loop_result["tracker"]
        final_epoch = loop_result["epoch"]

        def plot_gauge(x, y, name, color, filename):
            pts = jnp.stack([jnp.full_like(t_plot, x), jnp.full_like(t_plot, y), t_plot], axis=-1)
            U = model.apply(final_params, pts, train=False)
            min_depth_plot = cfg.get("numerics", {}).get("min_depth", 0.0)
            h_pred = jnp.where(U[..., 0] < min_depth_plot, 0.0, U[..., 0])
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
            tracker.log_image(path, filename)

        plot_gauge(150, 50.0, "Point 1", "blue", "P1_timeseries.png")
        plot_gauge(250.0, 50.0, "Point 2", "red", "P2_timeseries.png")
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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Test 3).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    main(args.config)
