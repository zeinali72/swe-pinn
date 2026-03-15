import os
import sys
import argparse
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from src.config import DTYPE
from src.data import (
    get_batches_tensor,
    bathymetry_fn,
    load_boundary_condition,
    sample_lhs,
)
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet_hu,
    loss_boundary_dirichlet_hv,
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
    get_sampling_count_from_config,
    get_boundary_segment_count,
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
    setup_experiment,
)

def make_compute_losses(bc_fn_static):
    """Return a compute_losses closure for Experiment 6 (split inflow + left wall)."""

    def compute_losses(model, params, batch, config, data_free):
        terms = {}
        terms['pde'] = compute_pde_loss(model, params, batch['pde'], config)
        terms['neg_h'] = compute_neg_h_loss(model, params, batch['pde'])

        # IC: dry bed
        U_ic = model.apply(params, batch['ic'], train=True)
        terms['ic'] = jnp.mean(U_ic[..., 0] ** 2) + jnp.mean(U_ic[..., 1] ** 2 + U_ic[..., 2] ** 2)

        # BC: inflow flux (hu + hv) + left wall + right/top/bottom slip walls
        t_inflow = batch['bc_inflow'][..., 2]
        Q_target_x = bc_fn_static(t_inflow)
        inflow_width = config["boundary_conditions"]["inflow_discharge_width"]
        flux_target_x = Q_target_x / inflow_width
        loss_bc_inflow = (
            loss_boundary_dirichlet_hu(model, params, batch['bc_inflow'], flux_target_x)
            + loss_boundary_dirichlet_hv(model, params, batch['bc_inflow'], jnp.zeros_like(flux_target_x))
        )
        loss_bc_left_wall = loss_boundary_wall_vertical(model, params, batch['bc_left_wall'])
        loss_bc_right = loss_boundary_wall_vertical(model, params, batch['bc_right'])
        loss_bc_top = loss_boundary_wall_horizontal(model, params, batch['bc_top'])
        loss_bc_bottom = loss_boundary_wall_horizontal(model, params, batch['bc_bottom'])
        terms['bc'] = loss_bc_inflow + loss_bc_left_wall + loss_bc_right + loss_bc_top + loss_bc_bottom

        data_batch_data = batch.get('data', jnp.empty((0, 6), dtype=DTYPE))
        if not data_free and data_batch_data.shape[0] > 0:
            terms['data'] = compute_data_loss(model, params, data_batch_data, config)

        return terms

    return compute_losses

def main(config_path: str):
    """
    Main training loop for Experiment 6.
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

    print("Info: Running Experiment 6 model training...")

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
    val_targets_all = validation["val_targets"]

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
        x_range = (0., domain_cfg["lx"])
        y_range = (0., domain_cfg["ly"])
        t_range = (0., domain_cfg["t_final"])

        pde_data = sample_and_batch(pde_key, sample_lhs, n_pde, batch_size, num_batches, x_range, y_range, t_range)
        ic_data = sample_and_batch(ic_key, sample_lhs, n_ic, batch_size, num_batches, x_range, y_range, (0., 0.))

        # Split Left Boundary
        inflow_segment = cfg["boundary_conditions"]["left_inflow_segment"]
        y_inflow_start = inflow_segment["y_start"]
        y_inflow_end = inflow_segment["y_end"]

        l_in_key, l_wall_bot_key, l_wall_top_key, r_key, b_key, t_key = random.split(bc_keys, 6)
        bc_inflow = sample_and_batch(l_in_key, sample_lhs, n_bc_inflow, batch_size, num_batches, (0., 0.), (y_inflow_start, y_inflow_end), t_range)
        # Sample each left-wall sub-segment with half the batch size so the
        # concatenated result has batch_size rows, matching the other walls.
        n_bc_left_half = max(batch_size // 2, n_bc_per_wall // 2)
        bc_left_wall_bottom = sample_and_batch(l_wall_bot_key, sample_lhs, n_bc_left_half, batch_size // 2, num_batches, (0., 0.), (0., y_inflow_start), t_range)
        bc_left_wall_above = sample_and_batch(l_wall_top_key, sample_lhs, n_bc_left_half, batch_size // 2, num_batches, (0., 0.), (y_inflow_end, domain_cfg["ly"]), t_range)
        bc_left_wall = jnp.concatenate([bc_left_wall_bottom, bc_left_wall_above], axis=1)
        bc_right = sample_and_batch(r_key, sample_lhs, n_bc_per_wall, batch_size, num_batches, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range)
        bc_bot = sample_and_batch(b_key, sample_lhs, n_bc_per_wall, batch_size, num_batches, x_range, (0., 0.), t_range)
        bc_top = sample_and_batch(t_key, sample_lhs, n_bc_per_wall, batch_size, num_batches, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range)

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc_inflow': bc_inflow,
            'bc_left_wall': bc_left_wall,
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
    )

    def plot_fn(final_params):
        print("Generating Experiment 6 plots...")
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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Experiment 6).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    main(args.config)
