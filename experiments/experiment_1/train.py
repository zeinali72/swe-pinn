"""Experiment 1 — Analytical dam-break on flat domain (Phase 1).

Verifies the PINN framework against an analytical dam-break solution.
Requires: configs/experiment_1_*.yaml
Builds on: None (baseline verification).

This script handles both data-free and data-driven (analytical) training modes.
It supports static loss weighting and provides
logging and result visualization through Aim.
"""

import os
import sys
import argparse
import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict
import numpy as np

# Local application imports
from src.config import get_dtype
from src.predict.predictor import _apply_min_depth
from src.data import sample_domain, get_batches_tensor
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_data_loss, compute_neg_h_loss,
    loss_boundary_dirichlet, loss_boundary_neumann_outflow_x,
    loss_boundary_wall_horizontal,
)
from src.utils import nse, rmse, plot_h_vs_x
from src.physics import h_exact, hu_exact, hv_exact
from src.training import (
    create_optimizer,
    calculate_num_batches,
    extract_loss_weights,
    get_active_loss_weights,
    get_boundary_segment_count,
    get_sampling_count_from_config,
    train_step_jitted,
    make_scan_body,
    sample_and_batch,
    empty_batch,
    maybe_batch_data,
    post_training_save,
    resolve_data_mode,
    run_training_loop,
    setup_experiment,
)

# To enable 64-bit precision, uncomment the following line:
# jax.config.update('jax_enable_x64', True)


def compute_losses(model, params, batch, config, data_free):
    """Compute all loss terms for Experiment 1 (analytical dam-break)."""
    terms = {}

    pde_batch_data = batch.get('pde', jnp.empty((0, 3), dtype=get_dtype()))
    if pde_batch_data.shape[0] > 0:
        terms['pde'] = compute_pde_loss(model, params, pde_batch_data, config)
        terms['neg_h'] = compute_neg_h_loss(model, params, pde_batch_data)

    ic_batch_data = batch.get('ic', jnp.empty((0, 3), dtype=get_dtype()))
    if ic_batch_data.shape[0] > 0:
        terms['ic'] = compute_ic_loss(model, params, ic_batch_data)

    bc_batches = batch.get('bc', {})
    if any(b.shape[0] > 0 for b in bc_batches.values() if hasattr(b, 'shape')):
        left = bc_batches.get('left', jnp.empty((0, 3), dtype=get_dtype()))
        right = bc_batches.get('right', jnp.empty((0, 3), dtype=get_dtype()))
        bottom = bc_batches.get('bottom', jnp.empty((0, 3), dtype=get_dtype()))
        top = bc_batches.get('top', jnp.empty((0, 3), dtype=get_dtype()))

        # Left: Dirichlet h + hu from analytical dam-break solution
        u_const = config["physics"]["u_const"]
        n_manning = config["physics"]["n_manning"]
        t_left = left[..., 2]
        h_true = h_exact(0.0, t_left, n_manning, u_const)
        hu_true = h_true * u_const
        loss_left = (loss_boundary_dirichlet(model, params, left, h_true, var_idx=0) +
                     loss_boundary_dirichlet(model, params, left, hu_true, var_idx=1))
        # Right: Neumann outflow
        loss_right = loss_boundary_neumann_outflow_x(model, params, right)
        # Top/Bottom: horizontal walls
        loss_bottom = loss_boundary_wall_horizontal(model, params, bottom)
        loss_top = loss_boundary_wall_horizontal(model, params, top)
        terms['bc'] = loss_left + loss_right + loss_bottom + loss_top

    data_batch_data = batch.get('data', jnp.empty((0, 6), dtype=get_dtype()))
    if not data_free and data_batch_data.shape[0] > 0:
        terms['data'] = compute_data_loss(model, params, data_batch_data, config)

    return terms


def main(config_path: str):
    """
    Main training function for the analytical scenario.
    """
    # --- 1. Load Config and Initialize Model ---
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

    print("Info: Running in analytical (no-building) mode.")

    # --- 4. Prepare Loss Weights ---
    static_weights_dict, _ = extract_loss_weights(cfg)
    
    # --- 5. Create Validation and Training Data (Analytical) ---
    
    # Create Analytical Validation Data
    val_points, h_true_val, hu_true_val, hv_true_val = None, None, None, None
    validation_data_loaded = False
    try:
        val_grid_cfg = cfg["validation_grid"]
        domain_cfg = cfg["domain"]
        print(f"Creating analytical validation set from 'validation_grid' config...")

        val_points = sample_domain(
            val_key,
            val_grid_cfg["n_points_val"],
            (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
        )
        n_manning = cfg["physics"]["n_manning"]
        u_const = cfg["physics"]["u_const"]
        h_true_val = h_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        hu_true_val = hu_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        hv_true_val = hv_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)

        if val_points.shape[0] > 0:
            validation_data_loaded = True
            print(f"Created analytical validation set with {val_points.shape[0]} points.")
        else:
            print("Warning: Analytical validation set is empty.")

    except KeyError:
        print("Warning: 'validation_grid' not found in config. Skipping NSE/RMSE calculation.")
    except Exception as e:
        print(f"Warning: Error creating analytical validation set: {e}. Skipping NSE/RMSE.")
        
    # Determine Data-Free Mode
    data_points_full = None
    data_free, has_data_loss = resolve_data_mode(cfg)

    # Create Analytical Training Data (if data_free is False)
    if not data_free:
        try:
            train_grid_cfg = cfg["train_grid"]
            domain_cfg = cfg["domain"]
            print(f"Creating analytical training dataset from 'train_grid' config...")
            
            # 1. Sample gauge locations (x, y) and expand to full time series
            n_gauges = train_grid_cfg["n_gauges"]
            dt_data = train_grid_cfg["dt_data"]
            t_final = domain_cfg["t_final"]

            if n_gauges <= 0 or dt_data <= 0:
                raise ValueError(
                    f"Gauge-based sampling requires n_gauges > 0 and dt_data > 0, "
                    f"got n_gauges={n_gauges}, dt_data={dt_data}. "
                    f"Set data_free: true to skip data sampling."
                )

            # Create time array at specified resolution
            t_steps = jnp.arange(0., t_final + dt_data * 0.5, dt_data, dtype=get_dtype())
            n_timesteps = t_steps.shape[0]

            # Sample n_gauges random spatial locations
            gauge_xy = sample_domain(
                train_key,
                n_gauges,
                (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.)
            )[:, :2]  # shape (n_gauges, 2) — keep only x, y

            # Expand: each gauge gets the full time series
            # gauge_xy_rep: (n_gauges * n_timesteps, 2)
            gauge_xy_rep = jnp.repeat(gauge_xy, n_timesteps, axis=0)
            # t_rep: (n_gauges * n_timesteps, 1)
            t_rep = jnp.tile(t_steps, n_gauges).reshape(-1, 1)
            # data_points_coords: (n_gauges * n_timesteps, 3) — [x, y, t]
            data_points_coords = jnp.hstack([gauge_xy_rep, t_rep])

            print(f"Gauge-based sampling: {n_gauges} gauges x {n_timesteps} timesteps (dt={dt_data}s) = {data_points_coords.shape[0]} data points")

            # 2. Calculate true values (h, u, v)
            h_true_train = h_exact(
                data_points_coords[:, 0], # x
                data_points_coords[:, 2], # t
                cfg["physics"]["n_manning"],
                cfg["physics"]["u_const"]
            )
            u_true_train = jnp.full_like(h_true_train, cfg["physics"]["u_const"])
            v_true_train = jnp.zeros_like(h_true_train)
            
            # 3. Stack into (N, 6) format: [t, x, y, h, u, v]
            data_points_full = jnp.stack([
                data_points_coords[:, 2], # t
                data_points_coords[:, 0], # x
                data_points_coords[:, 1], # y
                h_true_train,
                u_true_train,
                v_true_train
            ], axis=1).astype(get_dtype())

            if data_points_full.shape[0] == 0:
                 print("Warning: Analytical training data is empty. Disabling data loss.")
                 data_points_full = None
                 has_data_loss = False
            else:
                 print(f"Created {data_points_full.shape[0]} points for data loss term (weight={static_weights_dict.get('data', 0.0):.2e}).")

        except KeyError:
            print("Error: 'data_free: false' but 'train_grid' not found in config. Disabling data loss.")
            has_data_loss = False
            data_free = True # Revert to data-free
        except Exception as e:
            print(f"Error creating analytical training data: {e}. Disabling data loss.")
            has_data_loss = False
            data_free = True # Revert to data-free
    
    # --- 6. Determine Active Loss Terms ---
    current_weights_dict = get_active_loss_weights(
        static_weights_dict,
        data_free=data_free,
        excluded_keys={"building_bc"},
    )
    active_loss_term_keys = list(current_weights_dict.keys())

    # --- 7. Pre-calculate Batch Counts and Total Batches (for jax.lax.scan) ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]

    # Calculate expected points
    n_pde = get_sampling_count_from_config(cfg, "n_points_pde") if ('pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys) else 0
    n_ic = get_sampling_count_from_config(cfg, "n_points_ic") if 'ic' in active_loss_term_keys else 0
    n_bc_domain = get_sampling_count_from_config(cfg, "n_points_bc_domain") if 'bc' in active_loss_term_keys else 0
    n_bc_per_wall = get_boundary_segment_count(cfg, n_bc_domain) if n_bc_domain > 0 else 0
    
    # Calculate available batches per term
    num_batches = calculate_num_batches(
        batch_size,
        [
            n_pde,
            n_ic,
            n_bc_per_wall,
            n_bc_per_wall,
            n_bc_per_wall,
            n_bc_per_wall,
        ],
        data_points_full,
        data_free=data_free,
    )
    
    if num_batches == 0:
        print(f"Error: Batch size {batch_size} is too large for configured sample counts or data. No training will occur.")
        return -1.0

    # --- 3. Setup Optimizer (after num_batches is known for accumulation_factor) ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # --- Define JIT Data Generator ---
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, data_key = random.split(key, 5)
        x_range = (0., domain_cfg["lx"])
        y_range = (0., domain_cfg["ly"])
        t_range = (0., domain_cfg["t_final"])

        pde_data = sample_and_batch(pde_key, sample_domain, n_pde, batch_size, num_batches, x_range, y_range, t_range)
        ic_data = sample_and_batch(ic_key, sample_domain, n_ic, batch_size, num_batches, x_range, y_range, (0., 0.))

        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        bc_data = {
            'left':   sample_and_batch(l_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (0., 0.), y_range, t_range),
            'right':  sample_and_batch(r_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range),
            'bottom': sample_and_batch(b_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (0., 0.), t_range),
            'top':    sample_and_batch(t_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range),
        }

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc': bc_data,
            'data': maybe_batch_data(data_key, data_points_full, batch_size, num_batches, data_free),
            'building_bc': {},
        }

    generate_epoch_data_jit = jax.jit(generate_epoch_data)

    # --- Define Scan Body Function ---
    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights_dict, cfg, data_free,
        compute_losses_fn=compute_losses,
    )

    def validation_fn(model, params):
        nse_val, rmse_val = -jnp.inf, jnp.inf
        metrics = {}
        if validation_data_loaded:
            try:
                U_pred_val = model.apply({'params': params['params']}, val_points, train=False)
                min_depth_val = cfg.get("numerics", {}).get("min_depth", 0.0)
                U_pred_val = _apply_min_depth(U_pred_val, min_depth_val)
                h_pred_val = U_pred_val[..., 0]
                nse_val = float(nse(h_pred_val, h_true_val))
                rmse_val = float(rmse(h_pred_val, h_true_val))
                metrics = {'nse_h': nse_val, 'rmse_h': rmse_val}
                if hu_true_val is not None and hv_true_val is not None:
                    metrics['nse_hu'] = float(nse(U_pred_val[..., 1], hu_true_val))
                    metrics['rmse_hu'] = float(rmse(U_pred_val[..., 1], hu_true_val))
                    metrics['nse_hv'] = float(nse(U_pred_val[..., 2], hv_true_val))
                    metrics['rmse_hv'] = float(rmse(U_pred_val[..., 2], hv_true_val))
            except Exception as exc:
                print(f"Warning: Validation calculation failed: {exc}")
        if not metrics:
            metrics = {'nse_h': float(nse_val), 'rmse_h': float(rmse_val)}
        return metrics

    # --- Evaluate All Physics Losses (including zero-weight terms) ---
    n_eval = 200
    def compute_all_losses_fn(model, params):
        eval_key = random.PRNGKey(0)
        keys = random.split(eval_key, 5)
        x_range = (0., domain_cfg["lx"])
        y_range = (0., domain_cfg["ly"])
        t_range = (0., domain_cfg["t_final"])
        batch = {
            'pde': sample_domain(keys[0], n_eval, x_range, y_range, t_range),
            'ic': sample_domain(keys[1], n_eval, x_range, y_range, (0., 0.)),
            'bc': {
                'left': sample_domain(keys[2], n_eval, (0., 0.), y_range, t_range),
                'right': sample_domain(keys[2], n_eval, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range),
                'bottom': sample_domain(keys[3], n_eval, x_range, (0., 0.), t_range),
                'top': sample_domain(keys[3], n_eval, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range),
            },
            'data': jnp.empty((0, 6), dtype=get_dtype()),
            'building_bc': {},
        }
        return compute_losses(model, params, batch, cfg, data_free=True)

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
        experiment_name=experiment_name,
        trial_name=trial_name,
        results_dir=results_dir,
        model_dir=model_dir,
        config_path=config_path,
        validation_fn=validation_fn,
        source_script_path=__file__,
        compute_all_losses_fn=compute_all_losses_fn,
    )

    def plot_fn(final_params):
        print("  Generating 1D validation plot...")
        aim_tracker = loop_result["aim_tracker"]
        final_epoch = loop_result["epoch"]
        plot_cfg = cfg.get("plotting", {})
        min_depth_plot = cfg.get("numerics", {}).get("min_depth", 0.0)
        t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
        nx_val_plot = plot_cfg.get("nx_val", 101)
        y_const_plot = plot_cfg.get("y_const_plot", 0.0)
        x_val_plot = jnp.linspace(0.0, cfg["domain"]["lx"], nx_val_plot, dtype=get_dtype())
        plot_points_1d = jnp.stack([
            x_val_plot,
            jnp.full_like(x_val_plot, y_const_plot, dtype=get_dtype()),
            jnp.full_like(x_val_plot, t_const_val_plot, dtype=get_dtype()),
        ], axis=1)
        U_plot_pred_1d = model.apply({'params': final_params['params']}, plot_points_1d, train=False)
        U_plot_pred_1d = _apply_min_depth(U_plot_pred_1d, min_depth_plot)
        h_plot_pred_1d = U_plot_pred_1d[..., 0]
        plot_path_1d = os.path.join(results_dir, "final_validation_plot.png")
        plot_h_vs_x(x_val_plot, h_plot_pred_1d, t_const_val_plot, y_const_plot, cfg_dict, plot_path_1d)
        aim_tracker.log_image(plot_path_1d, 'validation_plot_1D', final_epoch)
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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Analytical Scenario).")
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