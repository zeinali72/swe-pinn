"""Experiment 1 — Analytical dam-break on flat domain (Phase 1).

Verifies the PINN framework against an analytical dam-break solution.
Requires: configs/experiment_1_*.yaml
Builds on: None (baseline verification).

This script handles both data-free and data-driven (analytical) training modes.
It supports static loss weighting and provides
logging and result visualization through W&B.
"""

import os
import sys
import argparse
import functools
import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict
import numpy as np

# Local application imports
from src.config import load_config, get_dtype
from src.predict.predictor import _apply_min_depth
from src.data import sample_domain, get_batches_tensor
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_data_loss, compute_neg_h_loss,
    loss_boundary_dirichlet, loss_boundary_neumann_outflow_x,
    loss_boundary_wall_horizontal,
)
from src.utils import nse, rmse, relative_l2, plot_h_vs_x
from src.physics import h_exact, hu_exact, hv_exact, SWEScaler
from src.training import (
    create_optimizer,
    calculate_num_batches,
    extract_loss_weights,
    get_active_loss_weights,
    get_boundary_segment_count,
    get_experiment_name,
    get_sampling_count_from_config,
    init_model_from_config,
    train_step_jitted,
    make_scan_body,
    sample_and_batch,
    empty_batch,
    maybe_batch_data,
    post_training_save,
    resolve_data_mode,
    run_training_loop,
    create_output_dirs,
)

# To enable 64-bit precision, uncomment the following line:
# jax.config.update('jax_enable_x64', True)


def compute_losses(model, params, batch, config, data_free, scaler=None):
    """Compute all loss terms for Experiment 1 (analytical dam-break).

    When ``scaler`` is provided, boundary targets are computed in dimensional
    space from the analytical solution and then scaled to match the network's
    dimensionless output.
    """
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
        # Use original dimensional physics values for the analytical solution
        dim_physics = config["physics"].get("dimensional", config["physics"])
        u_const = dim_physics.get("u_const", config["physics"]["u_const"])
        n_manning = dim_physics.get("n_manning", config["physics"]["n_manning"])
        # Recover dimensional time for the analytical solution
        t_left_dim = left[..., 2] * scaler.T0 if scaler is not None else left[..., 2]
        h_true_dim = h_exact(0.0, t_left_dim, n_manning, u_const)
        hu_true_dim = h_true_dim * u_const
        if scaler is not None:
            h_target = h_true_dim / scaler.H0
            hu_target = hu_true_dim / scaler._HU0
        else:
            h_target = h_true_dim
            hu_target = hu_true_dim
        loss_left = (loss_boundary_dirichlet(model, params, left, h_target, var_idx=0) +
                     loss_boundary_dirichlet(model, params, left, hu_target, var_idx=1))
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


def setup_trial(cfg_dict: dict, hpo_mode: bool = False) -> dict:
    """Set up all training components for Experiment 1 from a config dict.

    Args:
        cfg_dict: Mutable configuration dictionary (not a file path). This is the
            interface used by HPO to pass trial-specific configs directly.

    Returns:
        Dictionary containing all objects needed to call run_training_loop, plus
        production metadata fields (experiment_name, validation_data_loaded, etc.).
    """
    cfg = FrozenDict(cfg_dict)
    experiment_name = get_experiment_name(cfg_dict, "experiment_1")

    # --- Non-dimensionalization ---
    scaler = SWEScaler(cfg)
    nondim_cfg = scaler.nondim_physics_config(cfg_dict)
    print(scaler.summary())

    model, params, train_key, val_key = init_model_from_config(cfg)

    print("Info: Running in analytical (no-building) mode.")

    # --- Prepare Loss Weights ---
    static_weights_dict, _ = extract_loss_weights(cfg)

    # --- Create Validation Data (Analytical) ---
    val_points, h_true_val, hu_true_val, hv_true_val = None, None, None, None
    validation_data_loaded = False
    try:
        val_grid_cfg = cfg["validation_grid"]
        domain_cfg = cfg["domain"]
        print(f"Creating analytical validation set from 'validation_grid' config...")

        # Sample in dimensional space, then compute analytical targets
        val_points_dim = sample_domain(
            val_key,
            val_grid_cfg["n_points_val"],
            (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
        )
        n_manning = cfg["physics"]["n_manning"]
        u_const = cfg["physics"]["u_const"]
        h_true_dim = h_exact(val_points_dim[:, 0], val_points_dim[:, 2], n_manning, u_const)
        hu_true_dim = hu_exact(val_points_dim[:, 0], val_points_dim[:, 2], n_manning, u_const)
        hv_true_dim = hv_exact(val_points_dim[:, 0], val_points_dim[:, 2], n_manning, u_const)

        # Scale inputs and targets to dimensionless form
        val_points = scaler.scale_inputs(val_points_dim)
        h_true_val, hu_true_val, hv_true_val = scaler.scale_outputs(
            h_true_dim, hu_true_dim, hv_true_dim
        )

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

            n_gauges = train_grid_cfg["n_gauges"]
            dt_data = train_grid_cfg["dt_data"]
            t_final = domain_cfg["t_final"]

            if n_gauges <= 0 or dt_data <= 0:
                raise ValueError(
                    f"Gauge-based sampling requires n_gauges > 0 and dt_data > 0, "
                    f"got n_gauges={n_gauges}, dt_data={dt_data}. "
                    f"Set data_free: true to skip data sampling."
                )

            t_steps = jnp.arange(0., t_final + dt_data * 0.5, dt_data, dtype=get_dtype())
            n_timesteps = t_steps.shape[0]

            gauge_xy = sample_domain(
                train_key,
                n_gauges,
                (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.)
            )[:, :2]

            gauge_xy_rep = jnp.repeat(gauge_xy, n_timesteps, axis=0)
            t_rep = jnp.tile(t_steps, n_gauges).reshape(-1, 1)
            data_points_coords = jnp.hstack([gauge_xy_rep, t_rep])

            print(f"Gauge-based sampling: {n_gauges} gauges x {n_timesteps} timesteps (dt={dt_data}s) = {data_points_coords.shape[0]} data points")

            h_true_train = h_exact(
                data_points_coords[:, 0],
                data_points_coords[:, 2],
                cfg["physics"]["n_manning"],
                cfg["physics"]["u_const"]
            )
            u_true_train = jnp.full_like(h_true_train, cfg["physics"]["u_const"])
            v_true_train = jnp.zeros_like(h_true_train)

            # Scale coordinates and targets to dimensionless form
            coords_scaled = scaler.scale_inputs(data_points_coords)
            h_scaled, hu_scaled, hv_scaled = scaler.scale_outputs(
                h_true_train,
                h_true_train * u_true_train,
                h_true_train * v_true_train,
            )
            # data_loss expects [t, x, y, h, u, v] — recover u*, v* from (hu)*/h*
            eps_safe = 1e-12
            u_scaled = hu_scaled / jnp.maximum(h_scaled, eps_safe)
            v_scaled = hv_scaled / jnp.maximum(h_scaled, eps_safe)
            data_points_full = jnp.stack([
                coords_scaled[:, 2],   # t*
                coords_scaled[:, 0],   # x*
                coords_scaled[:, 1],   # y*
                h_scaled,
                u_scaled,
                v_scaled,
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
            data_free = True
        except Exception as e:
            print(f"Error creating analytical training data: {e}. Disabling data loss.")
            has_data_loss = False
            data_free = True

    # --- Determine Active Loss Terms ---
    current_weights_dict = get_active_loss_weights(
        static_weights_dict,
        data_free=data_free,
        excluded_keys={"building_bc"},
    )
    active_loss_term_keys = list(current_weights_dict.keys())

    # --- Pre-calculate Batch Counts ---
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]

    n_pde = get_sampling_count_from_config(cfg, "n_points_pde") if ('pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys) else 0
    n_ic = get_sampling_count_from_config(cfg, "n_points_ic") if 'ic' in active_loss_term_keys else 0
    n_bc_domain = get_sampling_count_from_config(cfg, "n_points_bc_domain") if 'bc' in active_loss_term_keys else 0
    n_bc_per_wall = get_boundary_segment_count(cfg, n_bc_domain) if n_bc_domain > 0 else 0

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
        raise ValueError(
            f"Batch size {batch_size} is too large for configured sample counts or data."
        )

    # --- Setup Optimizer ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # --- Scaled sampling ranges (dimensionless) ---
    x_range_s = scaler.scale_range(0., domain_cfg["lx"], "x")
    y_range_s = scaler.scale_range(0., domain_cfg["ly"], "y")
    t_range_s = scaler.scale_range(0., domain_cfg["t_final"], "t")
    x_left_s = scaler.scale_range(0., 0., "x")
    x_right_s = scaler.scale_range(domain_cfg["lx"], domain_cfg["lx"], "x")
    y_bottom_s = scaler.scale_range(0., 0., "y")
    y_top_s = scaler.scale_range(domain_cfg["ly"], domain_cfg["ly"], "y")

    # --- Define JIT Data Generator ---
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, data_key = random.split(key, 5)

        pde_data = sample_and_batch(pde_key, sample_domain, n_pde, batch_size, num_batches, x_range_s, y_range_s, t_range_s)
        ic_data = sample_and_batch(ic_key, sample_domain, n_ic, batch_size, num_batches, x_range_s, y_range_s, (0., 0.))

        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        bc_data = {
            'left':   sample_and_batch(l_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_left_s, y_range_s, t_range_s),
            'right':  sample_and_batch(r_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_right_s, y_range_s, t_range_s),
            'bottom': sample_and_batch(b_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range_s, y_bottom_s, t_range_s),
            'top':    sample_and_batch(t_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range_s, y_top_s, t_range_s),
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
    # Bind scaler into the loss function so the generic train_step signature is unchanged.
    _losses_fn = functools.partial(compute_losses, scaler=scaler)
    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights_dict, nondim_cfg, data_free,
        compute_losses_fn=_losses_fn,
    )

    # --- Validation Function ---
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
                metrics = {
                    'nse_h': nse_val,
                    'rmse_h': rmse_val,
                    'rel_l2_h': float(relative_l2(h_pred_val, h_true_val)),
                }
                if hu_true_val is not None and hv_true_val is not None:
                    metrics['nse_hu'] = float(nse(U_pred_val[..., 1], hu_true_val))
                    metrics['rmse_hu'] = float(rmse(U_pred_val[..., 1], hu_true_val))
                    metrics['rel_l2_hu'] = float(relative_l2(U_pred_val[..., 1], hu_true_val))
                    metrics['nse_hv'] = float(nse(U_pred_val[..., 2], hv_true_val))
                    metrics['rmse_hv'] = float(rmse(U_pred_val[..., 2], hv_true_val))
                    metrics['rel_l2_hv'] = float(relative_l2(U_pred_val[..., 2], hv_true_val))
            except Exception as exc:
                print(f"Warning: Validation calculation failed: {exc}")
        if not metrics:
            metrics = {'nse_h': float(nse_val), 'rmse_h': float(rmse_val)}
        return metrics

    # --- Evaluate All Physics Losses ---
    n_eval = 200

    def compute_all_losses_fn(model, params):
        eval_key = random.PRNGKey(0)
        keys = random.split(eval_key, 5)
        batch = {
            'pde': sample_domain(keys[0], n_eval, x_range_s, y_range_s, t_range_s),
            'ic': sample_domain(keys[1], n_eval, x_range_s, y_range_s, (0., 0.)),
            'bc': {
                'left': sample_domain(keys[2], n_eval, x_left_s, y_range_s, t_range_s),
                'right': sample_domain(keys[2], n_eval, x_right_s, y_range_s, t_range_s),
                'bottom': sample_domain(keys[3], n_eval, x_range_s, y_bottom_s, t_range_s),
                'top': sample_domain(keys[3], n_eval, x_range_s, y_top_s, t_range_s),
            },
            'data': jnp.empty((0, 6), dtype=get_dtype()),
            'building_bc': {},
        }
        return compute_losses(model, params, batch, nondim_cfg, data_free=True, scaler=scaler)

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
        "scaler": scaler,
        # Production extras
        "experiment_name": experiment_name,
        "validation_data_loaded": validation_data_loaded,
        "val_points_all": val_points,
        "h_true_val_all": h_true_val,
        "val_targets_all": None,
    }


def main(config_path: str):
    """Main training function for the analytical scenario."""
    cfg_dict = load_config(config_path)
    ctx = setup_trial(cfg_dict)

    experiment_name = ctx["experiment_name"]
    trial_name, results_dir, model_dir = create_output_dirs(ctx["cfg"], experiment_name)

    model = ctx["model"]
    cfg = ctx["cfg"]

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
        validation_fn=ctx["validation_fn"],
        compute_all_losses_fn=ctx["compute_all_losses_fn"],
    )

    scaler = SWEScaler(cfg)

    def plot_fn(final_params):
        print("  Generating 1D validation plot...")
        tracker = loop_result["tracker"]
        final_epoch = loop_result["epoch"]
        plot_cfg = cfg.get("plotting", {})
        min_depth_plot = cfg.get("numerics", {}).get("min_depth", 0.0)
        t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
        nx_val_plot = plot_cfg.get("nx_val", 101)
        y_const_plot = plot_cfg.get("y_const_plot", 0.0)
        # Build dimensional 1D line, then scale to dimensionless for the network
        x_val_dim = jnp.linspace(0.0, cfg["domain"]["lx"], nx_val_plot, dtype=get_dtype())
        plot_points_dim = jnp.stack([
            x_val_dim,
            jnp.full_like(x_val_dim, y_const_plot, dtype=get_dtype()),
            jnp.full_like(x_val_dim, t_const_val_plot, dtype=get_dtype()),
        ], axis=1)
        plot_points_nd = scaler.scale_inputs(plot_points_dim)
        U_plot_nd = model.apply({'params': final_params['params']}, plot_points_nd, train=False)
        U_plot_dim = scaler.unscale_output_array(U_plot_nd)
        U_plot_dim = _apply_min_depth(U_plot_dim, min_depth_plot)
        h_plot_pred_1d = U_plot_dim[..., 0]
        plot_path_1d = os.path.join(results_dir, "final_validation_plot.png")
        plot_h_vs_x(x_val_dim, h_plot_pred_1d, t_const_val_plot, y_const_plot, ctx["cfg_dict"], plot_path_1d)
        tracker.log_image(plot_path_1d, 'validation_plot_1D')
        print(f"Model and plot saved in {model_dir} and {results_dir}")

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
