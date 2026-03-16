"""Experiment 2 — Dam-break with building obstacle (Phase 1).

Introduces a building obstacle; motivates Fourier-MLP and DGM adoption.
Requires: configs/experiment_2_*.yaml, data/experiment_2/
Builds on: Experiment 1.

This script handles training for scenarios with building
structures. It supports static loss weighting and provides
comprehensive logging and result visualization through Aim.

This is derived from the unified 'src/train.py'.
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to path: {project_root}")

from src.config import DTYPE
from src.predict.predictor import _apply_min_depth
from src.data import sample_domain, get_batches_tensor, load_validation_data
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss,
    compute_building_bc_loss, compute_data_loss, compute_neg_h_loss
)
from src.utils import (
    nse, rmse, mask_points_inside_building,
    plot_comparison_scatter_2d
)
from src.training import (
    create_optimizer,
    calculate_num_batches,
    extract_loss_weights,
    get_active_loss_weights,
    get_boundary_segment_count,
    get_data_filename,
    get_sampling_count_from_config,
    load_training_data,
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

# To enable 64-bit precision, uncomment the following line:
# jax.config.update('jax_enable_x64', True)


def compute_losses(model, params, batch, config, data_free):
    """Compute all loss terms for Experiment 2 (building obstacle)."""
    terms = {}

    pde_batch_data = batch.get('pde', jnp.empty((0, 3), dtype=DTYPE))
    if pde_batch_data.shape[0] > 0:
        pde_mask = mask_points_inside_building(pde_batch_data, config["building"])
        terms['pde'] = compute_pde_loss(model, params, pde_batch_data, config, pde_mask)
        terms['neg_h'] = compute_neg_h_loss(model, params, pde_batch_data, pde_mask)

    ic_batch_data = batch.get('ic', jnp.empty((0, 3), dtype=DTYPE))
    if ic_batch_data.shape[0] > 0:
        terms['ic'] = compute_ic_loss(model, params, ic_batch_data)

    bc_batches = batch.get('bc', {})
    if any(b.shape[0] > 0 for b in bc_batches.values() if hasattr(b, 'shape')):
        terms['bc'] = compute_bc_loss(
            model, params,
            bc_batches.get('left', jnp.empty((0, 3), dtype=DTYPE)),
            bc_batches.get('right', jnp.empty((0, 3), dtype=DTYPE)),
            bc_batches.get('bottom', jnp.empty((0, 3), dtype=DTYPE)),
            bc_batches.get('top', jnp.empty((0, 3), dtype=DTYPE)),
            config,
        )

    bldg_batches = batch.get('building_bc', {})
    if bldg_batches and any(b.shape[0] > 0 for b in bldg_batches.values() if hasattr(b, 'shape')):
        terms['building_bc'] = compute_building_bc_loss(
            model, params,
            bldg_batches.get('left', jnp.empty((0, 3), dtype=DTYPE)),
            bldg_batches.get('right', jnp.empty((0, 3), dtype=DTYPE)),
            bldg_batches.get('bottom', jnp.empty((0, 3), dtype=DTYPE)),
            bldg_batches.get('top', jnp.empty((0, 3), dtype=DTYPE)),
        )

    data_batch_data = batch.get('data', jnp.empty((0, 6), dtype=DTYPE))
    if not data_free and data_batch_data.shape[0] > 0:
        terms['data'] = compute_data_loss(model, params, data_batch_data, config)

    return terms


def main(config_path: str):
    """
    Main training function for the BUILDING scenario.
    """
    # --- 1. Load Config and Initialize Model ---
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
    
    # --- BUILDING SCRIPT ASSERTION ---
    has_building = "building" in cfg
    if not has_building:
        print(f"Error: This script ('{__file__}') is for 'building' scenarios only.")
        print(f"Config '{config_path}' is missing the 'building' section.")
        print("Please use 'src/scenarios/analytical/analytical.py' for this config.")
        sys.exit(1)
    print("Info: Running in building mode.")
    # --- END ASSERTION ---

    # --- 4. Prepare Loss Weights ---
    static_weights_dict, _ = extract_loss_weights(cfg)

    # --- 5. Load Validation and Training Data ---
    val_points, h_true_val = None, None
    data_points_full = None
    
    try:
        experiment_paths = resolve_experiment_paths(cfg, experiment_name, require_scenario=True)
    except ValueError as exc:
        print(f"Error: {exc} in config '{config_path}'.")
        sys.exit(1)

    base_data_path = experiment_paths["base_data_path"]

    # === START MODIFIED BLOCK ===
    # This logic now mirrors analytical.py
    data_free, has_data_loss = resolve_data_mode(cfg)
    data_points_full, has_data_loss, data_free = load_training_data(
        base_data_path,
        has_data_loss,
        static_weights_dict,
        filename=get_data_filename(cfg, "training_file", "training_dataset_sample.npy"),
    )

    validation_data_file = os.path.join(
        base_data_path,
        get_data_filename(cfg, "validation_file", "validation_sample.npy"),
    )
    validation_data_loaded = False
    val_hu_true = None
    val_hv_true = None
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            _, val_points_all, val_targets_all = load_validation_data(validation_data_file, dtype=DTYPE)
            h_true_val_all = val_targets_all[:, 0]
            print("Applying building mask to validation metrics points...")
            mask_val = mask_points_inside_building(val_points_all, cfg["building"])
            val_points = val_points_all[mask_val]
            h_true_val = h_true_val_all[mask_val]
            if val_targets_all.shape[-1] >= 3:
                val_hu_true = (val_targets_all[:, 0] * val_targets_all[:, 1])[mask_val]
                val_hv_true = (val_targets_all[:, 0] * val_targets_all[:, 2])[mask_val]
            num_masked_val_points = val_points.shape[0]
            print(f"Masked validation metrics points remaining: {num_masked_val_points}.")
            if num_masked_val_points > 0:
                validation_data_loaded = True
            else:
                 print("Warning: No validation points remaining after masking. NSE/RMSE calculation will be skipped.")
        except Exception as e:
            print(f"Error loading or processing validation data file {validation_data_file}: {e}")
            val_points, h_true_val = None, None
            print("NSE/RMSE calculation using loaded data will be skipped.")
    else:
        print(f"Warning: Validation data file not found at {validation_data_file}.")
        print("Validation metrics (NSE/RMSE) for building scenario will be skipped.")

    # --- 6. Determine Active Loss Terms for the Run ---
    current_weights_dict = get_active_loss_weights(static_weights_dict, data_free=data_free)
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
    
    # Building BC points
    n_bldg_per_wall = 0
    if has_building and 'building_bc' in active_loss_term_keys:
        n_bldg = get_sampling_count_from_config(cfg, "n_points_bc_building")
        n_bldg_per_wall = get_boundary_segment_count(cfg, n_bldg)

    # Calculate available batches per term
    sample_sizes = [n_pde, n_ic, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall]
    if has_building and 'building_bc' in active_loss_term_keys:
        sample_sizes.extend([n_bldg_per_wall, n_bldg_per_wall, n_bldg_per_wall, n_bldg_per_wall])

    num_batches = calculate_num_batches(batch_size, sample_sizes, data_points_full, data_free=data_free)
    
    if num_batches == 0:
        print(f"Error: Batch size {batch_size} is too large for configured sample counts or data. No training will occur.")
        return -1.0
    print(f"Calculated number of batches per epoch: {num_batches}")

    # --- 3. Setup Optimizer (after num_batches is known for accumulation_factor) ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # --- Define JIT Data Generator ---
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, bldg_keys, data_key = random.split(key, 6)
        x_range = (0., domain_cfg["lx"])
        y_range = (0., domain_cfg["ly"])
        t_range = (0., domain_cfg["t_final"])

        pde_data = sample_and_batch(pde_key, sample_domain, n_pde, batch_size, num_batches, x_range, y_range, t_range)
        ic_data = sample_and_batch(ic_key, sample_domain, n_ic, batch_size, num_batches, x_range, y_range, (0., 0.))

        # Domain BCs
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        bc_data = {
            'left':   sample_and_batch(l_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (0., 0.), y_range, t_range),
            'right':  sample_and_batch(r_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range),
            'bottom': sample_and_batch(b_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (0., 0.), t_range),
            'top':    sample_and_batch(t_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range),
        }

        # Building BCs
        building_bc_data = {}
        if has_building and 'building_bc' in active_loss_term_keys:
             bldg_l_key, bldg_r_key, bldg_b_key, bldg_t_key = random.split(bldg_keys, 4)
             b_cfg = cfg["building"]
             building_bc_data['left']   = sample_and_batch(bldg_l_key, sample_domain, n_bldg_per_wall, batch_size, num_batches, (b_cfg["x_min"], b_cfg["x_min"]), (b_cfg["y_min"], b_cfg["y_max"]), t_range)
             building_bc_data['right']  = sample_and_batch(bldg_r_key, sample_domain, n_bldg_per_wall, batch_size, num_batches, (b_cfg["x_max"], b_cfg["x_max"]), (b_cfg["y_min"], b_cfg["y_max"]), t_range)
             building_bc_data['bottom'] = sample_and_batch(bldg_b_key, sample_domain, n_bldg_per_wall, batch_size, num_batches, (b_cfg["x_min"], b_cfg["x_max"]), (b_cfg["y_min"], b_cfg["y_min"]), t_range)
             building_bc_data['top']    = sample_and_batch(bldg_t_key, sample_domain, n_bldg_per_wall, batch_size, num_batches, (b_cfg["x_min"], b_cfg["x_max"]), (b_cfg["y_max"], b_cfg["y_max"]), t_range)

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc': bc_data,
            'data': maybe_batch_data(data_key, data_points_full, batch_size, num_batches, data_free),
            'building_bc': building_bc_data,
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
                if val_hu_true is not None and val_hv_true is not None:
                    metrics['nse_hu'] = float(nse(U_pred_val[..., 1], val_hu_true))
                    metrics['rmse_hu'] = float(rmse(U_pred_val[..., 1], val_hu_true))
                    metrics['nse_hv'] = float(nse(U_pred_val[..., 2], val_hv_true))
                    metrics['rmse_hv'] = float(rmse(U_pred_val[..., 2], val_hv_true))
            except Exception as exc:
                print(f"Warning: Validation calculation failed: {exc}")
        if not metrics:
            metrics = {'nse_h': float(nse_val), 'rmse_h': float(rmse_val)}
        return metrics

    # --- Evaluate All Physics Losses (including zero-weight terms) ---
    n_eval = 200
    def compute_all_losses_fn(model, params):
        eval_key = random.PRNGKey(0)
        keys = random.split(eval_key, 6)
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
            'data': jnp.empty((0, 6), dtype=DTYPE),
            'building_bc': {},
        }
        if has_building:
            b_cfg = cfg["building"]
            batch['building_bc'] = {
                'left': sample_domain(keys[4], n_eval, (b_cfg["x_min"], b_cfg["x_min"]), (b_cfg["y_min"], b_cfg["y_max"]), t_range),
                'right': sample_domain(keys[4], n_eval, (b_cfg["x_max"], b_cfg["x_max"]), (b_cfg["y_min"], b_cfg["y_max"]), t_range),
                'bottom': sample_domain(keys[5], n_eval, (b_cfg["x_min"], b_cfg["x_max"]), (b_cfg["y_min"], b_cfg["y_min"]), t_range),
                'top': sample_domain(keys[5], n_eval, (b_cfg["x_min"], b_cfg["x_max"]), (b_cfg["y_max"], b_cfg["y_max"]), t_range),
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
        print("  Generating 2D comparison plots...")
        aim_tracker = loop_result["aim_tracker"]
        final_epoch = loop_result["epoch"]
        plot_cfg = cfg.get("plotting", {})
        eps_plot = cfg.get("numerics", {}).get("eps", 1e-6)
        t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
        plot_data_time = t_const_val_plot
        plot_data_file = os.path.join(base_data_path, f"validation_plotting_t_{int(plot_data_time)}s.npy")
        if not os.path.exists(plot_data_file):
            print(f"  Warning: Plotting data file {plot_data_file} not found. Skipping comparison plot.")
            return

        plot_data = np.load(plot_data_file)
        plot_points_scatter = jnp.array(plot_data[:, [1, 2, 0]], dtype=DTYPE)
        x_coords_plot = jnp.array(plot_data[:, 1], dtype=DTYPE)
        y_coords_plot = jnp.array(plot_data[:, 2], dtype=DTYPE)
        h_true_plot = jnp.array(plot_data[:, 3], dtype=DTYPE)
        u_true_plot = jnp.array(plot_data[:, 4], dtype=DTYPE)
        v_true_plot = jnp.array(plot_data[:, 5], dtype=DTYPE)
        h_true_safe = jnp.maximum(h_true_plot, eps_plot)
        hu_true_plot = h_true_safe * u_true_plot
        hv_true_plot = h_true_safe * v_true_plot
        U_plot_pred_scatter = model.apply({'params': final_params['params']}, plot_points_scatter, train=False)
        min_depth_plot = cfg.get("numerics", {}).get("min_depth", 0.0)
        dry_mask = jnp.where(U_plot_pred_scatter[..., 0] >= min_depth_plot, 1.0, 0.0)
        U_plot_pred_scatter = U_plot_pred_scatter * dry_mask[..., None]
        h_pred_plot = U_plot_pred_scatter[..., 0]
        hu_pred_plot = U_plot_pred_scatter[..., 1]
        hv_pred_plot = U_plot_pred_scatter[..., 2]

        plot_path_h = os.path.join(results_dir, f"final_comparison_plot_t{int(plot_data_time)}s_h.png")
        plot_path_hu = os.path.join(results_dir, f"final_comparison_plot_t{int(plot_data_time)}s_hu.png")
        plot_path_hv = os.path.join(results_dir, f"final_comparison_plot_t{int(plot_data_time)}s_hv.png")
        plot_comparison_scatter_2d(x_coords_plot, y_coords_plot, h_pred_plot, h_true_plot, 'h', cfg_dict, plot_path_h)
        plot_comparison_scatter_2d(x_coords_plot, y_coords_plot, hu_pred_plot, hu_true_plot, 'hu', cfg_dict, plot_path_hu)
        plot_comparison_scatter_2d(x_coords_plot, y_coords_plot, hv_pred_plot, hv_true_plot, 'hv', cfg_dict, plot_path_hv)
        aim_tracker.log_image(plot_path_h, 'validation_plot_h', final_epoch)
        aim_tracker.log_image(plot_path_hu, 'validation_plot_hu', final_epoch)
        aim_tracker.log_image(plot_path_hv, 'validation_plot_hv', final_epoch)
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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Building Scenario).")
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