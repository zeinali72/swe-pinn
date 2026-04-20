"""Experiment 1 — Analytical dam-break with non-dimensionalized pipeline.

Isolated non-dim variant of ``experiments.experiment_1.train``. Uses
``SWEScaler`` to transform the SWE into a dimensionless system:

- Inputs (x, y, t) sampled in non-dim ranges and fed directly to the network.
- Model is initialised with a config whose ``domain.lx/ly/t_final`` are already
  scaled, so the internal ``Normalize`` layer maps the non-dim domain to
  ``[-1, 1]`` without double-scaling.
- PDE loss reads ``physics.Cf`` from a scaled-physics FrozenDict (g=1, n=0).
- BC Dirichlet targets are computed dimensionally via the analytical solution
  (using original ``n_manning`` / ``u_const`` preserved under
  ``physics.dimensional``) and scaled before entering the loss.
- Analytical gauge data is generated dimensionally, then coordinates + targets
  are scaled.
- Validation sampled dimensionally, inputs scaled before the forward pass,
  predictions unscaled to SI units before metrics.

When ``scaling.enabled: false`` the scaler is an identity and the script
reduces to the dimensional baseline.
"""

import os
import sys
import argparse
import copy
import functools

import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict

from src.config import load_config, get_dtype
from src.predict.predictor import _apply_min_depth
from src.data import sample_domain
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_data_loss, compute_neg_h_loss,
    loss_boundary_dirichlet, loss_boundary_neumann_outflow_x,
    loss_boundary_wall_horizontal,
)
from src.utils import nse, rmse, relative_l2, plot_h_vs_x
from src.physics import h_exact, hu_exact, hv_exact
from src.physics.scaling import SWEScaler
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
    maybe_batch_data,
    post_training_save,
    resolve_data_mode,
    run_training_loop,
    create_output_dirs,
)


def compute_losses(model, params, batch, config, data_free, scaler=None):
    """Compute all loss terms for Experiment 1 in non-dim space.

    ``config`` is expected to be the FrozenDict returned by
    ``SWEScaler.nondim_physics_config`` so the PDE loss reads ``physics.Cf``,
    ``physics.g=1``, ``physics.n_manning=0`` and the original dimensional
    values are recoverable under ``physics.dimensional``. When ``scaler`` is
    ``None`` or identity-mode, the function degrades to the dimensional path.
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

        # Pull original dimensional physics values. nondim_physics_config stores
        # these under physics.dimensional; dimensional mode keeps them at the
        # top level of physics.
        physics_cfg = config["physics"]
        dim_physics = physics_cfg.get("dimensional", physics_cfg)
        u_dim = dim_physics.get("u_const", physics_cfg.get("u_const"))
        n_dim = dim_physics.get("n_manning", physics_cfg.get("n_manning"))

        # Recover dimensional time for the analytical solution
        t_left_dim = left[..., 2] * scaler.T0 if scaler is not None else left[..., 2]
        h_true_dim = h_exact(0.0, t_left_dim, n_dim, u_dim)
        hu_true_dim = h_true_dim * u_dim
        if scaler is not None:
            h_target = h_true_dim / scaler.H0
            hu_target = hu_true_dim / scaler.HU0
        else:
            h_target = h_true_dim
            hu_target = hu_true_dim

        loss_left = (loss_boundary_dirichlet(model, params, left, h_target, var_idx=0) +
                     loss_boundary_dirichlet(model, params, left, hu_target, var_idx=1))
        # Right: Neumann outflow (scale-invariant)
        loss_right = loss_boundary_neumann_outflow_x(model, params, right)
        # Top/Bottom: horizontal wall (scale-invariant)
        loss_bottom = loss_boundary_wall_horizontal(model, params, bottom)
        loss_top = loss_boundary_wall_horizontal(model, params, top)
        terms['bc'] = loss_left + loss_right + loss_bottom + loss_top

    data_batch_data = batch.get('data', jnp.empty((0, 6), dtype=get_dtype()))
    if not data_free and data_batch_data.shape[0] > 0:
        terms['data'] = compute_data_loss(model, params, data_batch_data, config)

    return terms


def setup_trial(cfg_dict: dict, hpo_mode: bool = False) -> dict:
    """Set up the non-dim Experiment 1 trial.

    Returns the standard training-context dict plus ``scaler`` for downstream
    plotting / inference.
    """
    cfg = FrozenDict(cfg_dict)
    experiment_name = get_experiment_name(cfg_dict, "experiment_1")

    # --- Build the scaler and non-dim physics config ---
    scaler = SWEScaler(cfg)
    print(scaler.summary())
    nondim_cfg = scaler.nondim_physics_config(cfg_dict)

    # Build a model-init config with SCALED domain bounds so the internal
    # Normalize layer operates on dimensionless coordinates instead of
    # double-scaling already-scaled inputs.
    model_init_cfg_dict = copy.deepcopy(cfg_dict)
    model_init_cfg_dict["domain"]["lx"] = cfg_dict["domain"]["lx"] / scaler.L0
    model_init_cfg_dict["domain"]["ly"] = cfg_dict["domain"]["ly"] / scaler.L0
    model_init_cfg_dict["domain"]["t_final"] = cfg_dict["domain"]["t_final"] / scaler.T0
    model_init_cfg = FrozenDict(model_init_cfg_dict)

    model, params, train_key, val_key = init_model_from_config(model_init_cfg)

    print("Info: Running Experiment 1 in NON-DIM analytical mode.")

    static_weights_dict, _ = extract_loss_weights(cfg)

    # --- Validation data: sample dimensionally, scale inputs, keep targets SI ---
    val_points, h_true_val, hu_true_val, hv_true_val = None, None, None, None
    validation_data_loaded = False
    try:
        val_grid_cfg = cfg["validation_grid"]
        domain_cfg = cfg["domain"]
        print("Creating analytical validation set from 'validation_grid' config...")

        val_points_dim = sample_domain(
            val_key,
            val_grid_cfg["n_points_val"],
            (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
        )
        n_manning = cfg["physics"]["n_manning"]
        u_const = cfg["physics"]["u_const"]
        h_true_val = h_exact(val_points_dim[:, 0], val_points_dim[:, 2], n_manning, u_const)
        hu_true_val = hu_exact(val_points_dim[:, 0], val_points_dim[:, 2], n_manning, u_const)
        hv_true_val = hv_exact(val_points_dim[:, 0], val_points_dim[:, 2], n_manning, u_const)

        # Network inputs must be non-dim; metrics stay in SI
        val_points = scaler.scale_inputs(val_points_dim)

        if val_points.shape[0] > 0:
            validation_data_loaded = True
            print(f"Created analytical validation set with {val_points.shape[0]} points.")
        else:
            print("Warning: Analytical validation set is empty.")
    except KeyError:
        print("Warning: 'validation_grid' not found in config. Skipping NSE/RMSE calculation.")
    except Exception as e:
        print(f"Warning: Error creating analytical validation set: {e}. Skipping NSE/RMSE.")

    # --- Data mode resolution ---
    data_points_full = None
    data_free, has_data_loss = resolve_data_mode(cfg)

    if not data_free:
        try:
            train_grid_cfg = cfg["train_grid"]
            domain_cfg = cfg["domain"]
            print("Creating analytical training dataset from 'train_grid' config...")

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

            print(f"Gauge-based sampling: {n_gauges} gauges x {n_timesteps} timesteps "
                  f"(dt={dt_data}s) = {data_points_coords.shape[0]} data points")

            # Dimensional analytical targets
            h_true_train = h_exact(
                data_points_coords[:, 0],
                data_points_coords[:, 2],
                cfg["physics"]["n_manning"],
                cfg["physics"]["u_const"],
            )
            u_true_train = jnp.full_like(h_true_train, cfg["physics"]["u_const"])
            v_true_train = jnp.zeros_like(h_true_train)

            # Scale coordinates and conservative targets, then recover primitive
            # velocities from the scaled conservative quantities so the data
            # loss (which compares [h, u, v]) sees a self-consistent triplet.
            coords_scaled = scaler.scale_inputs(data_points_coords)
            h_sc, hu_sc, hv_sc = scaler.scale_outputs(
                h_true_train,
                h_true_train * u_true_train,
                h_true_train * v_true_train,
            )
            eps_safe = 1e-12
            u_sc = hu_sc / jnp.maximum(h_sc, eps_safe)
            v_sc = hv_sc / jnp.maximum(h_sc, eps_safe)

            data_points_full = jnp.stack([
                coords_scaled[:, 2],  # t*
                coords_scaled[:, 0],  # x*
                coords_scaled[:, 1],  # y*
                h_sc,
                u_sc,
                v_sc,
            ], axis=1).astype(get_dtype())

            if data_points_full.shape[0] == 0:
                print("Warning: Analytical training data is empty. Disabling data loss.")
                data_points_full = None
                has_data_loss = False
            else:
                print(f"Created {data_points_full.shape[0]} points for data loss term "
                      f"(weight={static_weights_dict.get('data', 0.0):.2e}).")
        except KeyError:
            print("Error: 'data_free: false' but 'train_grid' not found. Disabling data loss.")
            has_data_loss = False
            data_free = True
        except Exception as e:
            print(f"Error creating analytical training data: {e}. Disabling data loss.")
            has_data_loss = False
            data_free = True

    # --- Active loss terms ---
    current_weights_dict = get_active_loss_weights(
        static_weights_dict,
        data_free=data_free,
        excluded_keys={"building_bc"},
    )
    active_loss_term_keys = list(current_weights_dict.keys())

    # --- Batch counts ---
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]

    n_pde = get_sampling_count_from_config(cfg, "n_points_pde") if ('pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys) else 0
    n_ic = get_sampling_count_from_config(cfg, "n_points_ic") if 'ic' in active_loss_term_keys else 0
    n_bc_domain = get_sampling_count_from_config(cfg, "n_points_bc_domain") if 'bc' in active_loss_term_keys else 0
    n_bc_per_wall = get_boundary_segment_count(cfg, n_bc_domain) if n_bc_domain > 0 else 0

    num_batches = calculate_num_batches(
        batch_size,
        [n_pde, n_ic, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall],
        data_points_full,
        data_free=data_free,
    )

    if num_batches == 0:
        raise ValueError(
            f"Batch size {batch_size} is too large for configured sample counts or data."
        )

    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # --- Non-dim sampling ranges (computed once) ---
    x_range_s = scaler.scale_range(0., domain_cfg["lx"], "x")
    y_range_s = scaler.scale_range(0., domain_cfg["ly"], "y")
    t_range_s = scaler.scale_range(0., domain_cfg["t_final"], "t")
    x_left_s = scaler.scale_range(0., 0., "x")
    x_right_s = scaler.scale_range(domain_cfg["lx"], domain_cfg["lx"], "x")
    y_bottom_s = scaler.scale_range(0., 0., "y")
    y_top_s = scaler.scale_range(domain_cfg["ly"], domain_cfg["ly"], "y")

    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, data_key = random.split(key, 5)

        pde_data = sample_and_batch(pde_key, sample_domain, n_pde, batch_size, num_batches,
                                    x_range_s, y_range_s, t_range_s)
        ic_data = sample_and_batch(ic_key, sample_domain, n_ic, batch_size, num_batches,
                                   x_range_s, y_range_s, (0., 0.))

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

    # Bind the scaler into compute_losses so scan_body's signature is unchanged.
    _losses_fn = functools.partial(compute_losses, scaler=scaler)
    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights_dict, nondim_cfg, data_free,
        compute_losses_fn=_losses_fn,
    )

    def validation_fn(model, params):
        nse_val, rmse_val = -jnp.inf, jnp.inf
        metrics = {}
        if validation_data_loaded:
            try:
                U_pred_nd = model.apply({'params': params['params']}, val_points, train=False)
                U_pred_dim = scaler.unscale_output_array(U_pred_nd)
                min_depth_val = cfg.get("numerics", {}).get("min_depth", 0.0)
                U_pred_dim = _apply_min_depth(U_pred_dim, min_depth_val)
                h_pred = U_pred_dim[..., 0]
                nse_val = float(nse(h_pred, h_true_val))
                rmse_val = float(rmse(h_pred, h_true_val))
                metrics = {
                    'nse_h': nse_val,
                    'rmse_h': rmse_val,
                    'rel_l2_h': float(relative_l2(h_pred, h_true_val)),
                }
                if hu_true_val is not None and hv_true_val is not None:
                    metrics['nse_hu'] = float(nse(U_pred_dim[..., 1], hu_true_val))
                    metrics['rmse_hu'] = float(rmse(U_pred_dim[..., 1], hu_true_val))
                    metrics['rel_l2_hu'] = float(relative_l2(U_pred_dim[..., 1], hu_true_val))
                    metrics['nse_hv'] = float(nse(U_pred_dim[..., 2], hv_true_val))
                    metrics['rmse_hv'] = float(rmse(U_pred_dim[..., 2], hv_true_val))
                    metrics['rel_l2_hv'] = float(relative_l2(U_pred_dim[..., 2], hv_true_val))
            except Exception as exc:
                print(f"Warning: Validation calculation failed: {exc}")
        if not metrics:
            metrics = {'nse_h': float(nse_val), 'rmse_h': float(rmse_val)}
        return metrics

    n_eval = 200

    def compute_all_losses_fn(model, params):
        eval_key = random.PRNGKey(0)
        keys = random.split(eval_key, 6)
        batch = {
            'pde': sample_domain(keys[0], n_eval, x_range_s, y_range_s, t_range_s),
            'ic': sample_domain(keys[1], n_eval, x_range_s, y_range_s, (0., 0.)),
            'bc': {
                'left': sample_domain(keys[2], n_eval, x_left_s, y_range_s, t_range_s),
                'right': sample_domain(keys[3], n_eval, x_right_s, y_range_s, t_range_s),
                'bottom': sample_domain(keys[4], n_eval, x_range_s, y_bottom_s, t_range_s),
                'top': sample_domain(keys[5], n_eval, x_range_s, y_top_s, t_range_s),
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
        "experiment_name": experiment_name,
        "validation_data_loaded": validation_data_loaded,
        "val_points_all": val_points,
        "h_true_val_all": h_true_val,
        "val_targets_all": None,
    }


def main(config_path: str):
    """Non-dim Experiment 1 training entry point."""
    cfg_dict = load_config(config_path)
    ctx = setup_trial(cfg_dict)

    experiment_name = ctx["experiment_name"]
    trial_name, results_dir, model_dir = create_output_dirs(ctx["cfg"], experiment_name)

    model = ctx["model"]
    cfg = ctx["cfg"]
    scaler = ctx["scaler"]

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

    def plot_fn(final_params):
        print("  Generating 1D validation plot...")
        tracker = loop_result["tracker"]
        plot_cfg = cfg.get("plotting", {})
        min_depth_plot = cfg.get("numerics", {}).get("min_depth", 0.0)
        t_const_val_plot = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
        nx_val_plot = plot_cfg.get("nx_val", 101)
        y_const_plot = plot_cfg.get("y_const_plot", 0.0)

        # Build the line in dimensional space, then scale for the network.
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
        plot_h_vs_x(x_val_dim, h_plot_pred_1d, t_const_val_plot, y_const_plot,
                    ctx["cfg_dict"], plot_path_1d)
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
    parser = argparse.ArgumentParser(description="Non-dim PINN training for Experiment 1 (analytical).")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to path: {project_root}")

    try:
        final_nse = main(args.config)
        print("\n--- Script Finished ---")
        if isinstance(final_nse, (float, int)) and final_nse > -jnp.inf:
            print(f"Final best NSE reported: {final_nse:.6f}")
        else:
            print(f"Final best NSE value invalid or not achieved: {final_nse}")
        print("-----------------------")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the config file path.")
    except ValueError as e:
        print(f"Configuration or Model Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
