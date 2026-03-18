"""Experiment 1 — ReLoBRaLo adaptive loss balancing.

Identical physics to experiments/experiment_1/train.py but uses ReLoBRaLo
(Relative Loss Balancing Residual-based Loss) to dynamically tune loss weights.
Config ``loss_weights`` are ignored; weights are driven entirely by relative
training progress w.r.t. reference losses recorded at the end of the warmup.

Reference: Bischof & Kraus, "Multi-Objective Loss Balancing for Physics-Informed
Deep Learning", arXiv:2110.09813.

Usage
-----
::

    python experiments/experiment_1/train_relobralo.py --config configs/experiment_1.yaml
"""

import os
import sys

# Ensure project root is on path regardless of how the script is invoked.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import time
import argparse

import jax
import jax.numpy as jnp
from jax import lax, random
from flax.core import FrozenDict

from src.config import load_config, get_dtype
from src.predict.predictor import _apply_min_depth
from src.data import sample_domain
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_data_loss, compute_neg_h_loss,
    loss_boundary_dirichlet, loss_boundary_neumann_outflow_x,
    loss_boundary_wall_horizontal,
)
from src.utils import nse, rmse, relative_l2, plot_h_vs_x, save_model
from src.physics import h_exact, hu_exact, hv_exact
from src.training import (
    create_optimizer, calculate_num_batches,
    get_experiment_name, get_sampling_count_from_config,
    get_boundary_segment_count, init_model_from_config,
    sample_and_batch, maybe_batch_data,
    post_training_save, resolve_data_mode, create_output_dirs,
)
from src.training.loop import extract_lr
from src.monitoring import ConsoleLogger, AimTracker, compute_negative_depth_diagnostics
from src.checkpointing import CheckpointManager
from src.balancing import ReLoBRaLo, make_scan_body_relobralo


# ---------------------------------------------------------------------------
# Loss computation  (identical to train.py)
# ---------------------------------------------------------------------------

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
        left   = bc_batches.get('left',   jnp.empty((0, 3), dtype=get_dtype()))
        right  = bc_batches.get('right',  jnp.empty((0, 3), dtype=get_dtype()))
        bottom = bc_batches.get('bottom', jnp.empty((0, 3), dtype=get_dtype()))
        top    = bc_batches.get('top',    jnp.empty((0, 3), dtype=get_dtype()))

        u_const   = config["physics"]["u_const"]
        n_manning = config["physics"]["n_manning"]
        t_left    = left[..., 2]
        h_true    = h_exact(0.0, t_left, n_manning, u_const)
        hu_true   = h_true * u_const
        loss_left   = (loss_boundary_dirichlet(model, params, left, h_true, var_idx=0) +
                       loss_boundary_dirichlet(model, params, left, hu_true, var_idx=1))
        loss_right  = loss_boundary_neumann_outflow_x(model, params, right)
        loss_bottom = loss_boundary_wall_horizontal(model, params, bottom)
        loss_top    = loss_boundary_wall_horizontal(model, params, top)
        terms['bc'] = loss_left + loss_right + loss_bottom + loss_top

    data_batch_data = batch.get('data', jnp.empty((0, 6), dtype=get_dtype()))
    if not data_free and data_batch_data.shape[0] > 0:
        terms['data'] = compute_data_loss(model, params, data_batch_data, config)

    return terms


# ---------------------------------------------------------------------------
# Training loop with ReLoBRaLo weight updates
# ---------------------------------------------------------------------------

def _run_relobralo_loop(
    *,
    cfg, cfg_dict,
    model, params, opt_state,
    train_key, optimiser,
    generate_epoch_data_jit, scan_body,
    num_batches,
    relobralo,
    experiment_name,
    trial_name, results_dir, model_dir,
    config_path,
    validation_fn=None,
    source_script_path=None,
    compute_all_losses_fn=None,
):
    """Epoch loop with per-epoch ReLoBRaLo weight updates.

    The scan carry is ``(params, opt_state, loss_weights)``.  After every epoch
    :meth:`ReLoBRaLo.update` is called with the average unweighted MSE losses
    to produce new ``loss_weights`` for the next epoch.
    """
    aim_enabled = cfg_dict.get('aim', {}).get('enable', True)
    aim_tracker = AimTracker(cfg_dict, trial_name, enable=aim_enabled)
    aim_tracker.log_flags({"scenario_type": experiment_name, "loss_weighting": "relobralo"})
    if aim_enabled:
        try:
            aim_tracker.log_artifact(config_path, 'run_config.yaml')
            if source_script_path is not None:
                aim_tracker.log_artifact(os.path.abspath(source_script_path), 'source_script.py')
        except Exception:
            pass

    cfg_dict['scenario'] = cfg_dict.get('scenario', experiment_name)
    console = ConsoleLogger(cfg_dict)
    console.print_header()

    start_time = time.time()
    ckpt_mgr = CheckpointManager(model_dir, model=model)
    freq = cfg.get("reporting", {}).get("epoch_freq", 100)

    val_metrics = {}
    avg_losses_unweighted = {}
    avg_total_weighted_loss = 0.0
    neg_depth = {}
    global_step = 0
    current_lr = cfg["training"]["learning_rate"]
    epoch_history: list = []

    best_nse_stats = {
        'nse': -jnp.inf, 'rmse': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {}
    }
    best_loss_stats = {
        'total_weighted_loss': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'nse': -jnp.inf, 'rmse': jnp.inf, 'unweighted_losses': {}
    }
    best_params_nse = None
    best_params_loss = None
    epoch = 0

    loss_weights = relobralo.weights_array()

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()

            train_key, epoch_key = random.split(train_key)
            scan_inputs = generate_epoch_data_jit(epoch_key)

            carry_in = (params, opt_state, loss_weights)
            (params, opt_state, _), (batch_losses_stacked, batch_total_stacked) = lax.scan(
                scan_body, carry_in, scan_inputs
            )
            global_step += num_batches

            avg_losses_unweighted = {
                k: float(jnp.sum(v)) / num_batches
                for k, v in batch_losses_stacked.items()
            }
            avg_total_weighted_loss = float(jnp.sum(batch_total_stacked)) / num_batches

            # Update ReLoBRaLo weights for next epoch
            relobralo.update(avg_losses_unweighted)
            loss_weights = relobralo.weights_array()

            current_lr = extract_lr(opt_state, cfg["training"]["learning_rate"], epoch)

            # Validation
            if validation_fn is not None:
                try:
                    val_metrics = validation_fn(model, params)
                except Exception:
                    val_metrics = {}
            if not val_metrics:
                val_metrics = {'nse_h': float(-jnp.inf), 'rmse_h': float(jnp.inf)}

            selection_metric = float(val_metrics.get('nse_h', -jnp.inf))
            rmse_val = float(val_metrics.get('rmse_h', jnp.inf))

            if selection_metric > best_nse_stats['nse']:
                best_nse_stats.update({
                    'nse': selection_metric, 'rmse': rmse_val, 'epoch': epoch,
                    'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'total_weighted_loss': avg_total_weighted_loss,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()},
                    'validation_metrics': dict(val_metrics),
                })
                best_params_nse = jax.tree.map(jnp.copy, params)

            if avg_total_weighted_loss < best_loss_stats['total_weighted_loss']:
                best_loss_stats.update({
                    'total_weighted_loss': avg_total_weighted_loss, 'epoch': epoch,
                    'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'nse': selection_metric, 'rmse': rmse_val,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()},
                    'validation_metrics': dict(val_metrics),
                })
                best_params_loss = jax.tree.map(jnp.copy, params)

            epoch_time = time.time() - epoch_start_time
            elapsed_now = time.time() - start_time

            neg_depth = {'count': 0, 'fraction': 0.0, 'min': 0.0, 'mean': 0.0}
            if (epoch + 1) % freq == 0:
                try:
                    neg_depth = compute_negative_depth_diagnostics(
                        model, params, scan_inputs['pde'][0]
                    )
                except Exception:
                    pass

            saved_events = ckpt_mgr.update(
                epoch, params, opt_state, val_metrics,
                avg_losses_unweighted, avg_total_weighted_loss, cfg_dict, neg_depth,
                elapsed_time_s=elapsed_now,
            )
            for event in saved_events:
                event_type, value, ep, prev_value, prev_epoch = event
                if event_type == 'best_nse':
                    console.print_checkpoint_nse(value, ep, prev_value, prev_epoch)
                    aim_tracker.log_best_nse(value, ep, step=global_step)
                elif event_type == 'best_loss':
                    console.print_checkpoint_loss(value, ep, prev_value, prev_epoch)
                    aim_tracker.log_best_loss(value, ep, step=global_step)

            if (epoch + 1) % freq == 0:
                console.print_epoch(
                    epoch, cfg["training"]["epochs"],
                    avg_losses_unweighted, avg_total_weighted_loss,
                    current_lr, val_metrics,
                    neg_depth.get('fraction', 0.0), epoch_time,
                )
                w_str = "  ReLoBRaLo weights: " + ", ".join(
                    f"{k}={v:.4f}" for k, v in relobralo.weights.items()
                )
                print(w_str)

            epoch_history.append({
                'epoch': int(epoch),
                'total_loss': float(avg_total_weighted_loss),
                'losses': {k: float(v) for k, v in avg_losses_unweighted.items()},
                'relobralo_weights': dict(relobralo.weights),
                'val_metrics': {k: float(v) for k, v in val_metrics.items()},
                'lr': float(current_lr),
                'epoch_time_s': float(epoch_time),
                'elapsed_time_s': float(elapsed_now),
            })

            aim_tracker.log_epoch(
                epoch=epoch, step=global_step,
                losses=avg_losses_unweighted, total_loss=avg_total_weighted_loss,
                val_metrics=val_metrics, lr=current_lr,
                epoch_time=epoch_time, elapsed_time=elapsed_now,
                neg_depth=neg_depth if (epoch + 1) % freq == 0 else None,
            )
            aim_tracker.log_scalars(relobralo.weights, step=global_step, prefix="relobralo_weights")

            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))
            if epoch >= min_epochs and (epoch - best_nse_stats['epoch']) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch + 1} ---")
                print(f"Best NSE {best_nse_stats['nse']:.6f} at epoch {best_nse_stats['epoch'] + 1}.")
                break

    except KeyboardInterrupt:
        print("\n--- Training interrupted ---")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()

    finally:
        total_time = time.time() - start_time

        all_physics_losses = {}
        if compute_all_losses_fn is not None:
            eval_params = best_params_nse if best_params_nse is not None else params
            try:
                all_physics_losses = {k: float(v) for k, v in compute_all_losses_fn(model, eval_params).items()}
                print("\nAll physics losses (best-NSE params):")
                for k, v in all_physics_losses.items():
                    print(f"  {k}: {v:.6e}")
            except Exception as e:
                print(f"Warning: Failed to evaluate all physics losses: {e}")

        final_losses_for_ckpt = dict(avg_losses_unweighted)
        for k, v in all_physics_losses.items():
            if k not in final_losses_for_ckpt:
                final_losses_for_ckpt[k] = v

        ckpt_mgr.save_final(
            epoch, params, opt_state, val_metrics,
            final_losses_for_ckpt, avg_total_weighted_loss, cfg_dict, neg_depth,
            training_time_s=total_time,
        )

        try:
            history_path = os.path.join(model_dir, 'training_history.json')
            with open(history_path, 'w') as hf:
                json.dump({
                    'total_training_time_s': float(total_time),
                    'total_epochs': int(epoch + 1),
                    'loss_weighting': 'relobralo',
                    'relobralo_alpha': relobralo.alpha,
                    'relobralo_warmup': relobralo.warmup,
                    'epochs': epoch_history,
                }, hf, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write training_history.json: {e}")

        best_nse_ckpt = ckpt_mgr.get_best_nse_stats()
        best_loss_ckpt = ckpt_mgr.get_best_loss_stats()

        console.print_completion_summary(
            total_time=total_time, final_epoch=epoch,
            best_nse_stats=best_nse_ckpt, best_loss_stats=best_loss_ckpt,
            final_losses=final_losses_for_ckpt, final_val_metrics=val_metrics,
            neg_depth_final=neg_depth, neg_depth_best_nse={}, neg_depth_best_loss={},
            final_lr=current_lr,
        )

        if aim_tracker.enabled:
            try:
                aim_tracker.log_summary({
                    'best_validation_model': {**best_nse_stats, 'epoch': best_nse_stats.get('epoch', 0) + 1},
                    'best_loss_model': {**best_loss_stats, 'epoch': best_loss_stats.get('epoch', 0) + 1},
                    'final_system': {
                        'total_training_time_seconds': total_time,
                        'total_epochs_run': epoch + 1,
                        'total_steps_run': global_step,
                    },
                })
            except Exception as e:
                print(f"Warning: Error logging summary to Aim: {e}")

    return {
        "best_nse_stats": best_nse_stats,
        "best_loss_stats": best_loss_stats,
        "best_params_nse": best_params_nse,
        "best_params_loss": best_params_loss,
        "params": params,
        "opt_state": opt_state,
        "aim_tracker": aim_tracker,
        "epoch": epoch,
        "total_time": total_time,
    }


# ---------------------------------------------------------------------------
# Trial setup
# ---------------------------------------------------------------------------

def setup_trial(cfg_dict: dict) -> dict:
    """Set up all training components for Experiment 1 with ReLoBRaLo.

    Config ``loss_weights`` are ignored — ReLoBRaLo drives all weights.
    """
    cfg = FrozenDict(cfg_dict)
    experiment_name = get_experiment_name(cfg_dict, "experiment_1")

    model, params, train_key, val_key = init_model_from_config(cfg)
    print("Info: Running Experiment 1 with ReLoBRaLo adaptive loss balancing.")

    # --- Validation data (analytical) ---
    val_points, h_true_val, hu_true_val, hv_true_val = None, None, None, None
    validation_data_loaded = False
    try:
        val_grid_cfg = cfg["validation_grid"]
        domain_cfg   = cfg["domain"]
        val_points   = sample_domain(
            val_key,
            val_grid_cfg["n_points_val"],
            (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
        )
        n_manning  = cfg["physics"]["n_manning"]
        u_const    = cfg["physics"]["u_const"]
        h_true_val  = h_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        hu_true_val = hu_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        hv_true_val = hv_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        if val_points.shape[0] > 0:
            validation_data_loaded = True
            print(f"Created analytical validation set with {val_points.shape[0]} points.")
    except Exception as e:
        print(f"Warning: Validation set setup failed: {e}. Skipping NSE/RMSE.")

    # --- Data mode ---
    data_points_full = None
    data_free, _ = resolve_data_mode(cfg)

    if not data_free:
        try:
            train_grid_cfg = cfg["train_grid"]
            domain_cfg     = cfg["domain"]
            n_gauges = train_grid_cfg["n_gauges"]
            dt_data  = train_grid_cfg["dt_data"]
            t_final  = domain_cfg["t_final"]

            if n_gauges <= 0 or dt_data <= 0:
                raise ValueError(
                    f"Gauge sampling requires n_gauges>0 and dt_data>0, "
                    f"got n_gauges={n_gauges}, dt_data={dt_data}."
                )

            t_steps = jnp.arange(0., t_final + dt_data * 0.5, dt_data, dtype=get_dtype())
            n_timesteps = t_steps.shape[0]
            gauge_xy = sample_domain(
                train_key, n_gauges,
                (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.)
            )[:, :2]
            gauge_xy_rep = jnp.repeat(gauge_xy, n_timesteps, axis=0)
            t_rep = jnp.tile(t_steps, n_gauges).reshape(-1, 1)
            data_pts = jnp.hstack([gauge_xy_rep, t_rep])

            h_tr = h_exact(data_pts[:, 0], data_pts[:, 2], cfg["physics"]["n_manning"], cfg["physics"]["u_const"])
            u_tr = jnp.full_like(h_tr, cfg["physics"]["u_const"])
            v_tr = jnp.zeros_like(h_tr)
            data_points_full = jnp.stack(
                [data_pts[:, 2], data_pts[:, 0], data_pts[:, 1], h_tr, u_tr, v_tr], axis=1
            ).astype(get_dtype())
            print(f"Created {data_points_full.shape[0]} analytical training points for data loss.")
        except Exception as e:
            print(f"Error creating analytical training data: {e}. Disabling data loss.")
            data_free = True

    # --- Active loss keys ---
    loss_keys = ['pde', 'ic', 'bc', 'neg_h']
    if not data_free:
        loss_keys.append('data')
    loss_keys = tuple(loss_keys)

    relo_cfg = cfg.get("relobralo", {})
    relobralo = ReLoBRaLo(
        loss_keys=list(loss_keys),
        alpha=float(relo_cfg.get("alpha", 0.999)),
        warmup=int(relo_cfg.get("warmup", 20)),
        min_weight=float(relo_cfg.get("min_weight", 0.01)),
    )
    print(f"ReLoBRaLo initialised: keys={loss_keys}, alpha={relobralo.alpha}, "
          f"warmup={relobralo.warmup}, min_weight={relobralo.min_weight}.")

    # --- Batch counts ---
    batch_size    = cfg["training"]["batch_size"]
    domain_cfg    = cfg["domain"]
    n_pde         = get_sampling_count_from_config(cfg, "n_points_pde")
    n_ic          = get_sampling_count_from_config(cfg, "n_points_ic")
    n_bc_domain   = get_sampling_count_from_config(cfg, "n_points_bc_domain")
    n_bc_per_wall = get_boundary_segment_count(cfg, n_bc_domain)

    num_batches = calculate_num_batches(
        batch_size,
        [n_pde, n_ic, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall],
        data_points_full,
        data_free=data_free,
    )
    if num_batches == 0:
        raise ValueError(f"Batch size {batch_size} too large for configured sample counts.")

    # --- Optimizer ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # --- JIT data generator ---
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, data_key = random.split(key, 5)
        x_range = (0., domain_cfg["lx"])
        y_range = (0., domain_cfg["ly"])
        t_range = (0., domain_cfg["t_final"])

        pde_data = sample_and_batch(pde_key, sample_domain, n_pde, batch_size, num_batches, x_range, y_range, t_range)
        ic_data  = sample_and_batch(ic_key,  sample_domain, n_ic,  batch_size, num_batches, x_range, y_range, (0., 0.))

        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        bc_data = {
            'left':   sample_and_batch(l_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (0., 0.), y_range, t_range),
            'right':  sample_and_batch(r_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range),
            'bottom': sample_and_batch(b_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (0., 0.), t_range),
            'top':    sample_and_batch(t_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range),
        }

        return {
            'pde':         pde_data,
            'ic':          ic_data,
            'bc':          bc_data,
            'data':        maybe_batch_data(data_key, data_points_full, batch_size, num_batches, data_free),
            'building_bc': {},
        }

    generate_epoch_data_jit = jax.jit(generate_epoch_data)

    # --- Scan body ---
    scan_body = make_scan_body_relobralo(
        model, optimiser, loss_keys, cfg, data_free, compute_losses
    )

    # --- Validation function ---
    def validation_fn(model, params):
        metrics = {}
        if validation_data_loaded:
            try:
                U_pred = model.apply({'params': params['params']}, val_points, train=False)
                min_depth_val = cfg.get("numerics", {}).get("min_depth", 0.0)
                U_pred = _apply_min_depth(U_pred, min_depth_val)
                metrics = {
                    'nse_h':    float(nse(U_pred[..., 0], h_true_val)),
                    'rmse_h':   float(rmse(U_pred[..., 0], h_true_val)),
                    'rel_l2_h': float(relative_l2(U_pred[..., 0], h_true_val)),
                }
                if hu_true_val is not None:
                    metrics['nse_hu']    = float(nse(U_pred[..., 1], hu_true_val))
                    metrics['rmse_hu']   = float(rmse(U_pred[..., 1], hu_true_val))
                    metrics['rel_l2_hu'] = float(relative_l2(U_pred[..., 1], hu_true_val))
                    metrics['nse_hv']    = float(nse(U_pred[..., 2], hv_true_val))
                    metrics['rmse_hv']   = float(rmse(U_pred[..., 2], hv_true_val))
                    metrics['rel_l2_hv'] = float(relative_l2(U_pred[..., 2], hv_true_val))
            except Exception as exc:
                print(f"Warning: Validation failed: {exc}")
        if not metrics:
            metrics = {'nse_h': float(-jnp.inf), 'rmse_h': float(jnp.inf)}
        return metrics

    # --- All-physics evaluation ---
    n_eval = 200

    def compute_all_losses_fn(model, params):
        eval_key = random.PRNGKey(0)
        k_pde, k_ic, k_left, k_right, k_bottom, k_top = random.split(eval_key, 6)
        x_r = (0., domain_cfg["lx"])
        y_r = (0., domain_cfg["ly"])
        t_r = (0., domain_cfg["t_final"])
        batch = {
            'pde': sample_domain(k_pde, n_eval, x_r, y_r, t_r),
            'ic':  sample_domain(k_ic,  n_eval, x_r, y_r, (0., 0.)),
            'bc': {
                'left':   sample_domain(k_left,   n_eval, (0., 0.), y_r, t_r),
                'right':  sample_domain(k_right,  n_eval, (domain_cfg["lx"], domain_cfg["lx"]), y_r, t_r),
                'bottom': sample_domain(k_bottom, n_eval, x_r, (0., 0.), t_r),
                'top':    sample_domain(k_top,    n_eval, x_r, (domain_cfg["ly"], domain_cfg["ly"]), t_r),
            },
            'data':        jnp.empty((0, 6), dtype=get_dtype()),
            'building_bc': {},
        }
        return compute_losses(model, params, batch, cfg, data_free=True)

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
        "relobralo": relobralo,
        "validation_fn": validation_fn,
        "data_free": data_free,
        "compute_all_losses_fn": compute_all_losses_fn,
        "experiment_name": experiment_name,
        "validation_data_loaded": validation_data_loaded,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(config_path: str):
    cfg_dict = load_config(config_path)
    ctx = setup_trial(cfg_dict)

    experiment_name = ctx["experiment_name"]
    trial_name, results_dir, model_dir = create_output_dirs(ctx["cfg"], experiment_name)

    model = ctx["model"]
    cfg   = ctx["cfg"]

    loop_result = _run_relobralo_loop(
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
        relobralo=ctx["relobralo"],
        experiment_name=experiment_name,
        trial_name=trial_name,
        results_dir=results_dir,
        model_dir=model_dir,
        config_path=config_path,
        validation_fn=ctx["validation_fn"],
        source_script_path=__file__,
        compute_all_losses_fn=ctx["compute_all_losses_fn"],
    )

    def plot_fn(final_params):
        print("  Generating 1D validation plot...")
        aim_tracker = loop_result["aim_tracker"]
        final_epoch = loop_result["epoch"]
        plot_cfg = cfg.get("plotting", {})
        min_depth_plot = cfg.get("numerics", {}).get("min_depth", 0.0)
        t_const = plot_cfg.get("t_const_val", cfg["domain"]["t_final"] / 2.0)
        nx_val  = plot_cfg.get("nx_val", 101)
        y_const = plot_cfg.get("y_const_plot", 0.0)
        x_plot  = jnp.linspace(0.0, cfg["domain"]["lx"], nx_val, dtype=get_dtype())
        pts_1d  = jnp.stack([
            x_plot,
            jnp.full_like(x_plot, y_const),
            jnp.full_like(x_plot, t_const),
        ], axis=1)
        U_1d = model.apply({'params': final_params['params']}, pts_1d, train=False)
        U_1d = _apply_min_depth(U_1d, min_depth_plot)
        plot_path = os.path.join(results_dir, "final_validation_plot.png")
        plot_h_vs_x(x_plot, U_1d[..., 0], t_const, y_const, ctx["cfg_dict"], plot_path)
        aim_tracker.log_image(plot_path, 'validation_plot_1D', final_epoch)
        print(f"Plot saved to {plot_path}")

    post_training_save(
        loop_result=loop_result,
        model=model,
        model_dir=model_dir,
        results_dir=results_dir,
        trial_name=trial_name,
        plot_fn=plot_fn,
    )

    return (
        loop_result["best_nse_stats"]["nse"]
        if loop_result["best_nse_stats"]["nse"] > -jnp.inf
        else -1.0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1 — ReLoBRaLo adaptive loss balancing."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML (e.g. configs/experiment_1.yaml)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        final_nse = main(args.config)
        print(f"\n--- Script Finished ---")
        if isinstance(final_nse, float) and final_nse > -jnp.inf:
            print(f"Final best NSE: {final_nse:.6f}")
        print("-----------------------")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Config/Model Error: {e}")
    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}")
        traceback.print_exc()
