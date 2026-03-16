"""Shared training-loop body and post-training routines."""
import os
import time
import shutil

import jax
import jax.numpy as jnp
from jax import lax

from src.utils import nse, rmse, save_model, ask_for_confirmation
from src.utils.profiling import EpochTimer, log_memory_usage
from src.monitoring import ConsoleLogger, AimTracker, compute_negative_depth_diagnostics
from src.checkpointing import CheckpointManager
from src.predict.predictor import _apply_min_depth


def extract_lr(opt_state, base_lr, epoch):
    """Extract the current learning rate from the optimizer state."""
    current_lr = base_lr
    try:
        if hasattr(opt_state[-1], 'scale'):
            current_scale = float(opt_state[-1].scale)
            current_lr = base_lr * current_scale
    except Exception as e:
        if epoch == 0:
            print(f"Warning: Failed to extract LR scale: {e}")
    return current_lr


def run_training_loop(
    *,
    cfg, cfg_dict,
    model, params, opt_state,
    train_key, optimiser,
    generate_epoch_data_jit, scan_body,
    num_batches,
    experiment_name,
    trial_name, results_dir, model_dir,
    config_path,
    validation_data_loaded=False,
    val_points_all=None, h_true_val_all=None,
    val_targets_all=None,
    pde_key_for_diag="pde",
    validation_fn=None,
    selection_metric_key="nse_h",
    source_script_path=None,
    compute_all_losses_fn=None,
):
    """Execute the full epoch loop with logging, checkpointing, and early stopping.

    Parameters
    ----------
    generate_epoch_data_jit : callable
        JIT-compiled function that takes a PRNG key and returns a pytree of
        batched data shaped ``(num_batches, batch_size, ...)``.
    scan_body : callable
        Function ``(carry, batch_data) -> (carry, (terms, total))`` to pass
        to ``lax.scan``.
    pde_key_for_diag : str
        Key in the scan_inputs pytree where PDE points live (for negative-depth
        diagnostics).  Typically ``"pde"``.
    compute_all_losses_fn : callable, optional
        ``(model, params) -> dict[str, float]`` — evaluates **all** physics
        loss terms (PDE, IC, BC, neg_h, etc.) regardless of whether their
        training weights were zero.  Called once at the end of training on the
        best-NSE parameters so that physics compliance is always recorded in
        the final checkpoint metadata and summary.

    Returns
    -------
    dict with ``best_nse``, ``best_loss``, ``final_params``, ``params``, ``opt_state``, etc.
    """
    from jax import random

    aim_enabled = cfg_dict.get('aim', {}).get('enable', True)
    aim_tracker = AimTracker(cfg_dict, trial_name, enable=aim_enabled)
    aim_tracker.log_flags({"scenario_type": experiment_name})
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
    val_metrics = {}
    neg_depth = {}
    avg_losses_unweighted = {}
    avg_total_weighted_loss = 0.0
    global_step = 0
    current_lr = cfg["training"]["learning_rate"]

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

    enable_profiling = cfg_dict.get("debug", {}).get("profile_timing", False)
    epoch_timer = EpochTimer() if enable_profiling else None

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()

            train_key, epoch_key = random.split(train_key)

            if enable_profiling:
                with epoch_timer.time("data_generation"):
                    scan_inputs = generate_epoch_data_jit(epoch_key)
                    jax.block_until_ready(scan_inputs)
            else:
                scan_inputs = generate_epoch_data_jit(epoch_key)

            if enable_profiling:
                with epoch_timer.time("scan_step"):
                    (params, opt_state), (batch_losses_stacked, batch_total_stacked) = lax.scan(
                        scan_body, (params, opt_state), scan_inputs
                    )
                    jax.block_until_ready(batch_total_stacked)
            else:
                (params, opt_state), (batch_losses_stacked, batch_total_stacked) = lax.scan(
                    scan_body, (params, opt_state), scan_inputs
                )
            global_step += num_batches

            # Aggregate losses
            epoch_losses_sum = {k: jnp.sum(v) for k, v in batch_losses_stacked.items()}
            epoch_total_sum = jnp.sum(batch_total_stacked)
            avg_losses_unweighted = {k: float(v) / num_batches for k, v in epoch_losses_sum.items()}
            avg_total_weighted_loss = float(epoch_total_sum) / num_batches

            current_lr = extract_lr(opt_state, cfg["training"]["learning_rate"], epoch)

            # Validation
            nse_val, rmse_val = -jnp.inf, jnp.inf
            if validation_fn is not None:
                try:
                    val_metrics = validation_fn(model, params)
                except Exception:
                    val_metrics = {}
            else:
                val_metrics = {}
                if validation_data_loaded:
                    try:
                        U_val = model.apply(params, val_points_all, train=False)
                        min_depth = cfg.get("numerics", {}).get("min_depth", 0.0)
                        U_val = _apply_min_depth(U_val, min_depth)
                        nse_val = nse(U_val[..., 0], h_true_val_all)
                        rmse_val = rmse(U_val[..., 0], h_true_val_all)
                        val_metrics = {'nse_h': float(nse_val), 'rmse_h': float(rmse_val)}
                        # Compute hu/hv metrics when full targets [h, u, v] are available
                        if val_targets_all is not None and val_targets_all.shape[-1] >= 3:
                            hu_true = val_targets_all[..., 0] * val_targets_all[..., 1]
                            hv_true = val_targets_all[..., 0] * val_targets_all[..., 2]
                            val_metrics['nse_hu'] = float(nse(U_val[..., 1], hu_true))
                            val_metrics['rmse_hu'] = float(rmse(U_val[..., 1], hu_true))
                            val_metrics['nse_hv'] = float(nse(U_val[..., 2], hv_true))
                            val_metrics['rmse_hv'] = float(rmse(U_val[..., 2], hv_true))
                    except Exception:
                        pass
                if not val_metrics:
                    val_metrics = {'nse_h': float(nse_val), 'rmse_h': float(rmse_val)}

            selection_metric = float(val_metrics.get(selection_metric_key, -jnp.inf))
            rmse_val = float(val_metrics.get('rmse_h', jnp.inf))

            # Track best models
            if selection_metric > best_nse_stats['nse']:
                best_nse_stats.update({
                    'nse': selection_metric, 'rmse': rmse_val, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'total_weighted_loss': avg_total_weighted_loss,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()},
                    'validation_metrics': dict(val_metrics),
                })
                best_params_nse = jax.tree.map(jnp.copy, params)

            if avg_total_weighted_loss < best_loss_stats['total_weighted_loss']:
                best_loss_stats.update({
                    'total_weighted_loss': avg_total_weighted_loss, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'nse': selection_metric, 'rmse': rmse_val,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()},
                    'validation_metrics': dict(val_metrics),
                })
                best_params_loss = jax.tree.map(jnp.copy, params)

            freq = cfg.get("reporting", {}).get("epoch_freq", 100)
            epoch_time = time.time() - epoch_start_time

            neg_depth = {'count': 0, 'fraction': 0.0, 'min': 0.0, 'mean': 0.0}
            if (epoch + 1) % freq == 0:
                try:
                    neg_depth = compute_negative_depth_diagnostics(model, params, scan_inputs[pde_key_for_diag][0])
                except Exception:
                    pass

            saved_events = ckpt_mgr.update(
                epoch, params, opt_state, val_metrics,
                avg_losses_unweighted, avg_total_weighted_loss, cfg_dict, neg_depth
            )
            for event in saved_events:
                event_type, value, ep, prev_value, prev_epoch = event
                if event_type == 'best_nse':
                    console.print_checkpoint_nse(value, ep, prev_value, prev_epoch)
                    aim_tracker.log_best_nse(value, ep)
                elif event_type == 'best_loss':
                    console.print_checkpoint_loss(value, ep, prev_value, prev_epoch)
                    aim_tracker.log_best_loss(value, ep)

            if (epoch + 1) % freq == 0:
                console.print_epoch(
                    epoch, cfg["training"]["epochs"],
                    avg_losses_unweighted, avg_total_weighted_loss,
                    current_lr,
                    val_metrics, neg_depth.get('fraction', 0.0), epoch_time
                )

            aim_tracker.log_epoch(
                epoch=epoch, step=global_step,
                losses=avg_losses_unweighted, total_loss=avg_total_weighted_loss,
                val_metrics=val_metrics, lr=current_lr,
                epoch_time=epoch_time, elapsed_time=time.time() - start_time,
                neg_depth=neg_depth if (epoch + 1) % freq == 0 else None,
            )

            # Early stopping
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))
            if epoch >= min_epochs and (epoch - best_nse_stats['epoch']) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                print(f"Best NSE {best_nse_stats['nse']:.6f} achieved at epoch {best_nse_stats['epoch']+1}.")
                break

    except KeyboardInterrupt:
        print("\n--- Training interrupted ---")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        total_time = time.time() - start_time

        if enable_profiling and epoch_timer is not None:
            epoch_timer.print_summary()
            log_memory_usage("end_of_training")

        # Evaluate all physics losses on best-NSE params (even zero-weight terms)
        all_physics_losses = {}
        if compute_all_losses_fn is not None:
            eval_params = best_params_nse if best_params_nse is not None else params
            try:
                all_physics_losses = compute_all_losses_fn(model, eval_params)
                all_physics_losses = {k: float(v) for k, v in all_physics_losses.items()}
                print(f"\nAll physics losses (best-NSE params):")
                for k, v in all_physics_losses.items():
                    print(f"  {k}: {v:.6e}")
            except Exception as e:
                print(f"Warning: Failed to evaluate all physics losses: {e}")

        # Merge all_physics_losses into final losses for the checkpoint
        final_losses_for_ckpt = dict(avg_losses_unweighted)
        for k, v in all_physics_losses.items():
            if k not in final_losses_for_ckpt:
                final_losses_for_ckpt[k] = v

        ckpt_mgr.save_final(epoch, params, opt_state, val_metrics,
                            final_losses_for_ckpt, avg_total_weighted_loss, cfg_dict, neg_depth)
        best_nse_ckpt = ckpt_mgr.get_best_nse_stats()
        best_loss_ckpt = ckpt_mgr.get_best_loss_stats()

        console.print_completion_summary(
            total_time=total_time,
            final_epoch=epoch,
            best_nse_stats=best_nse_ckpt,
            best_loss_stats=best_loss_ckpt,
            final_losses=final_losses_for_ckpt,
            final_val_metrics=val_metrics,
            neg_depth_final=neg_depth,
            neg_depth_best_nse={},
            neg_depth_best_loss={},
            final_lr=current_lr,
        )

        if aim_tracker.enabled:
            try:
                summary_metrics = {
                    'best_validation_model': {**best_nse_stats, 'epoch': best_nse_stats.get('epoch', 0) + 1},
                    'best_loss_model': {**best_loss_stats, 'epoch': best_loss_stats.get('epoch', 0) + 1},
                    'final_system': {
                        'total_training_time_seconds': total_time,
                        'total_epochs_run': epoch + 1,
                        'total_steps_run': global_step,
                    }
                }
                if all_physics_losses:
                    summary_metrics['all_physics_losses'] = all_physics_losses
                aim_tracker.log_summary(summary_metrics)
                print("Summary metrics logged to Aim.")
            except Exception as e:
                print(f"Warning: Error logging summary metrics to Aim: {e}")

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


def post_training_save(
    *,
    loop_result,
    model, model_dir, results_dir, trial_name,
    prefer_loss_model=False,
    plot_fn=None,
):
    """Handle model saving, optional plotting, and cleanup after training.

    Parameters
    ----------
    loop_result : dict
        Return value of ``run_training_loop``.
    plot_fn : callable, optional
        ``plot_fn(final_params)`` will be called to generate experiment-specific plots.
    prefer_loss_model : bool
        If True, prefer best-loss params over best-NSE (used by some experiments).
    """
    aim_tracker = loop_result["aim_tracker"]
    best_params_nse = loop_result["best_params_nse"]
    best_params_loss = loop_result["best_params_loss"]

    if prefer_loss_model:
        final_params = best_params_loss if best_params_loss is not None else best_params_nse
    else:
        final_params = best_params_nse if best_params_nse is not None else best_params_loss

    if ask_for_confirmation():
        if final_params is not None:
            saved_model_path = save_model(final_params, model_dir, trial_name)

            if aim_tracker.enabled and saved_model_path:
                try:
                    aim_tracker.log_artifact(saved_model_path, 'model_weights.pkl')
                    print("Logged model artifact to Aim.")
                except Exception as e_mod:
                    print(f"Warning: Failed to log model artifact: {e_mod}")

            if plot_fn is not None:
                plot_fn(final_params)
        else:
            print("No model parameters found to save.")
    else:
        print("Save aborted by user. Deleting artifacts...")
        try:
            aim_tracker.delete_run()
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
                print(f"Deleted results directory: {results_dir}")
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                print(f"Deleted model directory: {model_dir}")
            if aim_tracker.run_hash:
                run_artifact_dir = os.path.join("aim_repo", "aim_artifacts", aim_tracker.run_hash)
                if os.path.exists(run_artifact_dir):
                    shutil.rmtree(run_artifact_dir)
                    print(f"Deleted run artifact directory: {run_artifact_dir}")
            print("Cleanup complete.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    aim_tracker.close()
    return final_params
