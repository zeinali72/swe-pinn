"""Weights & Biases experiment tracking wrapper.

Provides structured metric logging for all training experiments.
All JAX/NumPy values are sanitised to native Python types before being sent
to W&B.

Experiment grouping:
  Project  = "swe-pinn" (configurable via config["wandb"]["project"])
  Group    = scenario name  (experiment_1, experiment_2, …)
  Run name = trial_name     (date-time + arch, e.g. 2025-03-19_14-32_fourier)
  Tags     = scenario, model, variant flags
"""
import copy
import math
import os
import re
from typing import Optional

try:
    import jax as _jax
except Exception:
    _jax = None  # type: ignore[assignment]


def _safe_float(val, default=float('nan')):
    """Convert JAX/NumPy scalars to Python floats."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def sanitize_params(obj):
    """Recursively convert JAX/NumPy arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: sanitize_params(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_params(v) for v in obj)
    try:
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (float, int)):
            return obj
        return float(obj)
    except (TypeError, ValueError):
        return obj


def _flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict to dot-separated keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out


# Keys worth logging to wandb.config for grouping/filtering.
# Everything else is noise in the W&B UI.
_CONFIG_ALLOWLIST = [
    "scenario",
    "model.name", "model.width", "model.depth",
    "model.ff_dims", "model.fourier_scale",
    "domain.lx", "domain.ly", "domain.t_final",
    "physics.g", "physics.n_manning", "physics.u_const", "physics.inflow",
    "training.learning_rate", "training.epochs",
    "training.batch_size", "training.seed",
    "sampling.n_points_pde", "sampling.n_points_ic",
    "sampling.n_points_bc_domain",
    "device.dtype",
]

# All keys under loss_weights are kept (variable across experiments).
_CONFIG_ALLOW_PREFIXES = ["loss_weights."]


def _curated_config(config: dict) -> dict:
    """Extract only the meaningful config keys for W&B grouping."""
    flat = _flatten_dict(sanitize_params(copy.deepcopy(dict(config))))
    out = {}
    for key, val in flat.items():
        if key in _CONFIG_ALLOWLIST:
            out[key] = val
        elif any(key.startswith(p) for p in _CONFIG_ALLOW_PREFIXES):
            out[key] = val
    return out


class WandbTracker:
    """Wraps the W&B SDK with structured metric logging.

    If wandb is unavailable or ``enable=False``, all methods become no-ops.

    Grouping hierarchy in the W&B UI:

    * **Project** — ``config["wandb"]["project"]`` (default ``"swe-pinn"``).
    * **Group** — ``config["scenario"]`` (e.g. ``"experiment_1"``).
    * **Run name** — ``trial_name`` (date-time prefixed).
    * **Tags** — scenario, model architecture, data mode, phase.
    """

    _ENTITY = "zeinali72-exeter"

    def __init__(self, config: dict, trial_name: str, enable: bool = True):
        self.run = None
        self._enabled = enable
        self._run_files_artifact = None
        self._trial_name = trial_name

        if not enable:
            return

        try:
            import wandb

            wandb_cfg = config.get('wandb', {}) if isinstance(config, dict) else {}
            project = wandb_cfg.get('project', 'swe-pinn')
            entity = wandb_cfg.get('entity', self._ENTITY)

            scenario = config.get('scenario', '')
            group = scenario if scenario else None

            # Build tags
            arch_name = config.get('model', {}).get('name', '')
            data_weight = float(config.get('loss_weights', {}).get('data_weight', 0) or 0)
            data_mode = 'data_driven' if data_weight > 0 else 'data_free'

            phase = ''
            m = re.search(r'experiment_(\d+)', scenario)
            if m:
                n = int(m.group(1))
                phase = '1' if n <= 2 else ('2' if n <= 6 else '3')

            tags = [scenario, arch_name.lower(), data_mode]
            if phase:
                tags.append(f"phase_{phase}")
            tags = [t for t in tags if t]

            # Resolve JAX backend
            try:
                jax_backend = _jax.default_backend() if _jax is not None else 'unknown'
                jax_device_count = len(_jax.devices()) if _jax is not None else 0
            except Exception:
                jax_backend = 'unknown'
                jax_device_count = 0

            # Curated config — only meaningful keys for W&B grouping
            flat_config = _curated_config(config)

            self.run = wandb.init(
                project=project,
                entity=entity,
                name=trial_name,
                group=group,
                tags=tags,
                config=flat_config,
                reinit=True,
            )

            # Log additional metadata
            self.run.config.update({
                "jax_backend": jax_backend,
                "jax_device_count": jax_device_count,
                "data_mode": data_mode,
            }, allow_val_change=True)

            # Define x-axis options so the UI offers epoch, global_step,
            # and relative time as chart x-axes.
            wandb.define_metric("epoch")
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")

            print(f"W&B tracking initialised: {trial_name} [{project}/{group}] ({self.run.id})")

        except Exception as e:
            print(
                f"Warning: Failed to initialise W&B tracking: {e}. "
                "Training will continue without W&B."
            )
            self.run = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self.run is not None

    # ------------------------------------------------------------------
    # Per-epoch structured logging
    # ------------------------------------------------------------------
    def log_epoch(
        self,
        epoch: int,
        step: int,
        losses: dict,
        total_loss: float,
        val_metrics: dict,
        lr: float,
        epoch_time: float,
        elapsed_time: float,
        neg_depth: Optional[dict] = None,
    ):
        """Log per-epoch metrics.

        Metric groups logged as timeseries (step = epoch):
          train/   — training losses and learning rate
          val/     — validation accuracy metrics (NSE, RMSE, etc.)

        Logged only when negative points exist (count > 0):
          val/neg_depth_*  — negative-depth diagnostics

        Static values (timing) are written to run.summary only,
        not as timeseries, to avoid polluting the charts.
        """
        if not self.enabled:
            return
        try:
            metrics: dict = {}

            # X-axis metrics (logged so they can be selected as x-axes)
            metrics["epoch"] = int(epoch)
            metrics["global_step"] = int(step)

            # Training losses (skip terms that are exactly zero)
            metrics["train/total_loss"] = _safe_float(total_loss)
            for key, val in losses.items():
                fval = _safe_float(val)
                if fval != 0.0:
                    metrics[f"train/{key}"] = fval

            # Learning rate
            metrics["train/lr"] = _safe_float(lr)

            # Validation metrics (skip NaN — e.g. hv in 1D experiments)
            for key, val in val_metrics.items():
                default = -float('inf') if 'nse' in key else float('inf')
                fval = _safe_float(val, default)
                if not math.isnan(fval) and not math.isinf(fval):
                    metrics[f"val/{key}"] = fval

            # Negative-depth diagnostics — only when count > 0
            if neg_depth and neg_depth.get('count', 0) > 0:
                metrics["val/neg_depth_count"] = _safe_float(neg_depth['count'])
                metrics["val/neg_depth_fraction"] = _safe_float(neg_depth['fraction'])
                metrics["val/neg_depth_min"] = _safe_float(neg_depth['min'])
                metrics["val/neg_depth_mean"] = _safe_float(neg_depth['mean'])

            self.run.log(metrics)

            # Timing — summary only (overwritten each epoch so final values persist)
            self.run.summary["time/last_epoch_s"] = _safe_float(epoch_time)
            self.run.summary["time/elapsed_s"] = _safe_float(elapsed_time)
        except Exception as e:
            print(f"Warning: W&B logging failed at epoch {epoch}: {e}")

    # ------------------------------------------------------------------
    # Best-model tracking  (summary only — not timeseries)
    # ------------------------------------------------------------------
    def log_best_nse(self, nse_h: float, epoch: int, step: int):
        if not self.enabled:
            return
        try:
            self.run.summary["best/nse_h"] = _safe_float(nse_h)
            self.run.summary["best/nse_h_epoch"] = epoch + 1
            self.run.summary["best/nse_h_step"] = step
        except Exception as e:
            print(f"Warning: Failed to log best NSE to W&B: {e}")

    def log_best_loss(self, loss: float, epoch: int, step: int):
        if not self.enabled:
            return
        try:
            self.run.summary["best/loss"] = _safe_float(loss)
            self.run.summary["best/loss_epoch"] = epoch + 1
            self.run.summary["best/loss_step"] = step
        except Exception as e:
            print(f"Warning: Failed to log best loss to W&B: {e}")

    # ------------------------------------------------------------------
    # Run-level metadata
    # ------------------------------------------------------------------
    def log_flags(self, flags: dict):
        """Store flags as run config updates."""
        if not self.enabled:
            return
        try:
            safe_flags = sanitize_params(flags)
            self.run.config.update(
                {k: v for k, v in safe_flags.items() if v is not None},
                allow_val_change=True,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary and artifact logging
    # ------------------------------------------------------------------
    def log_summary(self, summary: dict):
        """Log key end-of-training values to run.summary.

        Only stores the most useful fields for the runs comparison table,
        not the full nested tree.  The complete data is already in the
        ``run-files`` artifact (training_history.json) and checkpoints.
        """
        if not self.enabled:
            return
        try:
            safe = sanitize_params(summary)

            # Best validation model
            bv = safe.get('best_validation_model', {})
            vm = bv.get('validation_metrics', {})
            for k, v in vm.items():
                fv = _safe_float(v)
                if not math.isnan(fv) and not math.isinf(fv):
                    self.run.summary[f"summary/best_val_{k}"] = fv
            if 'epoch' in bv:
                self.run.summary["summary/best_val_epoch"] = int(bv['epoch'])

            # Best loss model
            bl = safe.get('best_loss_model', {})
            if 'total_weighted_loss' in bl:
                self.run.summary["summary/best_loss_value"] = _safe_float(bl['total_weighted_loss'])
            if 'epoch' in bl:
                self.run.summary["summary/best_loss_epoch"] = int(bl['epoch'])

            # Training stats
            fs = safe.get('final_system', {})
            for k in ('total_training_time_seconds', 'total_epochs_run', 'total_steps_run'):
                if k in fs:
                    self.run.summary[f"summary/{k}"] = _safe_float(fs[k])

            # All physics losses (evaluated on best params)
            apl = safe.get('all_physics_losses', {})
            for k, v in apl.items():
                self.run.summary[f"summary/physics_{k}"] = _safe_float(v)

        except Exception as e:
            print(f"Warning: Failed to log summary to W&B: {e}")

    def log_artifact(self, path: str, name: str):
        """Attach a support file to this run's ``run-files`` artifact.

        All non-model files (config YAML, source script, training history)
        are bundled into a single ``type="run-files"`` artifact per run,
        keeping the artifact list compact.
        """
        if not self.enabled:
            return
        try:
            import wandb
            if self._run_files_artifact is None:
                self._run_files_artifact = wandb.Artifact(
                    name=f"files_{self._trial_name}",
                    type="run-files",
                )
            self._run_files_artifact.add_file(path, name=name)
        except Exception as e:
            print(f"Warning: Failed to stage artifact '{name}': {e}")

    def _flush_run_files(self):
        """Upload the accumulated run-files artifact (called at close)."""
        if self._run_files_artifact is not None:
            try:
                self.run.log_artifact(self._run_files_artifact)
            except Exception as e:
                print(f"Warning: Failed to upload run-files artifact: {e}")
            self._run_files_artifact = None

    def log_image(self, path: str, name: str):
        """Log an image so it appears in the W&B Media/Images panel."""
        if not self.enabled:
            return
        try:
            import wandb
            self.run.log({f"images/{name}": wandb.Image(path)}, commit=False)
        except Exception as e:
            print(f"Warning: Failed to log image '{name}': {e}")

    def log_scalars(self, scalars: dict, step: int, epoch: Optional[int] = None, prefix: str = "") -> None:
        """Track a flat dict of scalar values."""
        if not self.enabled:
            return
        try:
            metrics = {}
            for k, v in scalars.items():
                key = f"{prefix}/{k}" if prefix else k
                metrics[key] = _safe_float(v)
            self.run.log(metrics, step=epoch if epoch is not None else step)
        except Exception as e:
            print(f"Warning: W&B log_scalars failed at step {step}: {e}")

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------
    def log_run_status(self, status: str):
        """Set a terminal status in run summary."""
        if not self.enabled:
            return
        try:
            self.run.summary["run_status"] = status
        except Exception:
            pass

    def log_early_stopping(self, epoch: int, best_nse: float):
        """Record that early stopping fired."""
        if not self.enabled:
            return
        try:
            self.run.summary["early_stopped"] = True
            self.run.summary["early_stop_epoch"] = epoch + 1
            self.run.summary["early_stop_best_nse"] = best_nse
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Model registry (W&B Artifacts with type="model")
    # ------------------------------------------------------------------
    def register_best_model(self, model_path: str, metadata: dict = None) -> None:
        """Log the best checkpoint via ``run.log_model()``.

        Parameters
        ----------
        model_path : str
            Local path to the model weights file.
        metadata : dict, optional
            Extra metadata (NSE, RMSE, epoch, architecture, training time).
        """
        if not self.enabled:
            return
        try:
            self.run.log_model(
                path=model_path,
                name=f"model_{self._trial_name}",
                aliases=["best"],
                metadata=metadata,
            )
            print(f"Model logged: model_{self._trial_name}")
        except Exception as e:
            print(f"Warning: Failed to register model in W&B: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def delete_run(self):
        """Delete the run from W&B so rejected trials leave no trace."""
        if self.run is None:
            return
        try:
            import wandb

            # Capture identifiers before finishing
            run_id = self.run.id
            entity = self.run.entity
            project = self.run.project

            # Finish the run first (required before deletion)
            self.run.finish(exit_code=1)
            self.run = None

            # Delete the run via the W&B API
            api = wandb.Api()
            api_run = api.run(f"{entity}/{project}/{run_id}")
            api_run.delete()
            print(f"W&B run {run_id} deleted.")
        except Exception as e:
            print(f"Warning: Failed to delete W&B run: {e}")

    def close(self):
        if self.run is not None:
            try:
                self._flush_run_files()
                self.run.finish()
                self.run = None
                print("W&B run closed.")
            except Exception as e:
                print(f"Warning: Error closing W&B run: {e}")
