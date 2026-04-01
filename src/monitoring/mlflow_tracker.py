"""MLflow experiment tracking wrapper.

Provides structured metric logging using the key hierarchy defined in the
training monitoring specification. All JAX/NumPy values are sanitised to
native Python types before being sent to MLflow.

Logged metrics (x-axis = epoch):
  loss/total, loss/<component>   — training losses
  val/nse_h, val/rmse_h, ...     — validation metrics
  optim/lr                       — learning rate
  best/nse_h_value               — best NSE seen so far
  best/loss_value                — best total loss seen so far
  diagnostics/negative_h_*      — negative-depth stats (at report freq only)
  relobralo_weights/*            — adaptive loss weights (ReLoBRaLo only)

Summary (logged once as params at end of training, not as time-series):
  summary.*                      — best model stats, total time, etc.
"""
import copy
import json
import os
import tempfile
from typing import Optional

# MLflow backend limits
_PARAM_KEY_MAX = 250
_PARAM_VAL_MAX = 500


def _safe_float(val, default=float('nan')):
    """Convert JAX/NumPy scalars to Python floats."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def sanitize_params(obj):
    """Recursively convert JAX/NumPy arrays to native Python types for MLflow."""
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


class MLflowTracker:
    """Wraps the MLflow SDK with structured metric logging.

    If MLflow is unavailable or ``enable=False``, all methods become no-ops.

    Grouping hierarchy in the MLflow UI:

    * **Experiment** — ``config["scenario"]`` (e.g. ``"experiment_1"``).
      All runs for the same scenario appear under one experiment.
    * **Run name** — ``trial_name`` (date-time prefixed, e.g.
      ``"2025-03-19_14-32_experiment_1_fourier"``).
    * **Tags** — scenario, model architecture, and any variant labels
      passed via :meth:`log_flags`.
    """

    def __init__(self, config: dict, trial_name: str, enable: bool = True):
        self.run = None
        self.run_id = None
        self._enabled = enable
        self._tracking_uri = None

        if not enable:
            return

        try:
            import mlflow

            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlflow_repo")
            # Resolve to absolute path so scripts run from any directory write
            # to the same store.
            if not tracking_uri.startswith(("http://", "https://", "file://")):
                tracking_uri = os.path.abspath(tracking_uri)
            os.makedirs(tracking_uri, exist_ok=True)
            self._tracking_uri = tracking_uri
            mlflow.set_tracking_uri(tracking_uri)

            scenario = config.get('scenario', '')
            experiment_label = scenario if scenario else trial_name
            mlflow.set_experiment(experiment_label)

            self.run = mlflow.start_run(run_name=trial_name)
            self.run_id = self.run.info.run_id

            # Tags: scenario + model architecture + variant info
            tags = {"trial_name": trial_name}
            if scenario:
                tags["scenario"] = scenario
            model_name = config.get('model', {}).get('name', '')
            if model_name:
                tags["model"] = model_name
            mlflow.set_tags(tags)

            # Log hyperparameters as flattened params.
            # MLflow limits: key ≤ 250 chars, value ≤ 500 chars.
            try:
                flat = _flatten_dict(sanitize_params(copy.deepcopy(dict(config))))
                str_params = {
                    k[:_PARAM_KEY_MAX]: str(v)[:_PARAM_VAL_MAX]
                    for k, v in flat.items()
                }
                mlflow.log_params(str_params)
            except Exception as e:
                print(f"Warning: Failed to log hparams to MLflow: {e}")

            print(f"MLflow tracking initialised: {trial_name} [{experiment_label}] ({self.run_id})")

        except Exception as e:
            print(
                f"Warning: Failed to initialise MLflow tracking: {e}. "
                "Training will continue without MLflow."
            )
            self.run = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self.run is not None

    # ------------------------------------------------------------------
    # Per-epoch structured logging  (x-axis = epoch number)
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
        if not self.enabled:
            return
        try:
            import mlflow

            # Core training metrics — epoch is the x-axis so charts read
            # "epoch 0 … N" rather than raw batch-step counts.
            metrics = {"loss/total": _safe_float(total_loss)}
            for key, val in losses.items():
                metrics[f"loss/{key}"] = _safe_float(val)

            # Validation metrics
            for key, val in val_metrics.items():
                default = -float('inf') if 'nse' in key else float('inf')
                metrics[f"val/{key}"] = _safe_float(val, default)

            # Learning rate (useful for scheduler debugging)
            metrics["optim/lr"] = _safe_float(lr)

            # Negative-depth diagnostics — only logged at report frequency,
            # so neg_depth is None on non-reporting epochs.
            if neg_depth:
                for key, val in neg_depth.items():
                    metrics[f"diagnostics/negative_h_{key}"] = _safe_float(val)

            mlflow.log_metrics(metrics, step=epoch)
        except Exception as e:
            print(f"Warning: MLflow logging failed at epoch {epoch}: {e}")

    # ------------------------------------------------------------------
    # Best-model tracking  (x-axis = epoch)
    # ------------------------------------------------------------------
    def log_best_nse(self, nse_h: float, epoch: int, step: int):
        if not self.enabled:
            return
        try:
            import mlflow
            mlflow.log_metric("best/nse_h_value", _safe_float(nse_h), step=epoch)
            mlflow.set_tag("best_nse_h_epoch", str(epoch + 1))
        except Exception as e:
            print(f"Warning: Failed to log best NSE to MLflow: {e}")

    def log_best_loss(self, loss: float, epoch: int, step: int):
        if not self.enabled:
            return
        try:
            import mlflow
            mlflow.log_metric("best/loss_value", _safe_float(loss), step=epoch)
            mlflow.set_tag("best_loss_epoch", str(epoch + 1))
        except Exception as e:
            print(f"Warning: Failed to log best loss to MLflow: {e}")

    # ------------------------------------------------------------------
    # Run-level metadata
    # ------------------------------------------------------------------
    def log_flags(self, flags: dict):
        """Store flags as run tags."""
        if not self.enabled:
            return
        try:
            import mlflow
            safe_flags = sanitize_params(flags)
            tags = {k: str(v) for k, v in safe_flags.items() if v is not None}
            mlflow.set_tags(tags)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary and artifact logging
    # ------------------------------------------------------------------
    def log_summary(self, summary: dict):
        """Log end-of-training summary.

        Summary values are stored as:
        - A JSON artifact (full fidelity, nested structure preserved).
        - Flattened scalar leaves as run *params* (not metrics), so they
          appear as single values in the UI rather than single-point time
          series / vectors.
        """
        if not self.enabled:
            return
        try:
            import mlflow
            safe = sanitize_params(summary)

            # JSON artifact for full fidelity
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', prefix=f'mlflow_summary_{self.run_id}_',
                delete=False
            ) as f:
                json.dump(safe, f, indent=2)
                summary_path = f.name
            try:
                mlflow.log_artifact(summary_path, artifact_path="summary")
            finally:
                os.unlink(summary_path)

            # Scalar leaves as params — single values, not time-series.
            flat = _flatten_dict(safe)
            summary_params = {}
            for k, v in flat.items():
                try:
                    summary_params[f"summary.{k}"[:_PARAM_KEY_MAX]] = str(round(float(v), 6))[:_PARAM_VAL_MAX]
                except (TypeError, ValueError):
                    pass
            if summary_params:
                mlflow.log_params(summary_params)
        except Exception as e:
            print(f"Warning: Failed to log summary to MLflow: {e}")

    def log_artifact(self, path: str, name: str):
        """Log a file artifact.

        ``name`` is accepted for API compatibility but MLflow stores the file
        under its original filename inside the ``artifacts/`` directory.
        """
        if not self.enabled:
            return
        try:
            import mlflow
            mlflow.log_artifact(path, artifact_path="artifacts")
        except Exception as e:
            print(f"Warning: Failed to log artifact '{name}': {e}")

    def log_image(self, path: str, name: str):
        """Log an image file artifact.

        ``name`` is accepted for API compatibility but MLflow stores the file
        under its original filename inside the ``images/`` directory.
        """
        if not self.enabled:
            return
        try:
            import mlflow
            mlflow.log_artifact(path, artifact_path="images")
        except Exception as e:
            print(f"Warning: Failed to log image '{name}': {e}")

    def log_scalars(self, scalars: dict, step: int, epoch: Optional[int] = None, prefix: str = "") -> None:
        """Track a flat dict of scalar values under an optional name prefix.

        Uses ``epoch`` as the x-axis step when provided, otherwise falls back
        to ``step`` (global batch count).
        """
        if not self.enabled:
            return
        try:
            import mlflow
            metrics = {}
            for k, v in scalars.items():
                key = f"{prefix}/{k}" if prefix else k
                metrics[key] = _safe_float(v)
            mlflow.log_metrics(metrics, step=epoch if epoch is not None else step)
        except Exception as e:
            print(f"Warning: MLflow log_scalars failed at step {step}: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def delete_run(self):
        """End and soft-delete the active run."""
        if not self.run_id:
            return
        try:
            import mlflow
            # End the run before deleting so MLflow's active-run context is
            # cleared; prevents close() from attempting a double end_run.
            mlflow.end_run()
            mlflow.delete_run(self.run_id)
            self.run = None
            print("MLflow run deleted.")
        except Exception as e:
            print(f"Warning: Failed to delete MLflow run: {e}")

    def close(self):
        if self.run is not None:
            try:
                import mlflow
                mlflow.end_run()
                self.run = None
                print("MLflow run closed.")
            except Exception as e:
                print(f"Warning: Error closing MLflow run: {e}")
