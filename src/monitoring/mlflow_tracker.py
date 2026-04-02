"""MLflow experiment tracking wrapper.

Provides structured metric logging using the key hierarchy defined in the
training monitoring specification. All JAX/NumPy values are sanitised to
native Python types before being sent to MLflow.

X-axis options in the MLflow UI chart view:
  - "Step"          → epoch number  (primary, passed as MLflow step)
  - "Relative Time" → wall-clock seconds since run start  (automatic)
  - "Wall Time"     → absolute timestamp  (automatic)
  - Custom metric   → select "x.global_step" in the compare chart to plot
                      against total batch updates instead of epochs

Experiment grouping:
  Experiment = scenario name  (experiment_1, experiment_2, …)
  Run name   = trial_name     (date-time + arch, e.g. 2025-03-19_14-32_fourier)
  Tags       = scenario, model, variant flags

Model registry:
  Call register_model(model_uri, name) after saving to promote a run's
  checkpoint into the registry with a versioned, named entry.
"""
import copy
import json
import os
import re
import tempfile
from typing import Optional

try:
    import jax as _jax
except Exception:
    _jax = None  # type: ignore[assignment]

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
        self._registry_name = None

        if not enable:
            return

        try:
            import mlflow

            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
            if not tracking_uri:
                # Default: SQLite backend — single .db file instead of
                # thousands of small files per run.
                db_path = os.path.abspath("mlflow_repo/mlflow.db")
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                tracking_uri = f"sqlite:///{db_path}"
            elif not tracking_uri.startswith(("http://", "https://", "sqlite://", "postgresql://", "mysql://")):
                # Plain directory path — resolve to absolute for file backend.
                tracking_uri = f"file://{os.path.abspath(tracking_uri)}"
            self._tracking_uri = tracking_uri
            mlflow.set_tracking_uri(tracking_uri)

            scenario = config.get('scenario', '')
            experiment_label = scenario if scenario else trial_name

            # Ensure experiment exists with artifact root co-located under
            # mlflow_repo/artifacts/ so everything stays in one directory.
            # For remote tracking URIs (DagsHub, MLflow server) the server
            # manages artifact storage, so we only set artifact_location for
            # local SQLite/file backends.
            is_local = tracking_uri.startswith(("sqlite://", "file://"))
            artifact_root = f"file://{os.path.abspath('mlflow_repo/artifacts')}"
            experiment = mlflow.get_experiment_by_name(experiment_label)
            if experiment is None and is_local:
                mlflow.create_experiment(experiment_label, artifact_location=artifact_root)
            elif experiment is not None and is_local:
                if experiment.artifact_location and experiment.artifact_location != artifact_root:
                    print(
                        f"Warning: Experiment '{experiment_label}' has artifact location "
                        f"'{experiment.artifact_location}', expected '{artifact_root}'. "
                        "Artifacts may be stored in a different directory."
                    )
            mlflow.set_experiment(experiment_label)

            self.run = mlflow.start_run(run_name=trial_name)
            self.run_id = self.run.info.run_id

            # Build registry name: one entry per scenario+arch, versions accumulate
            # across runs (e.g. "experiment_1_mlp", "experiment_2_fourierpinn").
            arch_name = config.get('model', {}).get('name', '')
            if scenario and arch_name:
                self._registry_name = f"{scenario}_{arch_name.lower()}"
            elif scenario:
                self._registry_name = scenario

            # Tags for filtering / grouping in the UI
            data_weight = float(config.get('loss_weights', {}).get('data_weight', 0) or 0)
            data_mode = 'data_driven' if data_weight > 0 else 'data_free'

            phase = ''
            m = re.search(r'experiment_(\d+)', scenario)
            if m:
                n = int(m.group(1))
                phase = '1' if n <= 2 else ('2' if n <= 6 else '3')

            # Resolve JAX backend at run time so CPU-override envvars set
            # after import (e.g. JAX_PLATFORM_NAME=cpu in tests) are respected.
            try:
                jax_backend = _jax.default_backend() if _jax is not None else 'unknown'
                jax_device_count = str(len(_jax.devices())) if _jax is not None else '0'
            except Exception:
                jax_backend = 'unknown'
                jax_device_count = '0'

            tags = {
                "trial_name": trial_name,
                "data_mode": data_mode,
                "jax_backend": jax_backend,
                "jax_device_count": jax_device_count,
            }
            if scenario:
                tags["scenario"] = scenario
            if arch_name:
                tags["arch"] = arch_name.lower()
                tags["model"] = arch_name
            if phase:
                tags["phase"] = phase
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

        MLflow step = epoch, so the default x-axis reads "epoch 0 … N".
        ``x.global_step`` is logged as a metric so the compare chart can be
        switched to batch-step x-axis by selecting it as the custom x metric.
        Relative time and wall time are available automatically in the UI.
        """
        if not self.enabled:
            return
        try:
            import mlflow

            metrics: dict = {}

            # x-axis alternatives: global_step lets the UI compare view use
            # batch steps as a custom x-axis instead of epochs.
            metrics["x.global_step"] = float(step)

            # Training losses
            metrics["loss/total"] = _safe_float(total_loss)
            for key, val in losses.items():
                metrics[f"loss/{key}"] = _safe_float(val)

            # Validation metrics
            for key, val in val_metrics.items():
                default = -float('inf') if 'nse' in key else float('inf')
                metrics[f"val/{key}"] = _safe_float(val, default)

            # Learning rate
            metrics["optim/lr"] = _safe_float(lr)

            # Negative-depth diagnostics (only at report frequency)
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

        Stored as:
        - A JSON artifact for full fidelity.
        - Flattened scalar leaves as run *params* (static single values, not
          time-series), so they appear cleanly in the runs comparison table.
        """
        if not self.enabled:
            return
        try:
            import mlflow
            safe = sanitize_params(summary)

            # JSON artifact
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
    # Run lifecycle
    # ------------------------------------------------------------------
    def log_run_status(self, status: str):
        """Set a terminal status tag: ``'completed'``, ``'interrupted'``, or ``'error'``."""
        if not self.enabled:
            return
        try:
            import mlflow
            mlflow.set_tag("run_status", status)
        except Exception:
            pass

    def log_early_stopping(self, epoch: int, best_nse: float):
        """Record that early stopping fired, for quick filtering in the UI."""
        if not self.enabled:
            return
        try:
            import mlflow
            mlflow.set_tags({
                "early_stopped": "true",
                "early_stop_epoch": str(epoch + 1),
                "early_stop_best_nse": f"{best_nse:.6f}",
            })
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------
    def register_best_model(self, artifact_path: str) -> Optional[str]:
        """Register the best checkpoint under the canonical ``scenario_arch`` name.

        Versions accumulate across runs, e.g. ``experiment_1_mlp`` v1, v2, v3 …
        Falls back to a no-op when no registry name could be derived at init.
        """
        if not self.enabled:
            return None
        if not self._registry_name:
            print("Warning: register_best_model skipped — registry name could not be derived "
                  "(scenario or arch missing from config).")
            return None
        return self.register_model(artifact_path, self._registry_name)

    def register_model(self, artifact_path: str, model_name: str) -> Optional[str]:
        """Register a logged artifact in the MLflow Model Registry.

        Parameters
        ----------
        artifact_path : str
            Path within the run's artifact store, e.g. ``"artifacts/model.pkl"``.
        model_name : str
            Registry name, e.g. ``"experiment_1_fourier"``.

        Returns
        -------
        str or None
            The registered model version string, or None on failure.
        """
        if not self.enabled:
            return None
        try:
            import mlflow
            model_uri = f"runs:/{self.run_id}/{artifact_path}"
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            print(f"Model registered: {model_name} v{result.version}")
            return result.version
        except Exception as e:
            print(f"Warning: Failed to register model '{model_name}': {e}")
            return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def delete_run(self):
        """End and soft-delete the active run."""
        if not self.run_id:
            return
        try:
            import mlflow
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
