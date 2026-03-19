"""Aim experiment tracking wrapper.

Provides structured metric logging using the key hierarchy defined in the
training monitoring specification. All JAX/NumPy values are sanitised to
native Python types before being sent to Aim.
"""
import copy
from typing import Optional


def _safe_float(val, default=float('nan')):
    """Convert JAX/NumPy scalars to Python floats."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def sanitize_for_aim(obj):
    """Recursively convert JAX/NumPy arrays to native Python types for Aim."""
    if isinstance(obj, dict):
        return {k: sanitize_for_aim(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_for_aim(v) for v in obj)
    try:
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (float, int)):
            return obj
        return float(obj)
    except (TypeError, ValueError):
        return obj


class AimTracker:
    """Wraps the Aim SDK with structured metric logging.

    If Aim is unavailable or ``enable=False``, all methods become no-ops.

    Grouping hierarchy in the Aim UI:

    * **Experiment** — ``config["scenario"]`` (e.g. ``"experiment_1"``).
      All runs for the same scenario appear under one experiment group.
    * **Run name** — ``trial_name`` (date-time prefixed, e.g.
      ``"2025-03-19_14-32_experiment_1_fourier"``).
    * **Tags** — scenario, model architecture, and any variant labels
      passed via :meth:`log_flags` (e.g. ``"importance_sampling"``,
      ``"relobralo"``).
    """

    def __init__(self, config: dict, trial_name: str, enable: bool = True):
        self.aim_run = None
        self.aim_repo = None
        self.run_hash = None
        self._enabled = enable

        if not enable:
            return

        try:
            from aim import Repo, Run
            import os

            aim_repo_path = "aim_repo"
            os.makedirs(aim_repo_path, exist_ok=True)
            self.aim_repo = Repo(path=aim_repo_path, init=True)

            # Experiment = scenario (groups runs: experiment_1, experiment_2, …)
            scenario = config.get('scenario', '')
            experiment_label = scenario if scenario else trial_name
            self.aim_run = Run(repo=self.aim_repo, experiment=experiment_label)
            self.run_hash = self.aim_run.hash

            # Run name = trial_name (date-time prefix, visible in Aim UI)
            self.aim_run.name = trial_name

            # Tags: scenario + model architecture
            if scenario:
                self.aim_run.add_tag(scenario)
            model_name = config.get('model', {}).get('name', '')
            if model_name:
                self.aim_run.add_tag(model_name)

            artifact_path = os.path.join(aim_repo_path, "aim_artifacts")
            os.makedirs(artifact_path, exist_ok=True)
            abs_artifact_path = os.path.abspath(artifact_path)
            self.aim_run.set_artifacts_uri(f"file://{abs_artifact_path}")

            self.aim_run["hparams"] = sanitize_for_aim(
                copy.deepcopy(dict(config))
            )
            print(
                f"Aim tracking initialised: {trial_name} [{experiment_label}] ({self.run_hash})"
            )
        except Exception as e:
            print(
                f"Warning: Failed to initialise Aim tracking: {e}. "
                "Training will continue without Aim."
            )
            self.aim_run = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self.aim_run is not None

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
        if not self.enabled:
            return
        try:
            run = self.aim_run
            ctx_train = {'subset': 'train'}
            ctx_val = {'subset': 'validation'}
            ctx_sys = {'subset': 'system'}
            ctx_diag = {'subset': 'diagnostics'}

            # B.1 Loss components
            run.track(
                _safe_float(total_loss), name='loss/total',
                step=step, epoch=epoch, context=ctx_train,
            )
            for key, val in losses.items():
                run.track(
                    _safe_float(val), name=f'loss/{key}',
                    step=step, epoch=epoch, context=ctx_train,
                )

            # C Optimiser diagnostics
            run.track(
                _safe_float(lr), name='optim/lr',
                step=step, epoch=epoch, context=ctx_train,
            )
            run.track(
                _safe_float(epoch_time), name='optim/epoch_time_sec',
                step=step, epoch=epoch, context=ctx_sys,
            )
            run.track(
                _safe_float(elapsed_time), name='system/elapsed_time',
                step=step, epoch=epoch, context=ctx_sys,
            )

            # D Validation metrics
            for key, val in val_metrics.items():
                default = -float('inf') if 'nse' in key else float('inf')
                run.track(
                    _safe_float(val, default), name=f'val/{key}',
                    step=step, epoch=epoch, context=ctx_val,
                )

            # B.2 Negative depth diagnostics
            if neg_depth:
                for key, val in neg_depth.items():
                    run.track(
                        _safe_float(val),
                        name=f'diagnostics/negative_h_{key}',
                        step=step, epoch=epoch, context=ctx_diag,
                    )

        except Exception as e:
            print(f"Warning: Aim logging failed at epoch {epoch}: {e}")

    # ------------------------------------------------------------------
    # Best-model tracking (E.4)
    # ------------------------------------------------------------------
    def log_best_nse(self, nse_h: float, epoch: int, step: int):
        if not self.enabled:
            return
        try:
            self.aim_run.track(
                _safe_float(nse_h), name='best/nse_h_value',
                step=step, epoch=epoch,
            )
            self.aim_run['best_nse_h_epoch'] = epoch + 1
        except Exception as e:
            print(f"Warning: Failed to log best NSE to Aim: {e}")

    def log_best_loss(self, loss: float, epoch: int, step: int):
        if not self.enabled:
            return
        try:
            self.aim_run.track(
                _safe_float(loss), name='best/loss_value',
                step=step, epoch=epoch,
            )
            self.aim_run['best_loss_epoch'] = epoch + 1
        except Exception as e:
            print(f"Warning: Failed to log best loss to Aim: {e}")

    # ------------------------------------------------------------------
    # Run-level metadata (F)
    # ------------------------------------------------------------------
    def log_flags(self, flags: dict):
        """Store flags as run metadata and add variant labels as tags.

        String values in *flags* are added as tags so that runs can be
        filtered by variant within an experiment group (e.g. tag
        ``"relobralo"`` from ``loss_weighting`` flag).
        """
        if not self.enabled:
            return
        try:
            self.aim_run['flags'] = sanitize_for_aim(flags)
            for val in flags.values():
                if isinstance(val, str) and val:
                    self.aim_run.add_tag(val)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Summary and artifact logging
    # ------------------------------------------------------------------
    def log_summary(self, summary: dict):
        if not self.enabled:
            return
        try:
            self.aim_run['summary'] = sanitize_for_aim(summary)
        except Exception as e:
            print(f"Warning: Failed to log summary to Aim: {e}")

    def log_artifact(self, path: str, name: str):
        if not self.enabled:
            return
        try:
            self.aim_run.log_artifact(path, name=name)
        except Exception as e:
            print(f"Warning: Failed to log artifact '{name}': {e}")

    def log_image(self, path: str, name: str):
        if not self.enabled:
            return
        try:
            from aim import Image
            self.aim_run["images", name] = Image(path)
        except Exception as e:
            print(f"Warning: Failed to log image '{name}': {e}")

    def log_scalars(self, scalars: dict, step: int, epoch: Optional[int] = None, prefix: str = "") -> None:
        """Track a flat dict of scalar values under an optional name prefix."""
        if not self.enabled:
            return
        try:
            for k, v in scalars.items():
                name = f"{prefix}/{k}" if prefix else k
                self.aim_run.track(_safe_float(v), name=name, step=step, epoch=epoch)
        except Exception as e:
            print(f"Warning: Aim log_scalars failed at step {step}: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def delete_run(self):
        if self.aim_run and self.run_hash and self.aim_repo:
            try:
                self.aim_repo.delete_run(self.run_hash)
                print("Aim run deleted.")
            except Exception as e:
                print(f"Warning: Failed to delete Aim run: {e}")

    def close(self):
        if self.aim_run:
            try:
                self.aim_run.close()
                print("Aim run closed.")
            except Exception as e:
                print(f"Warning: Error closing Aim run: {e}")
