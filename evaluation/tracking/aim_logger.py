"""Aim experiment tracker wrapper for T1-T4 tracked values."""
import time
from typing import Optional, Any

try:
    from aim import Run, Image
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False


class AimTracker:
    """Thin wrapper over aim.Run for SWE-PINN training tracking.

    If Aim is not installed, all methods are no-ops.
    """

    def __init__(self, experiment_name: str, config: dict = None, repo: str = None):
        self._run = None
        if AIM_AVAILABLE:
            kwargs = {"experiment": experiment_name}
            if repo is not None:
                kwargs["repo"] = repo
            self._run = Run(**kwargs)
            if config is not None:
                self._run["config"] = config

    def log_losses(self, epoch: int, losses: dict) -> None:
        """T1: Log all loss components at current epoch.

        Args:
            epoch: Current epoch index.
            losses: Dict with keys like 'total', 'pde', 'bc', 'ic', 'data', etc.
        """
        if self._run is None:
            return
        for key, value in losses.items():
            self._run.track(value, name=f"loss/{key}", step=epoch)

    def log_optimisation_state(self, epoch: int, state: dict) -> None:
        """T2: Log optimisation state (lr, grad_norm, epoch_time, clip_count).

        Args:
            epoch: Current epoch index.
            state: Dict with keys 'learning_rate', 'grad_norm', 'epoch_time_s', 'clip_count'.
        """
        if self._run is None:
            return
        for key, value in state.items():
            self._run.track(value, name=f"optim/{key}", step=epoch)

    def log_validation(self, epoch: int, metrics: dict) -> None:
        """T3: Log validation metrics evaluated every N epochs.

        Args:
            epoch: Current epoch index.
            metrics: Dict with keys like 'nse_h', 'rmse_h', etc.
        """
        if self._run is None:
            return
        for key, value in metrics.items():
            self._run.track(value, name=f"val/{key}", step=epoch)

    def log_hpo_trial(self, trial_number: int, trial_info: dict) -> None:
        """T4: Log HPO trial outcome.

        Args:
            trial_number: Optuna trial index.
            trial_info: Dict with 'nse', 'total_epochs', 'elapsed_s', 'pruned', 'params'.
        """
        if self._run is None:
            return
        self._run.track(trial_info.get("nse", float("nan")), name="hpo/nse", step=trial_number)
        self._run.track(trial_info.get("total_epochs", 0), name="hpo/total_epochs", step=trial_number)
        self._run.track(trial_info.get("elapsed_s", 0.0), name="hpo/elapsed_s", step=trial_number)
        self._run.track(int(trial_info.get("pruned", False)), name="hpo/pruned", step=trial_number)
        params = trial_info.get("params", {})
        for k, v in params.items():
            if isinstance(v, (int, float)):
                self._run.track(v, name=f"hpo/param/{k}", step=trial_number)

    def log_image(self, path: str, name: str, epoch: int = 0) -> None:
        """Log an image artifact."""
        if self._run is None or not AIM_AVAILABLE:
            return
        try:
            img = Image(path)
            self._run.track(img, name=name, step=epoch)
        except Exception:
            pass

    def finish(self) -> None:
        """Close the Aim run."""
        if self._run is not None:
            self._run.close()
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
