"""
Standardised training logger and Aim tracker for SWE-PINN experiments.

Usage:
    from logger_setup import setup_training_logger, AimTracker

    logger = setup_training_logger("experiment_8")
    tracker = AimTracker(config, repo_path="./aim_repo")
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


def setup_training_logger(
    experiment_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger with standardised format for PINN training.

    Parameters
    ----------
    experiment_name : str
        Used as the logger name suffix: ``pinn.{experiment_name}``.
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(f"pinn.{experiment_name}")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class AimTracker:
    """Manages Aim run lifecycle and provides formatted console output.

    Parameters
    ----------
    config : object
        Training configuration with attributes for experiment metadata.
    repo_path : str
        Path to the Aim repository.
    experiment_name : str, optional
        Override experiment name (otherwise derived from config).
    """

    def __init__(
        self,
        config: Any,
        repo_path: str = "./aim_repo",
        experiment_name: Optional[str] = None,
    ):
        self.config = config
        self.repo_path = repo_path
        self.experiment_name = experiment_name or getattr(
            config, "experiment_name", "unknown"
        )
        self.start_time = time.time()
        self.run = None

        self._init_aim_run()

    def _init_aim_run(self):
        """Initialise the Aim run and log run-level parameters."""
        try:
            from aim import Run

            self.run = Run(
                repo=self.repo_path,
                experiment=self.experiment_name,
            )
            self._log_run_params()
        except ImportError:
            # Aim not installed — tracking disabled, console output still works
            self.run = None

    def _log_run_params(self):
        """Store run-level parameters in Aim."""
        if self.run is None:
            return

        cfg = self.config

        # Standard parameters
        params = {}

        # Training section
        if hasattr(cfg, "training"):
            t = cfg.training
            params["lr_init"] = getattr(t, "learning_rate", None)
            params["max_epochs"] = getattr(t, "epochs", None)
            params["scheduler"] = getattr(t, "scheduler", "none")
            params["convergence_threshold"] = getattr(
                t, "convergence_threshold", None
            )

        # Model section
        if hasattr(cfg, "model"):
            m = cfg.model
            params["architecture"] = getattr(m, "name", "unknown")
            params["hidden_layers"] = getattr(m, "depth", None)
            params["hidden_units"] = getattr(m, "width", None)
            params["activation"] = getattr(m, "activation", "tanh")
            # Fourier-specific
            params["fourier_mapping_size"] = getattr(
                m, "fourier_mapping_size", None
            )
            params["fourier_scale"] = getattr(m, "fourier_scale", None)

        # Grid / sampling section
        if hasattr(cfg, "grid"):
            g = cfg.grid
            params["n_pde"] = getattr(g, "n_pde", None)
            params["n_ic"] = getattr(g, "n_ic", None)
            params["n_bc"] = getattr(g, "n_bc", None)
            params["n_data"] = getattr(g, "n_data", None)

        # Device section
        if hasattr(cfg, "device"):
            d = cfg.device
            params["precision"] = getattr(d, "dtype", "float32")

        params["experiment_name"] = self.experiment_name

        # System info
        try:
            import jax

            params["jax_version"] = jax.__version__
            devices = jax.devices()
            params["device"] = str(devices[0]) if devices else "cpu"
        except ImportError:
            pass

        try:
            import flax

            params["flax_version"] = flax.__version__
        except ImportError:
            pass

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        self.run["hparams"] = params

    def format_header(self, config: Any) -> str:
        """Format the run header block for console output."""
        cfg = config
        arch = getattr(getattr(cfg, "model", None), "name", "unknown")
        phase = getattr(cfg, "phase", "N/A")
        phase_name = getattr(cfg, "phase_name", "")
        precision = getattr(
            getattr(cfg, "device", None), "dtype", "float32"
        )

        # Grid / sampling counts
        grid = getattr(cfg, "grid", None)
        n_pde = getattr(grid, "n_pde", "?") if grid else "?"
        n_ic = getattr(grid, "n_ic", "?") if grid else "?"
        n_bc = getattr(grid, "n_bc", "?") if grid else "?"
        n_data = getattr(grid, "n_data", "?") if grid else "?"

        # Training params
        training = getattr(cfg, "training", None)
        lr = getattr(training, "learning_rate", "?") if training else "?"
        scheduler = getattr(training, "scheduler", "none") if training else "none"
        max_epochs = getattr(training, "epochs", "?") if training else "?"
        threshold = (
            getattr(training, "convergence_threshold", "N/A")
            if training
            else "N/A"
        )

        # Device info
        device_info = "CPU"
        try:
            import jax

            devices = jax.devices()
            if devices:
                device_info = str(devices[0])
        except ImportError:
            pass

        # HPO trial (optional)
        trial_id = getattr(cfg, "optuna_trial_id", None)
        trial_line = f"HPO Trial: {trial_id}\n" if trial_id else ""

        return (
            "================================================================\n"
            f"EXPERIMENT: {self.experiment_name}\n"
            f"ARCHITECTURE: {arch}\n"
            f"PHASE: {phase} - {phase_name}\n"
            f"DATE: {datetime.now().isoformat()}\n"
            f"DEVICE: {device_info}\n"
            f"PRECISION: {precision}\n"
            "================================================================\n"
            f"{trial_line}"
            f"Total collocation points: PDE={n_pde}, IC={n_ic}, BC={n_bc}, Data={n_data}\n"
            f"Learning rate: {lr}, Scheduler: {scheduler}\n"
            f"Max epochs: {max_epochs}, Convergence threshold: {threshold}\n"
            "================================================================"
        )

    def format_epoch(
        self,
        epoch: int,
        losses: Dict[str, float],
        lr: float,
        epoch_time: float,
        val_nse: Optional[float] = None,
        max_epochs: Optional[int] = None,
    ) -> str:
        """Format a single epoch log line."""
        max_ep = max_epochs or "?"
        nse_str = f"{val_nse:.4f}" if val_nse is not None else "N/A"
        return (
            f"Epoch {epoch:>6d}/{max_ep} | "
            f"Loss: {losses.get('total', 0):.6e} | "
            f"PDE: {losses.get('pde', 0):.6e} | "
            f"IC: {losses.get('ic', 0):.6e} | "
            f"BC: {losses.get('bc', 0):.6e} | "
            f"Data: {losses.get('data', 0):.6e} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s | "
            f"NSE(val): {nse_str}"
        )

    def track_epoch(
        self,
        epoch: int,
        losses: Dict[str, float],
        lr: float,
        epoch_time: float,
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Track epoch metrics in Aim.

        Parameters
        ----------
        epoch : int
        losses : dict
            Keys: 'total', 'pde', 'ic', 'bc', 'data'.
        lr : float
            Current learning rate.
        epoch_time : float
            Wall-clock seconds for this epoch.
        val_metrics : dict, optional
            Keys like 'nse_h', 'nse_hu', 'nse_hv', 'rmse_h', etc.
        """
        if self.run is None:
            return

        # Loss components
        for key in ("total", "pde", "ic", "bc", "data"):
            if key in losses:
                self.run.track(losses[key], name=f"loss/{key}", step=epoch)

        self.run.track(lr, name="lr", step=epoch)
        self.run.track(epoch_time, name="epoch_time", step=epoch)

        # Validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                self.run.track(value, name=f"metrics/{key}", step=epoch)

    def format_summary(
        self,
        final_epoch: int,
        converged: bool,
        criterion_desc: str,
        best_nse: float,
        best_epoch: int,
        losses: Dict[str, float],
    ) -> str:
        """Format the training completion summary block."""
        elapsed = time.time() - self.start_time
        td = timedelta(seconds=int(elapsed))
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return (
            "================================================================\n"
            "TRAINING COMPLETE\n"
            f"Final epoch: {final_epoch}\n"
            f"Convergence: {'yes' if converged else 'no'} "
            f"(criterion: {criterion_desc})\n"
            f"Best validation NSE: {best_nse:.4f} at epoch {best_epoch}\n"
            f"Total training time: {hours}h {minutes}m {seconds}s\n"
            f"Final loss breakdown: "
            f"PDE={losses.get('pde', 0):.6e}, "
            f"IC={losses.get('ic', 0):.6e}, "
            f"BC={losses.get('bc', 0):.6e}, "
            f"Data={losses.get('data', 0):.6e}\n"
            "================================================================"
        )

    def close(self):
        """Finalise and close the Aim run."""
        if self.run is not None:
            self.run.close()
