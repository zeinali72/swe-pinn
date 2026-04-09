"""Training monitoring: console output, W&B tracking, and diagnostics."""
from src.monitoring.console_logger import ConsoleLogger
from src.monitoring.wandb_tracker import WandbTracker, sanitize_params
from src.monitoring.diagnostics import compute_negative_depth_diagnostics
