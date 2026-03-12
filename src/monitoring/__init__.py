"""Training monitoring: console output, Aim tracking, diagnostics, and legacy compat."""
from src.monitoring.console_logger import ConsoleLogger
from src.monitoring.aim_tracker import AimTracker, sanitize_for_aim
from src.monitoring.diagnostics import compute_negative_depth_diagnostics
from src.monitoring.legacy import print_epoch_stats, print_final_summary, log_metrics
