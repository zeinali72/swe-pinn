"""Shared validation metrics for training and inference pipelines."""
from src.metrics.accuracy import (
    nse, rmse, mae, relative_l2,
    compute_all_metrics, compute_validation_metrics,
)
from src.metrics.peak import (
    r_squared, peak_depth_error, time_to_peak_error, rmse_mae_ratio,
)
from src.metrics.negative_depth import negative_depth_stats
from src.metrics.flood_extent import flood_extent_metrics
from src.metrics.conservation import volume_balance, continuity_residual
from src.metrics.boundary import slip_violation, inflow_accuracy, initial_condition_accuracy
from src.metrics.decomposition import spatial_decomposition, temporal_decomposition
