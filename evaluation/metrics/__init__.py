"""Evaluation metrics sub-package."""
from evaluation.metrics.accuracy import (
    compute_nse,
    compute_rmse,
    compute_mae,
    compute_rel_l2,
    compute_all_accuracy,
)
from evaluation.metrics.flood_metrics import (
    peak_depth_error,
    time_to_peak_error,
    critical_success_index,
)
from evaluation.metrics.conservation import (
    compute_volume_balance,
    compute_continuity_residual,
)
from evaluation.metrics.boundary import (
    slip_violation,
    inflow_boundary_error,
    outflow_gradient_residual,
    initial_condition_error,
)
from evaluation.metrics.cost import (
    timed,
    training_cost,
    inference_cost,
    break_even_query_count,
)
from evaluation.metrics.data_efficiency import (
    data_fraction,
    data_efficiency_ratio,
)

__all__ = [
    "compute_nse",
    "compute_rmse",
    "compute_mae",
    "compute_rel_l2",
    "compute_all_accuracy",
    "peak_depth_error",
    "time_to_peak_error",
    "critical_success_index",
    "compute_volume_balance",
    "compute_continuity_residual",
    "slip_violation",
    "inflow_boundary_error",
    "outflow_gradient_residual",
    "initial_condition_error",
    "timed",
    "training_cost",
    "inference_cost",
    "break_even_query_count",
    "data_fraction",
    "data_efficiency_ratio",
]
