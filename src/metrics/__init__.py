"""Shared validation metrics for training and inference pipelines."""
# A1-A4: Accuracy
from src.metrics.accuracy import (
    nse, rmse, mae, relative_l2,
    compute_all_metrics, compute_validation_metrics, compute_all_accuracy,
)
# A5-A7: Flood / peak metrics
from src.metrics.peak import (
    r_squared, peak_depth_error, time_to_peak_error, rmse_mae_ratio,
)
from src.metrics.negative_depth import negative_depth_stats
from src.metrics.flood_extent import flood_extent_metrics

# B1-B3: Conservation
from src.metrics.conservation import volume_balance, continuity_residual

# C1-C4: Boundary conditions
from src.metrics.boundary import (
    slip_violation,
    inflow_accuracy,
    inflow_boundary_error,
    outflow_gradient_residual,
    initial_condition_accuracy,
)

# D1-D4: Computational cost
from src.metrics.cost import (
    timed, training_cost, inference_cost, break_even_query_count,
)

# F1-F2: Data efficiency
from src.metrics.data_efficiency import data_fraction, data_efficiency_ratio

# Decomposition
from src.metrics.decomposition import (
    spatial_decomposition, temporal_decomposition, classify_points,
)
