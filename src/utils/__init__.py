"""Utilities: naming, I/O, plotting, domain geometry, and user interaction."""
from src.utils.naming import generate_trial_name
from src.utils.io import save_model
from src.utils.plotting import plot_h_vs_x, plot_comparison_scatter_2d
from src.utils.domain import mask_points_inside_building
from src.utils.ui import ask_for_confirmation

# Re-export metrics for backward compatibility.
# New code should import directly from src.metrics.accuracy.
from src.metrics.accuracy import nse, rmse, mae, relative_l2
