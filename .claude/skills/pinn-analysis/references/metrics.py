"""
Validation metrics and error decomposition for SWE-PINN experiments.

Usage:
    from metrics import compute_all_metrics, spatial_decomposition, temporal_decomposition

All computations use NumPy only (no sklearn dependency).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Core Metrics
# =============================================================================

def nse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency.

    Returns
    -------
    float
        NSE value. 1.0 = perfect, 0.0 = mean predictor, <0 = worse than mean.
    """
    ss_res = np.sum((pred - obs) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - obs)))


def r_squared(pred: np.ndarray, obs: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def peak_error(pred: np.ndarray, obs: np.ndarray) -> float:
    """Absolute difference at peak value."""
    return float(np.abs(np.max(pred) - np.max(obs)))


def csi(
    pred: np.ndarray,
    obs: np.ndarray,
    threshold: float,
) -> float:
    """Critical Success Index at a given depth threshold.

    Parameters
    ----------
    pred, obs : np.ndarray
        Predicted and observed water depth arrays.
    threshold : float
        Depth threshold in metres.

    Returns
    -------
    float
        CSI value in [0, 1].
    """
    tp = np.sum((pred >= threshold) & (obs >= threshold))
    fp = np.sum((pred >= threshold) & (obs < threshold))
    fn = np.sum((pred < threshold) & (obs >= threshold))
    denom = tp + fp + fn
    if denom == 0:
        return float("nan")
    return float(tp / denom)


def mass_conservation_deficit(
    v_inflow: float,
    v_domain: float,
) -> float:
    """Percentage mass conservation deficit.

    Parameters
    ----------
    v_inflow : float
        Total inflow volume (m³).
    v_domain : float
        Volume present in domain at final time step (m³).

    Returns
    -------
    float
        Deficit as percentage.
    """
    if v_inflow == 0:
        return float("nan")
    return 100.0 * abs(v_inflow - v_domain) / v_inflow


# =============================================================================
# Composite Metric Computation
# =============================================================================

def compute_all_metrics(
    pred_h: np.ndarray,
    obs_h: np.ndarray,
    pred_hu: np.ndarray,
    obs_hu: np.ndarray,
    pred_hv: np.ndarray,
    obs_hv: np.ndarray,
    csi_thresholds: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the full validation metric set for all three variables.

    Parameters
    ----------
    pred_h, obs_h : np.ndarray
        Predicted and observed water depth.
    pred_hu, obs_hu : np.ndarray
        Predicted and observed x-discharge.
    pred_hv, obs_hv : np.ndarray
        Predicted and observed y-discharge.
    csi_thresholds : list of float, optional
        Depth thresholds for CSI (default: [0.01, 0.05, 0.1, 0.3]).
    mask : np.ndarray of bool, optional
        If provided, only compute metrics where mask is True.

    Returns
    -------
    dict
        Keys like 'nse_h', 'rmse_hu', 'csi_0.05', etc.
    """
    if csi_thresholds is None:
        csi_thresholds = [0.01, 0.05, 0.1, 0.3]

    if mask is not None:
        pred_h, obs_h = pred_h[mask], obs_h[mask]
        pred_hu, obs_hu = pred_hu[mask], obs_hu[mask]
        pred_hv, obs_hv = pred_hv[mask], obs_hv[mask]

    # Exclude dry cells (h=0 in both pred and obs) from h metrics
    wet_mask = ~((pred_h == 0) & (obs_h == 0))
    ph_wet = pred_h[wet_mask] if wet_mask.any() else pred_h
    oh_wet = obs_h[wet_mask] if wet_mask.any() else obs_h

    results = {}

    # Per-variable metrics
    for var_name, p, o in [
        ("h", ph_wet, oh_wet),
        ("hu", pred_hu, obs_hu),
        ("hv", pred_hv, obs_hv),
    ]:
        results[f"nse_{var_name}"] = nse(p, o)
        results[f"rmse_{var_name}"] = rmse(p, o)
        results[f"mae_{var_name}"] = mae(p, o)
        results[f"r2_{var_name}"] = r_squared(p, o)
        results[f"peak_error_{var_name}"] = peak_error(p, o)

    # CSI at each threshold (uses full h arrays including dry cells)
    for thresh in csi_thresholds:
        results[f"csi_{thresh}"] = csi(pred_h, obs_h, thresh)

    results["n_points"] = int(len(pred_h))
    results["n_wet"] = int(wet_mask.sum()) if mask is None else int(len(ph_wet))

    return results


# =============================================================================
# Spatial Error Decomposition
# =============================================================================

def spatial_decomposition(
    h_field: np.ndarray,
    boundary_distances: np.ndarray,
    gradient_percentile: float = 90.0,
    proximity_threshold: float = None,
    grid_spacing: float = None,
    n_spacings: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify evaluation points into three spatial categories.

    Parameters
    ----------
    h_field : np.ndarray
        2D water depth field at a specific time step.
    boundary_distances : np.ndarray
        Distance of each point to nearest solid boundary (same shape as h_field).
    gradient_percentile : float
        Percentile threshold for shock/wet-dry front (default 90).
    proximity_threshold : float, optional
        Explicit proximity distance in metres. If None, derived from
        grid_spacing * n_spacings.
    grid_spacing : float, optional
        Grid spacing in metres (used if proximity_threshold is None).
    n_spacings : int
        Number of grid spacings for boundary proximity (default 3).

    Returns
    -------
    cat1_mask, cat2_mask, cat3_mask : np.ndarray of bool
        Boolean masks for each category.
    """
    # Category 1: shock / wet-dry front
    grad_y, grad_x = np.gradient(h_field)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_thresh = np.percentile(grad_mag, gradient_percentile)
    cat1_mask = grad_mag > grad_thresh

    # Category 2: boundary interaction
    if proximity_threshold is None:
        if grid_spacing is None:
            raise ValueError(
                "Either proximity_threshold or grid_spacing must be provided"
            )
        proximity_threshold = grid_spacing * n_spacings

    cat2_mask = (boundary_distances <= proximity_threshold) & ~cat1_mask

    # Category 3: smooth interior (everything else)
    cat3_mask = ~cat1_mask & ~cat2_mask

    return cat1_mask, cat2_mask, cat3_mask


def metrics_by_category(
    pred: Dict[str, np.ndarray],
    obs: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute metrics separately for each spatial category.

    Parameters
    ----------
    pred : dict
        Keys 'h', 'hu', 'hv' with 1D arrays of predictions.
    obs : dict
        Keys 'h', 'hu', 'hv' with 1D arrays of observations.
    masks : dict
        Keys 'shock', 'boundary', 'interior' with boolean masks.

    Returns
    -------
    pd.DataFrame
        Rows = categories, columns = metrics.
    """
    rows = []
    total_points = len(pred["h"])

    for cat_name, mask in masks.items():
        if mask.sum() == 0:
            continue
        m = compute_all_metrics(
            pred["h"][mask], obs["h"][mask],
            pred["hu"][mask], obs["hu"][mask],
            pred["hv"][mask], obs["hv"][mask],
        )
        m["category"] = cat_name
        m["fraction"] = mask.sum() / total_points
        rows.append(m)

    return pd.DataFrame(rows).set_index("category")


# =============================================================================
# Temporal Error Decomposition
# =============================================================================

def temporal_phases(
    h_timeseries: np.ndarray,
    dt: float,
    rising_percentile: float = 75.0,
    peak_window: int = 5,
) -> Dict[str, np.ndarray]:
    """Classify time steps into temporal phases.

    Parameters
    ----------
    h_timeseries : np.ndarray
        Spatially-averaged water depth at each time step.
    dt : float
        Time step interval in seconds.
    rising_percentile : float
        Percentile of dh/dt to define "rising" threshold (default 75).
    peak_window : int
        Number of time steps around peak to include in peak phase.

    Returns
    -------
    dict
        Keys 'rising', 'peak', 'recession', 'steady' with boolean index arrays.
    """
    dh_dt = np.gradient(h_timeseries, dt)
    rising_thresh = np.percentile(np.abs(dh_dt), rising_percentile)
    n = len(h_timeseries)

    peak_idx = np.argmax(h_timeseries)
    peak_start = max(0, peak_idx - peak_window)
    peak_end = min(n, peak_idx + peak_window + 1)

    peak_mask = np.zeros(n, dtype=bool)
    peak_mask[peak_start:peak_end] = True

    rising_mask = (dh_dt > rising_thresh) & ~peak_mask
    recession_mask = (dh_dt < -rising_thresh) & ~peak_mask
    steady_mask = ~rising_mask & ~recession_mask & ~peak_mask

    return {
        "rising": rising_mask,
        "peak": peak_mask,
        "recession": recession_mask,
        "steady": steady_mask,
    }


# =============================================================================
# Output Formatting
# =============================================================================

def metrics_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
) -> str:
    """Convert a metrics DataFrame to a LaTeX table string.

    Parameters
    ----------
    df : pd.DataFrame
    caption : str
        Table caption.
    label : str
        LaTeX label (e.g., 'tab:exp8_metrics').

    Returns
    -------
    str
        LaTeX table source.
    """
    return df.to_latex(
        float_format="%.4f",
        caption=caption,
        label=label,
        escape=False,
    )


def print_summary(metrics: Dict[str, float], title: str = "Validation Metrics"):
    """Print a formatted console summary of metrics.

    Parameters
    ----------
    metrics : dict
        Output from compute_all_metrics().
    title : str
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    for var in ("h", "hu", "hv"):
        print(f"\n  {var.upper()}:")
        print(f"    NSE  = {metrics.get(f'nse_{var}', float('nan')):.4f}")
        print(f"    RMSE = {metrics.get(f'rmse_{var}', float('nan')):.6f}")
        print(f"    MAE  = {metrics.get(f'mae_{var}', float('nan')):.6f}")
        print(f"    R²   = {metrics.get(f'r2_{var}', float('nan')):.4f}")
        print(f"    Peak = {metrics.get(f'peak_error_{var}', float('nan')):.6f}")

    # CSI
    csi_keys = sorted(k for k in metrics if k.startswith("csi_"))
    if csi_keys:
        print("\n  CSI:")
        for k in csi_keys:
            thresh = k.replace("csi_", "")
            print(f"    h >= {thresh}m : {metrics[k]:.4f}")

    print(f"\n  Points: {metrics.get('n_points', '?')} "
          f"(wet: {metrics.get('n_wet', '?')})")
    print(f"{'=' * 60}\n")
