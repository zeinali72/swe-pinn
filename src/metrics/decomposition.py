"""Spatial and temporal error decomposition metrics."""
import numpy as np
import jax.numpy as jnp

from src.metrics.accuracy import compute_all_metrics


def spatial_decomposition(
    pred: jnp.ndarray,
    true: jnp.ndarray,
    coords: jnp.ndarray,
    domain_bounds: dict,
    boundary_threshold_frac: float = 0.05,
) -> dict:
    """Decompose errors into shock, boundary, and interior regions.

    Categories:
        - **shock**: points where |grad(h_true)| > 90th percentile
        - **boundary**: points within *boundary_threshold_frac* of domain edge
        - **interior**: everything else

    Args:
        pred: (N, 3) predictions [h, hu, hv].
        true: (N, 3) reference [h, hu, hv].
        coords: (N, 3) coordinates [x, y, t].
        domain_bounds: Dict with ``lx``, ``ly`` (and optional ``x_min``, ``y_min``).
        boundary_threshold_frac: Fraction of domain size defining the boundary strip.

    Returns:
        Dict keyed by region with per-variable metrics and point fraction.
    """
    h_true = true[..., 0]
    x = coords[:, 0]
    y = coords[:, 1]

    lx = domain_bounds["lx"]
    ly = domain_bounds["ly"]
    x_min = domain_bounds.get("x_min", 0.0)
    y_min = domain_bounds.get("y_min", 0.0)

    # Shock detection via finite-difference gradient magnitude estimate
    # Use pairwise differences of nearby points sorted by x then y
    dx = boundary_threshold_frac * lx
    dy = boundary_threshold_frac * ly

    # Simple gradient proxy: absolute difference from local mean
    # We approximate |grad(h)| as the local deviation
    h_sorted_idx = jnp.argsort(x)
    h_diff = jnp.abs(jnp.diff(h_true[h_sorted_idx]))
    h_diff = jnp.concatenate([h_diff, h_diff[-1:]])
    grad_mag = jnp.zeros_like(h_true)
    grad_mag = grad_mag.at[h_sorted_idx].set(h_diff)

    shock_threshold = jnp.percentile(grad_mag, 90)
    shock_mask = grad_mag > shock_threshold

    # Boundary mask
    dist_left = jnp.abs(x - x_min)
    dist_right = jnp.abs(x - (x_min + lx))
    dist_bottom = jnp.abs(y - y_min)
    dist_top = jnp.abs(y - (y_min + ly))
    min_dist = jnp.minimum(
        jnp.minimum(dist_left, dist_right),
        jnp.minimum(dist_bottom, dist_top),
    )
    boundary_mask = min_dist < jnp.minimum(dx, dy)

    # Interior = not shock and not boundary
    interior_mask = ~shock_mask & ~boundary_mask

    results = {}
    var_names = ["h", "hu", "hv"]
    for name, mask in [("shock", shock_mask), ("boundary", boundary_mask), ("interior", interior_mask)]:
        count = int(jnp.sum(mask))
        region = {"fraction": count / len(mask) if len(mask) > 0 else 0.0}
        if count > 0:
            for i, var in enumerate(var_names):
                p = pred[..., i][mask]
                t = true[..., i][mask]
                m = compute_all_metrics(p, t)
                for k, v in m.items():
                    region[f"{k}_{var}"] = v
        results[name] = region

    return results


def classify_points(
    coords: np.ndarray,
    true_h: np.ndarray,
    domain_bounds: dict,
    boundary_threshold_frac: float = 0.05,
) -> np.ndarray:
    """Classify evaluation points into spatial categories.

    Returns an integer label array:
      0 = interior
      1 = boundary  (within *boundary_threshold_frac* of any domain edge)
      2 = shock     (top-10% |grad h| proxy)

    Args:
        coords: (N, 3) coordinates [x, y, t].
        true_h: (N,) reference water depth.
        domain_bounds: Dict with ``lx``, ``ly`` (and optional ``x_min``, ``y_min``).
        boundary_threshold_frac: Fraction of domain size defining the boundary strip.

    Returns:
        (N,) integer numpy array of category labels.
    """
    coords = np.asarray(coords)
    true_h = np.asarray(true_h)
    x = coords[:, 0]
    y = coords[:, 1]

    lx = domain_bounds["lx"]
    ly = domain_bounds["ly"]
    x_min = domain_bounds.get("x_min", 0.0)
    y_min = domain_bounds.get("y_min", 0.0)

    dx = boundary_threshold_frac * lx
    dy = boundary_threshold_frac * ly

    # Boundary mask
    min_dist = np.minimum(
        np.minimum(np.abs(x - x_min), np.abs(x - (x_min + lx))),
        np.minimum(np.abs(y - y_min), np.abs(y - (y_min + ly))),
    )
    boundary_mask = min_dist < min(dx, dy)

    # Shock proxy: large local variation in h (sorted by x)
    sort_idx = np.argsort(x)
    h_sorted = true_h[sort_idx]
    h_diff = np.abs(np.diff(h_sorted))
    h_diff = np.append(h_diff, h_diff[-1])
    grad_mag = np.empty_like(true_h)
    grad_mag[sort_idx] = h_diff
    shock_threshold = np.percentile(grad_mag, 90)
    shock_mask = grad_mag > shock_threshold

    labels = np.zeros(len(x), dtype=np.int32)
    labels[boundary_mask] = 1
    labels[shock_mask & ~boundary_mask] = 2
    return labels


def temporal_decomposition(
    pred_h: jnp.ndarray,
    true_h: jnp.ndarray,
    t_coords: jnp.ndarray,
    pred_full: jnp.ndarray = None,
    true_full: jnp.ndarray = None,
) -> dict:
    """Decompose errors into rising, peak, and recession temporal phases.

    Phases:
        - **rising**: from first wet time to peak time
        - **peak**: t_peak +/- 10% of rise duration
        - **recession**: after peak window to end

    Args:
        pred_h: (N,) predicted water depth.
        true_h: (N,) reference water depth.
        t_coords: (N,) time values.
        pred_full: (N, 3) full predictions (optional, for per-variable metrics).
        true_full: (N, 3) full reference (optional).

    Returns:
        Dict keyed by phase with metrics.
    """
    t_peak = t_coords[jnp.argmax(true_h)]
    wet_mask = true_h > 1e-6
    if jnp.sum(wet_mask) == 0:
        return {"rising": {}, "peak": {}, "recession": {}}

    t_first_wet = jnp.min(jnp.where(wet_mask, t_coords, jnp.inf))
    rise_duration = float(t_peak - t_first_wet)
    if rise_duration < 1e-6:
        rise_duration = float(jnp.max(t_coords) - jnp.min(t_coords)) * 0.1

    peak_half_window = 0.1 * rise_duration
    t_peak_start = t_peak - peak_half_window
    t_peak_end = t_peak + peak_half_window

    rising_mask = (t_coords >= t_first_wet) & (t_coords < t_peak_start)
    peak_mask = (t_coords >= t_peak_start) & (t_coords <= t_peak_end)
    recession_mask = t_coords > t_peak_end

    results = {}
    for name, mask in [("rising", rising_mask), ("peak", peak_mask), ("recession", recession_mask)]:
        count = int(jnp.sum(mask))
        phase = {"fraction": count / len(mask) if len(mask) > 0 else 0.0}
        if count > 0:
            phase["h"] = compute_all_metrics(pred_h[mask], true_h[mask])
            if pred_full is not None and true_full is not None:
                for i, var in enumerate(["hu", "hv"]):
                    phase[var] = compute_all_metrics(
                        pred_full[..., i + 1][mask],
                        true_full[..., i + 1][mask],
                    )
        results[name] = phase

    return results
