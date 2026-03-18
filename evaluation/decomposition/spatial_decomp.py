"""Spatial error decomposition: interior, boundary, and shock regions."""
import numpy as np
import jax.numpy as jnp

from src.metrics.decomposition import spatial_decomposition as _spatial_decomposition


def classify_points(
    coords: np.ndarray,
    true_h: np.ndarray,
    domain_bounds: dict,
    boundary_threshold_frac: float = 0.05,
) -> np.ndarray:
    """Return integer labels for each point: 0=interior, 1=boundary, 2=shock.

    Args:
        coords: (N, 3) coordinates [x, y, t].
        true_h: (N,) reference water depth.
        domain_bounds: Dict with 'lx', 'ly' (and optional 'x_min', 'y_min').
        boundary_threshold_frac: Fraction of domain size defining the boundary strip.

    Returns:
        (N,) integer array: 0=interior, 1=boundary, 2=shock.
    """
    coords = jnp.asarray(coords)
    true_h = jnp.asarray(true_h)

    x = coords[:, 0]
    y = coords[:, 1]
    lx = domain_bounds["lx"]
    ly = domain_bounds["ly"]
    x_min = domain_bounds.get("x_min", 0.0)
    y_min = domain_bounds.get("y_min", 0.0)

    dx = boundary_threshold_frac * lx
    dy = boundary_threshold_frac * ly

    # Shock detection
    h_sorted_idx = jnp.argsort(x)
    h_diff = jnp.abs(jnp.diff(true_h[h_sorted_idx]))
    h_diff = jnp.concatenate([h_diff, h_diff[-1:]])
    grad_mag = jnp.zeros_like(true_h)
    grad_mag = grad_mag.at[h_sorted_idx].set(h_diff)
    shock_threshold = jnp.percentile(grad_mag, 90)
    shock_mask = grad_mag > shock_threshold

    # Boundary detection
    dist_left = jnp.abs(x - x_min)
    dist_right = jnp.abs(x - (x_min + lx))
    dist_bottom = jnp.abs(y - y_min)
    dist_top = jnp.abs(y - (y_min + ly))
    min_dist = jnp.minimum(jnp.minimum(dist_left, dist_right), jnp.minimum(dist_bottom, dist_top))
    boundary_mask = min_dist < jnp.minimum(dx, dy)

    labels = jnp.zeros(coords.shape[0], dtype=jnp.int32)
    labels = jnp.where(boundary_mask, 1, labels)
    labels = jnp.where(shock_mask, 2, labels)
    return np.array(labels)


def compute_spatial_decomposition(
    pred: np.ndarray,
    true: np.ndarray,
    coords: np.ndarray,
    domain_bounds: dict,
    boundary_threshold_frac: float = 0.05,
) -> dict:
    """Decompose errors into shock, boundary, and interior regions.

    Args:
        pred: (N, 3) predictions [h, hu, hv].
        true: (N, 3) reference [h, hu, hv].
        coords: (N, 3) coordinates [x, y, t].
        domain_bounds: Dict with 'lx', 'ly'.
        boundary_threshold_frac: Fraction of domain defining the boundary strip.

    Returns:
        Dict keyed by region ('shock', 'boundary', 'interior') with per-variable metrics.
    """
    return _spatial_decomposition(
        jnp.asarray(pred),
        jnp.asarray(true),
        jnp.asarray(coords),
        domain_bounds,
        boundary_threshold_frac=boundary_threshold_frac,
    )
