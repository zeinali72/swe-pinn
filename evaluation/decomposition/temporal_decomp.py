"""Temporal error decomposition: rising, peak, and recession phases."""
import numpy as np
import jax.numpy as jnp

from src.metrics.decomposition import temporal_decomposition as _temporal_decomposition


def compute_temporal_decomposition(
    pred_h: np.ndarray,
    true_h: np.ndarray,
    t_coords: np.ndarray,
    pred_full: np.ndarray = None,
    true_full: np.ndarray = None,
) -> dict:
    """Decompose errors into rising, peak, and recession temporal phases.

    Args:
        pred_h: (N,) predicted water depth.
        true_h: (N,) reference water depth.
        t_coords: (N,) time coordinate for each point.
        pred_full: (N, 3) full predictions [h, hu, hv] (optional).
        true_full: (N, 3) full reference [h, hu, hv] (optional).

    Returns:
        Dict keyed by phase ('rising', 'peak', 'recession') with per-variable metrics.
    """
    return _temporal_decomposition(
        jnp.asarray(pred_h),
        jnp.asarray(true_h),
        jnp.asarray(t_coords),
        pred_full=jnp.asarray(pred_full) if pred_full is not None else None,
        true_full=jnp.asarray(true_full) if true_full is not None else None,
    )
