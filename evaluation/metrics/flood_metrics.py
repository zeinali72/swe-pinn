"""Flood-specific metrics: peak depth error (A5), time-to-peak (A6), CSI (A7)."""
import numpy as np
import jax.numpy as jnp

from src.metrics.peak import peak_depth_error as _peak_depth_error
from src.metrics.flood_extent import flood_extent_metrics


def peak_depth_error(pred_h: np.ndarray, true_h: np.ndarray) -> float:
    """A5: Signed error between predicted and true peak water depth.

    Returns:
        max(h_pred) - max(h_ref), in metres.
    """
    return float(jnp.max(pred_h) - jnp.max(true_h))


def time_to_peak_error(
    pred_h: np.ndarray,
    true_h: np.ndarray,
    t_coords: np.ndarray,
) -> float:
    """A6: Signed time-to-peak error: t(max(h_pred)) - t(max(h_ref)).

    Positive = late; negative = early.
    """
    t_peak_pred = t_coords[jnp.argmax(pred_h)]
    t_peak_true = t_coords[jnp.argmax(true_h)]
    return float(t_peak_pred - t_peak_true)


def critical_success_index(
    pred_h: np.ndarray,
    true_h: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """A7: CSI = Hits / (Hits + Misses + False Alarms) at given depth threshold."""
    results = flood_extent_metrics(pred_h, true_h, thresholds=(threshold,))
    return results[f"threshold_{threshold}"]["csi"]
