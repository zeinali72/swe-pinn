"""Negative depth diagnostics for predicted water depth."""
import jax.numpy as jnp


def negative_depth_stats(pred_h: jnp.ndarray) -> dict:
    """Compute statistics on negative water-depth predictions.

    Args:
        pred_h: (N,) predicted water depth values.

    Returns:
        Dict with keys ``fraction``, ``min_h``, ``count``.
    """
    neg_mask = pred_h < 0.0
    count = int(jnp.sum(neg_mask))
    total = pred_h.shape[0]
    return {
        "fraction": count / total if total > 0 else 0.0,
        "min_h": float(jnp.min(pred_h)),
        "count": count,
    }
