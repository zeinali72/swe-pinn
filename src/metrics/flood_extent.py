"""Flood-extent classification metrics at multiple depth thresholds."""
import jax.numpy as jnp


def flood_extent_metrics(
    pred_h: jnp.ndarray,
    true_h: jnp.ndarray,
    thresholds: tuple = (0.01, 0.05, 0.1, 0.3),
) -> dict:
    """Compute CSI, hit rate, and FAR at each depth threshold.

    Args:
        pred_h: (N,) predicted water depth.
        true_h: (N,) reference water depth.
        thresholds: Depth thresholds (metres) for wet/dry classification.

    Returns:
        Dict keyed by threshold with sub-dicts of ``csi``, ``hit_rate``, ``far``.
    """
    results = {}
    for th in thresholds:
        pred_wet = pred_h >= th
        true_wet = true_h >= th

        tp = float(jnp.sum(pred_wet & true_wet))
        fp = float(jnp.sum(pred_wet & ~true_wet))
        fn = float(jnp.sum(~pred_wet & true_wet))

        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0

        results[f"threshold_{th}"] = {
            "csi": csi,
            "hit_rate": hit_rate,
            "far": far,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
    return results
