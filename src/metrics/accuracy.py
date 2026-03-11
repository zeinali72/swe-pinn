"""Accuracy metrics for PINN predictions: NSE, RMSE, MAE, Relative L2.

This module is the single source of truth for accuracy computation.
Both the training loop (periodic validation) and the inference pipeline
call these same functions, ensuring consistency.
"""
import jax.numpy as jnp


def nse(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency (NSE).

    Returns 1.0 for perfect prediction, 0.0 when prediction equals the mean,
    and negative values for worse-than-mean predictions.
    """
    num = jnp.sum((true - pred) ** 2)
    den = jnp.sum((true - jnp.mean(true)) ** 2)
    if den < 1e-9:
        return -jnp.inf
    return 1 - num / den


def rmse(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """Compute Root Mean Square Error (RMSE)."""
    return jnp.sqrt(jnp.mean((pred - true) ** 2))


def mae(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    return jnp.mean(jnp.abs(pred - true))


def relative_l2(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """Compute Relative L2 norm error: ||pred - true||_2 / ||true||_2."""
    num = jnp.sqrt(jnp.sum((pred - true) ** 2))
    den = jnp.sqrt(jnp.sum(true ** 2))
    if den < 1e-9:
        return jnp.inf
    return num / den


def compute_all_metrics(pred: jnp.ndarray, true: jnp.ndarray) -> dict:
    """Compute all accuracy metrics for a single variable.

    Returns:
        Dict with keys 'nse', 'rmse', 'mae', 'rel_l2'.
    """
    return {
        'nse': float(nse(pred, true)),
        'rmse': float(rmse(pred, true)),
        'mae': float(mae(pred, true)),
        'rel_l2': float(relative_l2(pred, true)),
    }


def compute_validation_metrics(U_pred: jnp.ndarray, U_true: jnp.ndarray) -> dict:
    """Compute accuracy metrics for all 3 output variables (h, hu, hv).

    Args:
        U_pred: Predicted outputs, shape (N, 3) -- columns [h, hu, hv].
        U_true: Reference outputs, shape (N, 3) -- columns [h, hu, hv].

    Returns:
        Dict with keys like 'nse_h', 'rmse_hu', 'mae_hv', 'rel_l2_h', etc.
    """
    results = {}
    var_names = ['h', 'hu', 'hv']
    for i, var in enumerate(var_names):
        metrics = compute_all_metrics(U_pred[..., i], U_true[..., i])
        for metric_name, value in metrics.items():
            results[f'{metric_name}_{var}'] = value
    return results
