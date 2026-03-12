"""Peak-related metrics: R-squared, peak depth error, time-to-peak, RMSE/MAE ratio."""
import jax.numpy as jnp


def r_squared(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """Coefficient of determination (R-squared).

    Returns 1.0 for perfect prediction.
    """
    ss_res = jnp.sum((true - pred) ** 2)
    ss_tot = jnp.sum((true - jnp.mean(true)) ** 2)
    if ss_tot < 1e-12:
        return float(-jnp.inf)
    return float(1.0 - ss_res / ss_tot)


def peak_depth_error(pred_h: jnp.ndarray, true_h: jnp.ndarray) -> float:
    """Absolute error between predicted and true peak water depth."""
    return float(jnp.abs(jnp.max(pred_h) - jnp.max(true_h)))


def time_to_peak_error(
    pred_h: jnp.ndarray, true_h: jnp.ndarray, t_coords: jnp.ndarray
) -> float:
    """Absolute difference in time-to-peak between prediction and reference.

    Args:
        pred_h: (N,) predicted water depth.
        true_h: (N,) reference water depth.
        t_coords: (N,) time coordinate for each point.

    Returns:
        |t_peak_pred - t_peak_true| in the same units as *t_coords*.
    """
    t_peak_pred = t_coords[jnp.argmax(pred_h)]
    t_peak_true = t_coords[jnp.argmax(true_h)]
    return float(jnp.abs(t_peak_pred - t_peak_true))


def rmse_mae_ratio(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """Ratio of RMSE to MAE.

    A value of 1.0 indicates uniform error magnitude; higher values
    indicate the presence of large outlier errors.
    """
    rmse_val = jnp.sqrt(jnp.mean((pred - true) ** 2))
    mae_val = jnp.mean(jnp.abs(pred - true))
    if mae_val < 1e-12:
        return float(jnp.inf) if rmse_val > 1e-12 else 1.0
    return float(rmse_val / mae_val)
