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
    """Signed peak water depth error: max(h_pred) - max(h_ref).

    Positive = over-prediction, negative = under-prediction.
    Units: m.  Used in Exp 4-11.
    """
    return float(jnp.max(pred_h) - jnp.max(true_h))


def time_to_peak_error(
    pred_h: jnp.ndarray, true_h: jnp.ndarray, t_coords: jnp.ndarray
) -> float:
    """Signed time-to-peak error: t(max(h_pred)) - t(max(h_ref)).

    Positive = peak arrives late, negative = peak arrives early.
    Units: same as *t_coords* (s or hrs).  Used in Exp 4-11.
    """
    t_peak_pred = t_coords[jnp.argmax(pred_h)]
    t_peak_true = t_coords[jnp.argmax(true_h)]
    return float(t_peak_pred - t_peak_true)


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
