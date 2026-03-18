"""Accuracy metrics re-exported from src.metrics.accuracy with spec-aligned signatures."""
import numpy as np

from src.metrics.accuracy import nse, rmse, mae, relative_l2, compute_all_metrics


def compute_nse(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    return float(nse(y_pred, y_ref))


def compute_rmse(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    return float(rmse(y_pred, y_ref))


def compute_mae(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    return float(mae(y_pred, y_ref))


def compute_rel_l2(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    return float(relative_l2(y_pred, y_ref))


def compute_all_accuracy(
    y_pred: dict,
    y_ref: dict,
) -> dict:
    """Compute all accuracy metrics for each variable.

    Args:
        y_pred: Dict keyed by variable name, e.g. {'h': ..., 'hu': ..., 'hv': ...}.
        y_ref: Dict keyed by variable name, same keys as y_pred.

    Returns:
        Dict of dicts: {'h': {'nse': ..., 'rmse': ..., 'mae': ..., 'rel_l2': ...}, ...}
    """
    results = {}
    for var in y_pred:
        results[var] = compute_all_metrics(y_pred[var], y_ref[var])
    return results
