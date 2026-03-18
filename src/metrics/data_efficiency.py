"""Data-efficiency metrics: F1 (data fraction) and F2 (efficiency ratio).

Used in Experiments 6 and 11 to quantify how much labelled data the hybrid
PINN needs compared with a data-only surrogate.
"""
import numpy as np


def data_fraction(n_training_points: int, n_total_icm_points: int) -> dict:
    """F1: Fraction of total ICM spatiotemporal points used for training.

    Args:
        n_training_points:  Number of labelled points used in training.
        n_total_icm_points: Total available spatiotemporal ICM output points.

    Returns:
        Dict with ``fraction``, ``pct``, ``n_training``, ``n_total``.
    """
    if n_total_icm_points <= 0:
        return {"fraction": 0.0, "pct": 0.0,
                "n_training": n_training_points, "n_total": n_total_icm_points}
    fraction = n_training_points / n_total_icm_points
    return {
        "fraction": fraction,
        "pct": fraction * 100.0,
        "n_training": n_training_points,
        "n_total": n_total_icm_points,
    }


def data_efficiency_ratio(
    fractions: list,
    nse_hybrid: list,
    nse_data_only: list,
    target_nse: float = None,
) -> dict:
    """F2: Data fraction at which PINN+data and data-only reach equivalent NSE.

    Finds the crossover fraction (smallest f where nse_hybrid[f] >= nse_data_only[f]).
    When *target_nse* is given, also computes the ratio of the fractions required
    by each approach to reach that target.

    Args:
        fractions:     Ascending list of data fractions.
        nse_hybrid:    NSE of PINN+data model at each fraction.
        nse_data_only: NSE of data-only model at each fraction.
        target_nse:    Optional NSE threshold for efficiency ratio calculation.

    Returns:
        Dict with ``crossover_fraction``, ``efficiency_ratio``, ``target_nse``.
    """
    fractions = np.asarray(fractions)
    nse_hybrid = np.asarray(nse_hybrid)
    nse_data_only = np.asarray(nse_data_only)

    crossover_fraction = float(fractions[-1])
    for f, fh, fd in zip(fractions, nse_hybrid, nse_data_only):
        if fh >= fd:
            crossover_fraction = float(f)
            break

    efficiency_ratio = float("nan")
    if target_nse is not None:
        target_fraction_hybrid = next(
            (float(f) for f, nh in zip(fractions, nse_hybrid) if nh >= target_nse), None
        )
        target_fraction_data = next(
            (float(f) for f, nd in zip(fractions, nse_data_only) if nd >= target_nse), None
        )
        if target_fraction_hybrid and target_fraction_data and target_fraction_hybrid > 0:
            efficiency_ratio = target_fraction_data / target_fraction_hybrid

    return {
        "crossover_fraction": crossover_fraction,
        "efficiency_ratio": efficiency_ratio,
        "target_nse": target_nse,
    }
