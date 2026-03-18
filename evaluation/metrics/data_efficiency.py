"""Data-efficiency metrics: F1 (data fraction) and F2 (efficiency ratio)."""
import numpy as np


def data_fraction(n_training_points: int, n_total_icm_points: int) -> dict:
    """F1: Fraction of total ICM spatiotemporal points used for training.

    Returns:
        Dict with 'fraction', 'n_training', 'n_total', 'pct'.
    """
    fraction = n_training_points / n_total_icm_points if n_total_icm_points > 0 else 0.0
    return {
        "fraction": fraction,
        "n_training": n_training_points,
        "n_total": n_total_icm_points,
        "pct": fraction * 100.0,
    }


def data_efficiency_ratio(
    fractions: list,
    nse_hybrid: list,
    nse_data_only: list,
    target_nse: float = None,
) -> dict:
    """F2: Data fraction at which PINN+data and data-only reach equivalent accuracy.

    Finds the crossover fraction where nse_hybrid >= nse_data_only.

    Args:
        fractions: List of data fractions (ascending).
        nse_hybrid: NSE values for PINN+data model at each fraction.
        nse_data_only: NSE values for data-only model at each fraction.
        target_nse: Optional target NSE threshold. If None, uses the crossover.

    Returns:
        Dict with 'crossover_fraction', 'efficiency_ratio', 'target_nse'.
    """
    fractions = np.asarray(fractions)
    nse_hybrid = np.asarray(nse_hybrid)
    nse_data_only = np.asarray(nse_data_only)

    crossover_fraction = None
    for i, (fh, fd) in enumerate(zip(nse_hybrid, nse_data_only)):
        if fh >= fd:
            crossover_fraction = float(fractions[i])
            break

    if crossover_fraction is None:
        crossover_fraction = float(fractions[-1])

    if target_nse is not None:
        # Find smallest fraction where hybrid meets target
        target_fraction_hybrid = None
        target_fraction_data = None
        for f, nh, nd in zip(fractions, nse_hybrid, nse_data_only):
            if target_fraction_hybrid is None and nh >= target_nse:
                target_fraction_hybrid = float(f)
            if target_fraction_data is None and nd >= target_nse:
                target_fraction_data = float(f)
        efficiency_ratio = (
            target_fraction_data / target_fraction_hybrid
            if target_fraction_hybrid and target_fraction_data and target_fraction_hybrid > 0
            else float("nan")
        )
    else:
        target_nse = None
        efficiency_ratio = float("nan")

    return {
        "crossover_fraction": crossover_fraction,
        "efficiency_ratio": efficiency_ratio,
        "target_nse": target_nse,
    }
