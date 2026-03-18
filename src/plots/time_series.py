"""Time-series plots: P1.1-P1.4."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

# Colour palette
EXETER_DEEP_GREEN = "#003C3C"
EXETER_TEAL = "#007D69"
EXETER_MINT = "#00C896"
BLUE_HEART_NAVY = "#0D2B45"
BLUE_HEART_OCEAN = "#1B5E8A"
BLUE_HEART_SKY = "#4FA3D1"

# Architecture colour mapping
ARCH_COLOURS: Dict[str, str] = {
    "MLP": EXETER_TEAL,
    "FourierPINN": BLUE_HEART_OCEAN,
    "DGM": EXETER_MINT,
    "PINN": EXETER_DEEP_GREEN,
    "Data-only": BLUE_HEART_SKY,
    "Reference": BLUE_HEART_NAVY,
}

# Loss component colour mapping (P1.3)
LOSS_COLOURS: Dict[str, str] = {
    "total": EXETER_DEEP_GREEN,
    "pde": EXETER_TEAL,
    "bc": BLUE_HEART_OCEAN,
    "ic": BLUE_HEART_SKY,
    "data": EXETER_MINT,
}

# Variable colour mapping (P1.4)
VAR_COLOURS: Dict[str, str] = {
    "h": EXETER_DEEP_GREEN,
    "hu": BLUE_HEART_OCEAN,
    "hv": EXETER_TEAL,
}


def _apply_style() -> None:
    """Apply project-standard matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_gauge_timeseries(
    t: np.ndarray,
    predictions: Dict[str, np.ndarray],
    h_ref: np.ndarray,
    gauge_name: str,
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P1.1 — Gauge time-series plot.

    Parameters
    ----------
    t:
        1-D time array (seconds).
    predictions:
        Mapping of architecture/model name to predicted water depth array.
    h_ref:
        Reference water depth array (plotted as dashed black line).
    gauge_name:
        Gauge identifier used in the plot title.
    metrics:
        Optional mapping of metric name to scalar value shown in legend,
        e.g. ``{'NSE': 0.94, 'RMSE': 0.01}``.
    save_path:
        If provided, the figure is saved to this path at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if t is None or len(t) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Gauge: {gauge_name}")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cycle_idx = 0

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot reference as dashed black line
    ax.plot(t, h_ref, color="black", linestyle="--", linewidth=1.5, label="Reference", zorder=5)

    # Plot each prediction
    for model_name, h_pred in predictions.items():
        colour = ARCH_COLOURS.get(model_name, prop_cycle[cycle_idx % len(prop_cycle)])
        if model_name not in ARCH_COLOURS:
            cycle_idx += 1

        label = model_name
        if metrics:
            metric_str = ", ".join(f"{k}={v:.3g}" for k, v in metrics.items())
            label = f"{model_name} ({metric_str})"

        ax.plot(t, h_pred, color=colour, linewidth=1.5, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Water depth h (m)")
    ax.set_title(f"Gauge: {gauge_name}")
    ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_mass_balance_timeseries(
    t_pinn: np.ndarray,
    e_mass_pinn: np.ndarray,
    t_icm: Optional[np.ndarray] = None,
    e_mass_icm: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P1.2 — Domain-integral volume balance time series.

    Parameters
    ----------
    t_pinn:
        Time array for the PINN mass-error curve (seconds).
    e_mass_pinn:
        Mass error (%) for the PINN at each time step.
    t_icm:
        Optional time array for the ICM reference curve.
    e_mass_icm:
        Optional mass error (%) for the ICM solver.
    save_path:
        If provided, the figure is saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if t_pinn is None or len(t_pinn) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Domain-Integral Volume Balance")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))

    # PINN line
    ax.plot(t_pinn, e_mass_pinn, color=EXETER_TEAL, linewidth=1.5, label="PINN")

    # Annotate max and final values for PINN
    if len(e_mass_pinn) > 0:
        max_idx = int(np.argmax(np.abs(e_mass_pinn)))
        max_val = e_mass_pinn[max_idx]
        final_val = e_mass_pinn[-1]
        ax.annotate(
            f"max={max_val:.3g}%",
            xy=(t_pinn[max_idx], max_val),
            xytext=(t_pinn[max_idx], max_val + 0.05 * (np.ptp(e_mass_pinn) or 1.0)),
            fontsize=8,
            color=EXETER_TEAL,
            ha="center",
        )
        ax.annotate(
            f"final={final_val:.3g}%",
            xy=(t_pinn[-1], final_val),
            xytext=(t_pinn[-1] * 0.95, final_val + 0.05 * (np.ptp(e_mass_pinn) or 1.0)),
            fontsize=8,
            color=EXETER_TEAL,
            ha="right",
        )

    # Optional ICM line
    if t_icm is not None and e_mass_icm is not None and len(t_icm) > 0:
        ax.plot(
            t_icm,
            e_mass_icm,
            color=BLUE_HEART_NAVY,
            linestyle="--",
            linewidth=1.5,
            label="ICM",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass error (%)")
    ax.set_title("Domain-Integral Volume Balance")
    ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_training_loss_curves(
    epochs: np.ndarray,
    losses_dict: Dict[str, np.ndarray],
    learning_rates: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P1.3 — Training loss curves on a log-scale y-axis.

    Parameters
    ----------
    epochs:
        1-D array of epoch indices.
    losses_dict:
        Mapping of loss component name to loss value array.
        Known keys: 'total', 'pde', 'bc', 'ic', 'data'. Others use the
        matplotlib default colour cycle.
    learning_rates:
        Optional 1-D array of learning rate values per epoch. Plotted on a
        secondary right y-axis.
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if epochs is None or len(epochs) == 0 or not losses_dict:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Training Loss Curves")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    fig, ax_loss = plt.subplots(figsize=(10, 5))

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cycle_idx = 0

    for name, values in losses_dict.items():
        if values is None or len(values) == 0:
            continue
        colour = LOSS_COLOURS.get(name, prop_cycle[cycle_idx % len(prop_cycle)])
        if name not in LOSS_COLOURS:
            cycle_idx += 1
        values_arr = np.asarray(values, dtype=float)
        # Replace non-positive values to avoid log issues
        safe_values = np.where(values_arr > 0, values_arr, np.nan)
        ax_loss.semilogy(epochs, safe_values, color=colour, linewidth=1.5, label=name)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (log scale)")
    ax_loss.set_title("Training Loss Curves")

    if learning_rates is not None and len(learning_rates) > 0:
        ax_lr = ax_loss.twinx()
        ax_lr.plot(
            epochs,
            learning_rates,
            color=BLUE_HEART_NAVY,
            linestyle="--",
            linewidth=1.0,
            label="Learning rate",
            alpha=0.7,
        )
        ax_lr.set_ylabel("Learning rate")
        ax_lr.spines["top"].set_visible(False)
        # Combine legends
        lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
        lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
        ax_loss.legend(lines_loss + lines_lr, labels_loss + labels_lr, frameon=False)
    else:
        ax_loss.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_validation_nse_during_training(
    epochs: np.ndarray,
    nse_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P1.4 — Validation NSE per variable during training.

    Parameters
    ----------
    epochs:
        1-D array of epoch indices at which validation was performed.
    nse_dict:
        Mapping of variable name ('h', 'hu', 'hv') to NSE value array.
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if epochs is None or len(epochs) == 0 or not nse_dict:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Validation NSE During Training")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cycle_idx = 0

    fig, ax = plt.subplots(figsize=(10, 5))

    # Horizontal reference at NSE = 0
    ax.axhline(y=0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label="NSE=0")

    for var_name, nse_values in nse_dict.items():
        if nse_values is None or len(nse_values) == 0:
            continue
        colour = VAR_COLOURS.get(var_name, prop_cycle[cycle_idx % len(prop_cycle)])
        if var_name not in VAR_COLOURS:
            cycle_idx += 1
        ax.plot(epochs, nse_values, color=colour, linewidth=1.5, label=var_name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("NSE")
    ax.set_title("Validation NSE During Training")
    ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
