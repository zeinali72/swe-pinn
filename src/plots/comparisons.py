"""Comparison plots: P3.1-P3.7."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

# Colour palette
EXETER_DEEP_GREEN = "#003C3C"
EXETER_TEAL = "#007D69"
EXETER_MINT = "#00C896"
BLUE_HEART_NAVY = "#0D2B45"
BLUE_HEART_OCEAN = "#1B5E8A"
BLUE_HEART_SKY = "#4FA3D1"


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


def plot_precision_comparison_bar(
    precisions: List[str],
    nse_values: List[float],
    training_times: List[float],
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P3.1 — Grouped bar chart comparing NSE and training time across precisions.

    Parameters
    ----------
    precisions:
        Precision labels, e.g. ``['float64', 'float32', 'bfloat16']``.
    nse_values:
        NSE for ``h`` at each precision, same order as *precisions*.
    training_times:
        Wall-clock training time (seconds) at each precision.
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if not precisions:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Precision Comparison")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    n = len(precisions)
    x = np.arange(n)
    bar_width = 0.35

    fig, ax_nse = plt.subplots(figsize=(max(6, 2 * n + 2), 5))
    ax_time = ax_nse.twinx()

    bars_nse = ax_nse.bar(
        x - bar_width / 2,
        nse_values,
        width=bar_width,
        color=EXETER_DEEP_GREEN,
        alpha=0.85,
        label="NSE (h)",
    )
    bars_time = ax_time.bar(
        x + bar_width / 2,
        training_times,
        width=bar_width,
        color=BLUE_HEART_OCEAN,
        alpha=0.85,
        label="Training time (s)",
    )

    # Value labels on bars
    for bar in bars_nse:
        height = bar.get_height()
        ax_nse.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=EXETER_DEEP_GREEN,
        )
    for bar in bars_time:
        height = bar.get_height()
        ax_time.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(training_times) * 0.01,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=BLUE_HEART_OCEAN,
        )

    ax_nse.set_xticks(x)
    ax_nse.set_xticklabels(precisions)
    ax_nse.set_xlabel("Precision")
    ax_nse.set_ylabel("NSE", color=EXETER_DEEP_GREEN)
    ax_time.set_ylabel("Training time (s)", color=BLUE_HEART_OCEAN)
    ax_nse.set_title("Precision Comparison: NSE vs Training Time")
    ax_time.spines["top"].set_visible(False)

    # Combined legend
    lines_nse, labels_nse = ax_nse.get_legend_handles_labels()
    lines_time, labels_time = ax_time.get_legend_handles_labels()
    ax_nse.legend(lines_nse + lines_time, labels_nse + labels_time, frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_architecture_comparison_bar(*args, **kwargs):
    """P3.2 — Architecture comparison bar chart. Not yet implemented."""
    raise NotImplementedError("Not yet implemented")


def plot_loss_weight_sensitivity(*args, **kwargs):
    """P3.3 — Loss weight sensitivity comparison. Not yet implemented."""
    raise NotImplementedError("Not yet implemented")


def plot_data_regime_comparison(*args, **kwargs):
    """P3.4 — Data regime comparison (physics-only vs data-driven). Not yet implemented."""
    raise NotImplementedError("Not yet implemented")


def plot_domain_complexity_scaling(*args, **kwargs):
    """P3.5 — Domain complexity scaling comparison. Not yet implemented."""
    raise NotImplementedError("Not yet implemented")


def plot_training_strategy_comparison(*args, **kwargs):
    """P3.6 — Training strategy comparison. Not yet implemented."""
    raise NotImplementedError("Not yet implemented")


def plot_benchmark_summary(*args, **kwargs):
    """P3.7 — Benchmark summary comparison across experiments. Not yet implemented."""
    raise NotImplementedError("Not yet implemented")
