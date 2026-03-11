"""
Plot configuration for SWE-PINN thesis figures.

Usage:
    from plot_config import *
    apply_style()  # Call once at the start of any plotting script
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# =============================================================================
# Colour Palette — Primary Scientific
# =============================================================================
EXETER_DEEP_GREEN = "#003C3C"
EXETER_TEAL = "#007D69"
EXETER_MINT = "#00C896"
BLUE_HEART_NAVY = "#1B2A4A"
BLUE_HEART_OCEAN = "#2E5C8A"
BLUE_HEART_SKY = "#7DB4D6"

# Architecture mapping (consistent across all figures)
ARCH_COLOURS = {
    "MLP": EXETER_DEEP_GREEN,
    "Fourier-MLP": EXETER_MINT,
    "FourierPINN": EXETER_MINT,
    "DGM": EXETER_TEAL,
    "DGMNetwork": EXETER_TEAL,
}

# Data source mapping
DATA_COLOURS = {
    "pinn": EXETER_DEEP_GREEN,
    "icm": BLUE_HEART_NAVY,
    "observed": BLUE_HEART_OCEAN,
    "uncertainty": BLUE_HEART_SKY,
}

# Loss component colours (for convergence curves)
LOSS_COLOURS = {
    "total": EXETER_DEEP_GREEN,
    "pde": BLUE_HEART_NAVY,
    "ic": EXETER_TEAL,
    "bc": EXETER_MINT,
    "data": BLUE_HEART_OCEAN,
}

# =============================================================================
# Colour Palette — Error / Diverging
# =============================================================================
ERROR_RED = "#C0392B"       # Overprediction
ERROR_BLUE = "#2980B9"      # Underprediction
ERROR_NEUTRAL = "#F0F0F0"   # Zero error

# Diverging colourmap centred at zero (blue → white → red)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "pinn_error",
    [ERROR_BLUE, ERROR_NEUTRAL, ERROR_RED],
    N=256,
)

# Sequential blue colourmap for flood extent (white → navy)
FLOOD_CMAP = LinearSegmentedColormap.from_list(
    "pinn_flood",
    ["#FFFFFF", BLUE_HEART_SKY, BLUE_HEART_OCEAN, BLUE_HEART_NAVY],
    N=256,
)

# =============================================================================
# Typography and Layout Constants
# =============================================================================
FONT_FAMILY = "Arial"
FONT_SIZE_AXIS_LABEL = 12
FONT_SIZE_TICK_LABEL = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_TITLE = 13
FONT_SIZE_SUBPLOT_LABEL = 12
FONT_SIZE_COLORBAR = 10

LINE_WIDTH_DATA = 1.5
LINE_WIDTH_REFERENCE = 1.0

GRID_COLOUR = "#E0E0E0"

# Figure sizes (inches)
FIG_SINGLE = (8, 6)
FIG_DOUBLE = (14, 6)

# Export settings
DPI = 300


def apply_style():
    """Apply the global SWE-PINN plotting style via matplotlib rcParams."""
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans", "Helvetica"],
        "font.size": FONT_SIZE_TICK_LABEL,
        # Axes
        "axes.labelsize": FONT_SIZE_AXIS_LABEL,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,  # grid behind data
        "axes.facecolor": "white",
        # Grid
        "grid.color": GRID_COLOUR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        # Ticks
        "xtick.labelsize": FONT_SIZE_TICK_LABEL,
        "ytick.labelsize": FONT_SIZE_TICK_LABEL,
        # Legend
        "legend.fontsize": FONT_SIZE_LEGEND,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # Lines
        "lines.linewidth": LINE_WIDTH_DATA,
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def add_subplot_labels(axes, labels=None, x=-0.05, y=1.05):
    """Add (a), (b), (c), ... labels to subplots.

    Parameters
    ----------
    axes : list of matplotlib Axes
        Subplots to label.
    labels : list of str, optional
        Custom labels. Defaults to (a), (b), (c), ...
    x, y : float
        Position in axes coordinates.
    """
    if labels is None:
        labels = [f"({chr(97 + i)})" for i in range(len(axes))]
    for ax, label in zip(axes, labels):
        ax.text(
            x, y, label,
            transform=ax.transAxes,
            fontsize=FONT_SIZE_SUBPLOT_LABEL,
            fontweight="bold",
            va="top",
            ha="left",
            fontfamily=FONT_FAMILY,
        )


def savefig(fig, path, formats=None):
    """Save figure in specified formats with correct settings.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str
        Base path without extension (e.g., 'figures/convergence_exp1').
    formats : list of str, optional
        File formats to save. Defaults to ['png', 'pdf'].
    """
    if formats is None:
        formats = ["png", "pdf"]
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(
            f"{path}.{fmt}",
            dpi=DPI,
            bbox_inches="tight",
            facecolor="white",
        )


def make_diverging_norm(vmin, vmax):
    """Create a TwoSlopeNorm centred at zero for error maps.

    Parameters
    ----------
    vmin, vmax : float
        Data range (should typically be symmetric, e.g., -0.1, 0.1).

    Returns
    -------
    matplotlib.colors.TwoSlopeNorm
    """
    from matplotlib.colors import TwoSlopeNorm
    return TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
