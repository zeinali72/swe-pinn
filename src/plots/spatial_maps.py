"""Spatial map plots: P2.1, P2.2, P2.7."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional, Tuple

# Colour palette
EXETER_DEEP_GREEN = "#003C3C"
EXETER_TEAL = "#007D69"
EXETER_MINT = "#00C896"
BLUE_HEART_NAVY = "#0D2B45"
BLUE_HEART_OCEAN = "#1B5E8A"
BLUE_HEART_SKY = "#4FA3D1"

# Category colours for error decomposition (P2.7)
CATEGORY_COLOURS = {
    0: EXETER_MINT,   # interior
    1: BLUE_HEART_SKY,  # boundary
    2: EXETER_TEAL,   # shock
}
CATEGORY_LABELS = {
    0: "Interior",
    1: "Boundary",
    2: "Shock",
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


def _draw_buildings(ax: "matplotlib.axes.Axes", buildings: List[Dict]) -> None:
    """Draw building footprints as filled black rectangles on *ax*."""
    for bldg in buildings:
        x_min = bldg.get("x_min", 0.0)
        x_max = bldg.get("x_max", 0.0)
        y_min = bldg.get("y_min", 0.0)
        y_max = bldg.get("y_max", 0.0)
        rect = mpatches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=0.8,
            edgecolor="black",
            facecolor="black",
            zorder=10,
        )
        ax.add_patch(rect)


def plot_error_map(
    x: np.ndarray,
    y: np.ndarray,
    error: np.ndarray,
    time_label: str,
    domain_type: str = "rectangular",
    buildings: Optional[List[Dict]] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P2.1 — Absolute error heatmap at a single time step.

    Parameters
    ----------
    x:
        1-D x-coordinates of evaluation points.
    y:
        1-D y-coordinates of evaluation points.
    error:
        1-D absolute error array ``|h_pred - h_ref|``.
    time_label:
        String used in the title, e.g. ``'t = 300 s'``.
    domain_type:
        ``'rectangular'`` or ``'irregular'``. Currently informational only.
    buildings:
        Optional list of dicts with keys ``x_min``, ``x_max``, ``y_min``,
        ``y_max`` to draw building footprints.
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if x is None or len(x) == 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"|h_pred - h_ref| at t = {time_label}")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(9, 5))

    tcf = ax.tricontourf(x, y, error, levels=20, cmap="YlOrRd")
    cbar = fig.colorbar(tcf, ax=ax)
    cbar.set_label("Error |Δh| (m)")

    if buildings:
        _draw_buildings(ax, buildings)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"|h_pred - h_ref| at t = {time_label}")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_error_maps_multi(
    snapshots: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    domain_type: str = "rectangular",
    buildings: Optional[List[Dict]] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P2.1 multi-timestep variant — error maps at 3–5 time steps.

    Parameters
    ----------
    snapshots:
        List of ``(x, y, error, time_label)`` tuples, one per time step.
        Between 1 and 5 snapshots are supported.
    domain_type:
        ``'rectangular'`` or ``'irregular'``. Currently informational only.
    buildings:
        Optional list of building footprint dicts.
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if not snapshots:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No snapshots provided", ha="center", va="center", transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    n = min(len(snapshots), 5)
    snapshots = snapshots[:n]

    # Determine global colour scale
    all_errors = np.concatenate([snap[2] for snap in snapshots if snap[2] is not None and len(snap[2]) > 0])
    vmin, vmax = (float(all_errors.min()), float(all_errors.max())) if len(all_errors) > 0 else (0.0, 1.0)
    levels = np.linspace(vmin, vmax, 21)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    tcf_last = None
    for ax, (x, y, error, time_label) in zip(axes, snapshots):
        if x is None or len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"t = {time_label}")
            continue
        tcf_last = ax.tricontourf(x, y, error, levels=levels, cmap="YlOrRd", vmin=vmin, vmax=vmax)
        if buildings:
            _draw_buildings(ax, buildings)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"t = {time_label}")
        ax.set_aspect("equal", adjustable="box")

    if tcf_last is not None:
        cbar = fig.colorbar(tcf_last, ax=axes, shrink=0.8, location="right")
        cbar.set_label("Error |Δh| (m)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_depth_map(
    x: np.ndarray,
    y: np.ndarray,
    h_pred: np.ndarray,
    h_ref: np.ndarray,
    time_label: str,
    domain_type: str = "rectangular",
    buildings: Optional[List[Dict]] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P2.2 — Side-by-side PINN vs reference water-depth map.

    Parameters
    ----------
    x:
        1-D x-coordinates.
    y:
        1-D y-coordinates.
    h_pred:
        PINN-predicted water depth.
    h_ref:
        Reference (analytical / ICM) water depth.
    time_label:
        String used in the overall title, e.g. ``'t = 600 s'``.
    domain_type:
        ``'rectangular'`` or ``'irregular'``. Currently informational only.
    buildings:
        Optional list of building footprint dicts.
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if x is None or len(x) == 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax in axes:
            ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        fig.suptitle(f"Water depth h at t = {time_label}")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    # Shared colour scale
    all_vals = np.concatenate([h_pred, h_ref])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    levels = np.linspace(vmin, vmax, 21)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, data, panel_title in zip(
        axes,
        [h_pred, h_ref],
        ["PINN Prediction", "Analytical / ICM Reference"],
    ):
        tcf = ax.tricontourf(x, y, data, levels=levels, cmap="Blues", vmin=vmin, vmax=vmax)
        if buildings:
            _draw_buildings(ax, buildings)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(panel_title)
        ax.set_aspect("equal", adjustable="box")

    # Shared colorbar below both panels
    cbar = fig.colorbar(tcf, ax=axes.tolist(), orientation="horizontal", shrink=0.6, pad=0.08)
    cbar.set_label("Water depth h (m)")

    fig.suptitle(f"Water depth h at t = {time_label}", y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_error_decomposition(
    x: np.ndarray,
    y: np.ndarray,
    error_h: np.ndarray,
    categories: np.ndarray,
    time_label: str,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P2.7 — Scatter plot of absolute error coloured by spatial category.

    Parameters
    ----------
    x:
        1-D x-coordinates.
    y:
        1-D y-coordinates.
    error_h:
        Absolute error ``|h_pred - h_ref|`` at each point.
    categories:
        Integer category array (0 = interior, 1 = boundary, 2 = shock).
    time_label:
        String used in the title.
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if x is None or len(x) == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Error decomposition at t = {time_label}")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(9, 6))

    error_arr = np.asarray(error_h, dtype=float)
    cat_arr = np.asarray(categories, dtype=int)
    total_pts = len(x)

    # Point size proportional to |error|, capped at 200
    max_err = float(np.nanmax(error_arr)) if len(error_arr) > 0 else 1.0
    if max_err == 0.0:
        max_err = 1.0
    sizes = np.clip((error_arr / max_err) * 150, 5, 200)

    legend_handles = []
    unique_cats = np.unique(cat_arr)

    for cat in sorted(unique_cats):
        mask = cat_arr == cat
        colour = CATEGORY_COLOURS.get(int(cat), "#888888")
        label_base = CATEGORY_LABELS.get(int(cat), f"Category {cat}")
        frac = float(mask.sum()) / total_pts * 100.0
        label = f"{label_base} ({frac:.1f}%)"
        ax.scatter(
            x[mask], y[mask],
            c=colour,
            s=sizes[mask],
            alpha=0.7,
            linewidths=0.0,
            label=label,
            zorder=3,
        )
        legend_handles.append(
            mpatches.Patch(color=colour, label=label)
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Error decomposition at t = {time_label}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(handles=legend_handles, frameon=False, title="Category (point size ∝ |error|)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
