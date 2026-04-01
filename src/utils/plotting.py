"""Visualization utilities for PINN predictions."""
import os

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as tri
import seaborn as sns
from typing import Dict, Any, Optional

from src.physics import h_exact


def plot_h_vs_x(x_line: jnp.ndarray, h_pred_line: jnp.ndarray, t_const: float, y_const: float,
                config: Dict[str, Any], filename: str = None) -> None:
    """Plot predicted and exact water depth along the x-axis."""
    n_manning = config["physics"]["n_manning"]
    u_const = config["physics"]["u_const"]
    h_exact_line = h_exact(x_line, jnp.full_like(x_line, t_const), n_manning, u_const)

    plt.figure(figsize=(10, 5))
    plt.plot(x_line, h_exact_line, 'b-', label="Exact $h$", linewidth=2.5)
    plt.plot(x_line, h_pred_line, 'r--', label="PINN $h$", linewidth=2)
    plt.xlabel("x (m)", fontsize=12)
    plt.ylabel("Depth $h$ (m)", fontsize=12)
    plt.title(f"h vs x at y={y_const:.2f}, t={t_const:.2f}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"1D plot saved as {filename}")
    plt.close()


def plot_comparison_scatter_2d(
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
    pred_data: jnp.ndarray,
    true_data: jnp.ndarray,
    var_name: str,
    config: Dict[str, Any],
    filename: str = None
) -> None:
    """Generate a stacked 2D comparison plot (Prediction, True, Error)
    using scattered data for a specific variable (h, hu, or hv).
    """
    sns.set_style("whitegrid")
    print(f"Generating 2D comparison plot for variable: {var_name}...")
    plot_cfg = config.get("plotting", {})
    building_cfg = config.get("building")
    domain_cfg = config.get("domain", {})
    t_const = plot_cfg.get("t_const_val", 1800.0)

    # Create Building Mask
    building_mask_pts = jnp.zeros_like(x_coords, dtype=bool)
    if building_cfg:
        x_min, x_max = building_cfg["x_min"], building_cfg["x_max"]
        y_min, y_max = building_cfg["y_min"], building_cfg["y_max"]
        building_mask_pts = (
            (x_coords >= x_min) & (x_coords <= x_max) &
            (y_coords >= y_min) & (y_coords <= y_max)
        )

    pred_data_masked = jnp.where(building_mask_pts, 0.0, pred_data)
    true_data_masked_for_error = jnp.where(building_mask_pts, 0.0, true_data)
    error_data = pred_data_masked - true_data_masked_for_error

    # Create Triangulation
    triang = None
    if len(x_coords) > 3:
        try:
            triang = tri.Triangulation(x_coords, y_coords)
        except Exception as e:
            print(f"  Warning: Could not create triangulation for '{var_name}' plot: {e}")
            return
    else:
        print(f"  Warning: Not enough points ({len(x_coords)}) to plot '{var_name}'.")
        return

    # Color Ranges & Labels
    label_map = {'h': 'h (m)', 'hu': 'hu (m\u00b2/s)', 'hv': 'hv (m\u00b2/s)'}
    cbar_label = label_map.get(var_name, var_name)

    cmap_val = 'vlag'
    vmax_abs = max(jnp.abs(pred_data).max(), jnp.abs(true_data).max(), 1e-9)
    if var_name == 'h':
        vmin_val = 0.0
        vmax_val = max(jnp.max(pred_data), jnp.max(true_data), 1e-9)
        extend_val = 'max'
    else:
        vmin_val, vmax_val = -vmax_abs, vmax_abs
        extend_val = 'both'

    levels_val = jnp.linspace(vmin_val, vmax_val, 51)

    v_err_max = max(jnp.max(jnp.abs(error_data)), 1e-9)
    levels_err = jnp.linspace(-v_err_max, v_err_max, 51)
    cmap_err = 'coolwarm'

    # Create Subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.3})
    (ax1, ax2, ax3) = axes

    cbar_kwargs = {'orientation': 'horizontal', 'pad': 0.2, 'aspect': 40, 'shrink': 0.8}

    contour1 = ax1.tricontourf(triang, pred_data_masked, levels=levels_val, cmap=cmap_val,
                                vmin=vmin_val, vmax=vmax_val, extend=extend_val)
    ax1.set_title(f"Predicted {var_name} at t = {t_const:.0f}s", fontsize=12)
    ax1.set_ylabel("y (m)", fontsize=11)
    fig.colorbar(contour1, ax=ax1, label=cbar_label, **cbar_kwargs)

    contour2 = ax2.tricontourf(triang, true_data, levels=levels_val, cmap=cmap_val,
                                vmin=vmin_val, vmax=vmax_val, extend=extend_val)
    ax2.set_title(f"True {var_name} at t = {t_const:.0f}s", fontsize=12)
    ax2.set_ylabel("y (m)", fontsize=11)
    fig.colorbar(contour2, ax=ax2, label=cbar_label, **cbar_kwargs)

    contour3 = ax3.tricontourf(triang, error_data, levels=levels_err, cmap=cmap_err,
                                vmin=-v_err_max, vmax=v_err_max, extend='both')
    ax3.set_title(f"Error (Predicted - True) for {var_name}", fontsize=12)
    ax3.set_xlabel("x (m)", fontsize=11)
    ax3.set_ylabel("y (m)", fontsize=11)
    fig.colorbar(contour3, ax=ax3, label="Error", **cbar_kwargs)

    # Building Patches
    building_patch_handle = None
    for ax in axes:
        if building_cfg:
            rect = patches.Rectangle(
                (building_cfg["x_min"], building_cfg["y_min"]),
                building_cfg["x_max"] - building_cfg["x_min"],
                building_cfg["y_max"] - building_cfg["y_min"],
                linewidth=1.5, edgecolor='black', facecolor='none',
                linestyle='--', label='Building Footprint'
            )
            ax.add_patch(rect)
            if building_patch_handle is None:
                building_patch_handle = rect
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(0, domain_cfg.get('ly', 100.0))
        ax.set_xlim(0, domain_cfg.get('lx', 1200.0))
        ax.tick_params(labelsize=10)

    if building_patch_handle:
        fig.legend(handles=[building_patch_handle], loc='upper center',
                   bbox_to_anchor=(0.5, 0.98), ncol=1, fontsize=10, frameon=True)

    fig.subplots_adjust(left=0.1, right=0.9, top=0.94, bottom=0.08)

    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"  Comparison plot saved as {filename}")
    plt.close(fig)


def plot_gauge_timeseries(
    x: float,
    y: float,
    name: str,
    filename: str,
    *,
    model,
    params,
    t_plot: jnp.ndarray,
    cfg: Dict[str, Any],
    results_dir: str,
    tracker=None,
    epoch: Optional[int] = None,
    full_val_data=None,
    color: Optional[str] = None,
) -> None:
    """Plot predicted water-depth time series at a single gauge location.

    This consolidates the ``plot_gauge()`` nested function that was duplicated
    across experiments 3-8.

    Parameters
    ----------
    x, y : float
        Spatial coordinates of the gauge point.
    name : str
        Human-readable label for the gauge (used in title/legend).
    filename : str
        Output filename (written inside *results_dir*).
    model : flax.linen.Module
        Trained PINN model.
    params : dict
        Model parameters (e.g. from training loop).
    t_plot : jnp.ndarray
        1-D array of time values at which to evaluate the model.
    cfg : dict or FrozenDict
        Experiment configuration — uses ``numerics.min_depth`` if present.
    results_dir : str
        Directory where the figure is saved.
    tracker : optional
        If provided, ``tracker.log_image(path, filename)`` is called.
    epoch : int, optional
        Unused — retained for call-site compatibility.
    full_val_data : ndarray or None, optional
        Full validation dataset (columns: t, x, y, h, ...).  When not None the
        function overlays the baseline (reference) time series for the gauge.
    color : str, optional
        Matplotlib colour for the predicted line.  When *None* the default
        colour cycle is used.
    """
    pts = jnp.stack(
        [jnp.full_like(t_plot, x), jnp.full_like(t_plot, y), t_plot],
        axis=-1,
    )
    U = model.apply(params, pts, train=False)
    min_depth_plot = cfg.get("numerics", {}).get("min_depth", 0.0)
    h_pred = jnp.where(U[..., 0] < min_depth_plot, 0.0, U[..., 0])

    plt.figure(figsize=(10, 6))

    # Overlay baseline validation data when available
    if full_val_data is not None:
        val_np = np.array(full_val_data)
        mask = np.isclose(val_np[:, 1], x) & np.isclose(val_np[:, 2], y)
        gauge_data = val_np[mask]
        if gauge_data.shape[0] > 0:
            gauge_data = gauge_data[gauge_data[:, 0].argsort()]
            plt.plot(
                gauge_data[:, 0],
                gauge_data[:, 3],
                "k--",
                linewidth=1.5,
                alpha=0.7,
                label=f"Baseline {name}",
            )

    plot_kwargs: Dict[str, Any] = {"label": f"Predicted h @ ({x},{y})"}
    if color is not None:
        plot_kwargs["color"] = color
    plt.plot(t_plot, h_pred, **plot_kwargs)
    plt.xlabel("Time (s)")
    plt.ylabel("Water Level h (m)")
    plt.title(f"{name} - Water Level vs Time")
    plt.legend()
    plt.grid(True)
    path = os.path.join(results_dir, filename)
    plt.savefig(path)
    plt.close()
    if tracker is not None:
        tracker.log_image(path, filename)
