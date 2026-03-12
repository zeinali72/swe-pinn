"""Visualization utilities for PINN predictions."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as tri
import seaborn as sns
from typing import Dict, Any

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
