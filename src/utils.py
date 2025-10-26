# src/utils.py
import os
import pickle
import datetime
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Import patches for building rectangle
import matplotlib.tri as tri # Needed for tricontourf
from mpl_toolkits.axes_grid1 import make_axes_locatable # For better colorbar placement
from typing import Dict, Any
import sys
import queue
import threading

# Import matplotlib settings for font consistency if desired
# plt.rcParams.update({'font.size': 12, 'font.family': 'serif'}) # Example

from src.physics import h_exact

def nse(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency (NSE) metric."""
    num = jnp.sum((true - pred)**2)
    den = jnp.sum((true - jnp.mean(true))**2)
    # Handle potential division by zero or near-zero variance
    if den < 1e-9:
        return -jnp.inf
    return 1 - num / den

def rmse(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """Compute Root Mean Square Error (RMSE)."""
    return jnp.sqrt(jnp.mean((pred - true)**2))

def generate_trial_name(config_filename):
    """Generate a unique trial name using the current date and config filename."""
    now = datetime.datetime.now()
    return f"{now.strftime('%Y-%m-%d_%H-%M')}_{config_filename}"

def save_model(params: Dict[str, Any], save_dir: str, trial_name: str) -> None:
    """Save model parameters to a pickle file."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{trial_name}_params.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(params, f)

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

def mask_points_inside_building(points: jnp.ndarray, building_config: Dict[str, Any]) -> jnp.ndarray:
    """
    Creates a boolean mask to exclude points inside a building's footprint.
    Points shape: (N, 3) where columns are (x, y, t)
    Returns: Boolean array of shape (N,), True for points OUTSIDE.
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min = building_config["x_min"]
    x_max = building_config["x_max"]
    y_min = building_config["y_min"]
    y_max = building_config["y_max"]

    # Mask is True for points *outside* the building
    mask = ~((x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max))
    return mask


# <<<--- REVISED FUNCTION for Stacked Comparison --- >>>
def plot_h_prediction_vs_true_2d(xx_pred: jnp.ndarray, yy_pred: jnp.ndarray, h_pred_mesh: jnp.ndarray,
                                 x_true: jnp.ndarray, y_true: jnp.ndarray, h_true_scatter: jnp.ndarray,
                                 config: Dict[str, Any], filename: str = None) -> None:
    """
    Generates and saves a stacked 2D contour plot comparing predicted (top)
    and true (bottom) water depth `h`. Aims for a professional look for papers.
    """
    print("Generating 2D prediction vs true comparison plot...")
    plot_cfg = config.get("plotting", {})
    building_cfg = config.get("building")
    domain_cfg = config.get("domain", {})
    t_const = plot_cfg.get("t_const_val", 1800.0)

    # --- Adjust Figure Size and Subplot Spacing ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True, gridspec_kw={'hspace': 0.1}) # Reduced hspace significantly

    # Determine common color scale based on the maximum of both datasets
    h_max_pred = jnp.max(h_pred_mesh) if h_pred_mesh.size > 0 else 0
    h_max_true = jnp.max(h_true_scatter) if h_true_scatter.size > 0 else 0
    h_max = max(h_max_pred, h_max_true, 0.1) # Use max, ensure at least 0.1 range
    h_min = 0
    levels = jnp.linspace(h_min, h_max, 51) # Increase levels for smoother contours
    cmap = 'viridis'

    # Common plot settings
    common_kwargs = {
        'cmap': cmap,
        'levels': levels,
        'vmin': h_min,
        'vmax': h_max,
        'extend': 'max' # Show values exceeding the range
    }

    # --- Top Plot: Predicted h (using meshgrid) ---
    ax1 = axes[0]
    contour1 = ax1.contourf(xx_pred, yy_pred, h_pred_mesh, **common_kwargs)
    ax1.set_title(f"Predicted Water Depth (h) at t = {t_const:.0f}s", fontsize=12) # Slightly smaller title
    ax1.set_ylabel("y (m)", fontsize=11)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5) # Thinner grid lines
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_ylim(0, domain_cfg.get('ly', 100.0))
    # ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis ticks/labels on top plot

    building_patches = [] # To store patches for a single legend

    if building_cfg:
        rect1 = patches.Rectangle(
            (building_cfg["x_min"], building_cfg["y_min"]),
            building_cfg["x_max"] - building_cfg["x_min"],
            building_cfg["y_max"] - building_cfg["y_min"],
            linewidth=1.2, edgecolor='r', facecolor=(0.5, 0.5, 0.5, 0.7), label='Building Footprint'
        )
        ax1.add_patch(rect1)
        building_patches.append(rect1) # Add to list


    # --- Bottom Plot: True h (using scattered data via tricontourf) ---
    ax2 = axes[1]
    contour2 = None
    if len(x_true) > 3:
        try:
            triang = tri.Triangulation(x_true, y_true)
            contour2 = ax2.tricontourf(triang, h_true_scatter, **common_kwargs)
        except Exception as e:
            print(f"  Warning: Could not create triangulation for true data plot: {e}")
    else:
        print("  Warning: Not enough points in true data to create contour plot.")

    ax2.set_title(f"True Water Depth (h) at t = {t_const:.0f}s", fontsize=12)
    ax2.set_xlabel("x (m)", fontsize=11)
    ax2.set_ylabel("y (m)", fontsize=11)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_aspect('equal', adjustable='box')
    ax2.tick_params(axis='both', labelsize=10) # Ensure ticks are visible

    if building_cfg:
        rect2 = patches.Rectangle(
            (building_cfg["x_min"], building_cfg["y_min"]),
            building_cfg["x_max"] - building_cfg["x_min"],
            building_cfg["y_max"] - building_cfg["y_min"],
            linewidth=1.2, edgecolor='r', facecolor=(0.5, 0.5, 0.5, 0.7) # No label here
        )
        ax2.add_patch(rect2)


    # Make axes look more professional
    for ax in axes:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.set_facecolor('white')
        ax.tick_params(labelsize=10)


    # --- Add Colorbar ---
    # Create an axes for the colorbar to the right of the plots
    # Adjust the fraction and pad as needed
    if contour1: # Only add colorbar if contour plots were successful
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) # [left, bottom, width, height] relative to figure
        cbar = fig.colorbar(contour1, cax=cbar_ax, label='Water Depth h (m)')
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Water Depth h (m)', size=11)

    # --- Add Building Legend Outside ---
    # Add legend to the figure, anchored outside the top right plot area
    if building_patches:
         fig.legend(handles=building_patches, loc='upper right', bbox_to_anchor=(0.87, 0.88), fontsize=10)

    # Adjust overall layout slightly to prevent overlap and reduce whitespace
    fig.subplots_adjust(left=0.1, right=0.85, top=0.92, bottom=0.1) # Leave space for colorbar on right

    # Remove the automatic suptitle as individual titles are clear enough
    # plt.suptitle(f"Comparison of Predicted vs True Water Depth at t = {t_const:.0f}s", fontsize=16, y=0.98)


    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight') # Increase DPI, use bbox_inches
        print(f"Stacked comparison plot saved as {filename}")
    plt.close()
# <<<--- END REVISED FUNCTION --- >>>


def ask_for_confirmation(timeout=60):
    """Asks the user for confirmation with a timeout, defaulting to yes."""
    q = queue.Queue()

    def get_input():
        try:
            sys.stderr.write(f"Save results and plots? (y/n) [auto-yes in {timeout}s]: ")
            sys.stderr.flush()
            q.put(sys.stdin.readline().strip().lower())
        except EOFError:
            q.put('y')
        except Exception as e:
            print(f"\nError reading input: {e}. Defaulting to yes.")
            q.put('y')


    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()

    try:
        answer = q.get(timeout=timeout)
        if answer == 'n':
            print("\nUser chose not to save.")
            return False
        else:
            if answer != 'y':
                print(f"\nReceived '{answer}', interpreting as yes.")
            return True
    except queue.Empty:
        print(f"\nTimeout ({timeout}s) reached. Proceeding to save automatically.")
        return True
    finally:
        if input_thread.is_alive():
             pass