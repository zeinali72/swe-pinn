# src/utils.py
import os
import pickle
import datetime
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Import patches for building rectangle
import matplotlib.tri as tri # Needed for tricontourf
import seaborn as sns  # <<<--- ADD THIS IMPORT
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def save_model(params: Dict[str, Any], save_dir: str, trial_name: str) -> str:
    """Save model parameters to a pickle file and return the path."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{trial_name}_params.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    return model_path # <<<--- FIX: Added the missing return statement

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

def mask_points_inside_building(points: jnp.ndarray, building_config) -> jnp.ndarray:
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
def plot_comparison_scatter_2d(
    x_coords: jnp.ndarray, 
    y_coords: jnp.ndarray, 
    pred_data: jnp.ndarray, 
    true_data: jnp.ndarray,
    var_name: str, 
    config: Dict[str, Any], 
    filename: str = None
) -> None:
    """
    Generates and saves a stacked 2D comparison plot (Prediction, True, Error)
    using scattered data for a specific variable (h, hu, or hv).
    
    - Uses seaborn styling for a professional look.
    - Sets building area in 'Predicted' and 'Error' plots to 0.0.
    - 'True' plot (middle) is NOT masked.
    - Uses horizontal color bars and a tight layout.
    - Uses 'vlag' colormap for data and 'coolwarm' for error.
    """
    sns.set_style("whitegrid") # Use seaborn styling
    print(f"Generating 2D comparison plot for variable: {var_name}...")
    plot_cfg = config.get("plotting", {})
    building_cfg = config.get("building")
    domain_cfg = config.get("domain", {})
    t_const = plot_cfg.get("t_const_val", 1800.0)

    # --- 1. Create Building Mask (for data points) ---
    building_mask_pts = jnp.zeros_like(x_coords, dtype=bool) # Default to False (no mask)
    if building_cfg:
        x_min, x_max = building_cfg["x_min"], building_cfg["x_max"]
        y_min, y_max = building_cfg["y_min"], building_cfg["y_max"]
        
        # Mask is True for points *inside* the building
        building_mask_pts = (
            (x_coords >= x_min) & (x_coords <= x_max) &
            (y_coords >= y_min) & (y_coords <= y_max)
        )
    
    # --- 2. Apply Mask and Calculate Error ---
    # Set predicted data inside building to 0.0 for plotting
    pred_data_masked = jnp.where(building_mask_pts, 0.0, pred_data)
    
    # Set true data inside building to 0.0 *for error calculation only*
    true_data_masked_for_error = jnp.where(building_mask_pts, 0.0, true_data)

    # Calculate error (this will be 0.0 inside the building)
    error_data = pred_data_masked - true_data_masked_for_error
    
    # --- 3. Create Base Triangulation ---
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
        
    # --- 4. Determine Color Ranges & Labels ---
    
    # Define labels with units (Request 2)
    label_map = {
        'h': 'h (m)',
        'hu': 'hu (m²/s)',
        'hv': 'hv (m²/s)',
    }
    cbar_label = label_map.get(var_name, var_name) # Get label with unit
    
    # Use diverging map 'vlag' for all data plots (Request 3)
    cmap_val = 'vlag' 
    vmax_abs = max(jnp.abs(pred_data).max(), jnp.abs(true_data).max(), 1e-9)
    # Special case for 'h': use [0, max] as range, but still with 'vlag' map
    # This will result in a map from white-to-red, as requested.
    if var_name == 'h':
        vmin_val = 0.0
        vmax_val = max(jnp.max(pred_data), jnp.max(true_data), 1e-9)
        extend_val = 'max'
    else:
        vmin_val, vmax_val = -vmax_abs, vmax_abs
        extend_val = 'both'
        
    levels_val = jnp.linspace(vmin_val, vmax_val, 51)

    # Error plot is always diverging 'coolwarm'
    v_err_max = jnp.max(jnp.abs(error_data)) # Error is already masked
    v_err_max = max(v_err_max, 1e-9)
    levels_err = jnp.linspace(-v_err_max, v_err_max, 51)
    cmap_err = 'coolwarm'

    # --- 5. Create 3 Subplots (Tighter layout) ---
    # Reduced figsize height and hspace (Request 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.3})
    (ax1, ax2, ax3) = axes

    # Common colorbar args (Horizontal)
    cbar_kwargs = {
        'orientation': 'horizontal',
        'pad': 0.2, 
        'aspect': 40,
        'shrink': 0.8
    }

    # --- Plot 1: Predicted (Masked to 0.0) ---
    contour1 = ax1.tricontourf(triang, pred_data_masked, levels=levels_val, cmap=cmap_val, vmin=vmin_val, vmax=vmax_val, extend=extend_val)
    ax1.set_title(f"Predicted {var_name} at t = {t_const:.0f}s", fontsize=12) # (Masked) removed
    ax1.set_ylabel("y (m)", fontsize=11)
    fig.colorbar(contour1, ax=ax1, label=cbar_label, **cbar_kwargs) # Use label with unit

    # --- Plot 2: True (Unmasked) ---
    contour2 = ax2.tricontourf(triang, true_data, levels=levels_val, cmap=cmap_val, vmin=vmin_val, vmax=vmax_val, extend=extend_val)
    ax2.set_title(f"True {var_name} at t = {t_const:.0f}s", fontsize=12)
    ax2.set_ylabel("y (m)", fontsize=11)
    fig.colorbar(contour2, ax=ax2, label=cbar_label, **cbar_kwargs) # Use label with unit

    # --- Plot 3: Error (Pred_Masked - True_Masked) ---
    contour3 = ax3.tricontourf(triang, error_data, levels=levels_err, cmap=cmap_err, vmin=-v_err_max, vmax=v_err_max, extend='both')
    ax3.set_title(f"Error (Predicted - True) for {var_name}", fontsize=12) # (Masked) removed
    ax3.set_xlabel("x (m)", fontsize=11)
    ax3.set_ylabel("y (m)", fontsize=11)
    fig.colorbar(contour3, ax=ax3, label=f"Error", **cbar_kwargs)

    # --- Add Building Patches and Styling ---
    building_patch_handle = None
    for ax in axes:
        if building_cfg:
            rect = patches.Rectangle(
                (building_cfg["x_min"], building_cfg["y_min"]),
                building_cfg["x_max"] - building_cfg["x_min"],
                building_cfg["y_max"] - building_cfg["y_min"],
                linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--', label='Building Footprint'
            )
            ax.add_patch(rect)
            if building_patch_handle is None:
                building_patch_handle = rect
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(0, domain_cfg.get('ly', 100.0))
        ax.set_xlim(0, domain_cfg.get('lx', 1200.0))
        ax.tick_params(labelsize=10)

    # Add a single legend for the building at the top (Tighter layout)
    if building_patch_handle:
         fig.legend(handles=[building_patch_handle], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=1, fontsize=10, frameon=True)

    # Adjust layout (tightened top margin)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.94, bottom=0.08)

    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"  Comparison plot saved as {filename}")
    plt.close(fig)
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