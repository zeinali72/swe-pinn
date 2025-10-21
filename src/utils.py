# src/utils.py
import os
import pickle
import datetime
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import queue
import threading

from src.physics import h_exact

def nse(pred: jnp.ndarray, true: jnp.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency (NSE) metric."""
    num = jnp.sum((true - pred)**2)
    den = jnp.sum((true - jnp.mean(true))**2)
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
        print(f"Plot saved as {filename}")
    plt.close()

def mask_points_inside_building(points: jnp.ndarray, building_config: Dict[str, Any]) -> jnp.ndarray:
    """
    Creates a boolean mask to exclude points inside a building's footprint.
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

def plot_h_2d_top_view(xx: jnp.ndarray, yy: jnp.ndarray, h_pred: jnp.ndarray, config: Dict[str, Any], filename: str = None) -> None:
    """
    Generates and saves a 2D top-view contour plot of the water depth `h`.
    """
    print("Generating 2D top-view plot...")
    plot_cfg = config["plotting"]
    building_cfg = config.get("building")
    t_const = plot_cfg["t_const_val"]

    plt.figure(figsize=(12, 6))
    contour = plt.contourf(xx, yy, h_pred, cmap='viridis', levels=50)
    plt.colorbar(contour, label='Water Depth h (m)')

    if building_cfg:
        rect = plt.Rectangle(
            (building_cfg["x_min"], building_cfg["y_min"]),
            building_cfg["x_max"] - building_cfg["x_min"],
            building_cfg["y_max"] - building_cfg["y_min"],
            linewidth=1.5, edgecolor='r', facecolor='grey', label='Building'
        )
        plt.gca().add_patch(rect)

    plt.title(f"Water Depth (h) at t = {t_const:.2f}s")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"2D plot saved as {filename}")
    plt.close()

def ask_for_confirmation(timeout=60):
    """Asks the user for confirmation with a timeout, defaulting to yes."""
    q = queue.Queue()

    def get_input():
        try:
            sys.stderr.write(f"Save results? (y/n) [auto-yes in {timeout}s]: ")
            sys.stderr.flush()
            q.put(input().lower())
        except EOFError:
            q.put('y')

    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()

    try:
        answer = q.get(timeout=timeout)
        return answer != 'n'
    except queue.Empty:
        print("\nNo input received, proceeding to save automatically.")
        return True