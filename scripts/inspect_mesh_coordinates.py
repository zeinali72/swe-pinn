import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from pathlib import Path

def inspect_mesh(
    file_path: Path,
    building_dims: dict,
    output_filename: str = "mesh_coordinates_inspection.png"
):
    """
    Loads validation data and plots the spatial coordinates (x, y) to inspect
    the mesh structure, particularly around a specified building area.

    Args:
        file_path (Path): Path to the validation_tensor.npy file.
        building_dims (dict): Dictionary with building's x_min, x_max, y_min, y_max.
        output_filename (str): The name of the output plot file.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"Mapping file: {file_path}...")
    data = np.load(file_path, mmap_mode='r')

    # We only need the coordinates, which are the same for every time step.
    # Let's find the points for the first time step to get a unique set of (x, y).
    time_column = data[:, 0]
    first_time_step = time_column[0]
    end_index_first_step = np.searchsorted(time_column, first_time_step, side='right')
    
    coords = data[:end_index_first_step, 1:3] # Get (x, y) columns
    print(f"Found {coords.shape[0]} unique mesh nodes.")

    x, y = coords[:, 0], coords[:, 1]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Validation Mesh Coordinate Inspection", fontsize=16)

    # --- Plot 1: Full Domain ---
    ax1.scatter(x, y, s=1, marker='.', alpha=0.5)
    ax1.set_title("Full Mesh Domain")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Draw building location on full plot
    rect = patches.Rectangle(
        (building_dims['x_min'], building_dims['y_min']),
        width=building_dims['x_max'] - building_dims['x_min'],
        height=building_dims['y_max'] - building_dims['y_min'],
        linewidth=1, edgecolor='r', facecolor='none', linestyle='--'
    )
    ax1.add_patch(rect)

    # --- Plot 2: Zoomed-in on Building ---
    ax2.scatter(x, y, s=10, marker='.')
    ax2.set_title("Zoomed-in View Around Building")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Draw ideal building box on zoomed plot
    rect_zoom = patches.Rectangle(
        (building_dims['x_min'], building_dims['y_min']),
        width=building_dims['x_max'] - building_dims['x_min'],
        height=building_dims['y_max'] - building_dims['y_min'],
        linewidth=2, edgecolor='r', facecolor='none', linestyle='--', label='Ideal Building Footprint'
    )
    ax2.add_patch(rect_zoom)
    
    # Set zoom limits
    ax2.set_xlim(building_dims['x_min'] - 20, building_dims['x_max'] + 20)
    ax2.set_ylim(building_dims['y_min'] - 20, building_dims['y_max'] + 20)
    ax2.legend()

    # Save the figure
    output_path = file_path.parent / output_filename
    plt.savefig(output_path, dpi=200)
    print(f"Inspection plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inspect the mesh coordinates from a validation tensor."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="The name of the scenario directory (e.g., 'one_building_DEM_zero')."
    )
    args = parser.parse_args()

    BASE_PATH = Path('data') / args.scenario
    SOURCE_FILE = BASE_PATH / 'validation_tensor.npy'
    
    # Define the ideal building dimensions for visualization
    BUILDING_DIMS = {
        'x_min': 325.0, 'x_max': 375.0,
        'y_min': 25.0,  'y_max': 75.0
    }

    inspect_mesh(
        file_path=SOURCE_FILE,
        building_dims=BUILDING_DIMS
    )