import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from pathlib import Path

def find_and_plot_footprint(
    file_path: Path,
    time_to_inspect: float,
    building_dims: dict,
    output_filename: str = "building_footprint_h0.png",
    h_tolerance: float = 1e-6
):
    """
    Finds and plots the building footprint by identifying points where h=0
    in the validation tensor at a specific time.

    Args:
        file_path (Path): Path to the validation_tensor.npy file.
        time_to_inspect (float): The timestamp to inspect (e.g., 5400.0).
        building_dims (dict): Dictionary with the ideal building's dimensions for plotting.
        output_filename (str): The name of the output plot file.
        h_tolerance (float): Tolerance to consider 'h' as zero.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"Mapping data from {file_path}...")
    data = np.load(file_path, mmap_mode='r')

    time_column = data[:, 0]
    print(f"Searching for data at t = {time_to_inspect}s...")
    
    start_index = np.searchsorted(time_column, time_to_inspect, side='left')
    end_index = np.searchsorted(time_column, time_to_inspect, side='right')

    if start_index == end_index:
        min_time, max_time = time_column[0], time_column[-1]
        print(f"No data found for t = {time_to_inspect}s.")
        print(f"Available time range in the file is from {min_time:.2f}s to {max_time:.2f}s.")
        return

    time_slice = data[start_index:end_index]
    print(f"Found {time_slice.shape[0]} total points at the specified time.")

    # Find points where h is effectively zero
    h_column = time_slice[:, 3]
    zero_h_mask = h_column < h_tolerance
    
    building_points = time_slice[zero_h_mask]

    if building_points.shape[0] == 0:
        print(f"No points with h=0 were found at t={time_to_inspect}s.")
        return
        
    print(f"Found {building_points.shape[0]} points with h=0, indicating the building footprint.")

    x_coords = building_points[:, 1]
    y_coords = building_points[:, 2]

    # --- Plotting ---
    plt.figure(figsize=(12, 9))
    plt.scatter(x_coords, y_coords, s=10, marker='.', label=f'Centroids with h=0 (t={time_to_inspect}s)')
    
    # Draw the ideal building footprint for comparison
    ideal_footprint = patches.Rectangle(
        (building_dims['x_min'], building_dims['y_min']),
        width=building_dims['x_max'] - building_dims['x_min'],
        height=building_dims['y_max'] - building_dims['y_min'],
        linewidth=2, edgecolor='r', facecolor='none', linestyle='--', label='Ideal Building Footprint'
    )
    plt.gca().add_patch(ideal_footprint)

    plt.title("Building Footprint from Simulation Data (h=0)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = file_path.parent / output_filename
    plt.savefig(output_path, dpi=200)
    print(f"Footprint plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Find and plot the building footprint from simulation data where h=0."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="The scenario directory inside 'data/' (e.g., 'one_building_DEM_zero')."
    )
    parser.add_argument(
        "--time",
        type=float,
        default=5400.0,
        help="The time in seconds to inspect."
    )
    args = parser.parse_args()

    BASE_PATH = Path('data') / args.scenario
    SOURCE_FILE = BASE_PATH / 'validation_tensor.npy'
    
    # Define the ideal building dimensions from the config for visualization
    BUILDING_DIMS = {
        'x_min': 325.0, 'x_max': 375.0,
        'y_min': 25.0,  'y_max': 75.0
    }

    find_and_plot_footprint(
        file_path=SOURCE_FILE,
        time_to_inspect=args.time,
        building_dims=BUILDING_DIMS
    )