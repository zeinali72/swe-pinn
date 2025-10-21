import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_validation_tensor_snapshot(
    file_path: Path,
    time_to_plot: float,
    output_filename: str = "validation_tensor_snapshot.png"
):
    """
    Efficiently loads a snapshot from a large validation tensor using memory-mapping,
    filters it for a specific time, and creates a 2D plot of the water depth.

    Args:
        file_path (Path): Path to the large validation_tensor.npy file.
        time_to_plot (float): The timestamp to visualize (e.g., 1800.0).
        output_filename (str): The name of the output plot file.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading data from {file_path} using memory-mapping...")
    # Use mmap_mode='r' to open the file without loading it all into memory
    data = np.load(file_path, mmap_mode='r')
    print(f"Successfully mapped file with shape: {data.shape}")

    time_column = data[:, 0]

    # Since the data is sorted by time, we can use np.searchsorted for efficiency
    print(f"Searching for data at t = {time_to_plot}s...")
    start_index = np.searchsorted(time_column, time_to_plot, side='left')
    end_index = np.searchsorted(time_column, time_to_plot, side='right')

    if start_index == end_index:
        print(f"No data found for t = {time_to_plot}s. Please check the time value.")
        # Check the available time range
        min_time, max_time = time_column[0], time_column[-1]
        print(f"Available time range in the file is from {min_time:.2f}s to {max_time:.2f}s.")
        return

    # Extract the slice of data for the specified time
    time_slice = data[start_index:end_index]
    print(f"Found {time_slice.shape[0]} points at t = {time_to_plot}s.")

    x = time_slice[:, 1]
    y = time_slice[:, 2]
    h = time_slice[:, 3]

    # Create the 2D scatter plot
    print("Generating plot...")
    plt.figure(figsize=(14, 7))
    scatter = plt.scatter(x, y, c=h, cmap='viridis', s=5, marker='.') # Use small markers for dense data
    plt.colorbar(scatter, label='Water Depth h (m)')

    plt.title(f"Validation Tensor Snapshot at t = {time_to_plot}s")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()

    # Save the figure
    output_path = file_path.parent / output_filename
    plt.savefig(output_path, dpi=300) # Save with higher resolution
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize a snapshot of the large validation_tensor.npy data."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="The name of the scenario directory inside 'data/' (e.g., 'one_building_DEM_zero')."
    )
    parser.add_argument(
        "--time",
        type=float,
        default=3600.0,
        help="The time in seconds to plot the snapshot for."
    )
    args = parser.parse_args()

    BASE_PATH = Path('data') / args.scenario
    SOURCE_FILE = BASE_PATH / 'validation_tensor.npy'

    plot_validation_tensor_snapshot(
        file_path=SOURCE_FILE,
        time_to_plot=args.time
    )