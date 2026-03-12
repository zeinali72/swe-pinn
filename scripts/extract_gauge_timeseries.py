"""Extract gauge time series from a large spatiotemporal NumPy array.

Given a .npy file containing spatiotemporal data ordered by time (all spatial
points for t=0, then all for t=1, etc.), extracts time series at specific
gauge locations using stride-based slicing.

Input format: .npy with columns [t, x, y, h, u, v] (or [t, x, y, h]),
    ordered by time with a fixed spatial grid repeated at each timestep.

Output format: .npy with the same column layout, containing only rows
    matching the specified gauge coordinates.

Usage:
    python extract_gauge_timeseries.py --input data.npy --output gauges.npy \\
        --gauges 400,50 600,50
"""

import numpy as np
import argparse
import os
import sys
import time


def extract_gauges_fast(
    input_path: str,
    output_path: str,
    gauge_coords: list[tuple[float, float]] | None = None,
):
    """Extract time series at gauge locations from a spatiotemporal tensor.

    Args:
        input_path: Path to input .npy file (time-ordered spatiotemporal data).
        output_path: Path to save the extracted gauge time series.
        gauge_coords: List of (x, y) tuples for gauge locations. Defaults to
            [(400.0, 50.0), (600.0, 50.0)] for backward compatibility.
    """
    print("--- Gauge Time Series Extraction ---")
    print(f"Input: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    t0 = time.time()

    # 1. Load Data
    try:
        data = np.load(input_path)
        print(f"Data loaded. Shape: {data.shape} ({data.nbytes / 1e9:.2f} GB)")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Determine Spatial Block Size (M)
    # The data is ordered by time. All points for t=0 come first.
    first_t = data[0, 0]
    t_changes = data[:, 0] != first_t

    if not np.any(t_changes):
        print("Warning: Only one timestep found in data.")
        spatial_block_size = data.shape[0]
        num_timesteps = 1
    else:
        spatial_block_size = np.argmax(t_changes)
        num_timesteps = data.shape[0] // spatial_block_size

    print(f"Spatial Grid Points (M): {spatial_block_size}")
    print(f"Estimated Timesteps (T): {num_timesteps}")

    # 3. Find Gauge Indices in the First Block
    spatial_view = data[:spatial_block_size]

    if gauge_coords is None:
        gauge_coords = [(400.0, 50.0), (600.0, 50.0)]

    selected_indices = []

    print("\nLocating gauges in spatial grid...")
    for i, (tx, ty) in enumerate(gauge_coords):
        dists_sq = (spatial_view[:, 1] - tx) ** 2 + (spatial_view[:, 2] - ty) ** 2
        local_idx = np.argmin(dists_sq)

        min_dist = np.sqrt(dists_sq[local_idx])
        found_x, found_y = spatial_view[local_idx, 1:3]
        print(
            f"  Gauge {i+1}: Target({tx},{ty}) -> Found({found_x:.2f},{found_y:.2f}) "
            f"[Dist: {min_dist:.4f}] @ Index {local_idx}"
        )

        selected_indices.append(local_idx)

    # 4. Extract All Times using Stride Slicing
    print("\nExtracting time series (Vectorized)...")

    extracted_rows = []
    for local_idx in selected_indices:
        gauge_series = data[local_idx::spatial_block_size]
        extracted_rows.append(gauge_series)

    final_data = np.vstack(extracted_rows)
    final_data = final_data[final_data[:, 0].argsort()]

    # 5. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, final_data)

    print(f"\nSuccess!")
    print(f"Output Shape: {final_data.shape}")
    print(f"Saved to: {output_path}")
    print(f"Total Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract gauge time series from a spatiotemporal .npy tensor."
    )
    parser.add_argument("--input", type=str, required=True, help="Input .npy file path")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file path")
    parser.add_argument(
        "--gauges",
        type=str,
        nargs="+",
        default=None,
        help="Gauge coordinates as x,y pairs (e.g., 400,50 600,50). "
        "Defaults to (400,50) and (600,50).",
    )

    args = parser.parse_args()

    gauge_coords = None
    if args.gauges:
        gauge_coords = []
        for g in args.gauges:
            parts = g.split(",")
            if len(parts) != 2:
                parser.error(f"Invalid gauge coordinate: {g}. Expected format: x,y")
            gauge_coords.append((float(parts[0]), float(parts[1])))

    extract_gauges_fast(args.input, args.output, gauge_coords)
