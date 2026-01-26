import numpy as np
import argparse
import os
import sys
import time

def extract_gauges_fast(input_path, output_path):
    print(f"--- Fast Gauge Extraction ---")
    print(f"Input: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    t0 = time.time()
    
    # 1. Load Data
    # We load fully into memory for max speed. 
    # If OOM occurs, use mmap_mode='r', but slice access might be slower.
    try:
        data = np.load(input_path)
        print(f"Data loaded. Shape: {data.shape} ({data.nbytes / 1e9:.2f} GB)")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # 2. Determine Spatial Block Size (M)
    # The data is ordered by time. All points for t=0 come first.
    # We count how many rows share the first timestamp.
    first_t = data[0, 0]
    
    # Find the first index where t changes
    # np.argmax on boolean gives index of first True
    t_changes = (data[:, 0] != first_t)
    
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
    # We only search within the first M rows (spatial_block)
    spatial_view = data[:spatial_block_size]
    
    gauges = {
        "P1_400_50": (400.0, 50.0),
        "P2_600_50": (600.0, 50.0)
    }
    
    selected_indices = []
    
    print("\nLocating gauges in spatial grid...")
    for name, (tx, ty) in gauges.items():
        # Euclidean distance in the first block
        dists_sq = (spatial_view[:, 1] - tx)**2 + (spatial_view[:, 2] - ty)**2
        local_idx = np.argmin(dists_sq)
        
        # Verify distance is reasonable
        min_dist = np.sqrt(dists_sq[local_idx])
        found_x, found_y = spatial_view[local_idx, 1:3]
        print(f"  {name}: Target({tx},{ty}) -> Found({found_x:.2f},{found_y:.2f}) [Dist: {min_dist:.4f}] @ Index {local_idx}")
        
        selected_indices.append(local_idx)

    # 4. Extract All Times using Stride Slicing
    # If gauge is at local_idx, it appears at local_idx, local_idx+M, local_idx+2M...
    print("\nExtracting time series (Vectorized)...")
    
    extracted_rows = []
    for local_idx in selected_indices:
        # Slice: start=local_idx, stop=None, step=spatial_block_size
        gauge_series = data[local_idx::spatial_block_size]
        extracted_rows.append(gauge_series)
        
    # Stack them: We want a 2D array (N_total, 6)
    # result shape: (2 * Timesteps, 6)
    final_data = np.vstack(extracted_rows)
    
    # Sort by time (optional, but keeps it clean: t0_p1, t0_p2, t1_p1...)
    # Or keep it grouped by gauge. Usually, sorting by time is better for playback.
    # To match 'sample_validation', usually order doesn't strictly matter, 
    # but let's sort by time (column 0) to be safe.
    final_data = final_data[final_data[:, 0].argsort()]

    # 5. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, final_data)
    
    print(f"\nSuccess!")
    print(f"Output Shape: {final_data.shape} (Matches sample_validation 2D format)")
    print(f"Saved to: {output_path}")
    print(f"Total Time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    extract_gauges_fast(args.input, args.output)