# scripts/create_samples.py
import numpy as np
import os
import argparse
from pathlib import Path
import time
import gc # Garbage collector

# --- GPU Imports ---
try:
    import cupy as cp
    import cudf # Import checks installation
    GPU_AVAILABLE = True
    print("✅ CuPy and cuDF detected. Using GPU acceleration.")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy or cuDF not found. Falling back to CPU-based operations.")
# --- End GPU Imports ---

def create_datasets(
    scenario_name: str,
    validation_samples: int = 15000,
    training_samples: int = 30000,
    training_max_time: float = 3600.0,
    plotting_time: float = 1800.0,
    seed: int = 42
):
    """
    Generates training sample (t <= training_max_time) and validation sample (full range)
    using separate random seeds, minimizing overlap statistically. Also extracts plotting data.
    Uses memory mapping and GPU acceleration.

    Args:
        scenario_name (str): The name of the scenario directory within 'data/'.
        validation_samples (int): Number of samples for the validation set.
        training_samples (int): Number of samples for the training set.
        training_max_time (float): Maximum time value (inclusive) for training samples.
        plotting_time (float): The specific time value (t) to extract for plotting data.
        seed (int): Base random seed for reproducibility.
    """
    print(f"--- Starting Dataset Creation for Scenario: {scenario_name} ---")
    start_script_time = time.time()

    # --- 1. Define File Paths ---
    base_path = Path('data') / scenario_name
    input_file = base_path / 'validation_tensor.npy'
    val_sample_file = base_path / 'validation_sample.npy'
    train_sample_file = base_path / 'training_dataset_sample.npy'
    plot_file = base_path / f'validation_plotting_t_{int(plotting_time)}s.npy'

    if not base_path.exists():
        print(f"Error: Scenario directory not found at {base_path}")
        return
    if not input_file.exists():
        print(f"Error: Input file validation_tensor.npy not found at {input_file}")
        return

    print(f"Input tensor: {input_file}")
    print(f"Output validation sample: {val_sample_file} ({validation_samples} points, full time range, separate RNG seed)")
    print(f"Output training sample: {train_sample_file} ({training_samples} points, t <= {training_max_time}s)")
    print(f"Output plotting data: {plot_file} (t={plotting_time}s)")

    # --- 2. Load Large File using Memory Mapping ---
    print("\nLoading large validation tensor using memory mapping...")
    try:
        data_mmap = np.load(input_file, mmap_mode='r')
        num_rows_total = data_mmap.shape[0]
        print(f"Successfully loaded tensor with shape: {data_mmap.shape}")
        if data_mmap.shape[1] != 6:
            print(f"Warning: Expected 6 columns [t, x, y, h, u, v], but found {data_mmap.shape[1]}.")
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    # --- Setup RNG Seeds ---
    # Use different seeds derived from the base seed
    train_seed = seed
    val_seed = seed + 1 # Simple way to get a different seed sequence

    np_rng_train = np.random.default_rng(train_seed)
    np_rng_val = np.random.default_rng(val_seed)
    # CuPy seed will be set before each use

    # --- 3. Find Training Index Range (Leveraging Sorted Data) ---
    print(f"\nFinding index range for training data (t <= {training_max_time}s)...")
    start_index_time = time.time()
    time_column_mmap = data_mmap[:, 0]
    end_train_idx_exclusive = np.searchsorted(time_column_mmap, training_max_time, side='right')
    num_eligible_train = int(end_train_idx_exclusive)
    del time_column_mmap
    gc.collect()

    if num_eligible_train == 0:
        print(f"Error: No data found with t <= {training_max_time}s. Cannot create training sample.")
        return

    print(f"Found {num_eligible_train} eligible rows for training (indices 0 to {num_eligible_train - 1}).")
    print(f"Index identification took {time.time() - start_index_time:.2f} seconds.")

    # --- 4. Sample Training Data (GPU or CPU) ---
    if training_samples > num_eligible_train:
        print(f"Warning: Requested {training_samples} training samples, but only {num_eligible_train} are available. Using all eligible rows.")
        training_samples = num_eligible_train

    print(f"Sampling {training_samples} unique training indices using seed {train_seed}...")
    start_train_sample_time = time.time()

    GPU_MEMORY_THRESHOLD = 100_000_000

    if GPU_AVAILABLE and num_eligible_train < GPU_MEMORY_THRESHOLD:
        print("  Using GPU for direct range sampling (training)...")
        try:
            # --- MODIFICATION START ---
            cp.random.seed(train_seed) # Set CuPy seed for training
            train_indices_gpu = cp.random.choice(num_eligible_train, size=training_samples, replace=False)
            # --- MODIFICATION END ---
            train_indices = cp.asnumpy(train_indices_gpu)
            del train_indices_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"  GPU sampling failed ({e}), falling back to CPU...")
            train_indices = np_rng_train.choice(num_eligible_train, size=training_samples, replace=False)
    else:
        if GPU_AVAILABLE: print(f"  Dataset range too large for GPU memory ({num_eligible_train}), using CPU sampling...")
        train_indices = np_rng_train.choice(num_eligible_train, size=training_samples, replace=False)

    print(f"Extracting training sample ({training_samples} points)...")
    training_data = data_mmap[train_indices, :]
    np.save(train_sample_file, training_data.astype(np.float32))
    print(f"Saved training sample to {train_sample_file}")
    print(f"Training sampling took {time.time() - start_train_sample_time:.2f} seconds.")

    # --- 5. Sample Validation Data (GPU or CPU - Simplified, Separate Seed) ---
    print(f"\nSampling validation indices using seed {val_seed} (potential small overlap)...")
    start_val_sample_time = time.time()

    if validation_samples > num_rows_total:
        print(f"Warning: Requested {validation_samples} validation samples, but only {num_rows_total} are available. Using all rows.")
        validation_samples = num_rows_total

    if GPU_AVAILABLE and num_rows_total < GPU_MEMORY_THRESHOLD:
        print("  Using GPU for direct range sampling (validation)...")
        try:
            # --- MODIFICATION START ---
            cp.random.seed(val_seed) # Set CuPy seed for validation
            val_indices_gpu = cp.random.choice(num_rows_total, size=validation_samples, replace=False)
             # --- MODIFICATION END ---
            val_indices = cp.asnumpy(val_indices_gpu)
            del val_indices_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"  GPU sampling failed ({e}), falling back to CPU...")
            val_indices = np_rng_val.choice(num_rows_total, size=validation_samples, replace=False)
    else:
        if GPU_AVAILABLE: print(f"  Dataset range too large for GPU memory ({num_rows_total}), using CPU sampling...")
        val_indices = np_rng_val.choice(num_rows_total, size=validation_samples, replace=False)


    print(f"Extracting validation sample ({validation_samples} points)...")
    validation_data = data_mmap[val_indices, :]
    np.save(val_sample_file, validation_data.astype(np.float32))
    print(f"Saved validation sample to {val_sample_file}")
    print(f"Validation sampling took {time.time() - start_val_sample_time:.2f} seconds.")


    # --- 6. Extract Plotting Data (Chunking, potentially GPU within chunk) ---
    print(f"\nExtracting plotting data for t = {plotting_time}s...")
    plotting_data = []
    chunk_size_plot = 10_000_000
    processed_rows_plot = 0
    start_plot_time = time.time()

    try:
        plot_indices = []
        for i in range(0, num_rows_total, chunk_size_plot):
            chunk_start = i
            chunk_end = min(i + chunk_size_plot, num_rows_total)
            time_chunk_np = data_mmap[chunk_start:chunk_end, 0]

            if GPU_AVAILABLE:
                time_chunk_gpu = cp.asarray(time_chunk_np)
                relative_indices_gpu = cp.where(cp.isclose(time_chunk_gpu, cp.float32(plotting_time)))[0]
                if relative_indices_gpu.size > 0:
                    plot_indices.extend(cp.asnumpy(relative_indices_gpu) + chunk_start)
                del time_chunk_gpu, relative_indices_gpu
                cp.get_default_memory_pool().free_all_blocks()
            else:
                 relative_indices_np = np.where(np.isclose(time_chunk_np, plotting_time))[0]
                 if relative_indices_np.size > 0:
                     plot_indices.extend(relative_indices_np + chunk_start)

            processed_rows_plot += (chunk_end - chunk_start)
            if processed_rows_plot % (chunk_size_plot * 5) == 0:
                 print(f"  Scanned {processed_rows_plot}/{num_rows_total} rows for plotting data...")

        if plot_indices:
            print(f"Found {len(plot_indices)} points for plotting. Extracting data...")
            plotting_data_final = data_mmap[plot_indices, :]
            np.save(plot_file, plotting_data_final.astype(np.float32))
            print(f"Saved plotting data to {plot_file}")
        else:
            print(f"No data points found for t = {plotting_time}s.")
        print(f"Plotting data extraction took {time.time() - start_plot_time:.2f} seconds.")

    except Exception as e:
        print(f"Error occurred during plotting data extraction: {e}")
        import traceback
        traceback.print_exc()

    # --- 7. Cleanup ---
    del data_mmap
    gc.collect()
    print(f"\n--- Dataset Creation Finished (Total time: {time.time() - start_script_time:.2f}s) ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create training (time-limited) and validation (separate RNG seed) samples."
    )
    # Arguments remain the same...
    parser.add_argument(
        "--scenario", type=str, default="one_building_DEM_zero", help="Scenario directory under ./data/"
    )
    parser.add_argument(
        "--val_samples", type=int, default=15000, help="Number of validation samples"
    )
    parser.add_argument(
        "--train_samples", type=int, default=30000, help="Number of training samples"
    )
    parser.add_argument(
        "--train_max_time", type=float, default=3600.0, help="Max time for training samples"
    )
    parser.add_argument(
        "--plot_time", type=float, default=1800.0, help="Time for plotting dataset"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed"
    )

    args = parser.parse_args()

    create_datasets(
        scenario_name=args.scenario,
        validation_samples=args.val_samples,
        training_samples=args.train_samples,
        training_max_time=args.train_max_time,
        plotting_time=args.plot_time,
        seed=args.seed
    )