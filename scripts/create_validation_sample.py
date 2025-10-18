import jax.numpy as jnp
import numpy as np
import argparse
from pathlib import Path
import jax

def create_validation_sample(
    source_path: Path,
    output_path: Path,
    num_samples: int,
    max_time: float,
    seed: int = 42,
    chunk_size: int = 1_000_000
):
    """
    Efficiently samples a subset of a large .npy file for validation using JAX and GPU.

    Args:
        source_path (Path): Path to the large source .npy file (e.g., validation_tensor.npy).
        output_path (Path): Path to save the smaller sampled .npy file.
        num_samples (int): The number of points to uniformly sample.
        max_time (float): The maximum timestamp to include in the sampling pool.
        seed (int): Random seed for reproducibility.
        chunk_size (int): Number of rows to process at once to avoid OOM.
    """
    print("--- Starting Validation Set Creation (JAX GPU) ---")
    print(f"JAX devices available: {jax.devices()}")
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source validation file not found at: {source_path}")

    # Use memory mapping to avoid loading entire file
    print(f"Loading source file: {source_path} (using memory mapping)")
    full_data = np.load(source_path, mmap_mode='r')
    print(f"Full dataset shape: {full_data.shape}")

    # --- Time Filtering (CPU-based for efficiency with mmap) ---
    print(f"Filtering data up to t = {max_time}s...")
    time_column = full_data[:, 0]
    valid_indices_count = np.searchsorted(time_column, max_time, side='right')
    
    if valid_indices_count == 0:
        raise ValueError(f"No data found in the time range [0, {max_time}]. Check your data or max_time setting.")

    print(f"Found {valid_indices_count} data points within the time range.")

    # --- Uniform Sampling (CPU-based to avoid OOM) ---
    print(f"Uniformly sampling {num_samples} points...")
    
    # Use numpy for sampling to avoid GPU OOM when pool is very large
    rng = np.random.default_rng(seed)
    random_indices_np = rng.choice(valid_indices_count, size=num_samples, replace=False)
    random_indices_np.sort()  # Sort for better mmap access patterns
    
    # --- Chunked Data Loading ---
    print(f"Loading sampled data in chunks (to avoid OOM)...")
    sampled_data_list = []
    
    for i in range(0, num_samples, chunk_size):
        end_idx = min(i + chunk_size, num_samples)
        chunk_indices = random_indices_np[i:end_idx]
        
        # Load chunk from mmap
        chunk_data = full_data[chunk_indices, :]
        
        # Convert to JAX array for GPU processing (if needed for further ops)
        chunk_jax = jnp.array(chunk_data)
        
        # Convert back to numpy for concatenation
        sampled_data_list.append(np.array(chunk_jax))
        
        print(f"  Processed {end_idx}/{num_samples} samples...")
    
    # Concatenate all chunks
    sampled_data = np.vstack(sampled_data_list)

    # --- Save the Output ---
    print(f"Saving sampled data to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sampled_data)

    print("\n--- ✅ Validation Set Creation Complete ---")
    print(f"Final sample shape: {sampled_data.shape}")
    print(f"Columns: [t, x, y, h, u, v]")
    print("-----------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a smaller, sampled validation set from a large .npy file.")
    parser.add_argument(
        "--scenario", 
        type=str, 
        required=True,
        help="The name of the scenario directory inside 'data/' (e.g., 'one_building_DEM_zero')."
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=10000,
        help="The number of points to sample."
    )
    parser.add_argument(
        "--time",
        type=float,
        default=3600.0,
        help="The maximum time to include in the sample (e.g., 3600.0)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of rows to process at once to avoid OOM (default: 1,000,000)."
    )
    
    args = parser.parse_args()
    
    BASE_PATH = Path('data') / args.scenario
    SOURCE_FILE = BASE_PATH / 'validation_tensor.npy'
    OUTPUT_FILE = BASE_PATH / 'validation_sample.npy'

    create_validation_sample(
        source_path=SOURCE_FILE,
        output_path=OUTPUT_FILE,
        num_samples=args.samples,
        max_time=args.time,
        chunk_size=args.chunk_size
    )