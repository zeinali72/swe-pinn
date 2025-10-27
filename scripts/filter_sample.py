import numpy as np
import os
import argparse
from pathlib import Path

def filter_validation_sample_by_time(input_npy_path: str, max_time: float = 3600.0) -> None:
    """
    Loads a validation_sample.npy file, filters it to keep rows where
    time (first column) is less than or equal to max_time, and saves
    the filtered data to a new file in the same directory.

    Args:
        input_npy_path (str): Path to the input validation_sample.npy file.
        max_time (float): The maximum time value (inclusive) to keep.
                          Defaults to 3600.0.
    """
    input_path = Path(input_npy_path)
    if not input_path.is_file():
        print(f"Error: Input file not found at {input_npy_path}")
        return

    # Construct the output filename
    output_filename = f"{input_path.stem}_tmax{int(max_time)}{input_path.suffix}"
    output_path = input_path.parent / output_filename

    print(f"Loading data from: {input_path}...")
    try:
        data = np.load(input_path)
        print(f"Original data shape: {data.shape}")

        if data.shape[1] < 1:
             print(f"Error: Data in {input_path} has no columns.")
             return

        # Assuming time 't' is the first column (index 0)
        time_column = data[:, 0]

        # Create a boolean mask for filtering
        time_mask = time_column <= max_time

        # Apply the mask
        filtered_data = data[time_mask]
        print(f"Filtered data shape (t <= {max_time}s): {filtered_data.shape}")

        if filtered_data.shape[0] == 0:
            print(f"Warning: No data found with t <= {max_time}s. Output file will be empty.")

        # Save the filtered data
        np.save(output_path, filtered_data)
        print(f"Filtered data saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Filter a validation_sample.npy file based on a maximum time value."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input validation_sample.npy file."
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=3600.0,
        help="Maximum time value (inclusive) to keep in the output file. Default is 3600.0."
    )

    args = parser.parse_args()

    # --- Example Usage ---
    # Assuming your file is in 'data/one_building_DEM_zero/validation_sample.npy'
    # You would run this script like:
    # python <script_name>.py data/one_building_DEM_zero/validation_sample.npy --max_time 3600.0
    # Or simply:
    # python <script_name>.py data/one_building_DEM_zero/validation_sample.npy
    # (since 3600.0 is the default max_time)

    filter_validation_sample_by_time(args.input_file, args.max_time)