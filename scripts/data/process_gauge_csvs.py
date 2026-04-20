"""Process InfoWorks ICM gauge CSV exports into NumPy arrays.

Reads three CSV files (depth, angle, speed) exported from InfoWorks ICM along
with a metadata CSV that maps Point IDs to (X, Y) coordinates. Converts
speed + angle (radians) to velocity components u, v and outputs a NumPy array
with columns [t, x, y, h, u, v].

Supports two modes:
  - Merged (default): All gauges are written to a single output file.
  - Split (--split): Gauges are randomly split into training (70%) and
    validation (30%) subsets, saved to separate files.

Column conventions:
  Input CSVs:
    - depth: water depth h (m)
    - angle: flow direction in radians (from InfoWorks ICM)
    - speed: flow speed (m/s)
  Output .npy:
    - [t, x, y, h, u, v] where u = speed * cos(angle), v = speed * sin(angle)
    - t in seconds, x/y in the coordinate system of the metadata file

Usage:
    # Merged mode (single output)
    python process_gauge_csvs.py --meta gauge_metadata.csv --depth gauge_depth.csv \\
        --angle gauge_angle.csv --speed gauge_speed.csv --output gauge_data.npy

    # Split mode (train/val outputs)
    python process_gauge_csvs.py --meta gauge_metadata.csv --depth gauge_depth.csv \\
        --angle gauge_angle.csv --speed gauge_speed.csv --split \\
        --output_train train_gauges.npy --output_val val_gauges_gt.npy
"""

import pandas as pd
import numpy as np
import argparse
import os
import re


def process_gauge_csvs(
    meta_file: str,
    depth_file: str,
    angle_file: str,
    speed_file: str,
    output_file: str = None,
    split: bool = False,
    output_train: str = None,
    output_val: str = None,
    train_fraction: float = 0.7,
    seed: int = 42,
):
    """Process gauge CSVs into NumPy arrays with columns [t, x, y, h, u, v].

    Args:
        meta_file: Path to metadata CSV with columns [Point ID, X, Y].
        depth_file: Path to depth CSV (columns: Seconds, <point_ids>...).
        angle_file: Path to angle CSV in radians (same column layout as depth).
        speed_file: Path to speed CSV (same column layout as depth).
        output_file: Output .npy path (merged mode only).
        split: If True, split gauges into train/val subsets.
        output_train: Output .npy path for training data (split mode only).
        output_val: Output .npy path for validation data (split mode only).
        train_fraction: Fraction of gauges for training (default 0.7).
        seed: Random seed for reproducible train/val split.
    """
    mode = "split" if split else "merged"
    print(f"--- Processing Gauge CSVs ({mode} mode) ---")

    # 1. Load coordinate metadata
    print(f"Loading coordinates from: {meta_file}")
    try:
        coords_df = pd.read_csv(meta_file)
        coords_df.columns = [c.strip() for c in coords_df.columns]

        if "Point ID" not in coords_df.columns:
            print(f"Error: 'Point ID' column missing. Found: {coords_df.columns.tolist()}")
            return

        coords_df["Point ID"] = (
            pd.to_numeric(coords_df["Point ID"], errors="coerce").fillna(-1).astype(int)
        )
        id_map = {row["Point ID"]: (row["X"], row["Y"]) for _, row in coords_df.iterrows()}
        print(f"Found {len(id_map)} coordinate points in metadata.")
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    # 2. Load data CSVs
    print("Loading data CSVs...")
    try:
        df_depth = pd.read_csv(depth_file)
        df_angle = pd.read_csv(angle_file)
        df_speed = pd.read_csv(speed_file)

        for df in [df_depth, df_angle, df_speed]:
            df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"Error reading data files: {e}")
        return

    if "Seconds" not in df_depth.columns:
        print("Error: 'Seconds' column missing in depth file.")
        return

    time_col = df_depth["Seconds"].values
    n_steps = len(time_col)

    # 3. Parse columns and compute velocities
    processed_gauges = []
    matched_count = 0

    print("Parsing columns and computing velocities (u = speed*cos(angle), v = speed*sin(angle))...")

    for col in df_depth.columns:
        if col.lower() in ["time", "seconds"]:
            continue

        # Parse point ID from column name
        point_id = None
        if col.isdigit():
            point_id = int(col)
        else:
            match = re.search(r"(?:^|p|P|_)([0-9]+)$", col)
            if match:
                point_id = int(match.group(1))

        if point_id is not None and point_id in id_map:
            matched_count += 1
            x, y = id_map[point_id]

            if col not in df_angle.columns or col not in df_speed.columns:
                print(f"Warning: Column '{col}' (ID {point_id}) missing in angle or speed files.")
                continue

            h = df_depth[col].values
            angle = df_angle[col].values  # radians
            speed = df_speed[col].values

            # Convert speed + angle to velocity components
            u = speed * np.cos(angle)
            v = speed * np.sin(angle)

            # Stack: [t, x, y, h, u, v]
            block = np.column_stack([
                time_col,
                np.full(n_steps, x),
                np.full(n_steps, y),
                h,
                u,
                v,
            ])
            block = block[block[:, 0].argsort()]
            processed_gauges.append(block)
        elif point_id is not None and matched_count == 0:
            print(
                f"Debug: Parsed ID {point_id} from '{col}' but not in metadata "
                f"(IDs: {list(id_map.keys())[:5]}...)"
            )

    total_found = len(processed_gauges)
    print(f"Total valid gauges processed: {total_found}")

    if total_found == 0:
        print("Error: No gauges matched.")
        return

    # 4. Output
    if split:
        # Split into train/val subsets
        np.random.seed(seed)
        indices = np.arange(total_found)
        np.random.shuffle(indices)

        split_idx = int(total_found * train_fraction)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        print(f"Splitting: {len(train_indices)} training, {len(val_indices)} validation.")

        # Save training data
        train_data = np.vstack([processed_gauges[i] for i in train_indices])
        output_dir = os.path.dirname(output_train)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.save(output_train, train_data.astype(np.float32))
        print(f"Saved training data to: {output_train} {train_data.shape}")

        # Save validation data
        if len(val_indices) > 0:
            val_data = np.vstack([processed_gauges[i] for i in val_indices])
            output_dir = os.path.dirname(output_val)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            np.save(output_val, val_data.astype(np.float32))
            print(f"Saved validation data to: {output_val} {val_data.shape}")
    else:
        # Merged mode: single output
        final_array = np.vstack(processed_gauges)
        final_array = final_array[final_array[:, 0].argsort()]

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.save(output_file, final_array.astype(np.float32))

        print(f"\nProcessed {matched_count} gauges.")
        print(f"Output shape: {final_array.shape} -> [N, 6] (t, x, y, h, u, v)")
        print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert InfoWorks ICM gauge CSVs (depth, angle, speed) to NumPy arrays."
    )
    parser.add_argument("--meta", required=True, help="Metadata CSV with columns: Point ID, X, Y")
    parser.add_argument("--depth", required=True, help="Depth CSV (Seconds + point columns)")
    parser.add_argument("--angle", required=True, help="Angle CSV in radians (same layout as depth)")
    parser.add_argument("--speed", required=True, help="Speed CSV (same layout as depth)")

    # Output mode
    parser.add_argument("--output", help="Output .npy path (merged mode)")
    parser.add_argument("--split", action="store_true", help="Split into train/val subsets")
    parser.add_argument("--output_train", help="Training output .npy path (split mode)")
    parser.add_argument("--output_val", help="Validation output .npy path (split mode)")
    parser.add_argument(
        "--train_fraction", type=float, default=0.7, help="Training fraction (default: 0.7)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split (default: 42)")

    args = parser.parse_args()

    if args.split:
        if not args.output_train or not args.output_val:
            parser.error("--split requires --output_train and --output_val")
    else:
        if not args.output:
            parser.error("Merged mode requires --output")

    process_gauge_csvs(
        meta_file=args.meta,
        depth_file=args.depth,
        angle_file=args.angle,
        speed_file=args.speed,
        output_file=args.output,
        split=args.split,
        output_train=args.output_train,
        output_val=args.output_val,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
