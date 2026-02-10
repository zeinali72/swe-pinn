import pandas as pd
import numpy as np
import argparse
import os
import re

def process_gauges_split(meta_file, depth_file, angle_file, speed_file, out_train, out_val):
    print(f"--- Processing Gauges with Split (Train=21, Val=9) ---")
    
    # 1. Load Coordinates
    print(f"Loading coordinates from: {meta_file}")
    try:
        coords_df = pd.read_csv(meta_file)
        coords_df.columns = [c.strip() for c in coords_df.columns]
        
        # Ensure 'Point ID' exists and is integer
        if 'Point ID' not in coords_df.columns:
            print(f"Error: 'Point ID' column missing. Found: {coords_df.columns.tolist()}")
            return
        
        # Force IDs to int
        coords_df['Point ID'] = pd.to_numeric(coords_df['Point ID'], errors='coerce').fillna(-1).astype(int)
        
        # Map ID -> (X, Y)
        id_map = {row['Point ID']: (row['X'], row['Y']) for _, row in coords_df.iterrows()}
        print(f"Found {len(id_map)} coordinate points in metadata.")
        
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    # 2. Load Data Files
    print("Loading data CSVs...")
    try:
        df_depth = pd.read_csv(depth_file)
        df_angle = pd.read_csv(angle_file)
        df_speed = pd.read_csv(speed_file)
        
        # Clean headers
        for df in [df_depth, df_angle, df_speed]:
            df.columns = [c.strip() for c in df.columns]
            
    except Exception as e:
        print(f"Error reading data files: {e}")
        return

    if 'Seconds' not in df_depth.columns:
        print("Error: 'Seconds' column missing in depth file.")
        return

    time_col = df_depth['Seconds'].values
    n_steps = len(time_col)
    
    # Store processed arrays here
    processed_gauges = [] 

    print("Parsing columns and computing (h, hu, hv)...")
    
    for col in df_depth.columns:
        # Skip time columns
        if col.lower() in ['time', 'seconds']:
            continue
            
        # Parse ID from column name
        point_id = None
        if col.isdigit():
            point_id = int(col)
        else:
            # Matches 'p1', 'point_1', 'Gauge 01', etc.
            match = re.search(r'(\d+)$', col)
            if match:
                point_id = int(match.group(1))

        if point_id is not None and point_id in id_map:
            # We have a valid gauge
            x, y = id_map[point_id]
            
            # Check other files
            if col not in df_angle.columns or col not in df_speed.columns:
                print(f"Warning: Column '{col}' missing in angle/speed files.")
                continue
            
            # Extract Data
            h = df_depth[col].values
            angle = df_angle[col].values
            speed = df_speed[col].values
            
            # Compute State
            u = speed * np.cos(angle)
            v = speed * np.sin(angle)

            
            # Create Block [N, 6] -> t, x, y, h, hu, hv
            # We assume x, y are constant for this gauge
            block = np.column_stack([
                time_col,
                np.full(n_steps, x),
                np.full(n_steps, y),
                h,
                u,
                v
            ])
            
            # Sort by time just in case
            block = block[block[:, 0].argsort()]
            processed_gauges.append(block)

    total_found = len(processed_gauges)
    print(f"Total valid gauges processed: {total_found}")
    
    if total_found == 0:
        print("Error: No gauges matched.")
        return

    # 3. Split Data (21 Train, 9 Val)
    # Using fixed seed for reproducibility
    np.random.seed(42)
    indices = np.arange(total_found)
    np.random.shuffle(indices)
    
    # Logic: 
    # If we have exactly 30, we take 21 and 9. 
    # If we have distinct count, we try to approximate or just split 70/30.
    # User requested 21 Train, 9 Val.
    
    n_train_target = 21
    n_val_target = 9
    
    if total_found == 30:
        train_indices = indices[:n_train_target]
        val_indices = indices[n_train_target:]
    else:
        print(f"Warning: Expected 30 gauges, found {total_found}. Splitting proportionally (approx 70/30).")
        split_idx = int(total_found * 0.7)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
    
    print(f"Splitting: {len(train_indices)} Training, {len(val_indices)} Validation.")

    # 4. Stack and Save
    # Train
    train_data_list = [processed_gauges[i] for i in train_indices]
    if train_data_list:
        train_array = np.vstack(train_data_list)
        os.makedirs(os.path.dirname(out_train), exist_ok=True)
        np.save(out_train, train_array.astype(np.float32))
        print(f"Saved TRAINING data to: {out_train} {train_array.shape}")
        
    # Val
    val_data_list = [processed_gauges[i] for i in val_indices]
    if val_data_list:
        val_array = np.vstack(val_data_list)
        os.makedirs(os.path.dirname(out_val), exist_ok=True)
        np.save(out_val, val_array.astype(np.float32))
        print(f"Saved VALIDATION data to: {out_val} {val_array.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Metadata CSV with Point ID, X, Y")
    parser.add_argument("--depth", required=True)
    parser.add_argument("--angle", required=True)
    parser.add_argument("--speed", required=True)
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_val", required=True)
    args = parser.parse_args()
    
    process_gauges_split(
        args.meta, args.depth, args.angle, args.speed, 
        args.output_train, args.output_val
    )