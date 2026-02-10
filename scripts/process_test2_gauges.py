import pandas as pd
import numpy as np
import argparse
import os
import re

def process_test2_gauges(meta_file, depth_file, angle_file, speed_file, output_file):
    print(f"--- Processing Test 2 Gauges (Merged) ---")
    
    # 1. Load Coordinate Map
    print(f"Loading coordinates from: {meta_file}")
    try:
        coords_df = pd.read_csv(meta_file)
        coords_df.columns = [c.strip() for c in coords_df.columns]
        
        if 'Point ID' not in coords_df.columns:
            print(f"Error: 'Point ID' column not found in metadata. Found: {coords_df.columns.tolist()}")
            return

        # Ensure Point ID is integer for consistent matching
        # (Handles cases where ID might be read as float 1.0 or string "01")
        coords_df['Point ID'] = pd.to_numeric(coords_df['Point ID'], errors='coerce').fillna(-1).astype(int)
        
        id_map = {row['Point ID']: (row['X'], row['Y']) for _, row in coords_df.iterrows()}
        print(f"Found {len(id_map)} coordinate points.")
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    # 2. Load Data Files
    print("Loading data files...")
    try:
        # Load header only first to check columns (optimization)
        df_depth = pd.read_csv(depth_file)
        df_angle = pd.read_csv(angle_file)
        df_speed = pd.read_csv(speed_file)
        
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
    all_points_list = []

    print("Merging data and computing velocities...")
    
    matched_count = 0
    
    for col in df_depth.columns:
        # Skip time columns
        if col.lower() in ['time', 'seconds']:
            continue
            
        point_id = None
        
        # Strategy 1: Is the column just a number? (e.g. "01", "12")
        if col.isdigit():
            point_id = int(col)
        # Strategy 2: Does it look like "p1" or "point_1"?
        else:
            match = re.search(r'(?:^|p|P|_)([0-9]+)$', col)
            if match:
                point_id = int(match.group(1))

        if point_id is not None:
            if point_id in id_map:
                matched_count += 1
                x, y = id_map[point_id]
                
                # Verify existence in other files
                if col not in df_angle.columns or col not in df_speed.columns:
                    print(f"Warning: Column '{col}' (ID {point_id}) missing in angle or speed files.")
                    continue
                    
                h = df_depth[col].values
                angle = df_angle[col].values # Radians
                speed = df_speed[col].values
                
                u = speed * np.cos(angle)
                v = speed * np.sin(angle)
                
                # Stack: [t, x, y, h, u, v]
                point_block = np.column_stack([
                    time_col,
                    np.full(n_steps, x),
                    np.full(n_steps, y),
                    h,
                    u,
                    v
                ])
                all_points_list.append(point_block)
            else:
                # Only print first few mismatches to avoid spam
                if matched_count == 0: 
                    print(f"Debug: Parsed ID {point_id} from '{col}' but it's not in metadata (IDs: {list(id_map.keys())[:5]}...)")
        
    # 3. Save
    if all_points_list:
        final_array = np.vstack(all_points_list)
        final_array = final_array[final_array[:, 0].argsort()]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, final_array.astype(np.float32))
        
        print(f"\nSuccess! Processed {matched_count} gauges.")
        print(f"Output Shape: {final_array.shape} -> [N, 6] (t, x, y, h, u, v)")
        print(f"Saved to: {output_file}")
    else:
        print("\nError: No valid points processed.")
        print(f"Depth columns: {df_depth.columns.tolist()[:10]}...")
        print(f"Metadata IDs: {list(id_map.keys())[:10]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--angle", required=True)
    parser.add_argument("--speed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    process_test2_gauges(args.meta, args.depth, args.angle, args.speed, args.output)