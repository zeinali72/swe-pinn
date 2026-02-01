import pandas as pd
import numpy as np
import argparse
import os

def process_test2_gauges(meta_file, data_file, output_file):
    print(f"--- Processing Test 2 Gauges ---")
    
    # 1. Load Coordinate Map (Test2output.csv)
    # Format: Point ID, X, Y
    print(f"Loading coordinates from: {meta_file}")
    try:
        coords_df = pd.read_csv(meta_file)
        # Create a dictionary: {1: (250.0, 250.0), 2: (250.0, 750.0), ...}
        # Ensure column names match your file (remove potential spaces)
        coords_df.columns = [c.strip() for c in coords_df.columns]
        id_map = {row['Point ID']: (row['X'], row['Y']) for _, row in coords_df.iterrows()}
        print(f"Found {len(id_map)} coordinate points.")
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    # 2. Load Time-Series Data (2D zone...csv)
    # Format: Time, Seconds, p1, p2, ...
    print(f"Loading data from: {data_file}")
    try:
        data_df = pd.read_csv(data_file)
        # Clean column names (remove spaces)
        data_df.columns = [c.strip() for c in data_df.columns]
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    if 'Seconds' not in data_df.columns:
        print("Error: Could not find 'Seconds' column in data file.")
        return

    time_col = data_df['Seconds'].values
    n_steps = len(time_col)
    
    all_points_list = []

    # 3. Iterate through columns (p1, p2...) and merge
    print("Merging coordinates with time series...")
    
    for col in data_df.columns:
        # Check if column is a point (starts with 'p' followed by digit)
        if col.startswith('p') and col[1:].isdigit():
            point_id = int(col[1:])
            
            if point_id in id_map:
                x, y = id_map[point_id]
                h_values = data_df[col].values
                
                # Stack into (N, 4) -> [t, x, y, h]
                # We repeat x and y for every timestep
                point_block = np.column_stack([
                    time_col,
                    np.full(n_steps, x),
                    np.full(n_steps, y),
                    h_values
                ])
                
                all_points_list.append(point_block)
            else:
                print(f"Warning: Data column '{col}' has no matching ID in {meta_file}")

    # 4. Save to .npy
    if all_points_list:
        final_array = np.vstack(all_points_list)
        
        # Sort by Time (column 0) just to be safe
        final_array = final_array[final_array[:, 0].argsort()]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, final_array.astype(np.float32))
        
        print(f"\nSuccess!")
        print(f"Output Shape: {final_array.shape} -> [N, 4] (t, x, y, h)")
        print(f"Saved to: {output_file}")
    else:
        print("Error: No valid points were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to Test2output.csv (Coords)")
    parser.add_argument("--data", required=True, help="Path to ...depth2d.csv (Values)")
    parser.add_argument("--output", required=True, help="Path to save .npy file")
    args = parser.parse_args()
    
    process_test2_gauges(args.meta, args.data, args.output)