import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
import gc

def prepare_validation_data_gpu(
    angle_path: Union[str, Path],
    depth_path: Union[str, Path],
    speed_path: Union[str, Path],
    output_path: Union[str, Path],
    chunk_size: int = 100000
):
    """
    GPU-accelerated data preparation using RAPIDS cuDF.
    """
    print("🚀 Starting GPU-accelerated data preparation...")
    
    try:
        import cudf
        import cupy as cp
        print("✅ RAPIDS cuDF and CuPy loaded successfully")
    except ImportError:
        print("❌ RAPIDS not available. Install with: pip install cudf-cu12 cupy-cuda12x")
        raise
    
    try:
        angle_path = Path(angle_path)
        depth_path = Path(depth_path)
        speed_path = Path(speed_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get column names and parse coordinates ONCE on CPU
        print("Extracting column information...")
        header_df = pd.read_csv(angle_path, nrows=0)
        time_col = 'Seconds' if 'Seconds' in header_df.columns else 'Time'
        coord_cols = [col for col in header_df.columns if col not in [time_col, 'Time']]
        print(f"Found {len(coord_cols)} coordinate points")

        # Parse ALL coordinates once and create x, y arrays
        coords_parsed = []
        for coord in coord_cols:
            try:
                x_str, y_str = coord.split(',')
                coords_parsed.append((float(x_str), float(y_str)))
            except:
                continue
        
        # Move coordinate arrays to GPU ONCE
        x_gpu = cp.array([c[0] for c in coords_parsed], dtype=cp.float32)
        y_gpu = cp.array([c[1] for c in coords_parsed], dtype=cp.float32)
        print(f"Moved {len(coords_parsed)} coordinates to GPU")

        # Store GPU arrays instead of CPU numpy arrays
        all_data_gpu = []
        processed_rows = 0
        
        # Read and process in chunks
        for chunk_idx, (chunk_angle, chunk_depth, chunk_speed) in enumerate(
            _read_chunks_aligned(angle_path, depth_path, speed_path, chunk_size)
        ):
            print(f"\n📦 Processing chunk {chunk_idx + 1}...")
            
            # Move to GPU
            df_angle = cudf.from_pandas(chunk_angle)
            df_depth = cudf.from_pandas(chunk_depth)
            df_speed = cudf.from_pandas(chunk_speed)
            
            time_data = df_angle[time_col].values
            n_timesteps = len(time_data)
            n_coords = len(coords_parsed)
            
            # Preallocate result array on GPU (much faster than concatenating)
            result_gpu = cp.zeros((n_timesteps * n_coords, 6), dtype=cp.float32)
            
            idx = 0
            for i, coord in enumerate(coord_cols[:len(coords_parsed)]):
                # Extract columns on GPU
                angle_gpu = df_angle[coord].values
                speed_gpu = df_speed[coord].values
                depth_gpu = df_depth[coord].values
                
                # Vectorized computation on GPU
                angle_rad = cp.deg2rad(angle_gpu)
                u_gpu = speed_gpu * cp.cos(angle_rad)
                v_gpu = speed_gpu * cp.sin(angle_rad)
                
                # Fill result array
                n = len(angle_gpu)
                result_gpu[idx:idx+n, 0] = time_data
                result_gpu[idx:idx+n, 1] = x_gpu[i]
                result_gpu[idx:idx+n, 2] = y_gpu[i]
                result_gpu[idx:idx+n, 3] = depth_gpu
                result_gpu[idx:idx+n, 4] = u_gpu
                result_gpu[idx:idx+n, 5] = v_gpu
                idx += n
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1} coordinates...")
            
            # Keep data on GPU! Store the GPU array
            all_data_gpu.append(result_gpu)
            
            processed_rows += result_gpu.shape[0]
            print(f"✓ Chunk processed: {processed_rows} rows | Chunk shape: {result_gpu.shape}")
            
            # Cleanup temporary GPU memory
            del df_angle, df_depth, df_speed, chunk_angle, chunk_depth, chunk_speed
            del angle_gpu, speed_gpu, depth_gpu, angle_rad, u_gpu, v_gpu
            gc.collect()

        # Concatenate all chunks ON GPU
        print("\n🔗 Concatenating chunks on GPU...")
        data_matrix_gpu = cp.vstack(all_data_gpu)
        
        # Free intermediate GPU arrays
        del all_data_gpu
        gc.collect()
        
        # Sort ON GPU using cuDF
        print("Sorting data on GPU...")
        df_gpu = cudf.DataFrame({
            't': data_matrix_gpu[:, 0],
            'x': data_matrix_gpu[:, 1],
            'y': data_matrix_gpu[:, 2],
            'h': data_matrix_gpu[:, 3],
            'u': data_matrix_gpu[:, 4],
            'v': data_matrix_gpu[:, 5]
        })
        df_gpu = df_gpu.sort_values(by=['t', 'x', 'y']).reset_index(drop=True)
        
        # Convert back to GPU array
        data_matrix_gpu = cp.column_stack([
            df_gpu['t'].values,
            df_gpu['x'].values,
            df_gpu['y'].values,
            df_gpu['h'].values,
            df_gpu['u'].values,
            df_gpu['v'].values
        ])
        
        # Transfer to CPU ONLY ONCE at the very end
        print("Transferring final result to CPU for saving...")
        data_matrix = cp.asnumpy(data_matrix_gpu)
        
        print(f"Saving array to '{output_path}'...")
        np.save(output_path, data_matrix)

        print("\n--- ✅ Data Preparation Complete ---")
        print(f"Final array shape: {data_matrix.shape}")
        print(f"Columns: [t, x, y, h, u, v]")
        print(f"Output file size: {output_path.stat().st_size / (1024**3):.2f} GB")
        print("-----------------------------------")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def _read_chunks_aligned(angle_path, depth_path, speed_path, chunk_size):
    """Generator that reads aligned chunks from all three CSVs."""
    angle_reader = pd.read_csv(angle_path, chunksize=chunk_size)
    depth_reader = pd.read_csv(depth_path, chunksize=chunk_size)
    speed_reader = pd.read_csv(speed_path, chunksize=chunk_size)
    
    for chunk_angle, chunk_depth, chunk_speed in zip(angle_reader, depth_reader, speed_reader):
        yield chunk_angle, chunk_depth, chunk_speed

if __name__ == '__main__':
    SCENARIO_NAME = 'one_building_DEM_zero'
    BASE_PATH = Path('data') / SCENARIO_NAME

    ANGLE_FILE = BASE_PATH / '2D zone_one_building_DEM_zero_DWF_angle2d.csv'
    DEPTH_FILE = BASE_PATH / '2D zone_one_building_DEM_zero_DWF_depth2d.csv'
    SPEED_FILE = BASE_PATH / '2D zone_one_building_DEM_zero_DWF_speed2d.csv'
    OUTPUT_FILE = BASE_PATH / 'validation_tensor.npy'

    prepare_validation_data_gpu(
        angle_path=ANGLE_FILE,
        depth_path=DEPTH_FILE,
        speed_path=SPEED_FILE,
        output_path=OUTPUT_FILE,
        chunk_size=1000000
    )