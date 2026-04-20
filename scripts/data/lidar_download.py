import requests
import xml.etree.ElementTree as ET
import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS
import os
import glob
import math
import shutil

# --- CONFIGURATION ---
WCS_BASE_URL = "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs"
TEMP_DIR = "temp_tiles"
OUTPUT_MERGED_TIFF = "merged_lidar_10km.tif"

# Your 10km x 10km Bounding Box (EPSG:27700)
MIN_X = 554771 
MIN_Y = 97959
MAX_X = 554771 + 10000
MAX_Y = 97959 + 10000

# Chunk size in meters (2000m = 2km x 2km tiles)
CHUNK_SIZE = 2000 
# ---------------------

def get_coverage_id(wcs_url):
    print("1. Fetching WCS Capabilities...")
    params = {"request": "GetCapabilities", "service": "WCS", "version": "2.0.1"}
    response = requests.get(wcs_url, params=params)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    ns = {'wcs': 'http://www.opengis.net/wcs/2.0'}
    coverage_id_elem = root.find('.//wcs:CoverageSummary/wcs:CoverageId', ns)
    
    if coverage_id_elem is not None:
        return coverage_id_elem.text
    raise ValueError("Could not find a CoverageId.")

def generate_tiles(min_x, min_y, max_x, max_y, chunk_size):
    """Slices the massive bounding box into smaller coordinate chunks."""
    tiles = []
    cols = math.ceil((max_x - min_x) / chunk_size)
    rows = math.ceil((max_y - min_y) / chunk_size)
    
    for c in range(cols):
        for r in range(rows):
            t_min_x = min_x + (c * chunk_size)
            t_min_y = min_y + (r * chunk_size)
            # Ensure the last tiles don't overshoot the maximum bounds
            t_max_x = min(t_min_x + chunk_size, max_x)
            t_max_y = min(t_min_y + chunk_size, max_y)
            
            tiles.append((t_min_x, t_min_y, t_max_x, t_max_y))
    return tiles

def download_tile(wcs_url, coverage_id, tile_bounds, filename):
    """Downloads a single chunk safely."""
    t_min_x, t_min_y, t_max_x, t_max_y = tile_bounds
    
    query_string = (
        f"service=WCS&version=2.0.1&request=GetCoverage"
        f"&coverageId={coverage_id}"
        f"&format=image/tiff"
        f"&subset=E({t_min_x},{t_max_x})"
        f"&subset=N({t_min_y},{t_max_y})"
    )
    
    request_url = f"{wcs_url}?{query_string}"
    
    try:
        # Added timeout to prevent infinite hanging
        response = requests.get(request_url, stream=True, timeout=(15, 60))
        
        # If the server returns 500, it usually means there is no LIDAR data for this specific tile
        if response.status_code != 200:
            print(f"      [Skipped] No data or server error for tile {filename}")
            return False

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except requests.exceptions.Timeout:
        print(f"      [Skipped] Server timed out on tile {filename}")
        return False
    except Exception as e:
        print(f"      [Skipped] Error on tile {filename}: {e}")
        return False

def merge_tiles(temp_dir, output_filename):
    """Uses rasterio to stitch all downloaded tiles into one seamless map."""
    print("\n3. Merging tiles into a single output file...")
    
    search_criteria = os.path.join(temp_dir, "*.tif")
    tile_files = glob.glob(search_criteria)
    
    if not tile_files:
        raise Exception("No tiles were successfully downloaded. Cannot merge.")
        
    src_files_to_mosaic = []
    for fp in tile_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
        
    # Perform the merge
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Copy metadata from the first tile and update it for the massive merged size
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": CRS.from_epsg(27700) # Force British National Grid
    })
    
    # Write the massive final file
    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    # Close all open files
    for src in src_files_to_mosaic:
        src.close()
        
    print(f"✅ Success! Merged file saved to '{output_filename}'")

if __name__ == "__main__":
    try:
        # Create temporary directory for tiles
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
            
        cov_id = get_coverage_id(WCS_BASE_URL)
        tiles = generate_tiles(MIN_X, MIN_Y, MAX_X, MAX_Y, CHUNK_SIZE)
        
        print(f"2. Downloading {len(tiles)} tiles...")
        
        # Download loop
        for i, tile_bounds in enumerate(tiles):
            tile_filename = os.path.join(TEMP_DIR, f"tile_{i}.tif")
            print(f"   -> Requesting tile {i+1}/{len(tiles)}...")
            download_tile(WCS_BASE_URL, cov_id, tile_bounds, tile_filename)
            
        # Merge the downloaded tiles
        merge_tiles(TEMP_DIR, OUTPUT_MERGED_TIFF)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    finally:
        # Clean up the temporary tiles to save hard drive space
        if os.path.exists(TEMP_DIR):
            print("4. Cleaning up temporary tile folder...")
            shutil.rmtree(TEMP_DIR)