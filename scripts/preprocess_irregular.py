import argparse
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import triangulate, unary_union
from shapely.validation import make_valid

def load_and_fix_geometry(shp_path):
    """Loads a shapefile, fixes invalid geometries, and dissolves into a single geometry."""
    gdf = gpd.read_file(shp_path)
    # Use union_all if available (newer geopandas), else unary_union
    try:
        geom = gdf.union_all()
    except AttributeError:
        geom = gdf.unary_union
        
    if not geom.is_valid:
        geom = make_valid(geom)
    return geom

def triangulate_polygon(geom):
    """Triangulates the polygon (filtering triangles outside the domain)."""
    triangles = triangulate(geom)
    valid_triangles = []
    check_geom = geom.buffer(0) 
    for tri in triangles:
        # Check centroid with a small buffer for robustness
        # This handles holes correctly (centroid of triangle in hole returns False)
        if check_geom.contains(tri.centroid):
            valid_triangles.append(tri)
    return valid_triangles

def get_triangle_data(triangles):
    """Converts shapely triangles to numpy arrays."""
    n_tri = len(triangles)
    coords = np.zeros((n_tri, 3, 2), dtype=np.float32)
    areas = np.zeros(n_tri, dtype=np.float32)
    for i, tri in enumerate(triangles):
        pts = np.array(tri.exterior.coords)[:3]
        coords[i] = pts
        areas[i] = tri.area
    return coords, areas

def geom_to_segments(geom):
    """Converts any geometry (LineString/MultiLineString) to segments."""
    segments = []
    if geom.is_empty:
        return segments
        
    parts = []
    if geom.geom_type == 'LineString':
        parts = [geom]
    elif geom.geom_type == 'MultiLineString':
        parts = geom.geoms
    elif geom.geom_type in ['Polygon', 'MultiPolygon']:
        parts = [geom.boundary]
        
    for line in parts:
        pts = list(line.coords)
        for i in range(len(pts) - 1):
            segments.append((pts[i], pts[i+1]))
    return segments

def build_boundary_arrays(segments):
    """Converts a list of segment tuples to arrays and CDF."""
    if not segments:
        return None, None, None
        
    n_seg = len(segments)
    starts = np.zeros((n_seg, 2), dtype=np.float32)
    vectors = np.zeros((n_seg, 2), dtype=np.float32)
    lengths = np.zeros(n_seg, dtype=np.float32)
    
    for i, (p1, p2) in enumerate(segments):
        starts[i] = p1
        vec = np.array(p2) - np.array(p1)
        vectors[i] = vec
        lengths[i] = np.linalg.norm(vec)
        
    probs = lengths / np.sum(lengths)
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0 # Ensure exact 1.0
    
    return starts, vectors, cdf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_shapefile", type=str, help="Path to domain polygon.")
    parser.add_argument("--out", type=str, default=None, help="Output directory. Defaults to input directory.")
    parser.add_argument("--bc", action='append', help="Additional BC file: 'label=path'")
    parser.add_argument("--buildings", type=str, help="Path to buildings shapefile (voids/obstacles).")
    args = parser.parse_args()
    
    # Determine output directory (default to same as input if not specified)
    if args.out is None:
        args.out = os.path.dirname(args.domain_shapefile)
        if not args.out: args.out = "."

    # 1. Process Domain Mesh (Interior)
    print(f"Processing Domain: {args.domain_shapefile}")
    domain_geom = load_and_fix_geometry(args.domain_shapefile)
    
    # NEW: Handle Buildings (Voids)
    bldg_bc_geom = None
    if args.buildings:
        print(f"  - Loading buildings from {args.buildings}...")
        bldg_gdf = gpd.read_file(args.buildings)
        try:
            bldg_geom = bldg_gdf.union_all()
        except AttributeError:
            bldg_geom = bldg_gdf.unary_union
            
        if not bldg_geom.is_valid:
            bldg_geom = make_valid(bldg_geom)
            
        print("    -> Subtracting buildings from domain (creating voids)...")
        domain_geom = domain_geom.difference(bldg_geom)
        
        # Save building boundary for BC processing
        bldg_bc_geom = bldg_geom.boundary
    
    # Triangulate (handles holes/voids via containment check)
    triangles = triangulate_polygon(domain_geom)
    tri_coords, tri_areas = get_triangle_data(triangles)
    tri_cdf = np.cumsum(tri_areas / np.sum(tri_areas))
    tri_cdf[-1] = 1.0
    
    save_dict = {
        'tri_coords': tri_coords,
        'tri_cdf': tri_cdf
    }
    
    # 2. Process Boundaries (Subtraction Logic)
    
    # Start with the full domain boundary as "Wall"
    # Note: domain_geom.boundary now includes the building perimeters (holes)
    wall_geom = domain_geom.boundary 
    
    # List to store processed specialized BCs
    special_bcs = []
    
    # Process explicit BCs passed via --bc
    if args.bc:
        for item in args.bc:
            try:
                label, path = item.split('=', 1)
                print(f"  - Loading specialized BC '{label}' from {path}...")
                
                bc_gdf = gpd.read_file(path)
                try:
                    bc_geom = bc_gdf.union_all()
                except AttributeError:
                    bc_geom = bc_gdf.unary_union
                
                special_bcs.append((label, bc_geom))
                
                print(f"    -> Subtracting '{label}' from Wall boundary...")
                eraser = bc_geom.buffer(0.01)
                wall_geom = wall_geom.difference(eraser)
                
            except Exception as e:
                print(f"Error processing BC {item}: {e}")

    # Process Building BC automatically if buildings were provided
    if bldg_bc_geom is not None:
        label = "building"
        print(f"  - Processing automatic BC '{label}' from buildings...")
        special_bcs.append((label, bldg_bc_geom))
        
        # Subtract building perimeter from the generic 'Wall' BC
        # so it is labeled as 'building' instead
        print(f"    -> Subtracting '{label}' from Wall boundary...")
        eraser = bldg_bc_geom.buffer(0.01) 
        wall_geom = wall_geom.difference(eraser)

    # 3. Discretize and Save 'Wall' (The Remainder)
    print(f"  - Finalizing 'wall' boundary segments...")
    wall_segs = geom_to_segments(wall_geom)
    w_starts, w_vecs, w_cdf = build_boundary_arrays(wall_segs)
    
    if w_starts is not None:
        save_dict['bc_wall_starts'] = w_starts
        save_dict['bc_wall_vectors'] = w_vecs
        save_dict['bc_wall_cdf'] = w_cdf
        print(f"    -> Wall segments: {len(w_starts)}")
    else:
        print("    -> Warning: No wall segments remaining!")

    # 4. Discretize and Save Specialized BCs
    for label, geom in special_bcs:
        segs = geom_to_segments(geom)
        starts, vecs, cdf = build_boundary_arrays(segs)
        if starts is not None:
            save_dict[f'bc_{label}_starts'] = starts
            save_dict[f'bc_{label}_vectors'] = vecs
            save_dict[f'bc_{label}_cdf'] = cdf
            print(f"    -> {label} segments: {len(starts)}")

    # 5. Save to Disk
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "domain_artifacts.npz")
    np.savez(out_path, **save_dict)
    print(f"Artifacts saved to {out_path}")

if __name__ == "__main__":
    main()