"""Scenario-specific data asset resolution helpers."""
import os
from typing import Dict, Tuple


SCENARIO_ASSET_ALIASES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "experiment_3": {
        "dem": ("experiment_3_dem.asc", "experiment_4_DEM.asc", "test1DEM.asc"),
        "boundary_condition": ("experiment_3_bc.csv", "experiment_4_BC.csv", "Test1BC.csv"),
    },
    "experiment_4": {
        "dem": ("experiment_4_dem.asc", "Test2DEM.asc"),
        "boundary_condition": ("experiment_4_bc.csv", "experiment_4_BC.csv", "Test2_BC.csv"),
        "boundary_condition_interpolated": ("experiment_4_bc_interpolated.csv", "experiment_4_BC_interpolated.csv"),
        "output_reference": ("experiment_4_output.csv", "Test2output.csv"),
        "domain_boundary_shapefile": ("domain_boundary.shp", "Test2ActiveArea_region.shp"),
        "inflow_boundary_shapefile": ("inflow_boundary_line.shp", "Test2BC_polyline.shp"),
    },
    "experiment_5": {
        "dem": ("experiment_5_dem.asc", "test3DEM.asc"),
        "boundary_condition": ("experiment_5_bc.csv", "Test3BC.csv"),
        "boundary_condition_interpolated": ("experiment_5_bc_interpolated.csv", "Test3BC_Interpolated.csv"),
        "output_reference": ("experiment_5_output.csv", "Test3output.csv"),
        "domain_boundary_shapefile": ("domain_boundary.shp", "Domain_Boundary.shp"),
        "inflow_boundary_shapefile": ("inflow_boundary_line.shp", "Inflow_BC.shp"),
    },
    "experiment_6": {
        "dem": ("experiment_6_dem.asc", "test4_dem.asc"),
        "boundary_condition": ("experiment_6_bc.csv", "Test4BC.csv"),
        "output_reference": ("experiment_6_output.csv", "Test4output.csv"),
        "domain_boundary_shapefile": ("domain_boundary.shp",),
        "inflow_boundary_shapefile": ("inflow_boundary_line.shp",),
        "mesh_geometry": ("mesh_cells.shp", "domain_boundary.shp"),
    },
    "experiment_7": {
        "domain_artifacts": ("domain.npz", "domain_artifacts.npz"),
        "dem": ("experiment_7_dem.asc", "Test5DEM.asc"),
        "boundary_condition": ("experiment_7_bc.csv", "Test5BC.csv", "Test4BC.csv"),
        "boundary_condition_interpolated": ("experiment_7_bc_interpolated.csv", "Test5_BC_interpolated.csv"),
        "output_reference": ("experiment_7_output.csv", "Test5output.csv"),
        "mesh_geometry": ("mesh_cells.shp", "mesh_zones.shp", "2D Zones.shp"),
        "domain_boundary_shapefile": ("domain_boundary.shp", "Test5ActiveArea_region.shp"),
        "inflow_boundary_shapefile": ("inflow_boundary_line.shp", "Test5BC_polyline.shp"),
        "fixed_boundary_shapefile": ("fixed_boundary.shp", "fixed_bc.shp"),
    },
    "experiment_8": {
        "domain_artifacts": ("domain.npz", "domain_artifacts.npz"),
        "dem": ("experiment_8_dem.asc", "DEM_v2_asc.asc"),
        "boundary_condition": ("experiment_8_bc_interpolated.csv", "Test6_BC_interpolated.csv"),
        "output_reference": ("experiment_8_output.csv", "Test6output.csv"),
        "mesh_geometry": ("mesh_cells.shp", "mesh_zones.shp", "2D Zones.shp"),
        "domain_boundary_shapefile": ("domain_boundary.shp", "boundary.shp"),
        "inflow_boundary_shapefile": (
            "inflow_boundary_line.shp",
            "upstream_bc_line_aligned.shp",
            "upstream_bc_line_allign.shp",
        ),
        "buildings_shapefile": ("building_obstacles.shp", "Buildings_cleaned.shp"),
    },
}


def get_asset_candidates(scenario_name: str, asset_name: str) -> Tuple[str, ...]:
    """Return candidate filenames for a scenario asset, canonical first."""
    scenario_assets = SCENARIO_ASSET_ALIASES.get(scenario_name, {})
    return scenario_assets.get(asset_name, (asset_name,))


def canonical_scenario_asset_path(base_data_path: str, scenario_name: str, asset_name: str) -> str:
    """Return the canonical path for a scenario asset."""
    return os.path.join(base_data_path, get_asset_candidates(scenario_name, asset_name)[0])


def resolve_scenario_asset_path(
    base_data_path: str,
    scenario_name: str,
    asset_name: str,
    *,
    required: bool = True,
) -> str:
    """Resolve the first existing path for a scenario asset."""
    candidates = get_asset_candidates(scenario_name, asset_name)
    checked_paths = []
    for filename in candidates:
        asset_path = os.path.join(base_data_path, filename)
        checked_paths.append(asset_path)
        if os.path.exists(asset_path):
            return asset_path

    if required:
        checked_display = ", ".join(checked_paths)
        raise FileNotFoundError(
            f"Missing '{asset_name}' for scenario '{scenario_name}'. Checked: {checked_display}"
        )

    return os.path.join(base_data_path, candidates[0])
