import os
import shutil
import tempfile
import unittest

from src.data.paths import canonical_scenario_asset_path, resolve_scenario_asset_path


class TestScenarioAssetPaths(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_data_paths_")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def touch(self, relative_path: str) -> str:
        full_path = os.path.join(self.test_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as handle:
            handle.write("test")
        return full_path

    def test_prefers_canonical_asset_name_when_present(self):
        base_data_path = os.path.join(self.test_dir, "experiment_8")
        canonical_path = self.touch("experiment_8/domain.npz")
        self.touch("experiment_8/domain_artifacts.npz")

        resolved_path = resolve_scenario_asset_path(base_data_path, "experiment_8", "domain_artifacts")

        self.assertEqual(resolved_path, canonical_path)

    def test_falls_back_to_legacy_asset_name(self):
        base_data_path = os.path.join(self.test_dir, "experiment_8")
        legacy_path = self.touch("experiment_8/upstream_bc_line_allign.shp")

        resolved_path = resolve_scenario_asset_path(base_data_path, "experiment_8", "inflow_boundary_shapefile")

        self.assertEqual(resolved_path, legacy_path)

    def test_optional_resolution_returns_canonical_target_path(self):
        base_data_path = os.path.join(self.test_dir, "experiment_4")

        resolved_path = resolve_scenario_asset_path(
            base_data_path,
            "experiment_4",
            "output_reference",
            required=False,
        )

        self.assertEqual(
            resolved_path,
            canonical_scenario_asset_path(base_data_path, "experiment_4", "output_reference"),
        )

    def test_required_resolution_raises_with_checked_paths(self):
        base_data_path = os.path.join(self.test_dir, "experiment_3")

        with self.assertRaises(FileNotFoundError) as exc_info:
            resolve_scenario_asset_path(base_data_path, "experiment_3", "dem")

        self.assertIn("experiment_3_dem.asc", str(exc_info.exception))
        self.assertIn("experiment_4_DEM.asc", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
