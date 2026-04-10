"""Tests for src.data.loader — LHS sampling and config-driven data resolution."""
import os
import shutil
import tempfile
import unittest

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax.numpy as jnp

from src.config import load_config
from src.data.loader import _lhs_indices, sample_training_data, resolve_training_data


class TestLhsIndices(unittest.TestCase):
    """Unit tests for the stratified index selection helper."""

    def test_returns_correct_count(self):
        indices = _lhs_indices(10000, 100, seed=0)
        self.assertEqual(len(indices), 100)

    def test_indices_within_bounds(self):
        n_total = 5000
        indices = _lhs_indices(n_total, 200, seed=42)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < n_total))

    def test_indices_are_sorted_approximately(self):
        """Stratified sampling should produce roughly monotonic indices."""
        indices = _lhs_indices(100000, 500, seed=7)
        # At least 90% of consecutive pairs should be increasing
        increasing = np.sum(np.diff(indices) > 0)
        self.assertGreater(increasing / (len(indices) - 1), 0.9)

    def test_no_duplicates_for_large_pool(self):
        indices = _lhs_indices(100000, 1000, seed=99)
        self.assertEqual(len(np.unique(indices)), len(indices))

    def test_deterministic_with_same_seed(self):
        a = _lhs_indices(10000, 50, seed=123)
        b = _lhs_indices(10000, 50, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_gives_different_result(self):
        a = _lhs_indices(10000, 50, seed=1)
        b = _lhs_indices(10000, 50, seed=2)
        self.assertFalse(np.array_equal(a, b))

    def test_edge_case_n_samples_equals_n_total(self):
        """When requesting as many samples as available, all indices should be covered."""
        indices = _lhs_indices(10, 10, seed=0)
        self.assertEqual(len(indices), 10)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 10))

    def test_single_sample(self):
        indices = _lhs_indices(1000, 1, seed=0)
        self.assertEqual(len(indices), 1)
        self.assertTrue(0 <= indices[0] < 1000)


class TestSampleTrainingData(unittest.TestCase):
    """Tests for the full load-sample-free pipeline."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_loader_")
        # Create a mock full-domain .npy with 1000 rows, 6 columns [t, x, y, h, u, v]
        rng = np.random.default_rng(42)
        self.n_rows = 1000
        self.data = np.zeros((self.n_rows, 6), dtype=np.float32)
        self.data[:, 0] = np.linspace(0, 3600, self.n_rows)  # t
        self.data[:, 1] = rng.uniform(0, 100, self.n_rows)   # x
        self.data[:, 2] = rng.uniform(0, 50, self.n_rows)    # y
        self.data[:, 3] = rng.uniform(0, 2, self.n_rows)     # h
        self.data[:, 4] = rng.uniform(-0.5, 0.5, self.n_rows)  # u
        self.data[:, 5] = rng.uniform(-0.5, 0.5, self.n_rows)  # v
        self.source_path = os.path.join(self.test_dir, "val_full_domain.npy")
        np.save(self.source_path, self.data)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_returns_correct_shape(self):
        result = sample_training_data(self.source_path, 50, seed=0, verbose=False)
        self.assertEqual(result.shape, (50, 6))

    def test_returns_jax_array(self):
        result = sample_training_data(self.source_path, 10, seed=0, verbose=False)
        self.assertIsInstance(result, jnp.ndarray)

    def test_returns_correct_dtype(self):
        result = sample_training_data(self.source_path, 10, seed=0, verbose=False)
        self.assertEqual(result.dtype, jnp.float32)

    def test_values_come_from_source(self):
        """Sampled values should all exist in the original data."""
        result = sample_training_data(self.source_path, 20, seed=0, verbose=False)
        result_np = np.array(result)
        for row in result_np:
            # Check that this t value exists in the source
            matches = np.isclose(self.data[:, 0], row[0], atol=1e-5)
            self.assertTrue(np.any(matches), f"t={row[0]} not found in source")

    def test_max_time_filter(self):
        """With max_time=1800, no sampled point should have t > 1800."""
        result = sample_training_data(
            self.source_path, 50, seed=0, max_time=1800.0, verbose=False
        )
        self.assertTrue(np.all(np.array(result[:, 0]) <= 1800.0))

    def test_request_more_than_available(self):
        """Requesting more samples than rows should return all rows."""
        result = sample_training_data(
            self.source_path, self.n_rows + 500, seed=0, verbose=False
        )
        self.assertEqual(result.shape[0], self.n_rows)

    def test_deterministic(self):
        a = sample_training_data(self.source_path, 30, seed=7, verbose=False)
        b = sample_training_data(self.source_path, 30, seed=7, verbose=False)
        np.testing.assert_array_equal(np.array(a), np.array(b))


class TestResolveTrainingData(unittest.TestCase):
    """Tests for the config-driven resolver with source file and fallback paths."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_resolve_")
        # Create source file
        rng = np.random.default_rng(0)
        self.source_data = rng.random((500, 6)).astype(np.float32)
        np.save(os.path.join(self.test_dir, "val_full_domain.npy"), self.source_data)
        # Create fallback file (smaller)
        self.fallback_data = rng.random((100, 6)).astype(np.float32)
        np.save(os.path.join(self.test_dir, "train_lhs_points.npy"), self.fallback_data)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_lhs_path_when_source_exists(self):
        cfg = {
            "data": {
                "source_file": "val_full_domain.npy",
                "n_train_samples": 50,
                "training_file": "train_lhs_points.npy",
            },
            "training": {"seed": 42},
        }
        result, has_data, data_free = resolve_training_data(
            cfg, self.test_dir, True, {"data": 1.0}, verbose=False
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (50, 6))
        self.assertTrue(has_data)
        self.assertFalse(data_free)

    def test_fallback_when_source_missing(self):
        cfg = {
            "data": {
                "source_file": "nonexistent.npy",
                "n_train_samples": 50,
                "training_file": "train_lhs_points.npy",
            },
            "training": {"seed": 42},
        }
        result, has_data, data_free = resolve_training_data(
            cfg, self.test_dir, True, {"data": 1.0}, verbose=False
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 100)  # fallback file has 100 rows
        self.assertTrue(has_data)
        self.assertFalse(data_free)

    def test_fallback_when_no_source_configured(self):
        cfg = {
            "data": {
                "training_file": "train_lhs_points.npy",
            },
        }
        result, has_data, data_free = resolve_training_data(
            cfg, self.test_dir, True, {"data": 1.0}, verbose=False
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 100)
        self.assertTrue(has_data)

    def test_returns_none_when_data_free(self):
        cfg = {"data": {}}
        result, has_data, data_free = resolve_training_data(
            cfg, self.test_dir, False, {}, verbose=False
        )
        self.assertIsNone(result)
        self.assertFalse(has_data)
        self.assertTrue(data_free)

    def test_disables_when_no_files_found(self):
        empty_dir = tempfile.mkdtemp(prefix="test_empty_")
        try:
            cfg = {"data": {"training_file": "missing.npy"}}
            result, has_data, data_free = resolve_training_data(
                cfg, empty_dir, True, {"data": 1.0}, verbose=False
            )
            self.assertIsNone(result)
            self.assertFalse(has_data)
            self.assertTrue(data_free)
        finally:
            shutil.rmtree(empty_dir)

    def test_max_time_passed_through(self):
        cfg = {
            "data": {
                "source_file": "val_full_domain.npy",
                "n_train_samples": 50,
                "train_max_time": 0.5,  # filter to first half of data
            },
            "training": {"seed": 42},
        }
        result, _, _ = resolve_training_data(
            cfg, self.test_dir, True, {"data": 1.0}, verbose=False
        )
        self.assertIsNotNone(result)
        self.assertTrue(np.all(np.array(result[:, 0]) <= 0.5))


if __name__ == "__main__":
    unittest.main()
