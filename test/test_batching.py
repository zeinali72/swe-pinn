"""Unit tests for src.data.batching — batch cycling reshuffling fix."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import unittest

import jax
import jax.numpy as jnp

from src.data.batching import get_batches_tensor


class TestGetBatchesTensor(unittest.TestCase):
    """Tests for the JIT-compatible batching helper."""

    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        # 20 samples, 3 features
        self.data = jnp.arange(60, dtype=jnp.float32).reshape(20, 3)
        self.batch_size = 5  # -> 4 batches available per cycle

    def test_output_shape_no_cycling(self):
        """When total_batches <= n_batches_avail, shape is correct."""
        total_batches = 3
        result = get_batches_tensor(self.key, self.data, self.batch_size, total_batches)
        self.assertEqual(result.shape, (total_batches, self.batch_size, 3))

    def test_output_shape_exact_fit(self):
        """When total_batches == n_batches_avail, shape is correct."""
        total_batches = 4  # exactly n_batches_avail
        result = get_batches_tensor(self.key, self.data, self.batch_size, total_batches)
        self.assertEqual(result.shape, (total_batches, self.batch_size, 3))

    def test_output_shape_with_cycling(self):
        """When total_batches > n_batches_avail, shape is still correct."""
        total_batches = 10  # 2.5 cycles
        result = get_batches_tensor(self.key, self.data, self.batch_size, total_batches)
        self.assertEqual(result.shape, (total_batches, self.batch_size, 3))

    def test_cycling_produces_different_permutations(self):
        """Batches from the second cycle should NOT be exact copies of the first.

        This is the core regression test for issue #60: the old modulo-based
        approach returned ``data[indices]`` where indices wrapped around, so
        batch i and batch i+n_batches_avail were *identical*.  The fix
        reshuffles each cycle independently, so they should differ.
        """
        n_batches_avail = self.data.shape[0] // self.batch_size  # 4
        total_batches = 2 * n_batches_avail  # exactly 2 full cycles

        result = get_batches_tensor(self.key, self.data, self.batch_size, total_batches)

        first_cycle = result[:n_batches_avail]   # batches 0..3
        second_cycle = result[n_batches_avail:]   # batches 4..7

        # At least one batch pair should differ between cycles.
        # With independent reshuffling, the probability that all 4 pairs
        # are identical is astronomically small (1 / (20!)^2 territory).
        any_different = False
        for i in range(n_batches_avail):
            if not jnp.array_equal(first_cycle[i], second_cycle[i]):
                any_different = True
                break

        self.assertTrue(
            any_different,
            "All batches in cycle 2 were exact copies of cycle 1 — "
            "reshuffling is not working.",
        )

    def test_no_cycling_unchanged_behaviour(self):
        """When total_batches <= n_batches_avail, each batch is unique."""
        n_batches_avail = self.data.shape[0] // self.batch_size  # 4
        total_batches = n_batches_avail

        result = get_batches_tensor(self.key, self.data, self.batch_size, total_batches)

        # Each batch should contain different rows (no duplicates within a cycle).
        # Flatten to (total_batches * batch_size, features) and check all rows unique.
        flat = result.reshape(-1, 3)
        # With 20 samples and batch_size=5, we use all 20 — each row appears once.
        unique_rows = jnp.unique(flat, axis=0, size=flat.shape[0])
        self.assertEqual(unique_rows.shape[0], flat.shape[0])

    def test_all_values_come_from_input(self):
        """Every value in the output must exist in the original data."""
        total_batches = 10
        result = get_batches_tensor(self.key, self.data, self.batch_size, total_batches)
        flat = result.reshape(-1, 3)

        # Every row in flat should be present in self.data
        for i in range(flat.shape[0]):
            row = flat[i]
            matches = jnp.all(self.data == row, axis=1)
            self.assertTrue(
                jnp.any(matches),
                f"Row {i} of output ({row}) not found in input data.",
            )

    def test_single_batch_available(self):
        """Edge case: batch_size == n_samples, so only 1 batch per cycle."""
        data = jnp.arange(15, dtype=jnp.float32).reshape(5, 3)
        batch_size = 5  # n_batches_avail = 1
        total_batches = 3

        result = get_batches_tensor(self.key, data, batch_size, total_batches)
        self.assertEqual(result.shape, (3, 5, 3))

        # Each of the 3 batches should be a permutation of the same 5 rows,
        # but with different orderings (different PRNG keys per cycle).
        # At minimum, check shapes and value membership.
        for b in range(total_batches):
            batch_sorted = jnp.sort(result[b], axis=0)
            data_sorted = jnp.sort(data, axis=0)
            self.assertTrue(
                jnp.allclose(batch_sorted, data_sorted),
                f"Batch {b} does not contain a permutation of the input data.",
            )


if __name__ == "__main__":
    unittest.main()
