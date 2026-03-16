"""Unit tests for src.checkpointing (CheckpointManager and load_checkpoint)."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import shutil
import tempfile
import unittest

import jax.numpy as jnp

from src.checkpointing.saver import CheckpointManager
from src.checkpointing.loader import load_checkpoint


class TestCheckpointManagerRoundTrip(unittest.TestCase):
    """Test save-then-load round-trip preserves parameter values."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_and_load_round_trip(self):
        """Saved params can be loaded back and match the original values."""
        mgr = CheckpointManager(experiment_dir=self.tmpdir)

        params = {'params': {'dense': {'kernel': jnp.ones((3, 16)),
                                        'bias': jnp.zeros((16,))}}}
        opt_state = {'count': 0}
        val_metrics = {'nse_h': 0.8}
        losses = {'pde': 1.0, 'ic': 0.5}

        mgr.update(epoch=1, params=params, opt_state=opt_state,
                   val_metrics=val_metrics, losses=losses,
                   total_loss=1.5, config={})

        # Load via CheckpointManager internal method
        loaded_params = mgr.get_best_nse_params()
        self.assertIsNotNone(loaded_params)
        self.assertTrue(
            jnp.allclose(loaded_params['params']['dense']['kernel'],
                         params['params']['dense']['kernel']),
            "Loaded kernel does not match saved kernel"
        )
        self.assertTrue(
            jnp.allclose(loaded_params['params']['dense']['bias'],
                         params['params']['dense']['bias']),
            "Loaded bias does not match saved bias"
        )

    def test_load_checkpoint_round_trip(self):
        """load_checkpoint returns matching params and metadata."""
        mgr = CheckpointManager(experiment_dir=self.tmpdir)

        params = {'params': {'layer': {'w': jnp.array([1.0, 2.0, 3.0])}}}
        val_metrics = {'nse_h': 0.9}
        losses = {'pde': 0.1}

        mgr.update(epoch=5, params=params, opt_state={},
                   val_metrics=val_metrics, losses=losses,
                   total_loss=0.1, config={})

        ckpt_path = os.path.join(self.tmpdir, 'checkpoints', 'best_nse')
        loaded_params, metadata = load_checkpoint(ckpt_path)

        self.assertIsNotNone(loaded_params)
        self.assertIsNotNone(metadata)
        self.assertTrue(
            jnp.allclose(loaded_params['params']['layer']['w'],
                         params['params']['layer']['w'])
        )
        self.assertEqual(metadata['epoch'], 5)
        self.assertAlmostEqual(metadata['total_loss'], 0.1, places=6)


class TestCheckpointManagerBestNSE(unittest.TestCase):
    """Test that best_nse tracking works correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_updates_on_improvement(self):
        """best_nse updates when NSE improves."""
        mgr = CheckpointManager(experiment_dir=self.tmpdir)
        dummy_params = {'params': {'w': jnp.array([1.0])}}

        # First update: NSE = 0.5
        saved = mgr.update(epoch=1, params=dummy_params, opt_state={},
                           val_metrics={'nse_h': 0.5}, losses={'pde': 1.0},
                           total_loss=1.0, config={})
        self.assertEqual(len(saved), 2)  # best_nse and best_loss both saved
        self.assertAlmostEqual(mgr.best_nse_h, 0.5, places=6)
        self.assertEqual(mgr.best_nse_epoch, 1)

        # Second update: NSE = 0.8 (improvement)
        saved = mgr.update(epoch=2, params=dummy_params, opt_state={},
                           val_metrics={'nse_h': 0.8}, losses={'pde': 0.9},
                           total_loss=0.9, config={})
        nse_events = [s for s in saved if s[0] == 'best_nse']
        self.assertEqual(len(nse_events), 1)
        self.assertAlmostEqual(mgr.best_nse_h, 0.8, places=6)
        self.assertEqual(mgr.best_nse_epoch, 2)

    def test_ignores_worse_nse(self):
        """best_nse does NOT update when NSE gets worse."""
        mgr = CheckpointManager(experiment_dir=self.tmpdir)
        dummy_params = {'params': {'w': jnp.array([1.0])}}

        mgr.update(epoch=1, params=dummy_params, opt_state={},
                   val_metrics={'nse_h': 0.9}, losses={'pde': 1.0},
                   total_loss=1.0, config={})

        # Worse NSE
        saved = mgr.update(epoch=2, params=dummy_params, opt_state={},
                           val_metrics={'nse_h': 0.3}, losses={'pde': 2.0},
                           total_loss=2.0, config={})
        nse_events = [s for s in saved if s[0] == 'best_nse']
        self.assertEqual(len(nse_events), 0, "Should not update best_nse for worse value")
        self.assertAlmostEqual(mgr.best_nse_h, 0.9, places=6)
        self.assertEqual(mgr.best_nse_epoch, 1)


class TestCheckpointManagerBestLoss(unittest.TestCase):
    """Test that best_loss tracking works correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_updates_on_improvement(self):
        """best_loss updates when total_loss decreases."""
        mgr = CheckpointManager(experiment_dir=self.tmpdir)
        dummy_params = {'params': {'w': jnp.array([1.0])}}

        mgr.update(epoch=1, params=dummy_params, opt_state={},
                   val_metrics={'nse_h': 0.5}, losses={'pde': 2.0},
                   total_loss=2.0, config={})
        self.assertAlmostEqual(mgr.best_loss, 2.0, places=6)

        # Improvement
        saved = mgr.update(epoch=2, params=dummy_params, opt_state={},
                           val_metrics={'nse_h': 0.5}, losses={'pde': 1.0},
                           total_loss=1.0, config={})
        loss_events = [s for s in saved if s[0] == 'best_loss']
        self.assertEqual(len(loss_events), 1)
        self.assertAlmostEqual(mgr.best_loss, 1.0, places=6)
        self.assertEqual(mgr.best_loss_epoch, 2)

    def test_ignores_worse_loss(self):
        """best_loss does NOT update when total_loss increases."""
        mgr = CheckpointManager(experiment_dir=self.tmpdir)
        dummy_params = {'params': {'w': jnp.array([1.0])}}

        mgr.update(epoch=1, params=dummy_params, opt_state={},
                   val_metrics={'nse_h': 0.5}, losses={'pde': 1.0},
                   total_loss=1.0, config={})

        saved = mgr.update(epoch=2, params=dummy_params, opt_state={},
                           val_metrics={'nse_h': 0.5}, losses={'pde': 5.0},
                           total_loss=5.0, config={})
        loss_events = [s for s in saved if s[0] == 'best_loss']
        self.assertEqual(len(loss_events), 0, "Should not update best_loss for worse value")
        self.assertAlmostEqual(mgr.best_loss, 1.0, places=6)
        self.assertEqual(mgr.best_loss_epoch, 1)


class TestLoadCheckpointMissing(unittest.TestCase):
    """Test load_checkpoint behavior for missing paths."""

    def test_missing_path_returns_none(self):
        """load_checkpoint returns (None, None) for a nonexistent directory."""
        params, metadata = load_checkpoint('/tmp/nonexistent_ckpt_path_xyz')
        self.assertIsNone(params)
        self.assertIsNone(metadata)


if __name__ == '__main__':
    unittest.main()
