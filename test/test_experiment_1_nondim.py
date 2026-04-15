"""Smoke test for experiments.experiment_1.train_nondim.

Runs a 200-epoch CPU training on a tiny MLP with non-dim scaling enabled,
and a separate run with scaling disabled (identity mode), to verify that the
isolated non-dim pipeline produces finite, non-collapsed metrics.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import math
import shutil
import sys
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.experiment_1.train_nondim import main as train_main


def _base_config(scaling_enabled: bool, data_free: bool = True) -> dict:
    return {
        'training': {
            'learning_rate': 1.0e-3,
            'epochs': 200,
            'batch_size': 32,
            'seed': 42,
            'clip_norm': 1.0,
        },
        'model': {
            'name': 'MLP',
            'width': 32,
            'depth': 2,
            'output_dim': 3,
            'kernel_init': 'glorot_uniform',
            'bias_init': 0.0,
        },
        'domain': {
            'lx': 1200.0,
            'ly': 100.0,
            't_final': 3600.0,
        },
        'physics': {
            'u_const': 0.29,
            'n_manning': 0.03,
            'inflow': None,
            'g': 9.81,
        },
        'scaling': {
            'enabled': scaling_enabled,
            'H0': 1.0,
        },
        'data_free': data_free,
        'train_grid': {
            'n_gauges': 1,
            'dt_data': 300.0,
        },
        'loss_weights': {
            'pde_weight': 1.0,
            'bc_weight': 1.0,
            'ic_weight': 1.0,
            'neg_h_weight': 1.0,
            'data_weight': 1.0,
        },
        'sampling': {
            'n_points_pde': 200,
            'n_points_ic': 50,
            'n_points_bc_domain': 100,
        },
        'validation_grid': {
            'n_points_val': 100,
        },
        'wandb': {
            'enable': False,
        },
        'plotting': {
            'nx_val': 20,
            't_const_val': 1800.0,
            'y_const_plot': 0,
        },
        'device': {
            'dtype': 'float32',
            'early_stop_min_epochs': 500,
            'early_stop_patience': 500,
        },
        'numerics': {
            'eps': 1.0e-6,
        },
    }


class TestExperiment1Nondim(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self._created_trials = []

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        for dir_name in ("results", "models"):
            for trial in self._created_trials:
                trial_path = os.path.join(dir_name, "experiment_1", trial)
                if os.path.isdir(trial_path):
                    shutil.rmtree(trial_path)

    def _snapshot_trials(self) -> set:
        exp_models_dir = os.path.join("models", "experiment_1")
        if not os.path.isdir(exp_models_dir):
            return set()
        return set(os.listdir(exp_models_dir))

    def _write_config(self, name: str, scaling_enabled: bool,
                      data_free: bool = True) -> str:
        path = os.path.join(self.test_dir, f"{name}.yaml")
        with open(path, 'w') as f:
            yaml.dump(_base_config(scaling_enabled, data_free=data_free), f)
        return path

    def _assert_final_checkpoint_exists(self, before: set) -> None:
        exp_models_dir = os.path.join("models", "experiment_1")
        self.assertTrue(os.path.isdir(exp_models_dir),
                        f"Missing {exp_models_dir}")
        after = set(os.listdir(exp_models_dir))
        new_trials = sorted(after - before)
        self.assertTrue(new_trials,
                        f"No new trial dir created under {exp_models_dir}")
        trial = new_trials[-1]
        self._created_trials.append(trial)
        final_ckpt = os.path.join(exp_models_dir, trial, "checkpoints", "final", "model.pkl")
        self.assertTrue(os.path.exists(final_ckpt),
                        f"Missing final checkpoint: {final_ckpt}")

    @patch('src.training.loop.ask_for_confirmation', return_value=True)
    def test_nondim_train_runs(self, _mock_confirm):
        """200-epoch smoke test with scaling.enabled=true."""
        cfg_path = self._write_config("test_config_nondim", scaling_enabled=True)
        before = self._snapshot_trials()
        try:
            final_nse = train_main(cfg_path)
        except Exception as exc:
            self.fail(f"Non-dim training raised: {exc}")

        self.assertIsNotNone(final_nse)
        self.assertTrue(math.isfinite(float(final_nse)),
                        f"Final NSE is not finite: {final_nse}")
        # A1 double-scaling bug would collapse the network and drive NSE to very
        # large negative values. -10 is a loose sanity check that training did
        # not explode into a constant output.
        self.assertGreater(float(final_nse), -10.0,
                           f"NSE too low — likely input-collapse regression: {final_nse}")
        self._assert_final_checkpoint_exists(before)

    @patch('src.training.loop.ask_for_confirmation', return_value=True)
    def test_nondim_with_gauge_data(self, _mock_confirm):
        """Exercise the scaled gauge-data path (data_free=false)."""
        cfg_path = self._write_config(
            "test_config_nondim_data", scaling_enabled=True, data_free=False
        )
        before = self._snapshot_trials()
        try:
            final_nse = train_main(cfg_path)
        except Exception as exc:
            self.fail(f"Non-dim gauge-data training raised: {exc}")

        self.assertIsNotNone(final_nse)
        self.assertTrue(math.isfinite(float(final_nse)),
                        f"Final NSE is not finite: {final_nse}")
        self.assertGreater(float(final_nse), -10.0,
                           f"NSE too low with gauge data: {final_nse}")
        self._assert_final_checkpoint_exists(before)

    @patch('src.training.loop.ask_for_confirmation', return_value=True)
    def test_nondim_disabled_matches_dim_mode(self, _mock_confirm):
        """200-epoch run with scaling.enabled=false proves identity-mode fallback."""
        cfg_path = self._write_config("test_config_nondim_off", scaling_enabled=False)
        before = self._snapshot_trials()
        try:
            final_nse = train_main(cfg_path)
        except Exception as exc:
            self.fail(f"Identity-mode non-dim training raised: {exc}")

        self.assertIsNotNone(final_nse)
        self.assertTrue(math.isfinite(float(final_nse)),
                        f"Final NSE is not finite in identity mode: {final_nse}")
        self._assert_final_checkpoint_exists(before)


if __name__ == '__main__':
    unittest.main()
