"""
Comprehensive test suite for train_gradnorm.py
Tests all training modes: data-free, data-driven, with/without GradNorm
"""
import os
import unittest
import shutil
import tempfile
import yaml
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Force JAX to use CPU for tests
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after setting up path
from src.train_gradnorm import main as train_main


class TestTrainGradNorm(unittest.TestCase):
    """Test suite for train_gradnorm.py"""

    @classmethod
    def setUpClass(cls):
        """Create temporary directory and mock data once for all tests."""
        cls.test_dir = tempfile.mkdtemp(prefix="test_train_gradnorm_")
        
        # Create mock datasets
        cls.mock_train_data = np.random.rand(20, 6).astype(np.float32)
        cls.mock_val_data = np.random.rand(15, 6).astype(np.float32)
        cls.mock_plot_data = np.random.rand(10, 6).astype(np.float32)
        cls.mock_plot_data[:, 0] = 1800.0  # Set time column
        
        print(f"\n{'='*60}")
        print(f"Test directory: {cls.test_dir}")
        print(f"{'='*60}\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        print(f"\n{'='*60}")
        print("Test cleanup completed")
        print(f"{'='*60}\n")

    def setUp(self):
        """Set up before each test."""
        self.config_path = os.path.join(self.test_dir, f"config_{self._testMethodName}.yaml")

    def create_minimal_config(self, **overrides):
        """
        Create a minimal but complete config for testing.
        
        Args:
            **overrides: Config values to override
        """
        config = {
            'scenario': 'test_scenario',
            'data_free': overrides.get('data_free', False),
            
            'training': {
                'seed': 42,
                'epochs': 5,  # Small number for fast tests
                'learning_rate': 0.001,
                'batch_size': 10,
            },
            
            'model': {
                'name': 'DGMNetwork',
                'output_dim': 3,
                'width': 32,  # Small network for tests
                'depth': 2,
            },
            
            'domain': {
                'lx': 1200.0,
                'ly': 100.0,
                't_final': 3600.0,
            },
            
            'grid': {
                'nx': 10,
                'ny': 10,
                'nt': 10,
            },
            
            'ic_bc_grid': {
                'nx_ic': 10,
                'ny_ic': 10,
                'ny_bc_left': 10,
                'nt_bc_left': 10,
                'ny_bc_right': 10,
                'nt_bc_right': 10,
                'nx_bc_bottom': 10,
                'nt_bc_other': 10,
                'nx_bc_top': 10,
            },
            
            'building': {
                'x_min': 325.0,
                'x_max': 375.0,
                'y_min': 25.0,
                'y_max': 75.0,
                'nx': 5,
                'ny': 5,
                'nt': 5,
            },
            
            'physics': {
                'u_const': 0.29,
                'n_manning': 0.03,
                'inflow': None,
                'g': 9.81,
            },
            
            'loss_weights': {
                'pde_weight': 1.0,
                'ic_weight': 1.0,
                'bc_weight': 1.0,
                'building_bc_weight': 1.0,
                'data_weight': overrides.get('data_weight', 0.0),
            },
            
            'gradnorm': {
                'enable': overrides.get('enable_gradnorm', False),
                'learning_rate': 0.01,
                'alpha': 1.5,
                'update_freq': 2,  # Update frequently in tests
            },
            
            'plotting': {
                'nx_val': 20,
                't_const_val': 1800.0,
                'y_const_plot': 0,
                'plot_resolution': 50,
            },
            
            'device': {
                'dtype': 'float32',
                'early_stop_min_epochs': 100,
                'early_stop_patience': 50,
            },
            
            'numerics': {
                'eps': 1e-6,
            },
            
            'aim': {
                'enable': False,
            },
        }
        
        # Apply any additional overrides
        for key, value in overrides.items():
            if key not in ['data_free', 'data_weight', 'enable_gradnorm']:
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
        
        # Write config file
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        return self.config_path

    def mock_load_data(self, filepath, *args, **kwargs):
        """Mock numpy/jax load to return in-memory test data."""
        filepath_str = str(filepath)
        
        if "training_dataset" in filepath_str:
            return self.mock_train_data
        elif "validation_plotting" in filepath_str:
            return self.mock_plot_data
        elif "validation" in filepath_str:
            return self.mock_val_data
        else:
            # Return empty array for unknown files
            return np.array([])

    def run_training_with_mocks(self, config_path):
        """
        Run training with all necessary mocks in place.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Best NSE value from training
        """
        # Create all mocks
        mock_makedirs = MagicMock()
        mock_save_model = MagicMock()
        mock_confirmation = MagicMock(return_value=True)
        mock_plt = MagicMock()
        
        # Patch everything
        patches = [
            patch('numpy.load', side_effect=self.mock_load_data),
            patch('jax.numpy.load', side_effect=self.mock_load_data),
            patch('src.train_gradnorm.os.makedirs', mock_makedirs),
            patch('src.train_gradnorm.save_model', mock_save_model),
            patch('src.train_gradnorm.ask_for_confirmation', mock_confirmation),
            patch('matplotlib.pyplot', mock_plt),
        ]
        
        for p in patches:
            p.start()
        
        try:
            best_nse = train_main(config_path)
            return best_nse
        finally:
            # Always stop patches
            for p in patches:
                p.stop()

    # ==================== Test Cases ====================

    def test_data_free_static_weights(self):
        """Test data-free mode with static loss weights."""
        print(f"\n{'─'*60}")
        print(f"TEST: {self._testMethodName}")
        print(f"{'─'*60}")
        
        config_path = self.create_minimal_config(
            data_free=True,
            data_weight=0.0,
            enable_gradnorm=False
        )
        
        best_nse = self.run_training_with_mocks(config_path)
        
        self.assertIsInstance(best_nse, float)
        self.assertGreater(best_nse, -np.inf)
        print(f"✓ Test passed | Best NSE: {best_nse:.6f}")

    def test_data_driven_static_weights(self):
        """Test data-driven mode with static loss weights."""
        print(f"\n{'─'*60}")
        print(f"TEST: {self._testMethodName}")
        print(f"{'─'*60}")
        
        config_path = self.create_minimal_config(
            data_free=False,
            data_weight=10.0,
            enable_gradnorm=False
        )
        
        best_nse = self.run_training_with_mocks(config_path)
        
        self.assertIsInstance(best_nse, float)
        self.assertGreater(best_nse, -np.inf)
        print(f"✓ Test passed | Best NSE: {best_nse:.6f}")

    def test_data_free_with_gradnorm(self):
        """Test data-free mode with GradNorm enabled."""
        print(f"\n{'─'*60}")
        print(f"TEST: {self._testMethodName}")
        print(f"{'─'*60}")
        
        config_path = self.create_minimal_config(
            data_free=True,
            data_weight=0.0,
            enable_gradnorm=True
        )
        
        best_nse = self.run_training_with_mocks(config_path)
        
        self.assertIsInstance(best_nse, float)
        self.assertGreater(best_nse, -np.inf)
        print(f"✓ Test passed | Best NSE: {best_nse:.6f}")

    def test_data_driven_with_gradnorm(self):
        """Test data-driven mode with GradNorm enabled."""
        print(f"\n{'─'*60}")
        print(f"TEST: {self._testMethodName}")
        print(f"{'─'*60}")
        
        config_path = self.create_minimal_config(
            data_free=False,
            data_weight=10.0,
            enable_gradnorm=True
        )
        
        best_nse = self.run_training_with_mocks(config_path)
        
        self.assertIsInstance(best_nse, float)
        self.assertGreater(best_nse, -np.inf)
        print(f"✓ Test passed | Best NSE: {best_nse:.6f}")

    def test_inferred_data_free_mode(self):
        """Test inferred data-free mode (data_weight=0, no flag)."""
        print(f"\n{'─'*60}")
        print(f"TEST: {self._testMethodName}")
        print(f"{'─'*60}")
        
        config_path = self.create_minimal_config(
            data_weight=0.0,
            enable_gradnorm=False
        )
        
        # Remove data_free flag from config to test inference
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config.pop('data_free', None)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        best_nse = self.run_training_with_mocks(config_path)
        
        self.assertIsInstance(best_nse, float)
        self.assertGreater(best_nse, -np.inf)
        print(f"✓ Test passed | Best NSE: {best_nse:.6f}")

    def test_inferred_data_driven_mode(self):
        """Test inferred data-driven mode (data_weight>0, no flag)."""
        print(f"\n{'─'*60}")
        print(f"TEST: {self._testMethodName}")
        print(f"{'─'*60}")
        
        config_path = self.create_minimal_config(
            data_weight=5.0,
            enable_gradnorm=False
        )
        
        # Remove data_free flag from config to test inference
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config.pop('data_free', None)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        best_nse = self.run_training_with_mocks(config_path)
        
        self.assertIsInstance(best_nse, float)
        self.assertGreater(best_nse, -np.inf)
        print(f"✓ Test passed | Best NSE: {best_nse:.6f}")


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestTrainGradNorm))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

