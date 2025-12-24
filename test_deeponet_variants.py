#!/usr/bin/env python3
"""
Verification script for DeepONet and FourierDeepONet implementations.
Tests model instantiation and output shapes.
"""

import sys
import os
import jax
import jax.numpy as jnp
from jax import random

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import load_config, DTYPE
from src.models import DeepONet, FourierDeepONet, init_deeponet_model

def test_model(model_name, config_path):
    """Test a specific model variant."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    # Load config
    cfg_dict = load_config(config_path)
    
    # Override model name
    cfg_dict["model"]["name"] = model_name
    
    # Get model class
    if model_name == "DeepONet":
        model_class = DeepONet
    elif model_name == "FourierDeepONet":
        model_class = FourierDeepONet
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Initialize model
    key = random.PRNGKey(42)
    try:
        model, params = init_deeponet_model(model_class, key, cfg_dict)
        print(f"✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False
    
    # Test forward pass with different batch sizes
    param_names = tuple(cfg_dict["physics"]["param_bounds"].keys())
    n_params = len(param_names)
    
    test_cases = [
        ("Single sample", 1),
        ("Batch of 16", 16),
        ("Batch of 512", 512),
    ]
    
    for test_name, batch_size in test_cases:
        # Create dummy inputs
        x_branch = jnp.zeros((batch_size, n_params), dtype=DTYPE)
        x_trunk = jnp.zeros((batch_size, 3), dtype=DTYPE)
        
        try:
            output = model.apply({'params': params['params']}, x_branch, x_trunk, train=False)
            expected_shape = (batch_size, cfg_dict["model"]["output_dim"])
            
            if output.shape == expected_shape:
                print(f"✓ {test_name}: Output shape {output.shape} matches expected {expected_shape}")
            else:
                print(f"✗ {test_name}: Output shape {output.shape} does not match expected {expected_shape}")
                return False
        except Exception as e:
            print(f"✗ {test_name}: Forward pass failed: {e}")
            return False
    
    # Count parameters
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params['params']))
    print(f"✓ Total parameters: {total_params:,}")
    
    return True

def main():
    config_path = "configs/analytical_deeponet.yaml"
    
    print("\n" + "="*60)
    print("DeepONet Variants Verification")
    print("="*60)
    
    results = {}
    
    # Test standard DeepONet
    results["DeepONet"] = test_model("DeepONet", config_path)
    
    # Test FourierDeepONet
    results["FourierDeepONet"] = test_model("FourierDeepONet", config_path)
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    for model_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{model_name:20s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
