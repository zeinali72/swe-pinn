# src/config.py
import os
import yaml
import jax.numpy as jnp

def _convert_str_floats(obj):
    """Recursively convert string representations of floats to actual floats."""
    if isinstance(obj, dict):
        return {k: _convert_str_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_str_floats(i) for i in obj]
    elif isinstance(obj, str):
        try:
            if any(c in obj for c in '.eE'):
                return float(obj)
        except ValueError:
            pass
    return obj

# Path is relative to the project root
CONFIG_PATH = "experiments/config_1.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Convert scientific notation strings to floats and add the config path
config = _convert_str_floats(config)
config['CONFIG_PATH'] = CONFIG_PATH # Add the path to the config dict

# Common numerical types derived from config for easy import
DTYPE = getattr(jnp, config["device"]["dtype"])
EPS = config["numerics"]["eps"]