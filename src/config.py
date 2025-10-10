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

def load_config(config_path: str):
    """Load and process the configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = _convert_str_floats(config)
    config['CONFIG_PATH'] = config_path

    global DTYPE, EPS
    DTYPE = getattr(jnp, config["device"]["dtype"])
    EPS = config["numerics"]["eps"]
    
    return config

# These will be set by load_config
DTYPE = None
EPS = None