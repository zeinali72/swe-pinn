# src/config.py
import os
import yaml
import jax
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
    dtype_str = config["device"]["dtype"]
    if dtype_str == "float64":
        jax.config.update("jax_enable_x64", True)
    DTYPE = getattr(jnp, dtype_str)
    EPS = config["numerics"]["eps"]

    return config

# These will be set by load_config
DTYPE = None
EPS = None


def get_dtype():
    """Return the current DTYPE value.

    Unlike ``from src.config import DTYPE``, which captures the value at
    import time, this function always returns the *current* module-level
    value so that callers see updates made by subsequent ``load_config``
    calls (e.g. during HPO trials).
    """
    return DTYPE


def get_eps():
    """Return the current EPS value.

    See ``get_dtype`` for rationale.
    """
    return EPS
