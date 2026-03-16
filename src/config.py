# src/config.py
import os
import warnings
import yaml
import jax.numpy as jnp

# Top-level sections that every config must contain.
_REQUIRED_SECTIONS = ("training", "model", "domain", "physics", "device", "numerics")


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


def _validate_required_keys(config: dict, config_path: str) -> None:
    """Raise ``ValueError`` if any required top-level section is missing."""
    missing = [s for s in _REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(
            f"Config '{config_path}' is missing required sections: "
            f"{', '.join(missing)}"
        )


def _migrate_deprecated_keys(config: dict) -> dict:
    """Emit deprecation warnings and migrate old config keys."""
    # accumulation_factor -> accumulation_size
    rop = config.get("training", {}).get("reduce_on_plateau", {})
    if "accumulation_factor" in rop and "accumulation_size" not in rop:
        warnings.warn(
            "Config key 'training.reduce_on_plateau.accumulation_factor' is "
            "deprecated; rename it to 'accumulation_size'.",
            DeprecationWarning,
            stacklevel=3,
        )
        rop["accumulation_size"] = rop.pop("accumulation_factor")
    return config


def load_config(config_path: str):
    """Load and process the configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = _convert_str_floats(config)

    _validate_required_keys(config, config_path)

    config = _migrate_deprecated_keys(config)

    # Override any stale CONFIG_PATH baked into the YAML with the actual path.
    config['CONFIG_PATH'] = config_path

    global DTYPE, EPS
    DTYPE = getattr(jnp, config["device"]["dtype"])
    EPS = config["numerics"]["eps"]

    return config

# These will be set by load_config
DTYPE = None
EPS = None
