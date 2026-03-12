"""Experiment setup: config loading, model initialisation, directory creation, and shared config helpers."""
import os
import importlib

import jax.numpy as jnp

from jax import random
from flax.core import FrozenDict

from src.config import load_config
from src.data import get_sample_count, resolve_scenario_asset_path
from src.models import init_model
from src.utils import generate_trial_name


DEFAULT_SAMPLING_COUNTS = {
    "n_points_pde": 1000,
    "n_points_ic": 100,
    "n_points_bc_domain": 100,
    "n_points_bc_inflow": 100,
    "n_points_bc_building": 100,
}


def init_model_from_config(cfg):
    """Instantiate a model from a frozen config and return train/val keys."""
    models_module = importlib.import_module("src.models")
    model_class = getattr(models_module, cfg["model"]["name"])

    key = random.PRNGKey(cfg["training"]["seed"])
    model_key, train_key, val_key = random.split(key, 3)
    model, params = init_model(model_class, model_key, cfg)
    return model, params, train_key, val_key


def create_output_dirs(cfg, experiment_name: str):
    """Create experiment-scoped results and model directories."""
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    results_dir = os.path.join("results", experiment_name, trial_name)
    model_dir = os.path.join("models", experiment_name, trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return trial_name, results_dir, model_dir


def get_experiment_name(cfg, default_experiment_name: str | None = None) -> str:
    """Resolve the logical experiment name from config, falling back to the caller's default."""
    experiment_cfg = cfg.get("experiment", {})
    return experiment_cfg.get("name") or default_experiment_name or cfg.get("scenario") or "experiment"


def resolve_experiment_paths(cfg, default_experiment_name: str | None = None, *, require_scenario: bool = False):
    """Resolve experiment name, scenario name, and data directory from config."""
    experiment_name = get_experiment_name(cfg, default_experiment_name)
    scenario_name = cfg.get("scenario", experiment_name)
    if require_scenario and not cfg.get("scenario"):
        raise ValueError("'scenario' key must be set in config for this experiment.")

    data_cfg = cfg.get("data", {})
    base_data_path = data_cfg.get("base_path")
    if not base_data_path:
        data_root = data_cfg.get("root_dir", "data")
        base_data_path = os.path.join(data_root, scenario_name)

    return {
        "experiment_name": experiment_name,
        "scenario_name": scenario_name,
        "base_data_path": base_data_path,
    }


def get_data_filename(cfg, key: str, default: str) -> str:
    """Resolve a training/validation data filename from config."""
    return cfg.get("data", {}).get(key, default)


def resolve_configured_asset_path(
    cfg,
    base_data_path: str,
    scenario_name: str,
    asset_key: str,
    *,
    required: bool = True,
):
    """Resolve a scenario asset, allowing explicit filename overrides in config."""
    configured_name = cfg.get("data", {}).get("assets", {}).get(asset_key)
    if configured_name:
        asset_path = os.path.join(base_data_path, configured_name)
        if required and not os.path.exists(asset_path):
            raise FileNotFoundError(
                f"Missing '{asset_key}' for scenario '{scenario_name}' at configured path '{asset_path}'"
            )
        return asset_path

    return resolve_scenario_asset_path(
        base_data_path,
        scenario_name,
        asset_key,
        required=required,
    )


def get_sampling_count_from_config(cfg, name: str) -> int:
    """Resolve a sampling count using config-provided defaults when needed."""
    sampling_cfg = cfg.get("sampling", {})
    sampling_defaults = cfg.get("sampling_defaults", {})
    default_value = sampling_defaults.get(name, DEFAULT_SAMPLING_COUNTS.get(name))
    if name not in sampling_cfg and default_value is None:
        raise KeyError(f"Missing sampling count '{name}' and no default is configured.")
    return get_sample_count(sampling_cfg, name, default_value)


def get_boundary_segment_count(cfg, total_points: int, *, default_segments: int = 4) -> int:
    """Compute per-segment boundary counts from config."""
    if total_points <= 0:
        return 0
    sampling_cfg = cfg.get("sampling", {})
    segment_count = sampling_cfg.get("boundary_segment_count", default_segments)
    min_points = sampling_cfg.get("min_points_per_boundary_segment", 5)
    return max(min_points, total_points // segment_count)


def calculate_num_batches(batch_size: int, sample_sizes, data_points_full=None, *, data_free: bool = True) -> int:
    """Compute the effective batch count across all active sample sources."""
    batch_counts = [size // batch_size for size in sample_sizes if size and size > 0]
    if not data_free and data_points_full is not None:
        batch_counts.append(data_points_full.shape[0] // batch_size)
    return max(batch_counts) if batch_counts else 0


def apply_irregular_domain_bounds(cfg_dict, domain_sampler):
    """Populate mutable config with extents derived from an irregular domain sampler."""
    all_coords = domain_sampler.tri_coords.reshape(-1, 2)
    min_vals = jnp.min(all_coords, axis=0)
    max_vals = jnp.max(all_coords, axis=0)

    x_min, y_min = float(min_vals[0]), float(min_vals[1])
    x_max, y_max = float(max_vals[0]), float(max_vals[1])

    domain_cfg = cfg_dict.setdefault("domain", {})
    domain_cfg["lx"] = x_max - x_min
    domain_cfg["ly"] = y_max - y_min
    domain_cfg["x_min"] = x_min
    domain_cfg["x_max"] = x_max
    domain_cfg["y_min"] = y_min
    domain_cfg["y_max"] = y_max

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "lx": domain_cfg["lx"],
        "ly": domain_cfg["ly"],
    }


def apply_output_scales(cfg_dict, default_output_scales):
    """Set model output scales from config, defaulting when absent."""
    model_cfg = cfg_dict.setdefault("model", {})
    output_scales = tuple(model_cfg.get("output_scales", default_output_scales))
    model_cfg["output_scales"] = output_scales
    return output_scales


def setup_experiment(config_path: str, experiment_name: str | None = None):
    """Load config, create model, set up output directories.

    Returns
    -------
    dict with keys:
        cfg_dict, cfg, model, params, train_key, val_key,
        trial_name, results_dir, model_dir
    """
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)

    resolved_experiment_name = get_experiment_name(cfg_dict, experiment_name)

    model, params, train_key, val_key = init_model_from_config(cfg)
    trial_name, results_dir, model_dir = create_output_dirs(cfg, resolved_experiment_name)

    return {
        "cfg_dict": cfg_dict,
        "cfg": cfg,
        "experiment_name": resolved_experiment_name,
        "model": model,
        "params": params,
        "train_key": train_key,
        "val_key": val_key,
        "trial_name": trial_name,
        "results_dir": results_dir,
        "model_dir": model_dir,
    }


def extract_loss_weights(cfg):
    """Extract active loss weights from config and return a FrozenDict.

    Returns
    -------
    (static_weights_dict, current_weights_dict)
        static_weights_dict : plain dict of all weight keys (stripped of '_weight' suffix)
        current_weights_dict : FrozenDict of only the active (> 0) keys, suitable for JIT static args
    """
    static_weights_dict = {k.replace('_weight', ''): v for k, v in cfg["loss_weights"].items()}
    active_loss_term_keys = [k for k, v in static_weights_dict.items() if v > 0]
    current_weights_dict = FrozenDict({k: static_weights_dict[k] for k in active_loss_term_keys})
    return static_weights_dict, current_weights_dict


def get_active_loss_weights(static_weights_dict, *, data_free: bool = False, excluded_keys=()):
    """Return the active loss weights after applying runtime exclusions."""
    active_keys = []
    excluded = set(excluded_keys)
    for key, value in static_weights_dict.items():
        if value <= 0 or key in excluded:
            continue
        if key == "data" and data_free:
            continue
        active_keys.append(key)
    return FrozenDict({key: static_weights_dict[key] for key in active_keys})
