"""Experiment setup: config loading, model initialisation, directory creation, loss weights."""
import os
import importlib

from jax import random
from flax.core import FrozenDict

from src.config import load_config
from src.models import init_model
from src.utils import generate_trial_name


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


def setup_experiment(config_path: str, experiment_name: str):
    """Load config, create model, set up output directories.

    Returns
    -------
    dict with keys:
        cfg_dict, cfg, model, params, train_key, val_key,
        trial_name, results_dir, model_dir
    """
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)

    model, params, train_key, val_key = init_model_from_config(cfg)
    trial_name, results_dir, model_dir = create_output_dirs(cfg, experiment_name)

    return {
        "cfg_dict": cfg_dict,
        "cfg": cfg,
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
