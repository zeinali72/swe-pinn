# optimisation/objective_function.py
import optuna
from flax.core import FrozenDict, unfreeze
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any
import copy

# Import the training loop function
from optimisation.optimization_train_loop import run_training_trial


def get_hpo_value(trial: optuna.trial.Trial,
                  param_name: str,
                  config_val: Any):
    """
    Parses the config value to decide whether to Fix, Suggest Range, or Suggest Category.
    All search spaces MUST be defined in the config — no hardcoded fallbacks.
    """
    if config_val is None:
        raise ValueError(
            f"Hyperparameter '{param_name}' not found in hpo_hyperparameters config. "
            "All tunable parameters must be explicitly defined."
        )

    # Fixed Value (Scalar)
    if isinstance(config_val, (int, float, str, bool)):
        return config_val

    # Categorical Choice (List)
    if isinstance(config_val, list):
        return trial.suggest_categorical(param_name, config_val)

    # Range (Dictionary with min/max)
    if isinstance(config_val, dict):
        val_type = config_val.get("type", "float")
        low = config_val["min"]
        high = config_val["max"]
        log = config_val.get("log", False)
        step = config_val.get("step", None)

        if val_type == "int" or isinstance(low, int):
            return trial.suggest_int(param_name, int(low), int(high), step=step or 1, log=log)
        else:
            return trial.suggest_float(param_name, float(low), float(high), step=step, log=log)

    raise ValueError(f"Cannot parse hpo_hyperparameters.{param_name}: {config_val!r}")


def objective(trial: optuna.trial.Trial, base_config_dict: Dict) -> float:
    """
    Config-Driven Objective Function.
    Reads 'hpo_hyperparameters' from YAML to define search spaces or fixed values.
    """
    base_cfg = FrozenDict(base_config_dict)
    model_name = base_cfg["model"]["name"]
    has_building = "building" in base_cfg

    hpo_cfg = base_config_dict.get("hpo_hyperparameters", {})
    hpo_settings = base_config_dict.get("hpo_settings", {})
    data_free = hpo_settings.get("data_free", True)
    objective_key = hpo_settings.get("objective_key", "nse_h")

    # Helper
    def suggest(name, section=hpo_cfg):
        return get_hpo_value(trial, name, section.get(name))

    # --- 1. Suggest Hyperparameters ---
    trial_params = {}

    # Training
    trial_params["learning_rate"] = suggest("learning_rate")
    trial_params["batch_size"] = suggest("batch_size")

    # Model
    trial_params["model_width"] = suggest("model_width")
    trial_params["model_depth"] = suggest("model_depth")

    if model_name == "FourierPINN":
        trial_params["ff_dims"] = suggest("ff_dims")
        trial_params["fourier_scale"] = suggest("fourier_scale")

    # Sampling
    samp_cfg = hpo_cfg.get("sampling", {})
    trial_params["sampling"] = {}
    for key in ["n_points_pde", "n_points_ic", "n_points_bc_domain"]:
        if key in samp_cfg:
            trial_params["sampling"][key] = get_hpo_value(trial, key, samp_cfg[key])
    if has_building and "n_points_bc_building" in samp_cfg:
        trial_params["sampling"]["n_points_bc_building"] = get_hpo_value(
            trial, "n_points_bc_building", samp_cfg["n_points_bc_building"])

    # Loss Weights
    weights_cfg = hpo_cfg.get("loss_weights", {})
    trial_params["loss_weights"] = {}
    for w in ["pde_weight", "ic_weight", "bc_weight", "neg_h_weight"]:
        if w in weights_cfg:
            trial_params["loss_weights"][w] = get_hpo_value(trial, w, weights_cfg[w])
    if has_building and "building_bc_weight" in weights_cfg:
        trial_params["loss_weights"]["building_bc_weight"] = get_hpo_value(
            trial, "building_bc_weight", weights_cfg["building_bc_weight"])
    if not data_free and "data_weight" in weights_cfg:
        trial_params["loss_weights"]["data_weight"] = get_hpo_value(
            trial, "data_weight", weights_cfg["data_weight"])
    elif data_free:
        trial_params["loss_weights"]["data_weight"] = 0.0

    # --- 2. Build trial config ---
    trial_config_dict = copy.deepcopy(base_config_dict)

    trial_config_dict["training"]["learning_rate"] = trial_params["learning_rate"]
    trial_config_dict["training"]["batch_size"] = trial_params["batch_size"]

    trial_config_dict["model"]["width"] = trial_params["model_width"]
    trial_config_dict["model"]["depth"] = trial_params["model_depth"]
    if model_name == "FourierPINN":
        trial_config_dict["model"]["ff_dims"] = trial_params["ff_dims"]
        trial_config_dict["model"]["fourier_scale"] = trial_params["fourier_scale"]

    trial_config_dict.setdefault("sampling", {}).update(trial_params["sampling"])
    trial_config_dict.setdefault("loss_weights", {}).update(trial_params["loss_weights"])
    trial_config_dict["data_free"] = data_free

    # Store full config as user attr
    config_to_store = unfreeze(trial_config_dict)
    trial.set_user_attr('full_config', config_to_store)
    trial_cfg_frozen = FrozenDict(trial_config_dict)

    # --- 3. Log (concise) ---
    flat = {}
    for k, v in trial_params.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v

    lines = []
    for k in sorted(flat):
        v = flat[k]
        if isinstance(v, (float, np.floating)):
            lines.append(f"  {k:<25}: {v:.6e}" if abs(v) < 1e-2 or abs(v) > 1e3 else f"  {k:<25}: {v:.6f}")
        else:
            lines.append(f"  {k:<25}: {v}")
    print(f"\n{'='*50}\nTrial {trial.number} | {model_name} | {objective_key}\n" + "\n".join(lines) + f"\n{'='*50}")

    # --- 4. Run ---
    minimize = objective_key in ("rmse_h", "rmse_hu", "rmse_hv",
                                  "mae_h", "mae_hu", "mae_hv",
                                  "rel_l2_h", "rel_l2_hu", "rel_l2_hv")
    fail_value = float("inf") if minimize else -1.0

    try:
        best_metric = run_training_trial(trial, trial_cfg_frozen)

        if hasattr(best_metric, 'item'): best_metric = best_metric.item()
        if jnp.isnan(best_metric):
             print(f"Trial {trial.number}: Invalid {objective_key}. Returning {fail_value}.")
             return fail_value
        return float(best_metric)

    except optuna.exceptions.TrialPruned as e:
         raise e
    except Exception as e:
         print(f"Trial {trial.number} Failed: {e}")
         import traceback; traceback.print_exc()
         return fail_value
