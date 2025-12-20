# optimisation/objective_function.py
import jax
import optuna
from flax.core import FrozenDict, unfreeze
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any
import copy
import yaml

# Import the training loop function
from optimisation.optimization_train_loop import run_training_trial

def get_hpo_value(trial: optuna.trial.Trial, 
                  param_name: str, 
                  config_val: Any, 
                  default_fn: callable = None):
    """
    Parses the config value to decide whether to Fix, Suggest Range, or Suggest Category.
    """
    # 1. If not in config, use the hardcoded default fallback (Exploration/Sensitivity)
    if config_val is None:
        if default_fn is None:
            raise ValueError(f"Parameter '{param_name}' not found in config and no default provided.")
        return default_fn()

    # 2. Fixed Value (Scalar) -> Exploitation (Fixed)
    if isinstance(config_val, (int, float, str, bool)):
        return config_val

    # 3. Categorical Choice (List) -> Exploration (Categorical)
    if isinstance(config_val, list):
        return trial.suggest_categorical(param_name, config_val)

    # 4. Range (Dictionary with min/max) -> Exploration (Range)
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

    return default_fn()

def objective(trial: optuna.trial.Trial, base_config_dict: Dict) -> float:
    """
    Config-Driven Objective Function.
    Reads 'hpo_hyperparameters' from YAML to define search spaces or fixed values.
    """
    base_cfg = FrozenDict(base_config_dict)
    model_name = base_cfg["model"]["name"]
    has_building = "building" in base_cfg
    
    # Load the HPO configuration section
    hpo_cfg = base_config_dict.get("hpo_hyperparameters", {})
    
    # Helper to simplify calls
    def suggest(name, section=hpo_cfg, default=None):
        val = section.get(name)
        return get_hpo_value(trial, name, val, default)

    # --- 1. Define Hyperparameters ---
    trial_params = {}

    # === Training ===
    trial_params["learning_rate"] = suggest("learning_rate", hpo_cfg, 
        lambda: trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True))
    
    trial_params["batch_size"] = suggest("batch_size", hpo_cfg,
        lambda: trial.suggest_categorical("batch_size", [256, 512, 1024]))

    # LR Boundaries (Calculated based on epochs)
    opt_epochs = base_config_dict.get("training", {}).get("epochs", 2000)
    boundary1 = int(opt_epochs * 0.6)
    boundary2 = int(opt_epochs * 0.8)
    trial_params["lr_boundaries"] = {str(boundary1): 0.1, str(boundary2): 0.1}

    # === Model ===
    trial_params["model_width"] = suggest("model_width", hpo_cfg,
        lambda: trial.suggest_categorical("model_width", [128, 256, 512, 1024]))
    
    trial_params["model_depth"] = suggest("model_depth", hpo_cfg,
        lambda: trial.suggest_int("model_depth", 3, 6))
    
    if model_name == "FourierPINN":
        trial_params["ff_dims"] = suggest("ff_dims", hpo_cfg,
            lambda: trial.suggest_categorical("ff_dims", [128, 256, 512]))
        trial_params["fourier_scale"] = suggest("fourier_scale", hpo_cfg,
            lambda: trial.suggest_float("fourier_scale", 5.0, 30.0))

    # === Sampling ===
    samp_cfg = hpo_cfg.get("sampling", hpo_cfg)
    trial_params["sampling"] = {}
    
    trial_params["sampling"]["n_points_pde"] = suggest("n_points_pde", samp_cfg,
        lambda: trial.suggest_int("n_points_pde", 10000, 120000, log=True))
        
    trial_params["sampling"]["n_points_ic"] = suggest("n_points_ic", samp_cfg,
        lambda: trial.suggest_int("n_points_ic", 1000, 20000, log=True))
        
    trial_params["sampling"]["n_points_bc_domain"] = suggest("n_points_bc_domain", samp_cfg,
        lambda: trial.suggest_int("n_points_bc_domain", 4000, 40000, log=True))

    if has_building:
        trial_params["sampling"]["n_points_bc_building"] = suggest("n_points_bc_building", samp_cfg,
            lambda: trial.suggest_int("n_points_bc_building", 1000, 20000, log=True))

    # === Loss Weights ===
    weights_cfg = hpo_cfg.get("loss_weights", hpo_cfg)
    trial_params["loss_weights"] = {}
    
    for w in ["pde_weight", "ic_weight", "bc_weight", "neg_h_weight", "building_bc_weight"]:
        if w == "building_bc_weight" and not has_building: continue
        
        min_w, max_w = (1.0, 1e6) if w == "pde_weight" else (1e-2, 1e3)
        trial_params["loss_weights"][w] = suggest(w, weights_cfg,
            lambda: trial.suggest_float(w, min_w, max_w, log=True))

    trial_params["loss_weights"]["data_weight"] = 0.0

    # === Construct Configuration ===
    trial_config_dict = copy.deepcopy(base_config_dict)

    # Update Training
    trial_config_dict["training"]["learning_rate"] = trial_params["learning_rate"]
    trial_config_dict["training"]["batch_size"] = trial_params["batch_size"]
    trial_config_dict["training"]["lr_boundaries"] = trial_params["lr_boundaries"]

    # Update Model
    trial_config_dict["model"]["width"] = trial_params["model_width"]
    trial_config_dict["model"]["depth"] = trial_params["model_depth"]
    if model_name == "FourierPINN":
        trial_config_dict["model"]["ff_dims"] = trial_params["ff_dims"]
        trial_config_dict["model"]["fourier_scale"] = trial_params["fourier_scale"]

    # Update Sampling & Weights
    trial_config_dict["sampling"] = trial_params["sampling"]
    trial_config_dict["loss_weights"] = trial_params["loss_weights"]
    
    # Cleanup
    for k in ["grid", "ic_bc_grid"]: trial_config_dict.pop(k, None)
    if has_building and "building" in trial_config_dict:
        for k in ["nx", "ny", "nt"]: trial_config_dict["building"].pop(k, None)

    # Ensure flags
    if "gradnorm" not in trial_config_dict: trial_config_dict["gradnorm"] = {}
    trial_config_dict["gradnorm"]["enable"] = False
    trial_config_dict["data_free"] = True

    # Store
    config_to_store = unfreeze(trial_config_dict)
    trial.set_user_attr('full_config', config_to_store)
    trial_cfg_frozen = FrozenDict(trial_config_dict)

    # --- Logging (Restored User's Format) ---
    # We flatten the params to log so we see EVERYTHING (Fixed + Tuned)
    params_to_log = {}
    for k, v in trial_params.items():
        if not isinstance(v, dict): params_to_log[k] = v
    if "sampling" in trial_params: params_to_log.update(trial_params["sampling"])
    if "loss_weights" in trial_params: params_to_log.update(trial_params["loss_weights"])

    print("-" * 50)
    print(f"Starting Trial {trial.number}")
    print("Suggested Hyperparameters (Fixed + Tuned):")
    for key in sorted(params_to_log.keys()):
        value = params_to_log[key]
        if value is not None:
             if isinstance(value, (float, np.floating)):
                  # Check bounds to decide formatting
                  print(f"  {key:<25}: {value:.6e}" if abs(value) < 1e-2 or abs(value) > 1e3 else f"  {key:<25}: {value:.6f}")
             else:
                  print(f"  {key:<25}: {value}")

    print("\nFull Configuration for this Trial:")
    print(yaml.dump(config_to_store, default_flow_style=False, sort_keys=False, indent=2))
    print("-" * 50)

    # --- Run ---
    try:
        best_nse = run_training_trial(trial, trial_cfg_frozen)
        
        if hasattr(best_nse, 'item'): best_nse = best_nse.item()
        if jnp.isnan(best_nse) or best_nse <= -float('inf'):
             print(f"Trial {trial.number}: Invalid NSE ({best_nse}). Returning -1.0.")
             return -1.0
        return float(best_nse)

    except optuna.exceptions.TrialPruned as e:
         raise e
    except Exception as e:
         print(f"Trial {trial.number} Failed: {e}")
         import traceback; traceback.print_exc()
         return -1.0