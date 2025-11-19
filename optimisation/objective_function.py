# optimisation/objective_function.py
"""
Defines the Optuna objective function: suggests hyperparameters based on flags,
builds the trial configuration, stores it, and calls the training loop.
"""
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

def objective(trial: optuna.trial.Trial,
              base_config_dict: Dict) -> float:
    """
    Objective function for Optuna HPO (Physics-Only, No GradNorm).
    """
    base_cfg = FrozenDict(base_config_dict)
    model_name = base_cfg["model"]["name"]
    has_building = "building" in base_cfg

    # --- 1. Define Hyperparameter Search Space ---
    trial_params = {}
    
    batch_size_choices = [256, 512, 1024]
    model_width_choices = [128, 256, 512, 1024]
    ff_dims_choices = [128, 256, 512] # For FourierPINN

    # === Training Hyperparameters ===
    trial_params["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    
    batch_size_index = trial.suggest_int("batch_size_index", 0, len(batch_size_choices) - 1)
    trial_params["batch_size"] = batch_size_choices[batch_size_index]

    opt_epochs = base_config_dict.get("training", {}).get("epochs", 2000)
    boundary1 = int(opt_epochs * 0.6)
    boundary2 = int(opt_epochs * 0.8)
    trial_params["lr_boundaries"] = {str(boundary1): 0.1, str(boundary2): 0.1}

    # === Model Hyperparameters ===
    model_width_index = trial.suggest_int("model_width_index", 0, len(model_width_choices) - 1)
    trial_params["model_width"] = model_width_choices[model_width_index]
    
    trial_params["model_depth"] = trial.suggest_int("model_depth", 3, 6)
    
    if model_name == "FourierPINN":
        ff_dims_index = trial.suggest_int("ff_dims_index", 0, len(ff_dims_choices) - 1)
        trial_params["ff_dims"] = ff_dims_choices[ff_dims_index]
        trial_params["fourier_scale"] = trial.suggest_float("fourier_scale", 5.0, 20.0)

    # === Grid Hyperparameters ===
    trial_params["sampling"] = {
        "n_points_pde": trial.suggest_int("n_points_pde", 10000, 120000, log=True),
        "n_points_ic": trial.suggest_int("n_points_ic", 1000, 20000, log=True),
        "n_points_bc_domain": trial.suggest_int("n_points_bc_domain", 4000, 40000, log=True)
    }
    if has_building:
        trial_params["sampling"]["n_points_bc_building"] = trial.suggest_int("n_points_bc_building", 1000, 20000, log=True)

    # === Loss Weights (Static) ===
    trial_params["loss_weights"] = {}
    
    print(f"Trial {trial.number}: Configuring independent static weights.")
    min_weight = 1e-2
    max_weight = 1e3
    
    trial_params["loss_weights"]["pde_weight"] = trial.suggest_float("pde_weight", 1.0, 1e6, log=True)
    trial_params["loss_weights"]["ic_weight"] = trial.suggest_float("ic_weight", min_weight, max_weight, log=True)
    trial_params["loss_weights"]["bc_weight"] = trial.suggest_float("bc_weight", min_weight, max_weight, log=True)
    trial_params["loss_weights"]["neg_h_weight"] = trial.suggest_float("neg_h_weight", min_weight, max_weight, log=True)

    if has_building:
        trial_params["loss_weights"]["building_bc_weight"] = trial.suggest_float("building_bc_weight", min_weight, max_weight, log=True)
    
    # Explicitly set data weight to 0
    trial_params["loss_weights"]["data_weight"] = 0.0
    trial.set_user_attr("data_weight", None) 

    # === Construct FULL Trial Configuration Dictionary ===
    trial_config_dict = copy.deepcopy(base_config_dict)

    trial_config_dict["training"]["learning_rate"] = trial_params["learning_rate"]
    trial_config_dict["training"]["batch_size"] = trial_params["batch_size"]
    trial_config_dict["training"]["lr_boundaries"] = trial_params["lr_boundaries"]

    trial_config_dict["model"]["width"] = trial_params["model_width"]
    trial_config_dict["model"]["depth"] = trial_params["model_depth"]
    if model_name == "FourierPINN":
        trial_config_dict["model"]["ff_dims"] = trial_params["ff_dims"]
        trial_config_dict["model"]["fourier_scale"] = trial_params["fourier_scale"]

    trial_config_dict["sampling"] = trial_params["sampling"]
    
    # Clean up old grid keys
    trial_config_dict.pop("grid", None)
    trial_config_dict.pop("ic_bc_grid", None)
    if has_building and "building" in trial_config_dict:
        trial_config_dict["building"].pop("nx", None)
        trial_config_dict["building"].pop("ny", None)
        trial_config_dict["building"].pop("nt", None)

    trial_config_dict["loss_weights"] = trial_params["loss_weights"]

    # Force gradnorm disabled
    if "gradnorm" not in trial_config_dict: trial_config_dict["gradnorm"] = {}
    trial_config_dict["gradnorm"]["enable"] = False
    trial_config_dict["gradnorm"].pop("alpha", None)
    trial_config_dict["gradnorm"].pop("update_freq", None)
    
    # Force data_free flag
    trial_config_dict["data_free"] = True

    # --- Store the complete configuration ---
    config_to_store = unfreeze(trial_config_dict)
    trial.set_user_attr('full_config', config_to_store)

    trial_cfg_frozen = FrozenDict(trial_config_dict)

    print("-" * 50)
    print(f"Starting Trial {trial.number}")
    print("Suggested Hyperparameters:")
    for key, value in trial.params.items():
         if value is not None:
             if isinstance(value, float):
                  print(f"  {key:<25}: {value:.6e}" if abs(value) < 1e-2 or abs(value) > 1e3 else f"  {key:<25}: {value:.6f}")
             else:
                  print(f"  {key:<25}: {value}")

    print("\nFull Configuration for this Trial:")
    print(yaml.dump(config_to_store, default_flow_style=False, sort_keys=False, indent=2))
    print("-" * 50)

    # --- Run the training trial ---
    try:
        best_nse = run_training_trial(trial, trial_cfg_frozen)

        if isinstance(best_nse, (jax.Array, jnp.ndarray, np.ndarray)):
            best_nse = float(best_nse)

        if jnp.isnan(best_nse) or not isinstance(best_nse, (float, int)) or best_nse <= -float('inf'):
             print(f"Trial {trial.number}: Invalid NSE ({best_nse}). Returning -1.0.")
             return -1.0

        return best_nse

    except optuna.exceptions.TrialPruned as e:
         raise e
    except Exception as e:
         print(f"Trial {trial.number}: UNHANDLED EXCEPTION in run_training_trial: {e}")
         import traceback
         traceback.print_exc()
         return -1.0