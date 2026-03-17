# optimisation/optimization_train_loop.py
"""
Thin HPO trial wrapper — delegates all experiment-specific setup to
the experiment's setup_trial() function, then runs a generic training
loop with Optuna integration.
"""
import importlib
import traceback

import jax.numpy as jnp
from jax import random, lax
from flax.core import FrozenDict, unfreeze
import optuna


def run_training_trial(trial: optuna.trial.Trial, trial_cfg: FrozenDict) -> float:
    """Run a single HPO trial. Returns best metric value (or -1.0 on failure)."""
    # Accept both FrozenDict and plain dict
    if isinstance(trial_cfg, FrozenDict):
        trial_cfg_dict = unfreeze(trial_cfg)
    else:
        trial_cfg_dict = trial_cfg

    scenario = trial_cfg_dict.get("scenario", "experiment_1")
    print(f"--- Starting Trial {trial.number} ---")

    # 1. Dynamically import the experiment's setup_trial function
    try:
        mod = importlib.import_module(f"experiments.{scenario}.train")
        setup_trial = mod.setup_trial
    except (ImportError, AttributeError) as e:
        print(f"Trial {trial.number}: Cannot import experiments.{scenario}.train.setup_trial: {e}")
        return -1.0

    # 2. Run experiment-specific setup (model init, terrain, closures, etc.)
    try:
        ctx = setup_trial(trial_cfg_dict)
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as e:
        print(f"Trial {trial.number}: ERROR during setup: {e}")
        traceback.print_exc()
        return -1.0

    # 3. Read HPO settings from config
    hpo_settings = trial_cfg_dict.get("hpo_settings", {})
    objective_key = hpo_settings.get("objective_key", "nse_h")
    epochs = trial_cfg_dict["training"]["epochs"]
    validation_freq = trial_cfg_dict.get("training", {}).get("validation_freq", 1)
    hpo_patience = trial_cfg_dict.get("training", {}).get("hpo_patience", 300)
    report_interval = trial_cfg_dict.get("training", {}).get("hpo_report_interval", 1)
    log_freq = validation_freq * 200

    # 4. Generic training loop with Optuna integration
    best_metric = -jnp.inf
    last_improvement = -hpo_patience
    train_key = ctx["train_key"]
    params = ctx["params"]
    opt_state = ctx["opt_state"]

    for epoch in range(epochs):
        train_key, epoch_key = random.split(train_key)
        scan_inputs = ctx["generate_epoch_data_jit"](epoch_key)
        (params, opt_state), (batch_terms, batch_totals) = lax.scan(
            ctx["scan_body"], (params, opt_state), scan_inputs,
        )

        if (epoch + 1) % validation_freq == 0:
            metrics = ctx["validation_fn"](ctx["model"], params)
            current = metrics.get(objective_key, -jnp.inf)

            if jnp.isnan(current):
                print(f"Trial {trial.number}, Epoch {epoch+1}: NaN metric. Pruning.")
                raise optuna.exceptions.TrialPruned()

            if current > best_metric:
                best_metric = current
                last_improvement = epoch

            # Intra-trial patience-based early stopping
            if epoch - last_improvement > hpo_patience:
                print(f"Trial {trial.number}: No improvement for "
                      f"{hpo_patience} epochs. Early stopping at epoch {epoch+1}.")
                break

            # Report to Optuna (every report_interval validations)
            if (epoch + 1) % (validation_freq * report_interval) == 0:
                trial.report(float(best_metric), epoch)
                if trial.should_prune():
                    print(f"Trial {trial.number}: Pruned at epoch {epoch+1}.")
                    raise optuna.exceptions.TrialPruned()

            # Log periodically
            if (epoch + 1) % log_freq == 0:
                avg_total = float(jnp.mean(batch_totals))
                avg_terms = {k: float(jnp.mean(v)) for k, v in batch_terms.items()}
                terms_str = " ".join(
                    f"{k}={v:.3e}" for k, v in sorted(avg_terms.items())
                )
                print(f"  Trial {trial.number}, Epoch {epoch+1}/{epochs}: "
                      f"loss={avg_total:.3e} [{terms_str}] "
                      f"{objective_key}={current:.6f}, Best={best_metric:.6f}")
                trial.set_user_attr(f"losses_epoch_{epoch+1}", avg_terms)

    if best_metric <= -jnp.inf:
        print(f"Trial {trial.number}: No valid metric achieved.")
        return -1.0

    print(f"Trial {trial.number}: Finished. Best {objective_key} = {best_metric:.6f}")
    return float(best_metric)
