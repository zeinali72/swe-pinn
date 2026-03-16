# optimisation/optimization_train_loop.py
"""
Thin wrapper that runs a single Optuna HPO trial using the shared
src/training/ modules and experiment-specific loss/sampling factories.
"""
import os
import jax.numpy as jnp
from jax import random, lax
from flax.core import FrozenDict
import numpy as np
import optuna

from src.config import get_dtype
from src.data import sample_domain
from src.utils import mask_points_inside_building
from src.physics import h_exact
from src.training import (
    init_model_from_config,
    create_optimizer,
    extract_loss_weights,
    get_active_loss_weights,
    calculate_num_batches,
    get_sampling_count_from_config,
    get_boundary_segment_count,
    train_step_jitted,
    make_scan_body,
)

from optimisation.experiment_registry import get_experiment_fns


def _load_validation_data(trial_cfg, val_key):
    """Load or create validation data. Returns (loaded, val_points, h_true_val)."""
    has_building = "building" in trial_cfg
    scenario = trial_cfg.get("scenario", "experiment_1")
    base_data_path = os.path.join("data", scenario)
    validation_file = os.path.join(base_data_path, "validation_sample.npy")

    if os.path.exists(validation_file):
        loaded = np.load(validation_file).astype(get_dtype())
        val_points = loaded[:, [1, 2, 0]]  # (x, y, t)
        h_true = loaded[:, 3]
        if has_building:
            mask = mask_points_inside_building(val_points, trial_cfg["building"])
            val_points, h_true = val_points[mask], h_true[mask]
        if val_points.shape[0] > 0:
            return True, val_points, h_true
        return False, None, None

    if not has_building and "validation_grid" in trial_cfg:
        val_grid_cfg = trial_cfg["validation_grid"]
        domain_cfg = trial_cfg["domain"]
        n_val = val_grid_cfg.get("n_points_val",
                                  val_grid_cfg.get("nx_val", 10) *
                                  val_grid_cfg.get("ny_val", 10) *
                                  val_grid_cfg.get("nt_val", 10))
        val_points = sample_domain(
            val_key, n_val,
            (0., domain_cfg["lx"]), (0., domain_cfg["ly"]),
            (0., domain_cfg["t_final"]),
        )
        h_true = h_exact(
            val_points[:, 0], val_points[:, 2],
            trial_cfg["physics"]["n_manning"], trial_cfg["physics"]["u_const"],
        )
        if val_points.shape[0] > 0:
            return True, val_points, h_true

    return False, None, None


def run_training_trial(trial: optuna.trial.Trial, trial_cfg: FrozenDict) -> float:
    """Run a single HPO trial. Returns best NSE (or -1.0 on failure)."""
    scenario = trial_cfg.get("scenario", "experiment_1")
    has_building = "building" in trial_cfg
    data_free = trial_cfg.get("hpo_settings", {}).get("data_free", True)
    registry = get_experiment_fns(scenario)

    print(f"--- Starting Trial {trial.number} ---")

    # 1. Model + keys
    try:
        model, params, train_key, val_key = init_model_from_config(trial_cfg)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Trial {trial.number}: ERROR during model initialization: {e}")
        return -1.0

    # 2. Loss weights (same logic as production)
    static_weights, _ = extract_loss_weights(trial_cfg)
    excluded = set() if has_building else {"building_bc"}
    current_weights = get_active_loss_weights(
        static_weights, data_free=data_free, excluded_keys=excluded,
    )
    active_keys = list(current_weights.keys())

    # 3. Sampling counts (same helpers as production)
    batch_size = trial_cfg["training"]["batch_size"]
    n_pde = get_sampling_count_from_config(trial_cfg, "n_points_pde") if ('pde' in active_keys or 'neg_h' in active_keys) else 0
    n_ic = get_sampling_count_from_config(trial_cfg, "n_points_ic") if 'ic' in active_keys else 0
    n_bc_domain = get_sampling_count_from_config(trial_cfg, "n_points_bc_domain") if 'bc' in active_keys else 0
    n_bc_per_wall = get_boundary_segment_count(trial_cfg, n_bc_domain) if n_bc_domain > 0 else 0

    n_bldg_per_wall = 0
    sample_sizes = [n_pde, n_ic, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall]
    if has_building and 'building_bc' in active_keys:
        n_bldg = get_sampling_count_from_config(trial_cfg, "n_points_bc_building")
        n_bldg_per_wall = get_boundary_segment_count(trial_cfg, n_bldg)
        sample_sizes.extend([n_bldg_per_wall] * 4)

    num_batches = calculate_num_batches(batch_size, sample_sizes, None, data_free=data_free)
    if num_batches == 0:
        print(f"Trial {trial.number}: batch_size {batch_size} too large. Returning -1.0.")
        return -1.0

    # 4. Optimizer (same as production — clip + adam + reduce_on_plateau)
    optimiser = create_optimizer(trial_cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # 5. Epoch data generator (experiment-specific factory)
    gen_kwargs = dict(
        cfg=trial_cfg, n_pde=n_pde, n_ic=n_ic, n_bc_per_wall=n_bc_per_wall,
        batch_size=batch_size, num_batches=num_batches, data_free=data_free,
    )
    if has_building:
        gen_kwargs.update(
            has_building=True,
            active_loss_term_keys=active_keys,
            n_bldg_per_wall=n_bldg_per_wall,
        )
    generate_epoch_data_jit = registry["make_generate_epoch_data"](**gen_kwargs)

    # 6. Scan body (same as production)
    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights,
        trial_cfg, data_free, compute_losses_fn=registry["compute_losses"],
    )

    # 7. Validation (experiment-specific factory)
    val_loaded, val_points, h_true_val = _load_validation_data(trial_cfg, val_key)
    validation_fn = registry["make_validation_fn"](
        trial_cfg, val_loaded, val_points, h_true_val,
    )

    # 8. Training loop with Optuna pruning
    epochs = trial_cfg["training"]["epochs"]
    validation_freq = trial_cfg.get("training", {}).get("validation_freq", 1)
    best_nse = -jnp.inf

    for epoch in range(epochs):
        train_key, epoch_key = random.split(train_key)
        scan_inputs = generate_epoch_data_jit(epoch_key)
        (params, opt_state), _ = lax.scan(scan_body, (params, opt_state), scan_inputs)

        if (epoch + 1) % validation_freq == 0:
            metrics = validation_fn(model, params)
            current_nse = metrics.get("nse_h", -jnp.inf)

            if jnp.isnan(current_nse):
                print(f"Trial {trial.number}, Epoch {epoch+1}: NaN NSE. Pruning.")
                raise optuna.exceptions.TrialPruned()

            best_nse = max(best_nse, current_nse if current_nse > -jnp.inf else -1.0)
            trial.report(best_nse, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number}: Pruned at epoch {epoch+1}.")
                raise optuna.exceptions.TrialPruned()

            if (epoch + 1) % (validation_freq * 200) == 0:
                print(f"  Trial {trial.number}, Epoch {epoch+1}/{epochs}: "
                      f"NSE={current_nse:.6f}, Best={best_nse:.6f}")

    if best_nse <= -jnp.inf:
        print(f"Trial {trial.number}: No valid NSE achieved.")
        return -1.0

    print(f"Trial {trial.number}: Finished. Best NSE = {best_nse:.6f}")
    return float(best_nse)
