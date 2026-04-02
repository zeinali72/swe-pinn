"""Experiment 1 — Importance Sampling variant (arXiv:2104.12325).

Analytical dam-break on a flat domain with residual-driven adaptive
collocation point sampling.  Mirrors ``train.py`` but replaces the uniform
per-epoch PDE resampling with Algorithm 2 from Wu et al. (2021):

  1. A large CPU pool of N candidate PDE points is generated once at startup.
  2. Every ``resample_freq`` epochs the PDE residual is evaluated on the pool.
  3. n_pde active points are selected with probabilities proportional to the
     residuals (mixed with a uniform baseline).
  4. Importance-correction weights are applied to the PDE loss term so that
     the gradient estimator remains unbiased.

Requires: configs/experiment_1.yaml (or any variant with an optional
``sampling.importance_sampling`` sub-section).

Config keys under ``sampling.importance_sampling`` (all optional):
  pool_size       : int   — total pool size (default 2 000 000)
  resample_freq   : int   — epochs between IS updates (default 40)
  eval_batch_size : int   — GPU batch for residual evaluation (default 100 000)
  alpha           : float — error/uniform mixture weight (default 0.8)
"""

import os
import sys
import time
import argparse

# Ensure project root is on sys.path before src imports.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
import jax.numpy as jnp
from jax import random, lax
from flax.core import FrozenDict
from src.config import load_config, get_dtype
from src.predict.predictor import _apply_min_depth
from src.data import sample_domain, sample_lhs
from src.losses import (
    compute_ic_loss, compute_data_loss, compute_neg_h_loss,
    loss_boundary_dirichlet, loss_boundary_neumann_outflow_x,
    loss_boundary_wall_horizontal,
)
from src.utils import nse, rmse, relative_l2, plot_h_vs_x, generate_trial_name, save_model, ask_for_confirmation
from src.physics import h_exact, hu_exact, hv_exact
from src.training import (
    create_optimizer,
    calculate_num_batches,
    extract_loss_weights,
    get_active_loss_weights,
    get_boundary_segment_count,
    get_experiment_name,
    get_sampling_count_from_config,
    init_model_from_config,
    train_step_jitted,
    make_scan_body,
    sample_and_batch,
    maybe_batch_data,
    resolve_data_mode,
)
from src.monitoring import ConsoleLogger, MLflowTracker, compute_negative_depth_diagnostics
from src.checkpointing import CheckpointManager
from src.balancing.importance_sampling import (
    compute_weighted_pde_loss,
    evaluate_pool_residuals,
    compute_sampling_probs,
    sample_from_pool,
)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_losses(model, params, batch, config, data_free):
    """Compute all loss terms for Experiment 1 IS variant.

    Reads ``batch['pde_weights']`` for importance-corrected PDE loss.
    All other terms (IC, BC, data) are identical to the uniform variant.
    """
    terms = {}

    pde_pts = batch.get('pde', jnp.empty((0, 3), dtype=get_dtype()))
    if pde_pts.shape[0] > 0:
        pde_weights = batch.get('pde_weights', jnp.ones(pde_pts.shape[0], dtype=get_dtype()))
        terms['pde'] = compute_weighted_pde_loss(model, params, pde_pts, pde_weights, config)
        terms['neg_h'] = compute_neg_h_loss(model, params, pde_pts)

    ic_pts = batch.get('ic', jnp.empty((0, 3), dtype=get_dtype()))
    if ic_pts.shape[0] > 0:
        terms['ic'] = compute_ic_loss(model, params, ic_pts)

    bc_batches = batch.get('bc', {})
    if any(b.shape[0] > 0 for b in bc_batches.values() if hasattr(b, 'shape')):
        left = bc_batches.get('left', jnp.empty((0, 3), dtype=get_dtype()))
        right = bc_batches.get('right', jnp.empty((0, 3), dtype=get_dtype()))
        bottom = bc_batches.get('bottom', jnp.empty((0, 3), dtype=get_dtype()))
        top = bc_batches.get('top', jnp.empty((0, 3), dtype=get_dtype()))

        u_const = config["physics"]["u_const"]
        n_manning = config["physics"]["n_manning"]
        t_left = left[..., 2]
        h_true = h_exact(0.0, t_left, n_manning, u_const)
        hu_true = h_true * u_const
        loss_left = (loss_boundary_dirichlet(model, params, left, h_true, var_idx=0) +
                     loss_boundary_dirichlet(model, params, left, hu_true, var_idx=1))
        loss_right = loss_boundary_neumann_outflow_x(model, params, right)
        loss_bottom = loss_boundary_wall_horizontal(model, params, bottom)
        loss_top = loss_boundary_wall_horizontal(model, params, top)
        terms['bc'] = loss_left + loss_right + loss_bottom + loss_top

    data_pts = batch.get('data', jnp.empty((0, 6), dtype=get_dtype()))
    if not data_free and data_pts.shape[0] > 0:
        terms['data'] = compute_data_loss(model, params, data_pts, config)

    return terms


# ---------------------------------------------------------------------------
# HPO setup
# ---------------------------------------------------------------------------

def setup_trial(cfg_dict: dict, hpo_mode: bool = False) -> dict:
    """Set up all training components for the IS variant.

    Used by the HPO framework via ``train_module: train_imp_samp`` in the
    HPO config.  The IS pool update requires the current model params each
    ``resample_freq`` epochs, so this function returns a ``pre_epoch_hook``
    that the HPO loop calls before each ``generate_epoch_data_jit`` invocation.

    Returns
    -------
    dict with all keys expected by ``optimization_train_loop.run_training_trial``
    plus ``pre_epoch_hook(epoch, params) -> None``.
    """
    cfg = FrozenDict(cfg_dict)
    experiment_name = get_experiment_name(cfg_dict, "experiment_1_is")

    model, params, train_key, val_key = init_model_from_config(cfg)
    domain_cfg = cfg["domain"]

    # --- IS config ---
    is_cfg = cfg.get("sampling", {}).get("importance_sampling", {})
    pool_size = int(is_cfg.get("pool_size", 2_000_000))
    resample_freq = int(is_cfg.get("resample_freq", 40))
    eval_batch_size = int(is_cfg.get("eval_batch_size", 100_000))
    alpha = float(is_cfg.get("alpha", 0.8))
    print(f"IS config: pool={pool_size}, freq={resample_freq}, eval_batch={eval_batch_size}, alpha={alpha}")

    # --- Loss weights ---
    static_weights_dict, _ = extract_loss_weights(cfg)
    data_free, _ = resolve_data_mode(cfg)
    current_weights_dict = get_active_loss_weights(
        static_weights_dict, data_free=data_free, excluded_keys={"building_bc"}
    )
    active_loss_term_keys = list(current_weights_dict.keys())

    # --- Validation (analytical) ---
    val_points, h_true_val, hu_true_val, hv_true_val = None, None, None, None
    validation_data_loaded = False
    try:
        val_grid_cfg = cfg["validation_grid"]
        val_points = sample_domain(
            val_key, val_grid_cfg["n_points_val"],
            (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
        )
        n_manning = cfg["physics"]["n_manning"]
        u_const = cfg["physics"]["u_const"]
        h_true_val = h_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        hu_true_val = hu_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        hv_true_val = hv_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        validation_data_loaded = val_points.shape[0] > 0
        print(f"Analytical validation set: {val_points.shape[0]} points.")
    except Exception as e:
        print(f"Warning: Validation setup failed: {e}")

    # --- Sample counts ---
    batch_size = cfg["training"]["batch_size"]
    n_pde = get_sampling_count_from_config(cfg, "n_points_pde") if ('pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys) else 0
    n_pde = (n_pde // batch_size) * batch_size
    n_ic = get_sampling_count_from_config(cfg, "n_points_ic") if 'ic' in active_loss_term_keys else 0
    n_bc_domain = get_sampling_count_from_config(cfg, "n_points_bc_domain") if 'bc' in active_loss_term_keys else 0
    n_bc_per_wall = get_boundary_segment_count(cfg, n_bc_domain) if n_bc_domain > 0 else 0

    num_batches = calculate_num_batches(
        batch_size,
        [n_pde, n_ic, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall],
        None,
        data_free=True,
    )
    if num_batches == 0:
        raise ValueError(f"Batch size {batch_size} is too large for configured sample counts.")

    # --- Build initial IS pool ---
    train_key, pool_key, sample_key = random.split(train_key, 3)
    print(f"Generating IS pool ({pool_size} points)...")
    pool_pde_init = sample_lhs(
        pool_key, pool_size,
        (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
    )
    initial_probs = jnp.ones(pool_size, dtype=get_dtype()) / pool_size

    # Closure-based JIT: captures model/cfg/eval_batch_size to avoid hashing them
    @jax.jit
    def _eval_pool(params, pool_pts):
        return evaluate_pool_residuals(model, params, pool_pts, cfg, eval_batch_size)

    # --- Optimizer ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # --- Epoch data generator (3-arg, JIT-compiled inner function) ---
    def _generate_epoch_data(key, pde_pts, pde_weights):
        k_ic, k_bc, k_data = random.split(key, 3)
        x_range = (0., domain_cfg["lx"])
        y_range = (0., domain_cfg["ly"])
        t_range = (0., domain_cfg["t_final"])
        pde_data = pde_pts.reshape((num_batches, batch_size, 3))
        pde_w = pde_weights.reshape((num_batches, batch_size))
        l_key, r_key, b_key, t_key = random.split(k_bc, 4)
        ic_data = sample_and_batch(k_ic, sample_domain, n_ic, batch_size, num_batches, x_range, y_range, (0., 0.))
        bc_data = {
            'left':   sample_and_batch(l_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (0., 0.), y_range, t_range),
            'right':  sample_and_batch(r_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range),
            'bottom': sample_and_batch(b_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (0., 0.), t_range),
            'top':    sample_and_batch(t_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range),
        }
        return {
            'pde': pde_data,
            'pde_weights': pde_w,
            'ic': ic_data,
            'bc': bc_data,
            'data': maybe_batch_data(k_data, None, batch_size, num_batches, True),
            'building_bc': {},
        }

    _generate_epoch_data_jit = jax.jit(_generate_epoch_data)

    # --- IS state: mutable object that the HPO loop updates via pre_epoch_hook ---
    class _ISState:
        """Encapsulates mutable IS pool state for the HPO trial loop."""
        def __init__(self):
            self.pool_pde = pool_pde_init
            self.current_probs = initial_probs
            self._key = sample_key
            # Draw initial active set
            self._key, sub = random.split(self._key)
            self.active_pde, self.active_weights = sample_from_pool(
                sub, self.pool_pde, self.current_probs, n_pde
            )

        def update(self, epoch, params):
            """Resample pool every resample_freq epochs, always draw new active set."""
            if epoch > 0 and epoch % resample_freq == 0:
                self._key, pk = random.split(self._key)
                self.pool_pde = sample_lhs(
                    pk, pool_size,
                    (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
                )
                residuals = _eval_pool(params, self.pool_pde)
                self.current_probs = compute_sampling_probs(residuals, alpha)
            self._key, sub = random.split(self._key)
            self.active_pde, self.active_weights = sample_from_pool(
                sub, self.pool_pde, self.current_probs, n_pde
            )

    is_state = _ISState()

    def generate_epoch_data_jit(key):
        """Single-arg wrapper used by the HPO loop; IS state updated via hook."""
        return _generate_epoch_data_jit(key, is_state.active_pde, is_state.active_weights)

    def pre_epoch_hook(epoch, params):
        is_state.update(epoch, params)

    # --- scan body ---
    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights_dict, cfg, data_free,
        compute_losses_fn=compute_losses,
    )

    # --- Validation function ---
    def validation_fn(model, params):
        metrics = {}
        if validation_data_loaded:
            try:
                U_pred = model.apply({'params': params['params']}, val_points, train=False)
                min_depth_val = cfg.get("numerics", {}).get("min_depth", 0.0)
                U_pred = _apply_min_depth(U_pred, min_depth_val)
                metrics = {
                    'nse_h': float(nse(U_pred[..., 0], h_true_val)),
                    'rmse_h': float(rmse(U_pred[..., 0], h_true_val)),
                    'rel_l2_h': float(relative_l2(U_pred[..., 0], h_true_val)),
                    'nse_hu': float(nse(U_pred[..., 1], hu_true_val)),
                    'rmse_hu': float(rmse(U_pred[..., 1], hu_true_val)),
                    'nse_hv': float(nse(U_pred[..., 2], hv_true_val)),
                    'rmse_hv': float(rmse(U_pred[..., 2], hv_true_val)),
                }
            except Exception as e:
                print(f"Warning: Validation failed: {e}")
        if not metrics:
            metrics = {'nse_h': float(-jnp.inf), 'rmse_h': float(jnp.inf)}
        return metrics

    return {
        "cfg": cfg,
        "cfg_dict": cfg_dict,
        "model": model,
        "params": params,
        "train_key": train_key,
        "optimiser": optimiser,
        "opt_state": opt_state,
        "generate_epoch_data_jit": generate_epoch_data_jit,
        "scan_body": scan_body,
        "num_batches": num_batches,
        "validation_fn": validation_fn,
        "pre_epoch_hook": pre_epoch_hook,
        # Production extras
        "experiment_name": experiment_name,
        "validation_data_loaded": validation_data_loaded,
        "val_points_all": val_points,
        "h_true_val_all": h_true_val,
        "val_targets_all": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)
    experiment_name = get_experiment_name(cfg_dict, "experiment_1_is")

    model, params, train_key, val_key = init_model_from_config(cfg)
    print("Info: Running Experiment 1 (Importance Sampling) training...")

    domain_cfg = cfg["domain"]

    # --- IS config ---
    is_cfg = cfg.get("sampling", {}).get("importance_sampling", {})
    POOL_SIZE = int(is_cfg.get("pool_size", 2_000_000))
    RESAMPLE_FREQ = int(is_cfg.get("resample_freq", 40))
    EVAL_BATCH_SIZE = int(is_cfg.get("eval_batch_size", 100_000))
    ALPHA = float(is_cfg.get("alpha", 0.8))
    print(f"IS config: pool={POOL_SIZE}, freq={RESAMPLE_FREQ}, eval_batch={EVAL_BATCH_SIZE}, alpha={ALPHA}")

    # --- Loss weights ---
    static_weights_dict, _ = extract_loss_weights(cfg)
    data_free, has_data_loss = resolve_data_mode(cfg)
    current_weights_dict = get_active_loss_weights(
        static_weights_dict, data_free=data_free, excluded_keys={"building_bc"}
    )
    active_loss_term_keys = list(current_weights_dict.keys())

    # --- Validation (analytical) ---
    val_points, h_true_val, hu_true_val, hv_true_val = None, None, None, None
    validation_data_loaded = False
    try:
        val_grid_cfg = cfg["validation_grid"]
        val_points = sample_domain(
            val_key, val_grid_cfg["n_points_val"],
            (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
        )
        n_manning = cfg["physics"]["n_manning"]
        u_const = cfg["physics"]["u_const"]
        h_true_val = h_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        hu_true_val = hu_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        hv_true_val = hv_exact(val_points[:, 0], val_points[:, 2], n_manning, u_const)
        validation_data_loaded = val_points.shape[0] > 0
        print(f"Analytical validation set: {val_points.shape[0]} points.")
    except Exception as e:
        print(f"Warning: Validation setup failed: {e}")

    # --- Training data (data-driven mode, mirrors train.py) ---
    data_points_full = None
    if not data_free:
        try:
            train_grid_cfg = cfg["train_grid"]
            n_gauges = train_grid_cfg["n_gauges"]
            dt_data = train_grid_cfg["dt_data"]
            t_final = domain_cfg["t_final"]
            t_steps = jnp.arange(0., t_final + dt_data * 0.5, dt_data, dtype=get_dtype())
            n_timesteps = t_steps.shape[0]
            gauge_xy = sample_domain(train_key, n_gauges, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))[:, :2]
            gauge_xy_rep = jnp.repeat(gauge_xy, n_timesteps, axis=0)
            t_rep = jnp.tile(t_steps, n_gauges).reshape(-1, 1)
            data_pts_coords = jnp.hstack([gauge_xy_rep, t_rep])
            h_tr = h_exact(data_pts_coords[:, 0], data_pts_coords[:, 2], cfg["physics"]["n_manning"], cfg["physics"]["u_const"])
            u_tr = jnp.full_like(h_tr, cfg["physics"]["u_const"])
            v_tr = jnp.zeros_like(h_tr)
            data_points_full = jnp.stack([data_pts_coords[:, 2], data_pts_coords[:, 0], data_pts_coords[:, 1], h_tr, u_tr, v_tr], axis=1).astype(get_dtype())
            if data_points_full.shape[0] == 0:
                data_points_full = None
                has_data_loss = False
        except Exception as e:
            print(f"Error creating training data: {e}. Disabling data loss.")
            data_free = True

    # --- Sample counts ---
    batch_size = cfg["training"]["batch_size"]
    n_pde = get_sampling_count_from_config(cfg, "n_points_pde") if ('pde' in active_loss_term_keys or 'neg_h' in active_loss_term_keys) else 0
    n_pde = (n_pde // batch_size) * batch_size  # align to batch_size
    n_ic = get_sampling_count_from_config(cfg, "n_points_ic") if 'ic' in active_loss_term_keys else 0
    n_bc_domain = get_sampling_count_from_config(cfg, "n_points_bc_domain") if 'bc' in active_loss_term_keys else 0
    n_bc_per_wall = get_boundary_segment_count(cfg, n_bc_domain) if n_bc_domain > 0 else 0

    num_batches = calculate_num_batches(
        batch_size,
        [n_pde, n_ic, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall, n_bc_per_wall],
        data_points_full,
        data_free=data_free,
    )
    if num_batches == 0:
        raise ValueError(f"Batch size {batch_size} is too large for configured sample counts.")
    print(f"Batches per epoch: {num_batches} | n_pde: {n_pde}")

    # --- Build IS pool on GPU (LHS for space-filling coverage) ---
    # Keys derived from the config seed via train_key (same root as model init)
    train_key, pool_key = random.split(train_key)
    print(f"Generating IS pool ({POOL_SIZE} points) on GPU via LHS...")
    pool_pde = sample_lhs(
        pool_key, POOL_SIZE,
        (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
    )
    print(f"Pool ready: {pool_pde.shape}")

    # Initial probabilities: uniform — active set redrawn every epoch
    current_probs = jnp.ones(POOL_SIZE, dtype=get_dtype()) / POOL_SIZE

    # JIT pool residual evaluator (model and config are static; chunk_size is static)
    eval_pool_jit = jax.jit(evaluate_pool_residuals, static_argnums=(0, 3, 4))

    # --- Optimizer ---
    optimiser = create_optimizer(cfg, num_batches=num_batches)
    opt_state = optimiser.init(params)

    # --- Epoch data generator ---
    def generate_epoch_data(key, pde_pts, pde_weights):
        k_ic, k_bc, k_data = random.split(key, 3)
        x_range = (0., domain_cfg["lx"])
        y_range = (0., domain_cfg["ly"])
        t_range = (0., domain_cfg["t_final"])

        pde_data = pde_pts.reshape((num_batches, batch_size, 3))
        pde_w = pde_weights.reshape((num_batches, batch_size))

        l_key, r_key, b_key, t_key = random.split(k_bc, 4)

        ic_data = sample_and_batch(k_ic, sample_domain, n_ic, batch_size, num_batches, x_range, y_range, (0., 0.))
        bc_data = {
            'left':   sample_and_batch(l_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (0., 0.), y_range, t_range),
            'right':  sample_and_batch(r_key, sample_domain, n_bc_per_wall, batch_size, num_batches, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range),
            'bottom': sample_and_batch(b_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (0., 0.), t_range),
            'top':    sample_and_batch(t_key, sample_domain, n_bc_per_wall, batch_size, num_batches, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range),
        }
        return {
            'pde': pde_data,
            'pde_weights': pde_w,
            'ic': ic_data,
            'bc': bc_data,
            'data': maybe_batch_data(k_data, data_points_full, batch_size, num_batches, data_free),
            'building_bc': {},
        }

    generate_epoch_data_jit = jax.jit(generate_epoch_data)

    # --- scan body ---
    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights_dict, cfg, data_free,
        compute_losses_fn=compute_losses,
    )

    # --- Output dirs and logging ---
    cfg_dict['scenario'] = cfg_dict.get('scenario', 'experiment_1')
    arch_name = cfg_dict.get('model', {}).get('name', '')
    trial_name = generate_trial_name(cfg_dict['scenario'], arch_name, variant="IS")
    results_dir = os.path.join("results", cfg_dict['scenario'], trial_name)
    model_dir = os.path.join("models", cfg_dict['scenario'], trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    tracking_enabled = cfg_dict.get('mlflow', {}).get('enable', True)
    tracker = MLflowTracker(cfg_dict, trial_name, enable=tracking_enabled)
    tracker.log_flags({"scenario_type": "experiment_1_importance_sampling"})
    if tracking_enabled:
        try:
            tracker.log_artifact(config_path, 'run_config.yaml')
            tracker.log_artifact(os.path.abspath(__file__), 'source_script.py')
        except Exception:
            pass
    console = ConsoleLogger(cfg_dict)
    console.print_header()

    ckpt_mgr = CheckpointManager(model_dir, model=model)
    start_time = time.time()
    global_step = 0
    epoch = 0
    val_metrics = {}
    neg_depth = {}
    avg_losses_unweighted = {}
    avg_total_weighted_loss = 0.0
    current_lr = cfg["training"]["learning_rate"]

    best_nse_stats = {
        'nse': -jnp.inf, 'rmse': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {}
    }
    best_loss_stats = {'total_weighted_loss': jnp.inf, 'epoch': 0}
    best_params_nse = None
    best_params_loss = None

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start = time.time()
            train_key, epoch_key, sample_key = random.split(train_key, 3)

            # --- IS pool update: resample pool + recompute probs ---
            if epoch > 0 and epoch % RESAMPLE_FREQ == 0:
                print(f"--- Epoch {epoch}: IS pool update ---")
                train_key, pool_key = random.split(train_key)

                # Resample pool from domain so new high-residual regions are reachable
                pool_pde = sample_lhs(
                    pool_key, POOL_SIZE,
                    (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
                )

                # Evaluate residuals on GPU, update probabilities
                all_residuals = eval_pool_jit(model, params, pool_pde, cfg, EVAL_BATCH_SIZE)
                current_probs = compute_sampling_probs(all_residuals, ALPHA)

                _w = 1.0 / (POOL_SIZE * current_probs)
                _w = _w / jnp.mean(_w)
                print(f"    residual: mean={float(jnp.mean(all_residuals)):.3e}  max={float(jnp.max(all_residuals)):.3e}")
                print(f"    weight:   min={float(jnp.min(_w)):.3f}  mean={float(jnp.mean(_w)):.3f}  max={float(jnp.max(_w)):.3f}")

            # --- Draw fresh active set every epoch from current pool + probs ---
            active_pde_pts, active_pde_weights = sample_from_pool(
                sample_key, pool_pde, current_probs, n_pde
            )
            scan_inputs = generate_epoch_data_jit(
                epoch_key, active_pde_pts, active_pde_weights
            )

            # --- Training scan ---
            (params, opt_state), (batch_losses_stacked, batch_total_stacked) = lax.scan(
                scan_body, (params, opt_state), scan_inputs
            )
            global_step += num_batches

            # --- Aggregate losses ---
            avg_losses_unweighted = {k: float(jnp.sum(v)) / num_batches for k, v in batch_losses_stacked.items()}
            avg_total_weighted_loss = float(jnp.sum(batch_total_stacked)) / num_batches

            # --- Extract LR ---
            try:
                if hasattr(opt_state[-1], 'scale'):
                    current_lr = cfg["training"]["learning_rate"] * float(opt_state[-1].scale)
            except Exception:
                pass

            # --- Validation ---
            nse_val, rmse_val = -jnp.inf, jnp.inf
            if validation_data_loaded:
                try:
                    U_pred = model.apply({'params': params['params']}, val_points, train=False)
                    min_depth = cfg.get("numerics", {}).get("min_depth", 0.0)
                    U_pred = _apply_min_depth(U_pred, min_depth)
                    nse_val = float(nse(U_pred[..., 0], h_true_val))
                    rmse_val = float(rmse(U_pred[..., 0], h_true_val))
                    val_metrics = {
                        'nse_h': nse_val,
                        'rmse_h': rmse_val,
                        'rel_l2_h': float(relative_l2(U_pred[..., 0], h_true_val)),
                        'nse_hu': float(nse(U_pred[..., 1], hu_true_val)),
                        'rmse_hu': float(rmse(U_pred[..., 1], hu_true_val)),
                        'rel_l2_hu': float(relative_l2(U_pred[..., 1], hu_true_val)),
                        'nse_hv': float(nse(U_pred[..., 2], hv_true_val)),
                        'rmse_hv': float(rmse(U_pred[..., 2], hv_true_val)),
                        'rel_l2_hv': float(relative_l2(U_pred[..., 2], hv_true_val)),
                    }
                except Exception as e:
                    print(f"Warning: Validation failed: {e}")
            if not val_metrics:
                val_metrics = {'nse_h': float(nse_val), 'rmse_h': float(rmse_val)}

            # --- Track best models ---
            if nse_val > best_nse_stats['nse']:
                best_nse_stats.update({
                    'nse': nse_val, 'rmse': rmse_val, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'total_weighted_loss': avg_total_weighted_loss,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()},
                })
                best_params_nse = jax.tree.map(jnp.copy, params)
                if nse_val > -jnp.inf:
                    print(f"    ---> New best NSE: {nse_val:.6f}")

            if avg_total_weighted_loss < best_loss_stats['total_weighted_loss']:
                best_loss_stats['total_weighted_loss'] = avg_total_weighted_loss
                best_loss_stats['epoch'] = epoch
                best_params_loss = jax.tree.map(jnp.copy, params)

            # --- Reporting ---
            freq = cfg.get("reporting", {}).get("epoch_freq", 100)
            epoch_time = time.time() - epoch_start

            neg_depth = {'count': 0, 'fraction': 0.0, 'min': 0.0, 'mean': 0.0}
            if (epoch + 1) % freq == 0:
                try:
                    neg_depth = compute_negative_depth_diagnostics(model, params, scan_inputs['pde'][0])
                except Exception:
                    pass

            elapsed = time.time() - start_time
            saved_events = ckpt_mgr.update(
                epoch, params, opt_state, val_metrics,
                avg_losses_unweighted, avg_total_weighted_loss, cfg_dict, neg_depth,
                elapsed_time_s=elapsed,
            )
            for event in saved_events:
                event_type, value, ep, prev_value, prev_epoch = event
                if event_type == 'best_nse':
                    console.print_checkpoint_nse(value, ep, prev_value, prev_epoch)
                    tracker.log_best_nse(value, ep, step=global_step)
                elif event_type == 'best_loss':
                    console.print_checkpoint_loss(value, ep, prev_value, prev_epoch)
                    tracker.log_best_loss(value, ep, step=global_step)

            if (epoch + 1) % freq == 0:
                console.print_epoch(
                    epoch, cfg["training"]["epochs"],
                    avg_losses_unweighted, avg_total_weighted_loss,
                    current_lr, val_metrics, neg_depth.get('fraction', 0.0), epoch_time
                )

            tracker.log_epoch(
                epoch=epoch, step=global_step,
                losses=avg_losses_unweighted, total_loss=avg_total_weighted_loss,
                val_metrics=val_metrics, lr=current_lr,
                epoch_time=epoch_time, elapsed_time=elapsed,
                neg_depth=neg_depth if (epoch + 1) % freq == 0 else None,
            )

            # --- Early stopping ---
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))
            if epoch >= min_epochs and (epoch - best_nse_stats['epoch']) >= patience:
                print(f"--- Early stopping at epoch {epoch + 1} ---")
                break

    except KeyboardInterrupt:
        print("\n--- Training interrupted ---")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        total_time = time.time() - start_time

        # Final physics-loss evaluation on best-NSE params
        all_physics_losses = {}
        eval_params = best_params_nse if best_params_nse is not None else params
        try:
            eval_key = random.PRNGKey(0)
            k_pde, k_ic, k_left, k_right, k_bottom, k_top = random.split(eval_key, 6)
            n_eval = 200
            x_range = (0., domain_cfg["lx"])
            y_range = (0., domain_cfg["ly"])
            t_range = (0., domain_cfg["t_final"])
            eval_batch = {
                'pde': sample_domain(k_pde, n_eval, x_range, y_range, t_range),
                'pde_weights': jnp.ones(n_eval, dtype=get_dtype()),
                'ic': sample_domain(k_ic, n_eval, x_range, y_range, (0., 0.)),
                'bc': {
                    'left':   sample_domain(k_left,   n_eval, (0., 0.), y_range, t_range),
                    'right':  sample_domain(k_right,  n_eval, (domain_cfg["lx"], domain_cfg["lx"]), y_range, t_range),
                    'bottom': sample_domain(k_bottom, n_eval, x_range, (0., 0.), t_range),
                    'top':    sample_domain(k_top,    n_eval, x_range, (domain_cfg["ly"], domain_cfg["ly"]), t_range),
                },
                'data': jnp.empty((0, 6), dtype=get_dtype()),
                'building_bc': {},
            }
            all_physics_losses = compute_losses(model, eval_params, eval_batch, cfg, data_free=True)
            all_physics_losses = {k: float(v) for k, v in all_physics_losses.items()}
            print("\nAll physics losses (best-NSE params):")
            for k, v in all_physics_losses.items():
                print(f"  {k}: {v:.6e}")
        except Exception as e:
            print(f"Warning: Physics loss evaluation failed: {e}")

        final_losses_for_ckpt = dict(avg_losses_unweighted)
        for k, v in all_physics_losses.items():
            if k not in final_losses_for_ckpt:
                final_losses_for_ckpt[k] = v

        ckpt_mgr.save_final(epoch, params, opt_state, val_metrics,
                            final_losses_for_ckpt, avg_total_weighted_loss, cfg_dict, neg_depth,
                            training_time_s=total_time)

        best_nse_ckpt = ckpt_mgr.get_best_nse_stats()
        best_loss_ckpt = ckpt_mgr.get_best_loss_stats()
        console.print_completion_summary(
            total_time=total_time,
            final_epoch=epoch,
            best_nse_stats=best_nse_ckpt,
            best_loss_stats=best_loss_ckpt,
            final_losses=final_losses_for_ckpt,
            final_val_metrics=val_metrics,
            neg_depth_final=neg_depth,
            neg_depth_best_nse={},
            neg_depth_best_loss={},
            final_lr=current_lr,
        )

        if tracker.enabled:
            try:
                tracker.log_summary({
                    'best_validation_model': best_nse_stats,
                    'best_loss_model': best_loss_stats,
                    'final_system': {
                        'total_training_time_seconds': total_time,
                        'total_epochs_run': epoch + 1,
                        'total_steps_run': global_step,
                    },
                    'importance_sampling': {
                        'pool_size': POOL_SIZE,
                        'resample_freq': RESAMPLE_FREQ,
                        'alpha': ALPHA,
                    },
                })
            except Exception as e:
                print(f"Warning: Aim summary logging failed: {e}")

        final_params = best_params_nse if best_params_nse is not None else best_params_loss

        if ask_for_confirmation():
            if final_params is not None:
                saved_model_path = save_model(final_params, model_dir, trial_name)
                if tracker.enabled and saved_model_path:
                    try:
                        tracker.log_artifact(saved_model_path, 'model_weights.pkl')
                    except Exception:
                        pass

                # 1D validation plot
                print("  Generating 1D validation plot...")
                plot_cfg = cfg.get("plotting", {})
                min_depth_plot = cfg.get("numerics", {}).get("min_depth", 0.0)
                t_val = plot_cfg.get("t_const_val", domain_cfg["t_final"] / 2.0)
                nx_val = plot_cfg.get("nx_val", 101)
                y_const = plot_cfg.get("y_const_plot", 0.0)
                x_plot = jnp.linspace(0., domain_cfg["lx"], nx_val, dtype=get_dtype())
                plot_pts = jnp.stack([
                    x_plot,
                    jnp.full_like(x_plot, y_const),
                    jnp.full_like(x_plot, t_val),
                ], axis=1)
                U_plot = model.apply({'params': final_params['params']}, plot_pts, train=False)
                U_plot = _apply_min_depth(U_plot, min_depth_plot)
                plot_path = os.path.join(results_dir, "final_validation_plot.png")
                plot_h_vs_x(x_plot, U_plot[..., 0], t_val, y_const, cfg_dict, plot_path)
                tracker.log_image(plot_path, 'validation_plot_1D')
                print(f"Model and plots saved in {model_dir} / {results_dir}")

        tracker.close()

    return best_nse_stats['nse'] if best_nse_stats['nse'] > -jnp.inf else -1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1 PINN training with importance sampling (arXiv:2104.12325)."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file, e.g. configs/experiment_1.yaml")
    args = parser.parse_args()

    try:
        final_nse = main(args.config)
        print(f"\n--- Script Finished ---")
        if final_nse > -jnp.inf:
            print(f"Final best NSE: {final_nse:.6f}")
        print("-----------------------")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
