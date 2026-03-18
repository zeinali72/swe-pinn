"""Experiment 8 — Importance sampling variant (Phase 3).

Same domain as Experiment 8 but uses error-driven adaptive importance
sampling to concentrate collocation points in high-residual regions.
Requires: configs/experiment_8.yaml, data/experiment_8/
Builds on: Experiment 8.
"""
import os
import sys
import time
import argparse
import importlib
import itertools
from typing import Any, Dict, Tuple, Optional
import shutil
import pandas as pd 

from jaxtyping import config as jax_config

import jax
import jax.numpy as jnp
from jax import random, lax
import optax
try:
    from aim import Image
except ImportError:
    Image = None
from flax.core import FrozenDict
import flax.linen as nn
import numpy as np 
import matplotlib
matplotlib.use('Agg') # Ensure headless plotting
import matplotlib.pyplot as plt 

# Local application imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.config import load_config, get_dtype
from src.data import (
    get_batches_tensor,
    get_sample_count,
    load_boundary_condition,
    IrregularDomainSampler,
    load_bathymetry,
    bathymetry_fn,
    load_validation_data,
    resolve_scenario_asset_path,
)
from src.models import init_model
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet,
    loss_slip_wall_generalized,
    compute_neg_h_loss,
    compute_data_loss,
)
from src.utils import (
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation
)
from src.monitoring import ConsoleLogger, AimTracker, compute_negative_depth_diagnostics
from src.checkpointing import CheckpointManager
from src.training import train_step_jitted, make_scan_body, maybe_batch_data

from src.physics import SWEPhysics
from src.balancing.importance_sampling import (
    evaluate_pool_residuals,
    compute_sampling_probs,
    sample_from_pool,
)

# ==============================================================================
# Helper: Shared PDE residual core (single Jacobian computation)
# ==============================================================================
def _compute_pde_squared_errors(model: nn.Module, params: Dict[str, Any],
                                pde_batch: jnp.ndarray,
                                config: FrozenDict) -> jnp.ndarray:
    """Compute per-point squared PDE residual errors with a single Jacobian pass.

    Returns:
        (N,) array of sum-of-squared residual errors per collocation point.
    """
    def U_fn(pts):
        return model.apply(params, pts, train=False)

    U_pred = U_fn(pde_batch)

    # Single Jacobian computation — the dominant cost per call
    jac_U = jax.vmap(jax.jacfwd(U_fn))(pde_batch)
    dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]

    x_batch = pde_batch[..., 0]
    y_batch = pde_batch[..., 1]

    _, bed_grad_x, bed_grad_y = bathymetry_fn(x_batch, y_batch)

    eps = config["numerics"]["eps"]
    physics = SWEPhysics(U_pred, eps=eps)

    g = config["physics"]["g"]
    n_manning = config["physics"]["n_manning"]
    inflow = config["physics"]["inflow"]

    JF, JG = physics.flux_jac(g=g)
    div_F = jnp.einsum('nij,nj->ni', JF, dU_dx)
    div_G = jnp.einsum('nij,nj->ni', JG, dU_dy)

    S = physics.source(g=g, n_manning=n_manning, inflow=inflow,
                       bed_grad_x=bed_grad_x, bed_grad_y=bed_grad_y)

    residual = dU_dt + div_F + div_G - S

    # Mask out dry areas for stability (h < eps)
    h_mask = jnp.where(U_pred[..., 0] < eps, 0.0, 1.0)
    final_residual = residual * h_mask[..., None]

    # Sum of squared errors of the 3 equations per point — shape (N,)
    return jnp.sum(final_residual ** 2, axis=1)



# ==============================================================================
# Helper: Weighted PDE Loss (Corrected for Importance Sampling)
# ==============================================================================
def compute_weighted_pde_loss(model: nn.Module, params: FrozenDict, pde_batch: jnp.ndarray,
                              weights: jnp.ndarray, config: FrozenDict) -> float:
    """
    Computes the weighted PDE loss.
    Essential for Unbiased Importance Sampling.
    """
    squared_error = _compute_pde_squared_errors(model, params, pde_batch, config)

    # Apply Importance Weights: Mean(Error * Weights)
    weighted_mse = jnp.mean(squared_error * weights)

    return weighted_mse

# ==============================================================================
# Loss computation factory
# ==============================================================================

def make_compute_losses(bc_fn_static):
    """Return a compute_losses closure for Experiment 8 importance sampling."""

    def compute_losses(model, params, batch, config, data_free):
        terms = {}

        # PDE loss weighted by importance sampling weights
        pde_pts = batch['pde']
        pde_weights = batch.get('pde_weights', jnp.ones(pde_pts.shape[0]))
        terms['pde'] = compute_weighted_pde_loss(model, params, pde_pts, pde_weights, config)
        terms['neg_h'] = compute_neg_h_loss(model, params, batch['pde'])

        # IC: dry bed
        U_ic = model.apply(params, batch['ic'], train=False)
        terms['ic'] = jnp.mean(U_ic[..., 0] ** 2) + jnp.mean(U_ic[..., 1] ** 2 + U_ic[..., 2] ** 2)

        # BC: upstream inflow + outer wall + building walls
        if batch['bc_upstream'].shape[0] > 0:
            t_inflow = batch['bc_upstream'][..., 2]
            Q_target = bc_fn_static(t_inflow)
            upstream_width = config["boundary_conditions"]["upstream_discharge_width"]
            flux_target_x = Q_target / upstream_width
            loss_bc_inflow = (
                loss_boundary_dirichlet(model, params, batch['bc_upstream'], flux_target_x, var_idx=1)
                + loss_boundary_dirichlet(model, params, batch['bc_upstream'], jnp.zeros_like(flux_target_x), var_idx=2)
            )
        else:
            loss_bc_inflow = 0.0

        loss_bc_wall = loss_slip_wall_generalized(model, params, batch['bc_wall'])
        loss_bldg = loss_slip_wall_generalized(model, params, batch['bc_building'])
        terms['bc'] = loss_bc_inflow + loss_bc_wall + loss_bldg
        terms['building'] = loss_bldg

        data_batch_data = batch.get('data', jnp.empty((0, 6), dtype=get_dtype()))
        if not data_free and data_batch_data.shape[0] > 0:
            terms['data'] = compute_data_loss(model, params, data_batch_data, config)

        return terms

    return compute_losses

# ==============================================================================
# Main Execution
# ==============================================================================

def main(config_path: str):
    """
    Main training loop for Experiment 8 Scenario (Importance Sampling).
    Includes Importance Sampling for PDE points (Memory Optimized).
    """
    #--- 1. LOAD CONFIGURATION (MUTABLE) ---
    cfg_dict = load_config(config_path)
    
    print("Info: Running Experiment 8 (Importance Sampling) training...")

    # --- 2. SETUP DATA & COMPUTE DOMAIN EXTENT ---
    scenario_name = cfg_dict.get('scenario', 'experiment_8')
    base_data_path = os.path.join("data", scenario_name)

    try:
        artifacts_path = resolve_scenario_asset_path(base_data_path, scenario_name, "domain_artifacts")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    
    print(f"Loading domain geometry from: {artifacts_path}")
    domain_sampler = IrregularDomainSampler(artifacts_path)

    # --- CALCULATE DOMAIN EXTENT ---
    all_coords = domain_sampler.tri_coords.reshape(-1, 2)
    min_vals = jnp.min(all_coords, axis=0)
    max_vals = jnp.max(all_coords, axis=0)
    
    x_min, y_min = float(min_vals[0]), float(min_vals[1])
    x_max, y_max = float(max_vals[0]), float(max_vals[1])
    
    calc_lx = x_max - x_min
    calc_ly = y_max - y_min
    
    if 'domain' not in cfg_dict: cfg_dict['domain'] = {}
    cfg_dict['domain']['lx'] = calc_lx
    cfg_dict['domain']['ly'] = calc_ly
    cfg_dict['domain']['x_min'] = x_min
    cfg_dict['domain']['x_max'] = x_max
    cfg_dict['domain']['y_min'] = y_min
    cfg_dict['domain']['y_max'] = y_max

    h_scale = 1.0  
    hu_scale = 1.0 
    hv_scale = 1.0

    if 'model' not in cfg_dict: cfg_dict['model'] = {}
    cfg_dict['model']['output_scales'] = (h_scale, hu_scale, hv_scale)

    # Derive upstream discharge width from the domain geometry
    bc_cfg = cfg_dict.setdefault("boundary_conditions", {})
    if 'upstream' in domain_sampler.boundaries:
        computed_width = domain_sampler.boundary_length('upstream')
        bc_cfg["upstream_discharge_width"] = computed_width
        print(f"Upstream discharge width derived from shapefile: {computed_width:.4f} m")

    # --- 3. FINALIZE CONFIG & INIT MODEL ---
    cfg = FrozenDict(cfg_dict)

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e
    
    key = random.PRNGKey(cfg["training"]["seed"])
    model_key, train_key, val_key, pool_key = random.split(key, 4)
    model, params = init_model(model_class, model_key, cfg)

    # --- 4. Setup Directories ---
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base + "_IS_Weighted")
    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- 5. Prepare Loss Weights ---
    static_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    active_loss_term_keys = [k for k, v in static_weights_dict.items() if v > 0]
    current_weights_dict = FrozenDict({k: static_weights_dict[k] for k in active_loss_term_keys})

    # --- 6. Load Remaining Assets ---
    try:
        dem_path = resolve_scenario_asset_path(base_data_path, scenario_name, "dem")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)
    
    try:
        bc_csv_path = resolve_scenario_asset_path(base_data_path, scenario_name, "boundary_condition")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    bc_fn_static = load_boundary_condition(bc_csv_path)

    val_points, h_true_val = None, None
    data_points_full = None
    
    data_free_flag = cfg.get("data_free")
    has_data_loss = not data_free_flag
    
    training_data_file = os.path.join(base_data_path, "training_dataset_sample.npy")
    if has_data_loss: 
        if os.path.exists(training_data_file):
            try:
                data_points_full = jnp.load(training_data_file).astype(get_dtype()) 
                if data_points_full.shape[0] == 0:
                     data_points_full = None
                     has_data_loss = False
            except Exception as e:
                data_points_full = None
                has_data_loss = False
        else:
            has_data_loss = False
    data_free = not has_data_loss 

    validation_data_file = os.path.join(base_data_path, "validation_gauges_ground_truth.npy")
    validation_data_loaded = False
    val_pts_batch = None
    val_h_true = None
    val_hu_true = None
    val_hv_true = None
    loaded_val_data = None

    if os.path.exists(validation_data_file):
        try:
            _, val_pts_batch, val_targets = load_validation_data(validation_data_file, dtype=get_dtype())
            val_h_true = val_targets[:, 0]
            u_temp = val_targets[:, 1]
            v_temp = val_targets[:, 2]
            val_hu_true = val_h_true * u_temp
            val_hv_true = val_h_true * v_temp
            if val_pts_batch.shape[0] > 0:
                validation_data_loaded = True
        except Exception as e:
            print(f"Error loading validation data: {e}")
            val_pts_batch = None

    # --- 7. Initialize Aim & Console Logger ---
    aim_enabled = cfg_dict.get('aim', {}).get('enable', True)
    aim_tracker = AimTracker(cfg_dict, trial_name, enable=aim_enabled)
    aim_tracker.log_flags({"scenario_type": "experiment_8_importance_sampling"})
    if aim_enabled:
        try:
            aim_tracker.log_artifact(config_path, 'run_config.yaml')
            aim_tracker.log_artifact(os.path.abspath(__file__), 'source_script.py')
        except Exception:
            pass

    cfg_dict['scenario'] = cfg_dict.get('scenario', 'experiment_8')
    console = ConsoleLogger(cfg_dict)
    console.print_header()

    # --- 8. Data Generation Setup ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]
    
    # --- IMPORTANCE SAMPLING SETUP (FROM CONFIG) ---
    is_cfg = sampling_cfg.get("importance_sampling", {})
    # Defaults provided if not in config
    POOL_SIZE = int(is_cfg.get("pool_size", 2_000_000))
    RESAMPLE_FREQ_EPOCHS = int(is_cfg.get("resample_freq", 5))
    EVAL_BATCH_SIZE = int(is_cfg.get("eval_batch_size", 10_000))
    P_ERROR_WEIGHT = float(is_cfg.get("p_error_weight", 0.8)) # Alpha

    print(f"Importance Sampling Config: Pool={POOL_SIZE}, Freq={RESAMPLE_FREQ_EPOCHS}, Alpha={P_ERROR_WEIGHT}")

    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000)
    n_pde = (n_pde // batch_size) * batch_size 
    
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100)
    n_bc_upstream = get_sample_count(sampling_cfg, "n_points_bc_inflow", 100) 
    n_bc_wall = get_sample_count(sampling_cfg, "n_points_bc_domain", 100)
    n_building = get_sample_count(sampling_cfg, "n_points_bc_building", 100)

    bc_counts = [n_pde//batch_size, n_ic//batch_size, n_bc_wall//batch_size, n_bc_upstream//batch_size, n_building//batch_size]
    if not data_free and data_points_full is not None:
        bc_counts.append(data_points_full.shape[0] // batch_size)

    num_batches = max(bc_counts) if bc_counts else 0
    print(f"Batches per epoch: {num_batches} | PDE Points per epoch: {n_pde}")
    
    # --- Generate the Pool (ON CPU) ---
    print(f"Generating Importance Sampling Pool ({POOL_SIZE} points)...")
    
    chunk_size = 1_000_000
    pool_chunks = []
    num_chunks = POOL_SIZE // chunk_size
    if POOL_SIZE % chunk_size != 0: num_chunks += 1

    pool_base_key = random.PRNGKey(12345)

    for i in range(num_chunks):
        subkey = random.fold_in(pool_base_key, i)
        current_chunk_size = min(chunk_size, POOL_SIZE - (i * chunk_size))
        pts = domain_sampler.sample_interior(subkey, current_chunk_size, (0., domain_cfg["t_final"]))
        pool_chunks.append(pts)
        if (i+1) % 2 == 0:
            print(f"  Generated {(i+1)*chunk_size} points...")

    pool_pde = jnp.concatenate(pool_chunks, axis=0)
    print(f"Pool ready (GPU): {pool_pde.shape}")
    del pool_chunks

    # JIT pool residual evaluator (model and config are static; chunk_size is static)
    eval_pool_jit = jax.jit(evaluate_pool_residuals, static_argnums=(0, 3, 4))

    # Initial probabilities: uniform — active set redrawn every epoch
    current_probs = jnp.ones(POOL_SIZE, dtype=get_dtype()) / POOL_SIZE

    # --- Optimizer ---
    reduce_on_plateau_cfg = cfg.get("training", {}).get("reduce_on_plateau", {})
    optimiser = optax.chain(
        optax.clip_by_global_norm(cfg.get("training", {}).get("clip_norm", 1.0)),
        optax.adam(learning_rate=cfg["training"]["learning_rate"]),
        optax.contrib.reduce_on_plateau(
            factor=float(reduce_on_plateau_cfg.get("factor", 0.5)),
            patience=int(reduce_on_plateau_cfg.get("patience", 5)),
            rtol=float(reduce_on_plateau_cfg.get("rtol", 1e-4)),
            atol=float(reduce_on_plateau_cfg.get("atol", 0.0)),
            cooldown=int(reduce_on_plateau_cfg.get("cooldown", 1)),
            accumulation_size=num_batches*int(reduce_on_plateau_cfg.get("accumulation_factor", 1)),
            min_scale=float(reduce_on_plateau_cfg.get("min_scale", 1e-6)),
        ),
    )
    opt_state = optimiser.init(params)

    # --- Epoch Data Generation ---
    def generate_epoch_data_with_IS(key, current_pde_pts, current_pde_weights):
        k1, k2, k3, k4, k5 = random.split(key, 5)
        t_range = (0., domain_cfg["t_final"])

        pde_data = current_pde_pts.reshape((num_batches, batch_size, 3))
        pde_w = current_pde_weights.reshape((num_batches, batch_size))

        ic_pts = domain_sampler.sample_interior(k2, n_ic, (0., 0.))
        ic_data = get_batches_tensor(k2, ic_pts, batch_size, num_batches)

        bc_upstream_pts = domain_sampler.sample_boundary(k3, n_bc_upstream, t_range, 'upstream')
        bc_upstream = get_batches_tensor(k3, bc_upstream_pts, batch_size, num_batches)

        bc_wall_pts = domain_sampler.sample_boundary(k4, n_bc_wall, t_range, 'wall')
        bc_wall = get_batches_tensor(k4, bc_wall_pts, batch_size, num_batches)

        bc_building_pts = domain_sampler.sample_boundary(k5, n_building, t_range, 'building')
        bc_building = get_batches_tensor(k5, bc_building_pts, batch_size, num_batches)

        return {
            'pde': pde_data, 'pde_weights': pde_w,
            'ic': ic_data,
            'bc_upstream': bc_upstream,
            'bc_wall': bc_wall,
            'bc_building': bc_building,
            'data': maybe_batch_data(k5, data_points_full, batch_size, num_batches, data_free),
        }

    generate_epoch_data_jitted = jax.jit(generate_epoch_data_with_IS)

    compute_losses_fn = make_compute_losses(bc_fn_static)
    scan_body = make_scan_body(
        train_step_jitted, model, optimiser, current_weights_dict,
        cfg, data_free, compute_losses_fn=compute_losses_fn,
    )

    # --- 10. Training Loop ---
    best_nse_stats = {
        'combined_nse': -jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {},
        'nse_h': -jnp.inf, 'nse_hu': -jnp.inf, 'nse_hv': -jnp.inf, 'rmse': jnp.inf
    }
    best_loss_stats = {'total_weighted_loss': jnp.inf, 'epoch': 0}
    best_params_nse = None
    best_params_loss = None 

    start_time = time.time()
    ckpt_mgr = CheckpointManager(model_dir, model=model)
    val_metrics = {}
    neg_depth = {}
    avg_losses_unweighted = {}
    avg_total_weighted_loss = 0.0
    global_step = 0

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()
            train_key, epoch_key, sample_key = random.split(train_key, 3)

            # --- IMPORTANCE SAMPLING UPDATE ---
            if epoch > 0 and epoch % RESAMPLE_FREQ_EPOCHS == 0:
                print(f"--- Epoch {epoch}: Updating Importance Sampling Pool ---")

                # 0. Resample pool from domain so new high-residual regions are reachable
                train_key, pool_base_key = random.split(train_key)
                new_pool_chunks = []
                for i in range(num_chunks):
                    subkey = random.fold_in(pool_base_key, i)
                    current_chunk_size = min(chunk_size, POOL_SIZE - i * chunk_size)
                    pts = domain_sampler.sample_interior(subkey, current_chunk_size, (0., domain_cfg["t_final"]))
                    new_pool_chunks.append(pts)
                pool_pde = jnp.concatenate(new_pool_chunks, axis=0)
                del new_pool_chunks

                # 1. Evaluate residuals on entire pool via lax.map (single JIT, no CPU round-trip)
                all_residuals = eval_pool_jit(model, params, pool_pde, cfg, EVAL_BATCH_SIZE)

                # 2. Compute sampling probabilities on GPU
                current_probs = compute_sampling_probs(all_residuals, P_ERROR_WEIGHT)

                print(f"    mean_residual={float(jnp.mean(all_residuals)):.3e}, "
                      f"max_residual={float(jnp.max(all_residuals)):.3e}")
                print(f"    Pool probabilities updated.")

            # --- Draw fresh active set every epoch from current pool + probs ---
            current_epoch_pde_pts, current_epoch_pde_weights = sample_from_pool(
                sample_key, pool_pde, current_probs, n_pde
            )

            # --- Generate Data ---
            scan_inputs = generate_epoch_data_jitted(epoch_key, current_epoch_pde_pts, current_epoch_pde_weights)
            
            # --- Train Scan ---
            (params, opt_state), (batch_losses_unweighted_stacked, batch_total_weighted_loss_stacked) = lax.scan(
                scan_body, (params, opt_state), scan_inputs
            )
            
            global_step += num_batches
            
            # --- Aggregate Losses ---
            epoch_losses_unweighted_sum = {k: jnp.sum(v) for k, v in batch_losses_unweighted_stacked.items()}
            epoch_total_weighted_loss_sum = jnp.sum(batch_total_weighted_loss_stacked)

            avg_losses_unweighted = {k: float(v) / num_batches for k, v in epoch_losses_unweighted_sum.items()}
            avg_total_weighted_loss = float(epoch_total_weighted_loss_sum) / num_batches

            # --- Validation ---
            combined_nse_val = -float('inf')
            nse_h_val, nse_hu_val, nse_hv_val = -float('inf'), -float('inf'), -float('inf')
            rmse_h_val, rmse_hu_val, rmse_hv_val = float('inf'), float('inf'), float('inf')

            current_lr = cfg["training"]["learning_rate"]
            try:
                if hasattr(opt_state[-1], 'scale'):
                    current_lr = cfg["training"]["learning_rate"] * float(opt_state[-1].scale)
            except: pass

            if validation_data_loaded:
                try:
                    U_pred = model.apply(params, val_pts_batch, train=False)
                    h_pred = U_pred[..., 0]
                    hu_pred = U_pred[..., 1]
                    hv_pred = U_pred[..., 2]

                    nse_h_val = float(nse(h_pred, val_h_true))
                    nse_hu_val = float(nse(hu_pred, val_hu_true))
                    nse_hv_val = float(nse(hv_pred, val_hv_true))
                    combined_nse_val = (nse_h_val + nse_hu_val + nse_hv_val)/3.0
                    rmse_h_val = float(rmse(h_pred, val_h_true))
                    rmse_hu_val = float(rmse(hu_pred, val_hu_true))
                    rmse_hv_val = float(rmse(hv_pred, val_hv_true))
                except Exception as e:
                    print(f"Validation Error: {e}")

            val_metrics = {
                'nse_h': float(nse_h_val), 'nse_hu': float(nse_hu_val), 'nse_hv': float(nse_hv_val),
                'rmse_h': float(rmse_h_val), 'rmse_hu': float(rmse_hu_val), 'rmse_hv': float(rmse_hv_val),
                'combined_nse': float(combined_nse_val),
            }

            # --- Update Best Model ---
            if combined_nse_val > best_nse_stats['combined_nse']:
                best_nse_stats.update({
                    'combined_nse': combined_nse_val,
                    'nse_h': nse_h_val, 'nse_hu': nse_hu_val, 'nse_hv': nse_hv_val,
                    'rmse': rmse_val, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'total_weighted_loss': avg_total_weighted_loss,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()}
                })
                best_params_nse = jax.tree.map(jnp.copy, params)
                if combined_nse_val > -jnp.inf:
                    print(f"    ---> New Best Combined NSE: {combined_nse_val:.4f}")

            if avg_total_weighted_loss < best_loss_stats['total_weighted_loss']:
                best_loss_stats['total_weighted_loss'] = avg_total_weighted_loss
                best_loss_stats['epoch'] = epoch
                best_params_loss = jax.tree.map(jnp.copy, params)

            # Negative depth diagnostics and checkpoint tracking
            freq = cfg.get("reporting", {}).get("epoch_freq", 100)
            epoch_time = time.time() - epoch_start_time

            neg_depth = {'count': 0, 'fraction': 0.0, 'min': 0.0, 'mean': 0.0}
            if (epoch + 1) % freq == 0:
                try:
                    neg_depth = compute_negative_depth_diagnostics(model, params, scan_inputs['pde'][0])
                except Exception:
                    pass

            saved_events = ckpt_mgr.update(
                epoch, params, opt_state, val_metrics,
                avg_losses_unweighted, avg_total_weighted_loss, cfg_dict, neg_depth
            )
            for event in saved_events:
                event_type, value, ep, prev_value, prev_epoch = event
                if event_type == 'best_nse':
                    console.print_checkpoint_nse(value, ep, prev_value, prev_epoch)
                    aim_tracker.log_best_nse(value, ep, step=global_step)
                elif event_type == 'best_loss':
                    console.print_checkpoint_loss(value, ep, prev_value, prev_epoch)
                    aim_tracker.log_best_loss(value, ep, step=global_step)

            # --- Reporting ---
            if (epoch + 1) % freq == 0:
                console.print_epoch(
                    epoch, cfg["training"]["epochs"],
                    avg_losses_unweighted, avg_total_weighted_loss,
                    current_lr,
                    val_metrics, neg_depth.get('fraction', 0.0), epoch_time
                )

            aim_tracker.log_epoch(
                epoch=epoch, step=global_step,
                losses=avg_losses_unweighted, total_loss=avg_total_weighted_loss,
                val_metrics=val_metrics, lr=current_lr,
                epoch_time=epoch_time, elapsed_time=time.time() - start_time,
                neg_depth=neg_depth if (epoch + 1) % freq == 0 else None,
            )

            # --- Early Stopping ---
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))
            if epoch >= min_epochs and (epoch - best_nse_stats['epoch']) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                break

    except KeyboardInterrupt:
        print("\n--- Training interrupted ---")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    # --- 11. Post-Training ---
    finally:
        total_time = time.time() - start_time

        # Evaluate all physics losses on best-NSE params (even zero-weight terms)
        all_physics_losses = {}
        eval_params = best_params_nse if best_params_nse is not None else params
        try:
            eval_key = random.PRNGKey(0)
            eval_keys = random.split(eval_key, 6)
            n_eval = 200
            t_range = (0., domain_cfg["t_final"])
            eval_batch = {
                'pde': domain_sampler.sample_interior(eval_keys[0], n_eval, t_range),
                'ic': domain_sampler.sample_interior(eval_keys[1], n_eval, (0., 0.)),
                'bc_upstream': domain_sampler.sample_boundary(eval_keys[2], n_eval, t_range, 'upstream'),
                'bc_wall': domain_sampler.sample_boundary(eval_keys[3], n_eval, t_range, 'wall'),
                'bc_building': domain_sampler.sample_boundary(eval_keys[4], n_eval, t_range, 'building'),
                'data': jnp.empty((0, 6), dtype=get_dtype()),
            }
            all_physics_losses = compute_losses_fn(model, eval_params, eval_batch, cfg, data_free=True)
            all_physics_losses = {k: float(v) for k, v in all_physics_losses.items()}
            print(f"\nAll physics losses (best-NSE params):")
            for k, v in all_physics_losses.items():
                print(f"  {k}: {v:.6e}")
        except Exception as e:
            print(f"Warning: Failed to evaluate all physics losses: {e}")

        # Merge all_physics_losses into final losses for the checkpoint
        final_losses_for_ckpt = dict(avg_losses_unweighted)
        for k, v in all_physics_losses.items():
            if k not in final_losses_for_ckpt:
                final_losses_for_ckpt[k] = v

        ckpt_mgr.save_final(epoch if 'epoch' in locals() else 0, params, opt_state, val_metrics, final_losses_for_ckpt, avg_total_weighted_loss, cfg_dict, neg_depth)

        best_nse_ckpt = ckpt_mgr.get_best_nse_stats()
        best_loss_ckpt = ckpt_mgr.get_best_loss_stats()

        console.print_completion_summary(
            total_time=total_time,
            final_epoch=epoch if 'epoch' in locals() else 0,
            best_nse_stats=best_nse_ckpt,
            best_loss_stats=best_loss_ckpt,
            final_losses=final_losses_for_ckpt,
            final_val_metrics=val_metrics,
            neg_depth_final=neg_depth,
            neg_depth_best_nse={},
            neg_depth_best_loss={},
            final_lr=current_lr if 'current_lr' in locals() else cfg["training"]["learning_rate"],
        )

        final_params = best_params_loss if best_params_loss is not None else best_params_nse

        summary_metrics = {
            'best_validation_model': best_nse_stats,
            'best_loss_model': best_loss_stats,
            'final_system': {
                'total_training_time_seconds': total_time,
                'total_epochs_run': (epoch + 1) if 'epoch' in locals() else 0,
                'total_steps_run': global_step
            }
        }
        if all_physics_losses:
            summary_metrics['all_physics_losses'] = all_physics_losses
        aim_tracker.log_summary(summary_metrics)

        if ask_for_confirmation():
            if final_params is not None:
                saved_model_path = save_model(final_params, model_dir, trial_name)

                if aim_tracker.enabled and saved_model_path:
                    aim_tracker.log_artifact(saved_model_path, 'model_weights.pkl')

                print("Generating plots...")
                t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=get_dtype())
                output_points = []

                output_csv_path = resolve_scenario_asset_path(
                    base_data_path, scenario_name, "output_reference", required=False
                )
                if os.path.exists(output_csv_path):
                     try:
                        df_out = pd.read_csv(output_csv_path)
                        df_out.columns = [c.strip() for c in df_out.columns]
                        if 'X' in df_out.columns and 'Y' in df_out.columns:
                            for idx, row in df_out.iterrows():
                                output_points.append((row['X'], row['Y'], f"Point_{idx+1}"))
                     except: pass

                if not output_points:
                    cx, cy = (x_max+x_min)/2, (y_max+y_min)/2
                    output_points = [(cx, cy, "Center_Point")]

                for px, py, pname in output_points:
                    pts = jnp.stack([jnp.full_like(t_plot, px), jnp.full_like(t_plot, py), t_plot], axis=-1)
                    U = model.apply(final_params, pts, train=False)
                    min_depth_plot = cfg.get("numerics", {}).get("min_depth", 0.0)
                    h_pred = jnp.where(U[..., 0] < min_depth_plot, 0.0, U[..., 0])

                    plt.figure(figsize=(10, 6))
                    if loaded_val_data is not None:
                        dists = jnp.sqrt((loaded_val_data[:, 1] - px)**2 + (loaded_val_data[:, 2] - py)**2)
                        mask = dists < 2.0
                        subset = loaded_val_data[mask]
                        if subset.shape[0] > 0:
                            subset = subset[jnp.argsort(subset[:, 0])]
                            plt.plot(subset[:, 0], subset[:, 3], 'k--', label='Baseline', alpha=0.7)

                    plt.plot(t_plot, h_pred, label=f'Predicted h')
                    plt.title(f'{pname} ({px:.1f}, {py:.1f})')
                    plt.legend()
                    plt.grid(True)
                    path = os.path.join(results_dir, f"{pname}_timeseries.png")
                    plt.savefig(path)
                    plt.close()
                    aim_tracker.log_image(path, f"{pname}_timeseries.png", epoch if 'epoch' in locals() else 0)

                print(f"Plots saved to {results_dir}")

        aim_tracker.close()

    return best_nse_stats['combined_nse']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Experiment 6 - Importance Sampling V2).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)
