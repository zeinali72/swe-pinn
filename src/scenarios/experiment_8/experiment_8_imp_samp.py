import os
import sys
import time
import copy
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
from aim import Repo, Run, Image
from flax.core import FrozenDict
import flax.linen as nn
import numpy as np 
import matplotlib
matplotlib.use('Agg') # Ensure headless plotting
import matplotlib.pyplot as plt 

# Local application imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.config import load_config, DTYPE
from src.data import (
    get_batches_tensor,
    get_sample_count,
    load_boundary_condition,
    IrregularDomainSampler,
    load_bathymetry,
    bathymetry_fn,
    load_validation_data,
)
from src.models import init_model
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet_hu,
    loss_boundary_dirichlet_hv,
    loss_slip_wall_generalized,
    compute_neg_h_loss,
    compute_data_loss,
    total_loss
)
from src.utils import ( 
   nse, rmse, generate_trial_name, save_model, ask_for_confirmation
)

from src.reporting import (
    print_epoch_stats, log_metrics, print_final_summary
)

from src.physics import SWEPhysics 

# ==============================================================================
# Helper: Vectorized Residual Calculation for Importance Sampling
# ==============================================================================
def compute_pde_residual_vector(model: nn.Module, params: Dict[str, Any], pde_batch: jnp.ndarray,
                                config: FrozenDict) -> jnp.ndarray:
    """
    Computes the scalar residual error for EACH point in the batch.
    Used for Importance Sampling evaluation.
    Returns: (Batch_Size,) array of errors.
    """
    # Create physics instance
    U_pred = model.apply({'params': params['params']}, pde_batch, train=False)
    
    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)

    # Calculate Gradients
    jac_U = jax.vmap(jax.jacfwd(U_fn))(pde_batch)
    dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]

    x_batch = pde_batch[..., 0]
    y_batch = pde_batch[..., 1]

    # Bathymetry gradients
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

    # Residual vector: [N, 3]
    residual = (dU_dt + div_F + div_G - S)
    
    # Mask out dry areas for stability (h < eps)
    h_mask = jnp.where(U_pred[..., 0] < eps, 0.0, 1.0)
    final_residual = residual * h_mask[..., None]
    
    # Reduce to scalar error per point: Sum of squared errors of the 3 equations
    # Shape: (N,)
    error_per_point = jnp.sum(final_residual ** 2, axis=1)
    
    return error_per_point

# JIT the residual function for fast evaluation
get_residuals_jitted = jax.jit(compute_pde_residual_vector, static_argnums=(0, 3))

# ==============================================================================
# Helper: Weighted PDE Loss (Corrected for Importance Sampling)
# ==============================================================================
def compute_weighted_pde_loss(model: nn.Module, params: FrozenDict, pde_batch: jnp.ndarray, 
                              weights: jnp.ndarray, config: FrozenDict) -> float:
    """
    Computes the weighted PDE loss.
    Essential for Unbiased Importance Sampling.
    """
    U_pred = model.apply(params, pde_batch, train=True)
    
    def U_fn(pts):
        return model.apply(params, pts, train=True)

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

    residual = (dU_dt + div_F + div_G - S)
    h_mask = jnp.where(U_pred[..., 0] < eps, 0.0, 1.0)
    final_residual = residual * h_mask[..., None]
    
    # Squared error per point
    squared_error = jnp.sum(final_residual ** 2, axis=1)
    
    # Apply Importance Weights: Mean(Error * Weights)
    weighted_mse = jnp.mean(squared_error * weights)
    
    return weighted_mse

# ==============================================================================
# Training Step
# ==============================================================================

def train_step(
        model: Any, 
        optimiser: optax.GradientTransformation, 
        params: FrozenDict, 
        opt_state: optax.OptState, 
        batch: Dict[str, jnp.ndarray], 
        config: Dict[str, Any],
        data_free: bool,
        bc_fn_static: Any,
        weights_dict: FrozenDict 
        ) -> Tuple[FrozenDict, optax.OptState, Dict[str, float], float]:
    
    active_loss_keys_base = list(weights_dict.keys())

    def loss_fn(params):
        
        # --- 1. PDE Loss (Weighted for IS) ---
        pde_pts = batch['pde']
        # Default to ones if not present
        pde_weights = batch.get('pde_weights', jnp.ones(pde_pts.shape[0]))
        
        loss_pde = compute_weighted_pde_loss(model, params, pde_pts, pde_weights, config)
        loss_neg_h = compute_neg_h_loss(model, params, batch['pde'])
        
        # --- 2. Initial Condition Loss ---
        U_ic = model.apply(params, batch['ic'], train=True)        
        loss_ic = jnp.mean(U_ic[..., 0]**2) + jnp.mean(U_ic[..., 1]**2 + U_ic[..., 2]**2)

        # --- 3. Boundary Conditions ---
        if batch['bc_upstream'].shape[0] > 0:
            t_inflow = batch['bc_upstream'][..., 2]
            Q_target = bc_fn_static(t_inflow) 
            flux_target_x = Q_target / 372.92  
            loss_inflow_x = loss_boundary_dirichlet_hu(model, params, batch['bc_upstream'], flux_target_x)
            loss_inflow_y = loss_boundary_dirichlet_hv(model, params, batch['bc_upstream'], jnp.zeros_like(flux_target_x))
            loss_bc_inflow = loss_inflow_x + loss_inflow_y
        else:
            loss_bc_inflow = 0.0

        loss_bc_wall = loss_slip_wall_generalized(model, params, batch['bc_wall'])
        loss_bldg = loss_slip_wall_generalized(model, params, batch['bc_building'])
        total_bc = loss_bc_inflow + loss_bc_wall + loss_bldg

        # Data Loss
        data_batch_data = batch.get('data', jnp.empty((0,6), dtype=DTYPE))
        if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0:
             loss_data = compute_data_loss(model, params, data_batch_data, config)
        else:
             loss_data = 0.0

        terms = {
            'pde': loss_pde,
            'neg_h': loss_neg_h,
            'ic': loss_ic,
            'bc': total_bc, 
            'data': loss_data,
            'building': loss_bldg 
        }

        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params, value=loss_val)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics, loss_val

# JIT Compile
train_step_jitted = jax.jit(train_step, static_argnames=['model', 'optimiser', 'config', 'bc_fn_static', 'weights_dict', 'data_free'])

# ==============================================================================
# Main Execution
# ==============================================================================

def main(config_path: str):
    """
    Main training loop for Benchmark Test 6 Scenario (Experiment 6).
    Includes Importance Sampling for PDE points (Memory Optimized).
    """
    #--- 1. LOAD CONFIGURATION (MUTABLE) ---
    cfg_dict = load_config(config_path)
    
    print("Info: Running Benchmark Test 6 (Experiment 6 - Importance Sampling V3 - Configurable)...")

    # --- 2. SETUP DATA & COMPUTE DOMAIN EXTENT ---
    scenario_name = cfg_dict.get('scenario', 'experiment_6')
    base_data_path = os.path.join("data", scenario_name)

    artifacts_path = os.path.join(base_data_path, "domain_artifacts.npz")
    if not os.path.exists(artifacts_path):
        artifacts_path = os.path.join(base_data_path, "domain.npz")
        
    if not os.path.exists(artifacts_path):
        print(f"Error: Domain artifacts file not found at {artifacts_path}")
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
    dem_path = os.path.join(base_data_path, "DEM_v2_asc.asc")
    if not os.path.exists(dem_path):
        print(f"Error: DEM file not found at {dem_path}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)
    
    bc_csv_path = os.path.join(base_data_path, "Test6_BC_interpolated.csv")
    if not os.path.exists(bc_csv_path):
        print(f"Error: Boundary condition CSV file not found at {bc_csv_path}.")
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
                data_points_full = jnp.load(training_data_file).astype(DTYPE) 
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
            _, val_pts_batch, val_targets = load_validation_data(validation_data_file, dtype=DTYPE)
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

    # --- 7. Initialize Aim ---
    aim_repo = None
    aim_run = None
    try:
        aim_repo_path = "aim_repo"
        os.makedirs(aim_repo_path, exist_ok=True)
        aim_repo = Repo(path=aim_repo_path, init=True)
        aim_run = Run(repo=aim_repo, experiment=trial_name)
        
        artifact_storage_path = os.path.join(aim_repo_path, "aim_artifacts")
        os.makedirs(artifact_storage_path, exist_ok=True)
        abs_artifact_path = os.path.abspath(artifact_storage_path)
        aim_run.set_artifacts_uri(f"file://{abs_artifact_path}")
        
        aim_run["hparams"] = cfg_dict
        aim_run['flags'] = {"scenario_type": "experiment_6_IS_Configurable"}
    except Exception as e:
        print(f"Warning: Aim tracking failed to initialize: {e}")

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
    # Handle remainder if POOL_SIZE not div by chunk_size
    if POOL_SIZE % chunk_size != 0: num_chunks += 1
    
    pool_base_key = random.PRNGKey(12345)
    
    for i in range(num_chunks):
        subkey = random.fold_in(pool_base_key, i)
        # Determine actual size for this chunk
        current_chunk_size = min(chunk_size, POOL_SIZE - (i * chunk_size))
        
        pts = domain_sampler.sample_interior(subkey, current_chunk_size, (0., domain_cfg["t_final"]))
        pts_cpu = np.array(pts)
        pool_chunks.append(pts_cpu)
        if (i+1) % 2 == 0:
            print(f"  Generated {(i+1)*chunk_size} points...")
            
    pool_pde_cpu = np.concatenate(pool_chunks, axis=0)
    print(f"Pool Generation Complete. Stored on CPU. Shape: {pool_pde_cpu.shape}")
    del pool_chunks, pts
    
    # Initial Active Set (Random Selection)
    np_rng = np.random.default_rng(42)
    active_indices = np_rng.choice(POOL_SIZE, size=n_pde, replace=False)
    
    active_pde_pts = jnp.array(pool_pde_cpu[active_indices])
    # Initialize weights to 1.0 (Uniform sampling initially)
    active_pde_weights = jnp.ones((n_pde,), dtype=DTYPE)

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
        # PDE: Use provided active set (assumed shuffled)
        pde_data = current_pde_pts.reshape((num_batches, batch_size, 3))
        pde_w = current_pde_weights.reshape((num_batches, batch_size))
        
        ic_pts = domain_sampler.sample_interior(k2, n_ic, (0., 0.))
        ic_data = get_batches_tensor(k2, ic_pts, batch_size, num_batches)
        
        bc_upstream_pts = domain_sampler.sample_boundary(k3, n_bc_upstream, (0., domain_cfg["t_final"]), 'upstream')
        bc_upstream = get_batches_tensor(k3, bc_upstream_pts, batch_size, num_batches)

        bc_wall_pts = domain_sampler.sample_boundary(k4, n_bc_wall, (0., domain_cfg["t_final"]), 'wall')
        bc_wall = get_batches_tensor(k4, bc_wall_pts, batch_size, num_batches)
        
        bc_building_pts = domain_sampler.sample_boundary(k5, n_building, (0., domain_cfg["t_final"]), 'building')
        bc_building = get_batches_tensor(k5, bc_building_pts, batch_size, num_batches)
        
        if not data_free and data_points_full is not None:
             data_d = get_batches_tensor(k5, data_points_full, batch_size, num_batches)
        else:
             data_d = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)

        return {
            'pde': pde_data, 'pde_weights': pde_w,
            'ic': ic_data, 
            'bc_upstream': bc_upstream,
            'bc_wall': bc_wall, 
            'bc_building': bc_building,
            'data': data_d
        }

    generate_epoch_data_jitted = jax.jit(generate_epoch_data_with_IS)

    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        current_all_batches = {
            'pde': batch_data['pde'],
            'pde_weights': batch_data['pde_weights'], 
            'ic': batch_data['ic'],
            'bc_upstream': batch_data['bc_upstream'],
            'bc_wall': batch_data['bc_wall'], 
            'bc_building': batch_data['bc_building'],
            'data': batch_data['data']
        }
        new_params, new_opt_state, terms, total = train_step_jitted(
            model, optimiser, curr_params, curr_opt_state,
            current_all_batches, cfg, data_free, bc_fn_static, current_weights_dict
        )
        return (new_params, new_opt_state), (terms, total)

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
    global_step = 0

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()
            train_key, epoch_key, shuffle_key = random.split(train_key, 3)
            
            # --- IMPORTANCE SAMPLING UPDATE ---
            if epoch > 0 and epoch % RESAMPLE_FREQ_EPOCHS == 0:
                print(f"--- Epoch {epoch}: Updating Importance Sampling Pool ---")
                
                # 1. Evaluate residuals on the WHOLE pool
                num_eval_batches = int(np.ceil(POOL_SIZE / EVAL_BATCH_SIZE))
                all_residuals_list = []
                
                for i in range(num_eval_batches):
                    idx_start = i * EVAL_BATCH_SIZE
                    idx_end = min((i + 1) * EVAL_BATCH_SIZE, POOL_SIZE)
                    
                    batch_pts_cpu = pool_pde_cpu[idx_start:idx_end]
                    batch_pts_gpu = jax.device_put(batch_pts_cpu)
                    
                    batch_errs_gpu = get_residuals_jitted(model, params, batch_pts_gpu, cfg)
                    batch_errs_cpu = np.array(batch_errs_gpu.block_until_ready())
                    
                    all_residuals_list.append(batch_errs_cpu)
                
                all_residuals = np.concatenate(all_residuals_list, axis=0)
                
                # 2. Compute Selection Probabilities (CPU)
                err_sum = np.sum(all_residuals)
                if err_sum < 1e-9:
                    probs = np.ones(POOL_SIZE) / POOL_SIZE
                else:
                    # Use configurable alpha
                    alpha = P_ERROR_WEIGHT 
                    p_error = all_residuals / err_sum
                    p_uniform = 1.0 / POOL_SIZE
                    probs = alpha * p_error + (1 - alpha) * p_uniform
                
                # 3. Sample indices
                new_indices = np_rng.choice(POOL_SIZE, size=n_pde, p=probs, replace=False)

                # 4. Compute Importance Weights
                # w_i = 1 / (N * p_i)
                selected_probs = probs[new_indices]
                weights_unnormalized = 1.0 / (POOL_SIZE * selected_probs)
                
                # Normalize weights to have mean 1.0
                weights_normalized = weights_unnormalized / np.mean(weights_unnormalized)
                
                # 5. Update Active Set
                active_pde_pts = jnp.array(pool_pde_cpu[new_indices])
                active_pde_weights = jnp.array(weights_normalized)
                
                mean_res = np.mean(all_residuals)
                max_res = np.max(all_residuals)
                print(f"    IS Update Stats: Mean Error={mean_res:.6f}, Max Error={max_res:.6f}, Max Weight={np.max(weights_normalized):.2f}")
                print(f"    Resampled {n_pde} points.")

            # --- Shuffle Active Set ---
            shuffled_indices = random.permutation(shuffle_key, n_pde)
            limit = num_batches * batch_size
            shuffled_indices = shuffled_indices[:limit]
            
            current_epoch_pde_pts = active_pde_pts[shuffled_indices]
            current_epoch_pde_weights = active_pde_weights[shuffled_indices]
            
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
            rmse_val = float('inf')
            
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

                    nse_h_val = float(nse(val_h_true, h_pred))
                    nse_hu_val = float(nse(val_hu_true, hu_pred))
                    nse_hv_val = float(nse(val_hv_true, hv_pred))
                    combined_nse_val = (nse_h_val + nse_hu_val + nse_hv_val)/3.0
                    rmse_val = float(rmse(val_h_true, h_pred))
                except Exception as e:
                    print(f"Validation Error: {e}")

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
                best_params_nse = copy.deepcopy(params)
                if combined_nse_val > -jnp.inf:
                    print(f"    ---> New Best Combined NSE: {combined_nse_val:.4f}")
            
            if avg_total_weighted_loss < best_loss_stats['total_weighted_loss']:
                best_loss_stats['total_weighted_loss'] = avg_total_weighted_loss
                best_loss_stats['epoch'] = epoch
                best_params_loss = copy.deepcopy(params)

            # --- Reporting ---
            freq = cfg.get("reporting", {}).get("epoch_freq", 100)
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % freq == 0:
                print_epoch_stats(
                    epoch, global_step, start_time, avg_total_weighted_loss,
                    avg_losses_unweighted.get('pde', 0.0), 
                    avg_losses_unweighted.get('ic', 0.0), 
                    avg_losses_unweighted.get('bc', 0.0),
                    0.0, 
                    avg_losses_unweighted.get('data', 0.0),
                    avg_losses_unweighted.get('neg_h', 0.0),
                    combined_nse_val, rmse_val, epoch_time
                )

            if aim_run:
                epoch_metrics_to_log = {
                    'validation_metrics': {'nse': combined_nse_val, 'combined_nse': combined_nse_val, 'rmse': rmse_val},
                    'epoch_avg_losses': avg_losses_unweighted,
                    'epoch_avg_total_weighted_loss': avg_total_weighted_loss,
                    'optimization': {'total_loss': avg_total_weighted_loss}, 
                    'system_metrics': {'epoch_time': epoch_time},
                    'training_metrics': {'learning_rate': float(current_lr)}
                }
                log_metrics(aim_run, step=global_step, epoch=epoch, metrics=epoch_metrics_to_log)

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
        print("\n" + "="*50)
        print("                 FINAL SUMMARY")
        print("="*50)
        print(f"Total Time: {total_time:.2f}s")
        print(f"Best NSE: {best_nse_stats['combined_nse']:.6f} at epoch {best_nse_stats['epoch']+1}")
        
        final_params = best_params_loss if best_params_loss is not None else best_params_nse

        if aim_run:
            aim_run['summary'] = {
                'best_validation_model': best_nse_stats,
                'final_system': {'total_training_time_seconds': total_time, 'total_epochs_run': (epoch + 1)}
            }

        if ask_for_confirmation():
            if final_params is not None:
                save_model(final_params, model_dir, trial_name)
                
                print("Generating plots...")
                t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=DTYPE)
                output_points = []
                
                output_csv_path = os.path.join(base_data_path, "Test6output.csv")
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
                    h_pred = U[..., 0]
                    
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
                    if aim_run: aim_run.track(Image(path), name=f"{pname}_timeseries.png")
                
                print(f"Plots saved to {results_dir}")

        if aim_run: aim_run.close()

    return best_nse_stats['combined_nse']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Experiment 6 - Importance Sampling V2).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)