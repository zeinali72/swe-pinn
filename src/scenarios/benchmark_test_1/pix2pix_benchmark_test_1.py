import os
import time
import copy
import argparse
from functools import partial
from typing import Any, Dict, Tuple, List, Optional

import jax
import jax.numpy as jnp
from jax import random, lax
import optax
from aim import Repo, Run, Image
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from src.config import load_config, DTYPE
from src.data import bathymetry_fn, load_boundary_condition, load_bathymetry
from src.utils import generate_trial_name, save_model, ask_for_confirmation
from src.reporting import print_epoch_stats, print_final_summary

# ==============================================================================
# 1. MODELS
# ==============================================================================

class GN(nn.Module):
    max_groups: int = 32
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        c = x.shape[-1]
        g = 1
        for cand in range(min(self.max_groups, c), 0, -1):
            if c % cand == 0:
                g = cand
                break
        return nn.GroupNorm(num_groups=g, epsilon=self.eps)(x)

class ResBlock(nn.Module):
    filters: int
    act: Any = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        y = nn.Conv(self.filters, kernel_size=(3, 3), strides=1, padding="SAME")(x)
        y = GN()(y)
        y = self.act(y)
        y = nn.Conv(self.filters, kernel_size=(3, 3), strides=1, padding="SAME")(y)
        y = GN()(y)
        return self.act(residual + y)

class Pix2PixGenerator(nn.Module):
    output_dim: int
    filters: int = 64
    n_downsample: int = 3
    n_res_blocks: int = 6
    lx: float = 700.0
    ly: float = 100.0
    t_final: float = 72000.0
    z_min: float = 0.0
    z_max: float = 10.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Added training arg to fix TypeError
        
        # Normalize inputs [-1, 1]
        x_norm = 2.0 * x[..., 0:1] / self.lx - 1.0
        y_norm = 2.0 * x[..., 1:2] / self.ly - 1.0
        t_norm = 2.0 * x[..., 2:3] / self.t_final - 1.0
        
        z_range = (self.z_max - self.z_min) + 1e-6
        z_norm = 2.0 * (x[..., 3:4] - self.z_min) / z_range - 1.0
        
        h_in = jnp.concatenate([x_norm, y_norm, t_norm, z_norm], axis=-1)

        # Network
        h_net = nn.Conv(self.filters, kernel_size=(7, 7), strides=1, padding="SAME")(h_in)
        h_net = GN()(h_net)
        h_net = nn.relu(h_net)

        curr_filters = self.filters
        for _ in range(self.n_downsample):
            curr_filters *= 2
            h_net = nn.Conv(curr_filters, kernel_size=(3, 3), strides=2, padding="SAME")(h_net)
            h_net = GN()(h_net)
            h_net = nn.relu(h_net)

        for _ in range(self.n_res_blocks):
            h_net = ResBlock(curr_filters)(h_net)

        for _ in range(self.n_downsample):
            curr_filters //= 2
            h_net = nn.ConvTranspose(curr_filters, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(h_net)
            h_net = GN()(h_net)
            h_net = nn.relu(h_net)

        out_raw = nn.Conv(self.output_dim, kernel_size=(7, 7), strides=1, padding="SAME")(h_net)
        
        # Physics constraints: h >= 0
        h_out = nn.softplus(out_raw[..., 0:1] + 2.0) 
        hu_out = out_raw[..., 1:2]
        hv_out = out_raw[..., 2:3]
        
        return jnp.concatenate([h_out, hu_out, hv_out], axis=-1)

# ==============================================================================
# 2. CAUSAL DATA GENERATOR
# ==============================================================================

class CausalDataGenerator:
    def __init__(self, cfg: Dict[str, Any]):
        self.lx = float(cfg['domain']['lx'])
        self.ly = float(cfg['domain']['ly'])
        self.t_final = float(cfg['domain']['t_final'])
        
        self.num_chunks = int(cfg['training'].get('causal_chunks', 4))
        
        data_cfg = cfg.get("data", {})
        self.nx = int(data_cfg.get("nx", 256))
        self.ny = int(data_cfg.get("ny", 64))
        
        x = jnp.linspace(0.0, self.lx, self.nx, dtype=DTYPE)
        y = jnp.linspace(0.0, self.ly, self.ny, dtype=DTYPE)
        
        self.x_coords = x
        self.y_coords = y
        
        self.X, self.Y = jnp.meshgrid(x, y)
        self.dx = float(x[1] - x[0])
        self.dy = float(y[1] - y[0])
        
        z_vals, _, _ = bathymetry_fn(self.X.flatten(), self.Y.flatten())
        self.Z = z_vals.reshape((self.ny, self.nx, 1)).astype(DTYPE)
        self.z_min, self.z_max = float(jnp.min(self.Z)), float(jnp.max(self.Z))

        self.dz_dx, self.dz_dy = self._compute_static_gradients()

    def _compute_static_gradients(self):
        pad_x = jnp.pad(self.Z, ((0,0), (1,1), (0,0)), mode='edge')
        pad_y = jnp.pad(self.Z, ((1,1), (0,0), (0,0)), mode='edge')
        dz_dx = (pad_x[:, 2:, :] - pad_x[:, :-2, :]) / (2.0 * self.dx)
        dz_dy = (pad_y[2:, :, :] - pad_y[:-2, :, :]) / (2.0 * self.dy)
        return dz_dx, dz_dy

    def get_causal_batch(self, key, batch_size_per_chunk: int):
        keys = random.split(key, self.num_chunks)
        dt_chunk = self.t_final / self.num_chunks
        
        chunks = []
        for i in range(self.num_chunks):
            t_start = i * dt_chunk
            t_end = (i + 1) * dt_chunk
            
            t = random.uniform(keys[i], (batch_size_per_chunk, 1, 1, 1), 
                               minval=t_start, maxval=t_end, dtype=DTYPE)
            
            X_b = jnp.broadcast_to(self.X[None, ..., None], (batch_size_per_chunk, self.ny, self.nx, 1))
            Y_b = jnp.broadcast_to(self.Y[None, ..., None], (batch_size_per_chunk, self.ny, self.nx, 1))
            Z_b = jnp.broadcast_to(self.Z[None, ...], (batch_size_per_chunk, self.ny, self.nx, 1))
            T_b = jnp.broadcast_to(t, (batch_size_per_chunk, self.ny, self.nx, 1))
            
            chunk = jnp.concatenate([X_b, Y_b, T_b, Z_b], axis=-1)
            chunks.append(chunk)
            
        return jnp.stack(chunks)

    def get_ic_batch(self, batch_size: int):
        t = jnp.zeros((batch_size, 1, 1, 1), dtype=DTYPE)
        X_b = jnp.broadcast_to(self.X[None, ..., None], (batch_size, self.ny, self.nx, 1))
        Y_b = jnp.broadcast_to(self.Y[None, ..., None], (batch_size, self.ny, self.nx, 1))
        Z_b = jnp.broadcast_to(self.Z[None, ...], (batch_size, self.ny, self.nx, 1))
        T_b = jnp.broadcast_to(t, (batch_size, self.ny, self.nx, 1))
        return jnp.concatenate([X_b, Y_b, T_b, Z_b], axis=-1)

# ==============================================================================
# 3. VALIDATION METRICS
# ==============================================================================

def _preprocess_val_data(val_data, data_gen, n_samples: int = 50):
    val_np = np.array(val_data)
    unique_times = np.unique(val_np[:, 0])
    if len(unique_times) > n_samples:
        idx = np.linspace(0, len(unique_times) - 1, n_samples).astype(int)
        sampled_times = unique_times[idx]
    else:
        sampled_times = unique_times

    gauge_coords = np.unique(val_np[:, 1:3], axis=0) 

    x_coords_np = np.array(data_gen.x_coords)
    y_coords_np = np.array(data_gen.y_coords)

    gauge_ix = np.abs(x_coords_np[None, :] - gauge_coords[:, 0:1]).argmin(axis=1)
    gauge_iy = np.abs(y_coords_np[None, :] - gauge_coords[:, 1:2]).argmin(axis=1)
    gauge_pixel_indices = np.stack([gauge_iy, gauge_ix], axis=1)

    h_true_lookup = {}
    for row in val_np:
        t, gx, gy, h = row[0], row[1], row[2], row[3]
        h_true_lookup[(t, gx, gy)] = h

    return sampled_times, gauge_coords, h_true_lookup, gauge_pixel_indices

def compute_validation_metrics_fast(model, params, data_gen, val_data, n_samples: int = 50):
    if val_data is None or len(val_data) == 0:
        return -jnp.inf, jnp.inf

    sampled_times, gauge_coords, h_true_lookup, gauge_pixel_indices = \
        _preprocess_val_data(val_data, data_gen, n_samples)

    n_times = len(sampled_times)
    
    X_grid = data_gen.X[None, ..., None]
    Y_grid = data_gen.Y[None, ..., None]
    Z_grid = data_gen.Z[None, ...]

    X_batch = jnp.broadcast_to(X_grid, (n_times, data_gen.ny, data_gen.nx, 1))
    Y_batch = jnp.broadcast_to(Y_grid, (n_times, data_gen.ny, data_gen.nx, 1))
    Z_batch = jnp.broadcast_to(Z_grid, (n_times, data_gen.ny, data_gen.nx, 1))
    T_batch = jnp.array(sampled_times)[:, None, None, None]
    T_batch = jnp.broadcast_to(T_batch, (n_times, data_gen.ny, data_gen.nx, 1))

    inputs = jnp.concatenate([X_batch, Y_batch, T_batch, Z_batch], axis=-1).astype(DTYPE)

    # Pass training=False here
    outputs = model.apply({'params': params}, inputs, training=False)
    h_pred_grid = np.array(outputs[..., 0])

    h_preds_list = []
    h_trues_list = []
    
    for gi, (gx, gy) in enumerate(gauge_coords):
        iy, ix = gauge_pixel_indices[gi]
        for ti, t_val in enumerate(sampled_times):
            key = (t_val, gx, gy)
            if key in h_true_lookup:
                h_preds_list.append(h_pred_grid[ti, iy, ix])
                h_trues_list.append(h_true_lookup[key])

    if len(h_preds_list) == 0:
        return -jnp.inf, jnp.inf

    h_preds = np.array(h_preds_list, dtype=np.float32)
    h_trues = np.array(h_trues_list, dtype=np.float32)

    rmse_val = np.sqrt(np.mean((h_preds - h_trues) ** 2))
    ss_res = np.sum((h_preds - h_trues) ** 2)
    ss_tot = np.sum((h_trues - np.mean(h_trues)) ** 2)
    
    if ss_tot < 1e-9:
        nse_val = -jnp.inf 
    else:
        nse_val = 1.0 - ss_res / (ss_tot + 1e-10)

    return float(nse_val), float(rmse_val)

# ==============================================================================
# 4. LOSS FUNCTIONS
# ==============================================================================

def fd_gradient(u, dx, dy):
    """Finite Difference Spatial Gradients (Convolution)."""
    u_pad_x = jnp.pad(u, ((0,0), (0,0), (1,1), (0,0)), mode='edge')
    u_pad_y = jnp.pad(u, ((0,0), (1,1), (0,0), (0,0)), mode='edge')
    
    du_dx = (u_pad_x[:, :, 2:, :] - u_pad_x[:, :, :-2, :]) / (2.0 * dx)
    du_dy = (u_pad_y[:, 2:, :, :] - u_pad_y[:, :-2, :, :]) / (2.0 * dy)
    return du_dx, du_dy

def compute_chunk_residual(params, model_apply, chunk_batch, dx, dy, dz_dx, dz_dy, config):
    tangent_t = jnp.concatenate([jnp.zeros_like(chunk_batch[..., :2]), 
                                 jnp.ones_like(chunk_batch[..., 2:3]), 
                                 jnp.zeros_like(chunk_batch[..., 3:])], axis=-1)
    
    # Pass training=True during JVP
    def model_fn(x): return model_apply({'params': params}, x, training=True)
    U, dU_dt = jax.jvp(model_fn, (chunk_batch,), (tangent_t,))
    
    h, hu, hv = U[..., 0:1], U[..., 1:2], U[..., 2:3]
    
    dh_dx, dh_dy = fd_gradient(h, dx, dy)
    dhu_dx, dhu_dy = fd_gradient(hu, dx, dy)
    dhv_dx, dhv_dy = fd_gradient(hv, dx, dy)
    
    g = float(config['physics']['g'])
    n_m = float(config['physics']['n_manning'])
    eps = float(config['numerics']['eps'])
    
    h_safe = jnp.maximum(h, eps)
    u, v = hu / h_safe, hv / h_safe
    vel = jnp.sqrt(u**2 + v**2 + eps)
    
    F_hu = hu * u + 0.5 * g * h**2
    G_hu = hu * v
    F_hv = hu * v
    G_hv = hv * v + 0.5 * g * h**2
    
    dFhu_dx, _ = fd_gradient(F_hu, dx, dy)
    _, dGhu_dy = fd_gradient(G_hu, dx, dy)
    dFhv_dx, _ = fd_gradient(F_hv, dx, dy)
    _, dGhv_dy = fd_gradient(G_hv, dx, dy)
    
    S_fx = g * n_m**2 * hu * vel / (h_safe**(4/3))
    S_fy = g * n_m**2 * hv * vel / (h_safe**(4/3))
    
    dz_dx_b = jnp.broadcast_to(dz_dx[None, ...], h.shape)
    dz_dy_b = jnp.broadcast_to(dz_dy[None, ...], h.shape)
    
    src_x = -g * h * dz_dx_b - S_fx
    src_y = -g * h * dz_dy_b - S_fy
    
    res_h = dU_dt[..., 0:1] + dhu_dx + dhv_dy
    res_hu = dU_dt[..., 1:2] + dFhu_dx + dGhu_dy - src_x
    res_hv = dU_dt[..., 2:3] + dFhv_dx + dGhv_dy - src_y
    
    mask = jnp.ones_like(res_h)
    mask = mask.at[:, 0, :, :].set(0).at[:, -1, :, :].set(0)
    mask = mask.at[:, :, 0, :].set(0).at[:, :, -1, :].set(0)
    
    loss_chunk = jnp.mean((res_h * mask)**2 + (res_hu * mask)**2 + (res_hv * mask)**2)
    return loss_chunk

def compute_causal_loss(params, model_apply, causal_batches, ic_batch, bc_fn, data_gen, config):
    loss_causal_total = 0.0
    cumulative_loss = 0.0
    epsilon = float(config['training']['causal_epsilon'])
    
    dz_dx, dz_dy = data_gen.dz_dx, data_gen.dz_dy
    dx, dy = data_gen.dx, data_gen.dy
    
    num_chunks = causal_batches.shape[0]
    
    for i in range(num_chunks):
        batch_i = causal_batches[i]
        L_i = compute_chunk_residual(params, model_apply, batch_i, dx, dy, dz_dx, dz_dy, config)
        
        if i == 0:
            w_i = 1.0
        else:
            w_i = jnp.exp(-epsilon * cumulative_loss)
            w_i = lax.stop_gradient(w_i)
        
        loss_causal_total += w_i * L_i
        cumulative_loss += lax.stop_gradient(L_i)
        
    # IC (Training=True)
    U_ic = model_apply({'params': params}, ic_batch, training=True)
    h_target = jnp.maximum(0.0, float(config['physics']['ic_eta0']) - ic_batch[..., 3:4])
    loss_ic = jnp.mean((U_ic[..., 0:1] - h_target)**2) + \
              jnp.mean(U_ic[..., 1:2]**2) + \
              jnp.mean(U_ic[..., 2:3]**2)

    # BC (Training=True)
    all_points = causal_batches.reshape((-1, data_gen.ny, data_gen.nx, 4))
    U_all = model_apply({'params': params}, all_points, training=True)
    
    t_left = all_points[:, :, 0, 2]
    z_left = all_points[:, :, 0, 3]
    h_bc_target = jnp.maximum(0.0, bc_fn(t_left)[:, :, None] - z_left[:, :, None])
    
    loss_bc = jnp.mean((U_all[:, :, 0, 0:1] - h_bc_target)**2) + \
              jnp.mean(U_all[:, :, -1, 1:2]**2) + \
              jnp.mean(U_all[:, 0, :, 2:3]**2) + \
              jnp.mean(U_all[:, -1, :, 2:3]**2)
    
    return {'pde': loss_causal_total, 'ic': loss_ic, 'bc': loss_bc}

# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================

class TrainState(train_state.TrainState):
    key: random.PRNGKey

@partial(jax.jit, static_argnames=['model_apply', 'bc_fn', 'data_gen', 'config'])
def train_step_scan(state, causal_batches, ic_batch, model_apply, bc_fn, data_gen, config, weights):
    def loss_fn(params):
        terms = compute_causal_loss(params, model_apply, causal_batches, ic_batch, bc_fn, data_gen, config)
        total = (weights['pde'] * terms['pde'] + 
                 weights['ic'] * terms['ic'] + 
                 weights['bc'] * terms['bc'])
        return total, terms

    (loss, terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, terms)

def scan_epoch(state, data_gen, batch_size, steps, model_apply, bc_fn, config, weights):
    def step_fn(carry, _):
        key, k1, k2 = random.split(carry.key, 3)
        bs_chunk = max(1, batch_size // data_gen.num_chunks)
        
        causal_batches = data_gen.get_causal_batch(k1, bs_chunk)
        ic_batch = data_gen.get_ic_batch(batch_size)
        
        new_state = carry.replace(key=key)
        final_state, metrics = train_step_scan(
            new_state, causal_batches, ic_batch, model_apply, bc_fn, data_gen, config, weights
        )
        return final_state, metrics

    return jax.lax.scan(step_fn, state, None, length=steps)

# ==============================================================================
# 6. MAIN
# ==============================================================================

def main(config_path: str):
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)
    
    print(f"--- Causal Pix2Pix Training (Fix Validation + Config Chunks) ---")

    # Setup
    scenario_name = cfg['scenario']
    base_data_path = os.path.join("data", scenario_name)
    load_bathymetry(os.path.join(base_data_path, "test1DEM.asc"))
    bc_fn = load_boundary_condition(os.path.join(base_data_path, "Test1BC.csv"))
    
    data_gen = CausalDataGenerator(cfg_dict)
    
    model = Pix2PixGenerator(
        output_dim=3, filters=int(cfg['model'].get('filters', 32)),
        lx=data_gen.lx, ly=data_gen.ly, t_final=data_gen.t_final,
        z_min=data_gen.z_min, z_max=data_gen.z_max
    )
    
    key = random.PRNGKey(int(cfg['training']['seed']))
    dummy = jnp.zeros((1, data_gen.ny, data_gen.nx, 4), dtype=DTYPE)
    params = model.init(key, dummy, training=False)['params']
    
    tx = optax.adam(learning_rate=float(cfg['training']['learning_rate']))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, key=key)
    
    # Directories
    trial_name = generate_trial_name("causal_pix2pix")
    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Validation Data
    validation_data_file = os.path.join(base_data_path, "validation_gauges.npy")
    loaded_val_data = None
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            loaded_val_data = jnp.load(validation_data_file).astype(DTYPE)
        except Exception as e:
            print(f"Error loading validation: {e}")
            
    # Training Loop
    epochs = int(cfg['training']['epochs'])
    batch_size = int(cfg['training']['batch_size'])
    steps_per_epoch = 100
    
    weights = {
        'pde': float(cfg['loss_weights']['pde_weight']),
        'ic': float(cfg['loss_weights']['ic_weight']),
        'bc': float(cfg['loss_weights']['bc_weight'])
    }
    
    scan_jitted = jax.jit(partial(
        scan_epoch, data_gen=data_gen, batch_size=batch_size, steps=steps_per_epoch,
        model_apply=model.apply, bc_fn=bc_fn, config=cfg, weights=weights
    ))
    
    start_time = time.time()
    best_nse = -np.inf
    
    for epoch in range(epochs):
        epoch_start = time.time()
        state, (losses, terms) = scan_jitted(state)
        
        avg_loss = jnp.mean(losses)
        avg_pde = jnp.mean(terms['pde'])
        avg_ic = jnp.mean(terms['ic'])
        avg_bc = jnp.mean(terms['bc'])
        
        if (epoch + 1) % int(cfg['reporting']['epoch_freq']) == 0:
            # Validation
            nse_val, rmse_val = -jnp.inf, jnp.inf
            val_freq = int(cfg.get("reporting", {}).get("val_freq", 100))
            if loaded_val_data is not None and ((epoch + 1) % val_freq == 0):
                nse_val, rmse_val = compute_validation_metrics_fast(
                    model, state.params, data_gen, loaded_val_data
                )
                if nse_val > best_nse:
                    best_nse = nse_val
            
            print_epoch_stats(epoch, epoch, start_time, float(avg_loss), 
                              float(avg_pde), float(avg_ic), float(avg_bc), 
                              0, 0, 0, float(nse_val), float(rmse_val), time.time()-epoch_start)
            
    # Save & Plot
    save_model(state.params, model_dir, trial_name)
    
    print("Generating Plot...")
    t_final = data_gen.t_final
    test_inp = data_gen.get_ic_batch(1).at[..., 2].set(t_final)
    U_out = model.apply({'params': state.params}, test_inp, training=False)
    
    plt.figure()
    plt.imshow(U_out[0, :, :, 0], origin='lower', cmap='viridis', aspect='auto')
    plt.title(f"H at t={t_final}")
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, "final_causal.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)