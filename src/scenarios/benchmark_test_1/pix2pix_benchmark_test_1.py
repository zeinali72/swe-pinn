import os
import sys
import time
import copy
import math
import argparse
from functools import partial
from typing import Any, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import random
import optax
from aim import Repo, Run, Image
from flax import linen as nn
from flax.core import FrozenDict
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from src.config import load_config, DTYPE
from src.data import (
    bathymetry_fn,
    load_boundary_condition,
    load_bathymetry
)
from src.utils import (
    nse, rmse, generate_trial_name, save_model, ask_for_confirmation
)
from src.reporting import (
    print_epoch_stats, log_metrics, print_final_summary
)

# ==============================================================================
# Helpers
# ==============================================================================

def parse_compute_dtype(x: Any) -> Any:
    """Allow config strings like 'float32', 'bf16', 'fp16'."""
    if isinstance(x, str):
        s = x.lower().strip()
        if s in ("bf16", "bfloat16"):
            return jnp.bfloat16
        if s in ("fp16", "float16", "half"):
            return jnp.float16
        return jnp.float32
    return x

def central_diff_roll(u: jnp.ndarray, dx: float, dy: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Central difference using roll. Edge cells are later masked out, so wrap-around is harmless."""
    du_dx = (jnp.roll(u, -1, axis=2) - jnp.roll(u, 1, axis=2)) / (2.0 * dx)
    du_dy = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2.0 * dy)
    return du_dx, du_dy

def make_edge_mask(shape_like: jnp.ndarray) -> jnp.ndarray:
    """Mask out outermost boundary where central differences are invalid."""
    mask = jnp.ones_like(shape_like)
    mask = mask.at[:, 0, :].set(0).at[:, -1, :].set(0).at[:, :, 0].set(0).at[:, :, -1].set(0)
    return mask

# ==============================================================================
# 1. PIX2PIX MODEL ARCHITECTURE (GroupNorm, optional remat, mixed-precision compute)
# ==============================================================================

class GN(nn.Module):
    max_groups: int = 32
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        c = x.shape[-1]
        # choose largest divisor <= max_groups
        g = 1
        for cand in range(min(self.max_groups, c), 0, -1):
            if c % cand == 0:
                g = cand
                break
        return nn.GroupNorm(num_groups=g, epsilon=self.eps)(x)

class ResBlock(nn.Module):
    filters: int
    use_remat: bool = False
    act: Any = nn.relu
    compute_dtype: Any = jnp.float32  # activation dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x

        def body(y):
            y = nn.Conv(self.filters, kernel_size=(3, 3), strides=1, padding="SAME",
                        param_dtype=jnp.float32, dtype=self.compute_dtype)(y)
            y = GN()(y)
            y = self.act(y)
            y = nn.Conv(self.filters, kernel_size=(3, 3), strides=1, padding="SAME",
                        param_dtype=jnp.float32, dtype=self.compute_dtype)(y)
            y = GN()(y)
            return y

        if self.use_remat:
            y = nn.remat(body)(x)
        else:
            y = body(x)

        return self.act(residual + y)

class Pix2PixGenerator(nn.Module):
    """
    Pix2Pix-style Generator: Downscaling -> Residual Blocks -> Upscaling.
    Includes input normalization from physical units to [-1, 1].

    Notes for memory:
      - GroupNorm removes batch_stats (less state & less memory).
      - compute_dtype can be bf16/fp16 to cut activation memory.
      - use_remat can reduce activation memory at a compute cost.
    """
    output_dim: int
    filters: int = 64
    n_downsample: int = 3
    n_res_blocks: int = 6
    use_remat: bool = False
    compute_dtype: Any = jnp.float32

    # Domain scaling parameters (set from config/data)
    lx: float = 700.0
    ly: float = 100.0
    t_final: float = 72000.0
    z_min: float = 0.0
    z_max: float = 10.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # x shape: [B, H, W, 4] with channels [x, y, t, z] in PHYSICAL units

        # 0) Input normalization to [-1, 1] (in float32 for safety)
        x = x.astype(jnp.float32)
        x_phys = x[..., 0:1]
        y_phys = x[..., 1:2]
        t_phys = x[..., 2:3]
        z_phys = x[..., 3:4]

        x_norm = 2.0 * x_phys / self.lx - 1.0
        y_norm = 2.0 * y_phys / self.ly - 1.0
        t_norm = 2.0 * t_phys / self.t_final - 1.0

        z_range = (self.z_max - self.z_min) + 1e-6
        z_norm = 2.0 * (z_phys - self.z_min) / z_range - 1.0

        x = jnp.concatenate([x_norm, y_norm, t_norm, z_norm], axis=-1)

        # Cast to compute dtype for activations (memory savings)
        x = x.astype(self.compute_dtype)

        # 1) Initial conv
        x = nn.Conv(self.filters, kernel_size=(7, 7), strides=1, padding="SAME",
                    param_dtype=jnp.float32, dtype=self.compute_dtype)(x)
        x = GN()(x)
        x = nn.relu(x)

        # 2) Downscaling
        curr_filters = self.filters
        for _ in range(self.n_downsample):
            curr_filters *= 2
            x = nn.Conv(curr_filters, kernel_size=(3, 3), strides=2, padding="SAME",
                        param_dtype=jnp.float32, dtype=self.compute_dtype)(x)
            x = GN()(x)
            x = nn.relu(x)

        # 3) ResBlocks
        for _ in range(self.n_res_blocks):
            x = ResBlock(curr_filters, use_remat=self.use_remat,
                         compute_dtype=self.compute_dtype)(x)

        # 4) Upscaling
        for _ in range(self.n_downsample):
            curr_filters //= 2
            x = nn.ConvTranspose(curr_filters, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                 param_dtype=jnp.float32, dtype=self.compute_dtype)(x)
            x = GN()(x)
            x = nn.relu(x)

        # 5) Output conv
        x = nn.Conv(self.output_dim, kernel_size=(7, 7), strides=1, padding="SAME",
                    param_dtype=jnp.float32, dtype=self.compute_dtype)(x)

        # Return in float32 for physics math stability
        x = x.astype(jnp.float32)

        h_raw = x[..., 0:1]
        hu_raw = x[..., 1:2]
        hv_raw = x[..., 2:3]
        h_out = jax.nn.softplus(h_raw)  # h >= 0

        return jnp.concatenate([h_out, hu_raw, hv_raw], axis=-1)

# ==============================================================================
# 2. STRUCTURED DATA PIPELINE (supports full-grid or patch sampling)
# ==============================================================================

class StructuredDataGenerator:
    """Generates dense grid or patch batches (B, H, W, C) with raw physical units."""
    def __init__(self, cfg: Dict[str, Any]):
        self.lx = float(cfg['domain']['lx'])
        self.ly = float(cfg['domain']['ly'])
        self.t_final = float(cfg['domain']['t_final'])

        data_cfg = cfg.get("data", {})
        self.nx = int(data_cfg.get("nx", 256))
        self.ny = int(data_cfg.get("ny", 64))

        # Time-grid support (nt) for discrete sampling
        tr_cfg = cfg.get("training", {})
        self.nt = int(tr_cfg.get("nt", 0))
        if self.nt and self.nt > 1:
            self.t_grid = jnp.linspace(0.0, self.t_final, self.nt, dtype=jnp.float32)
        else:
            self.t_grid = None

        # IMPORTANT: keep resolutions divisible by 2**n_downsample to preserve exact shapes
        x = jnp.linspace(0.0, self.lx, self.nx, dtype=jnp.float32)
        y = jnp.linspace(0.0, self.ly, self.ny, dtype=jnp.float32)
        self.x_coords = x
        self.y_coords = y

        self.X, self.Y = jnp.meshgrid(x, y)  # (ny, nx)

        # Precompute bathymetry (physical units)
        z_vals, _, _ = bathymetry_fn(self.X.flatten(), self.Y.flatten())
        Z = z_vals.reshape((self.ny, self.nx)).astype(jnp.float32)
        self.Z2 = Z  # (ny, nx)
        self.Z = Z[..., None]  # (ny, nx, 1)

        self.z_min = float(jnp.min(self.Z2))
        self.z_max = float(jnp.max(self.Z2))

        self.dx = float(x[1] - x[0])
        self.dy = float(y[1] - y[0])

        # Gauge indices (kept)
        self.gauge1_ix = int(jnp.argmin(jnp.abs(x - 600.0)))
        self.gauge1_iy = int(jnp.argmin(jnp.abs(y - 50.0)))
        self.gauge2_ix = int(jnp.argmin(jnp.abs(x - 400.0)))
        self.gauge2_iy = int(jnp.argmin(jnp.abs(y - 50.0)))

    def sample_times(
        self,
        key,
        batch_size: int,
        mode: str = "random",
        sampling: str = "uniform",
        sweep_index: int = 0,
    ) -> jnp.ndarray:
        """
        sampling:
          - "uniform"     : continuous uniform over [0, t_final]
          - "grid_random" : sample indices from a fixed time grid (nt)
          - "grid_sweep"  : deterministic sweep over time grid across epochs (uses sweep_index)
        mode:
          - "zero"        : all zeros (IC)
          - "random"      : use sampling strategy above
        """
        if mode == "zero":
            return jnp.zeros((batch_size,), dtype=jnp.float32)

        # If no grid is defined, fall back to uniform continuous sampling
        if sampling == "uniform" or self.t_grid is None:
            t = random.uniform(key, (batch_size,), minval=0.0, maxval=self.t_final, dtype=jnp.float32)
            return t

        if sampling == "grid_random":
            idx = random.randint(key, (batch_size,), 0, self.nt)
            return self.t_grid[idx]

        if sampling == "grid_sweep":
            # Deterministic: same time for whole batch, swept by epoch
            t_val = self.t_grid[sweep_index % self.nt]
            return jnp.full((batch_size,), t_val, dtype=jnp.float32)

        raise ValueError(f"Unknown sampling mode: {sampling}")

    def get_full_batch(
        self,
        key,
        batch_size: int,
        t_mode: str = "random",
        time_sampling: str = "uniform",
        sweep_index: int = 0,
    ):
        kt, _ = random.split(key)
        t_samples = self.sample_times(
            kt,
            batch_size,
            mode=t_mode,
            sampling=time_sampling,
            sweep_index=sweep_index,
        )  # (B,)

        X_b = jnp.broadcast_to(self.X[None, ..., None], (batch_size, self.ny, self.nx, 1))
        Y_b = jnp.broadcast_to(self.Y[None, ..., None], (batch_size, self.ny, self.nx, 1))
        Z_b = jnp.broadcast_to(self.Z[None, ...], (batch_size, self.ny, self.nx, 1))
        T_b = jnp.broadcast_to(t_samples[:, None, None, None], (batch_size, self.ny, self.nx, 1))

        inputs = jnp.concatenate([X_b, Y_b, T_b, Z_b], axis=-1).astype(DTYPE)
        return inputs, t_samples.astype(DTYPE)

    def get_full_batch_multi_t(
        self,
        key,
        n_times: int,
        t_mode: str = "random",
        time_sampling: str = "uniform",
        sweep_index: int = 0,
    ):
        """
        Sample n_times different time values over the full spatial grid.
        Returns (n_times, ny, nx, 4) with each time as a separate batch element.
        This improves temporal coverage per training step.
        """
        kt, _ = random.split(key)

        if t_mode == "zero":
            t_samples = jnp.zeros((n_times,), dtype=jnp.float32)
        else:
            # If using a fixed time grid, sample or sweep from it
            if time_sampling != "uniform" and self.t_grid is not None:
                if time_sampling == "grid_random":
                    idx = random.randint(kt, (n_times,), 0, self.nt)
                    t_samples = self.t_grid[idx]
                elif time_sampling == "grid_sweep":
                    # Sweep n_times sequential indices each call
                    base = (sweep_index * n_times) % self.nt
                    idx = (base + jnp.arange(n_times)) % self.nt
                    t_samples = self.t_grid[idx]
                else:
                    raise ValueError(f"Unknown time_sampling: {time_sampling}")
            else:
                # Continuous uniform sampling
                t_samples = random.uniform(kt, (n_times,), minval=0.0, maxval=self.t_final, dtype=jnp.float32)

        X_b = jnp.broadcast_to(self.X[None, ..., None], (n_times, self.ny, self.nx, 1))
        Y_b = jnp.broadcast_to(self.Y[None, ..., None], (n_times, self.ny, self.nx, 1))
        Z_b = jnp.broadcast_to(self.Z[None, ...], (n_times, self.ny, self.nx, 1))
        T_b = jnp.broadcast_to(t_samples[:, None, None, None], (n_times, self.ny, self.nx, 1))

        inputs = jnp.concatenate([X_b, Y_b, T_b, Z_b], axis=-1).astype(DTYPE)
        return inputs, t_samples.astype(DTYPE)

    def get_patch_batch(
        self,
        key,
        batch_size: int,
        patch_nx: int,
        patch_ny: int,
        mode: str = "random",
        t_mode: str = "random",
        t_samples: Optional[jnp.ndarray] = None,  # <-- NEW: allow caller to reuse same time samples
        time_sampling: str = "uniform",           # <-- optional if t_samples is None
        sweep_index: int = 0,                     # <-- optional if time_sampling == grid_sweep
    ):
        """
        mode:
          - 'random' : random (iy0, ix0)
          - 'left'   : ix0=0, random iy0
          - 'right'  : ix0=nx-patch_nx, random iy0
          - 'bottom' : iy0=0, random ix0
          - 'top'    : iy0=ny-patch_ny, random ix0

        If t_samples is provided, those times are used directly (fixes BC time-mismatch).
        """
        kt, kxy = random.split(key)

        if t_samples is None:
            t_samples = self.sample_times(
                kt, batch_size, mode=t_mode, sampling=time_sampling, sweep_index=sweep_index
            )  # (B,)
        else:
            t_samples = t_samples.astype(jnp.float32)

        max_ix0 = self.nx - patch_nx
        max_iy0 = self.ny - patch_ny
        if max_ix0 < 0 or max_iy0 < 0:
            raise ValueError(f"Patch ({patch_nx}x{patch_ny}) larger than grid ({self.nx}x{self.ny}).")

        if mode == "random":
            ix0 = random.randint(kxy, (batch_size,), 0, max_ix0 + 1)
            iy0 = random.randint(kxy, (batch_size,), 0, max_iy0 + 1)
        elif mode == "left":
            ix0 = jnp.zeros((batch_size,), dtype=jnp.int32)
            iy0 = random.randint(kxy, (batch_size,), 0, max_iy0 + 1)
        elif mode == "right":
            ix0 = jnp.full((batch_size,), max_ix0, dtype=jnp.int32)
            iy0 = random.randint(kxy, (batch_size,), 0, max_iy0 + 1)
        elif mode == "bottom":
            iy0 = jnp.zeros((batch_size,), dtype=jnp.int32)
            ix0 = random.randint(kxy, (batch_size,), 0, max_ix0 + 1)
        elif mode == "top":
            iy0 = jnp.full((batch_size,), max_iy0, dtype=jnp.int32)
            ix0 = random.randint(kxy, (batch_size,), 0, max_ix0 + 1)
        else:
            raise ValueError(f"Unknown patch mode: {mode}")

        def slice_one(iy, ix):
            Xp = jax.lax.dynamic_slice(self.X, (iy, ix), (patch_ny, patch_nx))
            Yp = jax.lax.dynamic_slice(self.Y, (iy, ix), (patch_ny, patch_nx))
            Zp = jax.lax.dynamic_slice(self.Z2, (iy, ix), (patch_ny, patch_nx))
            return Xp, Yp, Zp

        Xp, Yp, Zp = jax.vmap(slice_one)(iy0, ix0)  # each: (B, patch_ny, patch_nx)

        Xp = Xp[..., None]
        Yp = Yp[..., None]
        Zp = Zp[..., None]
        Tp = jnp.broadcast_to(t_samples[:, None, None, None], (batch_size, patch_ny, patch_nx, 1))

        inputs = jnp.concatenate([Xp, Yp, Tp, Zp], axis=-1).astype(DTYPE)
        return inputs, t_samples.astype(DTYPE)

# ==============================================================================
# 3. VALIDATION METRICS - FAST VECTORIZED VERSION
# ==============================================================================

def _preprocess_val_data(val_data, data_gen, n_samples: int = 50):
    """
    Pre-process validation data into numpy arrays for fast lookup.
    Returns: (times_arr, gauge_coords, h_true_arr, gauge_pixel_indices)
    where gauge_pixel_indices is (n_gauges, 2) with (iy, ix) for each gauge.
    """
    val_np = np.array(val_data)  # Convert once to numpy

    # Get unique times
    unique_times = np.unique(val_np[:, 0])
    if len(unique_times) > n_samples:
        idx = np.linspace(0, len(unique_times) - 1, n_samples).astype(int)
        sampled_times = unique_times[idx]
    else:
        sampled_times = unique_times

    # Get unique gauge locations
    gauge_coords = np.unique(val_np[:, 1:3], axis=0)  # (n_gauges, 2) with (x, y)

    # Compute pixel indices for each gauge (vectorized)
    x_coords_np = np.array(data_gen.x_coords)
    y_coords_np = np.array(data_gen.y_coords)

    gauge_ix = np.abs(x_coords_np[None, :] - gauge_coords[:, 0:1]).argmin(axis=1)
    gauge_iy = np.abs(y_coords_np[None, :] - gauge_coords[:, 1:2]).argmin(axis=1)
    gauge_pixel_indices = np.stack([gauge_iy, gauge_ix], axis=1)  # (n_gauges, 2)

    # Build lookup dict for h_true: (t, gx, gy) -> h
    h_true_lookup = {}
    for row in val_np:
        t, gx, gy, h = row[0], row[1], row[2], row[3]
        h_true_lookup[(t, gx, gy)] = h

    return sampled_times, gauge_coords, h_true_lookup, gauge_pixel_indices


def compute_validation_metrics_fast(
    model,
    params,
    data_gen,
    val_data,
    n_samples: int = 50,
):
    """
    Fast vectorized validation using batched inference.
    Builds all patches at once and runs a single batched forward pass.
    """
    if val_data is None or len(val_data) == 0:
        return -jnp.inf, jnp.inf

    sampled_times, gauge_coords, h_true_lookup, gauge_pixel_indices = \
        _preprocess_val_data(val_data, data_gen, n_samples)

    n_times = len(sampled_times)
    n_gauges = len(gauge_coords)

    # Use full grid inference at sampled times (more efficient than patches)
    # Stack all times into a batch
    X_grid = data_gen.X[None, ..., None]  # (1, ny, nx, 1)
    Y_grid = data_gen.Y[None, ..., None]
    Z_grid = data_gen.Z  # (ny, nx, 1)
    Z_grid = Z_grid[None, ...]  # (1, ny, nx, 1)

    # Broadcast to (n_times, ny, nx, 1)
    X_batch = jnp.broadcast_to(X_grid, (n_times, data_gen.ny, data_gen.nx, 1))
    Y_batch = jnp.broadcast_to(Y_grid, (n_times, data_gen.ny, data_gen.nx, 1))
    Z_batch = jnp.broadcast_to(Z_grid, (n_times, data_gen.ny, data_gen.nx, 1))
    T_batch = jnp.array(sampled_times)[:, None, None, None]
    T_batch = jnp.broadcast_to(T_batch, (n_times, data_gen.ny, data_gen.nx, 1))

    inputs = jnp.concatenate([X_batch, Y_batch, T_batch, Z_batch], axis=-1).astype(DTYPE)

    # Single batched forward pass (JIT will compile this)
    outputs = model.apply({'params': params}, inputs, training=False)
    h_pred_grid = np.array(outputs[..., 0])  # (n_times, ny, nx) - move to numpy once

    # Vectorized extraction: build (time_idx, gauge_idx) pairs that have h_true
    # Then use advanced indexing to extract predictions all at once
    val_np = np.array(val_data)
    
    # Create arrays for vectorized matching
    h_preds_list = []
    h_trues_list = []
    
    # For each gauge, find which sampled_times have data and extract
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
    nse_val = 1.0 - ss_res / (ss_tot + 1e-10)

    return float(nse_val), float(rmse_val)


# Keep old function as alias for compatibility
def compute_validation_metrics(
    model, params, data_gen, val_data, n_samples=50, gauge_patch_nx=64, gauge_patch_ny=64
):
    """Wrapper that calls the fast version."""
    return compute_validation_metrics_fast(model, params, data_gen, val_data, n_samples)

# ==============================================================================
# 4. LOSS FUNCTIONS (PDE + BC + IC) with corrected friction + roll FD
# ==============================================================================

def pde_terms_with_time_jvp(
    model,
    params,
    inputs: jnp.ndarray,  # (B,H,W,4)
    dx: float,
    dy: float,
    eps: float,
    n_manning: float,
    g: float = 9.81,
):
    """
    Returns loss_pde and residual components based on shallow water equations.
    Corrected Manning friction term (conservative form in hu/hv).
    """
    # JVP w.r.t. physical time input (channel 2)
    tangent = jnp.zeros_like(inputs)
    tangent = tangent.at[..., 2].set(1.0)

    def forward_fn(x):
        return model.apply({'params': params}, x, training=True)

    U_pred, dU_dt = jax.jvp(forward_fn, (inputs,), (tangent,))

    # Ensure float32 for physics
    U_pred = U_pred.astype(jnp.float32)
    dU_dt = dU_dt.astype(jnp.float32)

    # If shapes drift (non-multiple-of-2**n), crop to input shape
    H, W = inputs.shape[1], inputs.shape[2]
    U_pred = U_pred[:, :H, :W, :]
    dU_dt = dU_dt[:, :H, :W, :]

    h, hu, hv = U_pred[..., 0], U_pred[..., 1], U_pred[..., 2]
    dh_dt, dhu_dt, dhv_dt = dU_dt[..., 0], dU_dt[..., 1], dU_dt[..., 2]

    z = inputs[..., 3].astype(jnp.float32)

    h_safe = jnp.maximum(h, eps)
    u = hu / h_safe
    v = hv / h_safe

    # Flux divergences
    d_hu_dx, _ = central_diff_roll(hu, dx, dy)
    _, d_hv_dy = central_diff_roll(hv, dx, dy)

    F_hu = hu * u + 0.5 * g * (h ** 2)
    G_hu = hu * v
    F_hv = hu * v
    G_hv = hv * v + 0.5 * g * (h ** 2)

    dFhu_dx, _ = central_diff_roll(F_hu, dx, dy)
    _, dGhu_dy = central_diff_roll(G_hu, dx, dy)

    dFhv_dx, _ = central_diff_roll(F_hv, dx, dy)
    _, dGhv_dy = central_diff_roll(G_hv, dx, dy)

    # Bathymetry slopes
    dz_dx, dz_dy = central_diff_roll(z, dx, dy)

    vel_mag = jnp.sqrt(u * u + v * v + eps)

    # Correct conservative Manning friction for hu/hv:
    # Sfx = g*n^2 * hu * |U| / h^(4/3)
    # Sfy = g*n^2 * hv * |U| / h^(4/3)
    friction_x = g * (n_manning ** 2) * hu * vel_mag / (h_safe ** (4.0 / 3.0))
    friction_y = g * (n_manning ** 2) * hv * vel_mag / (h_safe ** (4.0 / 3.0))

    rhs_h = 0.0
    rhs_hu = -g * h * dz_dx - friction_x
    rhs_hv = -g * h * dz_dy - friction_y

    # Residuals: LHS - RHS
    res_h = dh_dt + d_hu_dx + d_hv_dy - rhs_h
    res_hu = dhu_dt + dFhu_dx + dGhu_dy - rhs_hu
    res_hv = dhv_dt + dFhv_dx + dGhv_dy - rhs_hv

    mask = make_edge_mask(res_h)
    loss_pde = jnp.mean((res_h * mask) ** 2 + (res_hu * mask) ** 2 + (res_hv * mask) ** 2)
    return loss_pde

def bc_loss_on_patch_edges(
    model,
    params,
    inputs_left: jnp.ndarray,
    inputs_right: jnp.ndarray,
    inputs_top: jnp.ndarray,
    inputs_bottom: jnp.ndarray,
    bc_times: jnp.ndarray,
    bc_fn,
):
    """
    Apply BCs on patches pinned to the domain boundaries.
    IMPORTANT: This assumes bc_fn returns STAGE (free surface elevation eta),
    so depth h_target = max(0, eta - z). If your BC file is already DEPTH, remove the '- z'.
    """
    # forward passes (no time derivatives)
    out_left = model.apply({'params': params}, inputs_left, training=False).astype(jnp.float32)
    out_right = model.apply({'params': params}, inputs_right, training=False).astype(jnp.float32)
    out_top = model.apply({'params': params}, inputs_top, training=False).astype(jnp.float32)
    out_bottom = model.apply({'params': params}, inputs_bottom, training=False).astype(jnp.float32)

    # Crop to match input shapes (in case of odd patch sizes)
    Hl, Wl = inputs_left.shape[1], inputs_left.shape[2]
    Hr, Wr = inputs_right.shape[1], inputs_right.shape[2]
    Ht, Wt = inputs_top.shape[1], inputs_top.shape[2]
    Hb, Wb = inputs_bottom.shape[1], inputs_bottom.shape[2]

    out_left = out_left[:, :Hl, :Wl, :]
    out_right = out_right[:, :Hr, :Wr, :]
    out_top = out_top[:, :Ht, :Wt, :]
    out_bottom = out_bottom[:, :Hb, :Wb, :]

    hL = out_left[..., 0]
    huR = out_right[..., 1]
    hvT = out_top[..., 2]
    hvB = out_bottom[..., 2]

    z_left = inputs_left[..., 3].astype(jnp.float32)
    z_left_col0 = z_left[:, :, 0]  # (B, Hpatch)

    eta = bc_fn(bc_times).astype(jnp.float32)  # (B,)
    eta = eta[:, None]  # (B,1)
    h_target = jnp.maximum(0.0, eta - z_left_col0)

    loss_bc_left = jnp.mean((hL[:, :, 0] - h_target) ** 2)
    loss_bc_right = jnp.mean(huR[:, :, -1] ** 2)
    loss_bc_top = jnp.mean(hvT[:, -1, :] ** 2)
    loss_bc_bottom = jnp.mean(hvB[:, 0, :] ** 2)

    total_bc = loss_bc_left + loss_bc_right + loss_bc_top + loss_bc_bottom
    return total_bc, (loss_bc_left, loss_bc_right, loss_bc_top, loss_bc_bottom)

def ic_loss(
    model,
    params,
    inputs_ic: jnp.ndarray,
    eta0: float,
    t_window: float,
):
    """
    Initial condition: enforce (near t=0) depth h = max(0, eta0 - z), hu=hv=0.
    Uses a soft time mask to allow patch IC batches.
    """
    out = model.apply({'params': params}, inputs_ic, training=False).astype(jnp.float32)
    H, W = inputs_ic.shape[1], inputs_ic.shape[2]
    out = out[:, :H, :W, :]

    h, hu, hv = out[..., 0], out[..., 1], out[..., 2]
    t_grid = inputs_ic[..., 2].astype(jnp.float32)
    z = inputs_ic[..., 3].astype(jnp.float32)

    mask_ic = (t_grid < t_window).astype(jnp.float32)
    h_target = jnp.maximum(0.0, eta0 - z)

    num = jnp.sum(mask_ic * ((h - h_target) ** 2 + hu ** 2 + hv ** 2))
    den = jnp.sum(mask_ic) + 1e-6
    return num / den

def compute_all_losses(
    model,
    params,
    inputs_pde: jnp.ndarray,
    inputs_bc_left: jnp.ndarray,
    inputs_bc_right: jnp.ndarray,
    inputs_bc_top: jnp.ndarray,
    inputs_bc_bottom: jnp.ndarray,
    bc_times: jnp.ndarray,
    bc_fn,
    dx: float,
    dy: float,
    eps: float,
    n_manning: float,
    g: float,
    ic_gate: jnp.ndarray,          # float scalar (0 or 1)
    eta0: float,
    ic_t_window: float,
):
    loss_pde = pde_terms_with_time_jvp(model, params, inputs_pde, dx, dy, eps, n_manning, g)

    loss_bc, bc_components = bc_loss_on_patch_edges(
        model, params,
        inputs_bc_left, inputs_bc_right, inputs_bc_top, inputs_bc_bottom,
        bc_times, bc_fn
    )

    # Always compute, gate it (JAX-friendly)
    loss_ic_raw = ic_loss(model, params, inputs_pde, eta0=eta0, t_window=ic_t_window)
    loss_ic_val = ic_gate * loss_ic_raw

    loss_neg_h = 0.0

    terms = {
        "pde": loss_pde,
        "ic": loss_ic_val,
        "bc": loss_bc,
        "neg_h": loss_neg_h,
        "bc_left": bc_components[0],
        "bc_right": bc_components[1],
        "bc_top": bc_components[2],
        "bc_bottom": bc_components[3],
    }
    return terms


def train_step(
    model: Any,
    optimiser: optax.GradientTransformation,
    params: FrozenDict,
    opt_state: optax.OptState,
    inputs_pde: jnp.ndarray,
    inputs_bc_left: jnp.ndarray,
    inputs_bc_right: jnp.ndarray,
    inputs_bc_top: jnp.ndarray,
    inputs_bc_bottom: jnp.ndarray,
    bc_times: jnp.ndarray,
    bc_fn_static: Any,
    dx: float,
    dy: float,
    eps: float,
    n_manning: float,
    g: float,
    ic_gate: jnp.ndarray,   # float scalar (0 or 1)
    eta0: float,
    ic_t_window: float,
    w_pde: float,
    w_ic: float,
    w_bc: float,
    w_neg_h: float,
):
    def loss_fn(p):
        terms = compute_all_losses(
            model=model,
            params=p,
            inputs_pde=inputs_pde,
            inputs_bc_left=inputs_bc_left,
            inputs_bc_right=inputs_bc_right,
            inputs_bc_top=inputs_bc_top,
            inputs_bc_bottom=inputs_bc_bottom,
            bc_times=bc_times,
            bc_fn=bc_fn_static,
            dx=dx,
            dy=dy,
            eps=eps,
            n_manning=n_manning,
            g=g,
            ic_gate=ic_gate,
            eta0=eta0,
            ic_t_window=ic_t_window,
        )
        total = (
            w_pde * terms["pde"] +
            w_ic * terms["ic"] +
            w_bc * terms["bc"] +
            w_neg_h * terms["neg_h"]
        )
        return total, terms

    (loss_val, terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, terms, loss_val


train_step_jitted = jax.jit(train_step, static_argnames=["model", "optimiser", "bc_fn_static"])

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main(config_path: str):
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)

    print("Info: Running Benchmark Test 1 - Pix2Pix Version (patched & fixed)...")

    key = random.PRNGKey(int(cfg["training"]["seed"]))

    # --- Resolve scenario/data paths early ---
    scenario_name = cfg.get("scenario")
    base_data_path = os.path.join("data", scenario_name)
    dem_path = os.path.join(base_data_path, "test1DEM.asc")
    bc_csv_path = os.path.join(base_data_path, "Test1BC.csv")

    # --- FIX: load bathymetry BEFORE building generator (bathymetry_fn may depend on DEM) ---
    load_bathymetry(dem_path)

    # Data generator
    data_gen = StructuredDataGenerator(cfg_dict)

    # Model config
    compute_dtype = parse_compute_dtype(cfg.get("model", {}).get("compute_dtype", "float32"))

    model = Pix2PixGenerator(
        output_dim=3,
        filters=int(cfg.get("model", {}).get("filters", 32)),
        n_downsample=int(cfg.get("model", {}).get("n_downsample", 3)),
        n_res_blocks=int(cfg.get("model", {}).get("n_res_blocks", 6)),
        use_remat=bool(cfg.get("model", {}).get("use_remat", False)),
        compute_dtype=compute_dtype,
        lx=float(cfg["domain"]["lx"]),
        ly=float(cfg["domain"]["ly"]),
        t_final=float(cfg["domain"]["t_final"]),
        z_min=float(data_gen.z_min),
        z_max=float(data_gen.z_max),
    )

    # Init
    dummy_input = jnp.zeros((1, data_gen.ny, data_gen.nx, 4), dtype=DTYPE)
    variables = model.init(key, dummy_input)
    params = variables["params"]

    # Directories
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Loss weights (extract scalars, avoid passing dicts into jit)
    lw = cfg.get("loss_weights", {})
    w_pde = float(lw.get("pde_weight", 1.0))
    w_ic = float(lw.get("ic_weight", 1.0))
    w_bc = float(lw.get("bc_weight", 1.0))
    w_neg_h = float(lw.get("neg_h_weight", 0.0))

    # Physics/numerics scalars
    eps = float(cfg.get("numerics", {}).get("eps", 1e-6))
    n_manning = float(cfg.get("physics", {}).get("n_manning", 0.02))
    g = float(cfg.get("physics", {}).get("g", 9.81))

    # IC settings
    eta0 = float(cfg.get("physics", {}).get("ic_eta0", 9.7))
    ic_t_window = float(cfg.get("physics", {}).get("ic_t_window", 720.0))

    # Boundary condition function
    bc_fn_static = load_boundary_condition(bc_csv_path)

    # Validation data
    validation_data_file = os.path.join(base_data_path, "validation_gauges.npy")
    validation_loaded = False
    loaded_val_data = None
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            loaded_val_data = jnp.load(validation_data_file).astype(DTYPE)
            validation_loaded = True
        except Exception as e:
            print(f"Error loading validation: {e}")

    # Aim
    aim_run = None
    try:
        aim_repo_path = "aim_repo"
        os.makedirs(aim_repo_path, exist_ok=True)
        aim_repo = Repo(path=aim_repo_path, init=True)
        aim_run = Run(repo=aim_repo, experiment=trial_name)
        aim_run["hparams"] = copy.deepcopy(cfg_dict)
        aim_run["flags"] = {"type": "pix2pix_grid_fixed"}
        print(f"Aim tracking initialized: {trial_name}")
    except Exception as e:
        print(f"Warning: Aim failed: {e}")

    # Optimizer
    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=float(cfg["training"]["learning_rate"])),
    )
    opt_state = optimiser.init(params)

    # Patch training knobs (key fix for VRAM)
    use_patches = bool(cfg.get("training", {}).get("use_patches", False))
    patch_nx = int(cfg.get("training", {}).get("patch_nx", 256))
    patch_ny = int(cfg.get("training", {}).get("patch_ny", 64))

    # Time sampling knobs (nt / grid sampling)
    time_sampling = str(cfg.get("training", {}).get("time_sampling", "uniform"))  # uniform|grid_random|grid_sweep

    # Strong recommendation: patch dims divisible by 2**n_downsample
    down = int(cfg.get("model", {}).get("n_downsample", 3))
    mult = 2 ** down
    if use_patches:
        if (patch_nx % mult) != 0 or (patch_ny % mult) != 0:
            print(f"WARNING: patch sizes should be divisible by {mult} to preserve exact shapes. "
                  f"Got patch_nx={patch_nx}, patch_ny={patch_ny}.")

    # Training loop
    start_time = time.time()
    best_nse = -jnp.inf
    best_loss = jnp.inf
    best_params = copy.deepcopy(params)

    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])

    print(f"\n--- Training Started: {trial_name} ---")
    print(f"Grid: nx={data_gen.nx}, ny={data_gen.ny} | "
          f"use_patches={use_patches} patch=({patch_nx}x{patch_ny}) | "
          f"compute_dtype={compute_dtype} | time_sampling={time_sampling} nt={getattr(data_gen, 'nt', 0)}")

    for epoch in range(epochs):
        epoch_start = time.time()
        key, step_key = random.split(key)

        do_ic = (epoch % 4 == 0)

        if use_patches:
            # ==========================================================
            # IMPORTANT FIX:
            # Sample times ONCE and reuse for PDE patch and BC patches,
            # so BC enforcement time matches the patch time.
            # ==========================================================
            key, kt = random.split(key)
            t_mode = "zero" if do_ic else "random"

            # One shared t_samples for this epoch/batch
            t_samples = data_gen.sample_times(
                kt,
                batch_size=batch_size,
                mode=t_mode,
                sampling=time_sampling,
                sweep_index=epoch,
            )
            bc_times = t_samples.astype(DTYPE)

            # PDE/IC patch (random) with shared time samples
            key, k0 = random.split(key)
            pde_inputs, _ = data_gen.get_patch_batch(
                k0, batch_size, patch_nx, patch_ny,
                mode="random",
                t_samples=t_samples,
                time_sampling=time_sampling,
                sweep_index=epoch,
            )

            # Boundary patches for BCs (all share same t_samples)
            key, k1 = random.split(key)
            left_inputs, _ = data_gen.get_patch_batch(
                k1, batch_size, patch_nx, patch_ny,
                mode="left",
                t_samples=t_samples,
                time_sampling=time_sampling,
                sweep_index=epoch,
            )
            key, k2 = random.split(key)
            right_inputs, _ = data_gen.get_patch_batch(
                k2, batch_size, patch_nx, patch_ny,
                mode="right",
                t_samples=t_samples,
                time_sampling=time_sampling,
                sweep_index=epoch,
            )
            key, k3 = random.split(key)
            top_inputs, _ = data_gen.get_patch_batch(
                k3, batch_size, patch_nx, patch_ny,
                mode="top",
                t_samples=t_samples,
                time_sampling=time_sampling,
                sweep_index=epoch,
            )
            key, k4 = random.split(key)
            bottom_inputs, _ = data_gen.get_patch_batch(
                k4, batch_size, patch_nx, patch_ny,
                mode="bottom",
                t_samples=t_samples,
                time_sampling=time_sampling,
                sweep_index=epoch,
            )

        else:
            # Full-grid training with multi-time sampling (optional)
            n_times = int(cfg.get("training", {}).get("n_times_per_batch", 1))
            if n_times > 1:
                pde_inputs, bc_times = data_gen.get_full_batch_multi_t(
                    step_key,
                    n_times,
                    t_mode="zero" if do_ic else "random",
                    time_sampling=time_sampling,
                    sweep_index=epoch,
                )
            else:
                pde_inputs, bc_times = data_gen.get_full_batch(
                    step_key,
                    batch_size,
                    t_mode="zero" if do_ic else "random",
                    time_sampling=time_sampling,
                    sweep_index=epoch,
                )
            left_inputs = pde_inputs
            right_inputs = pde_inputs
            top_inputs = pde_inputs
            bottom_inputs = pde_inputs

        # Convert do_ic bool to JAX-compatible float gate
        ic_gate = jnp.array(1.0 if do_ic else 0.0, dtype=jnp.float32)

        params, opt_state, terms, loss = train_step_jitted(
            model, optimiser, params, opt_state,
            pde_inputs,
            left_inputs, right_inputs, top_inputs, bottom_inputs,
            bc_times,
            bc_fn_static,
            data_gen.dx, data_gen.dy,
            eps, n_manning, g,
            ic_gate, eta0, ic_t_window,
            w_pde, w_ic, w_bc, w_neg_h
        )

        # Reporting
        epoch_freq = int(cfg.get("reporting", {}).get("epoch_freq", 100))
        if (epoch + 1) % epoch_freq == 0:
            val_freq = int(cfg.get("reporting", {}).get("val_freq", 1000))
            if validation_loaded and ((epoch + 1) % val_freq == 0):
                nse_val, rmse_val = compute_validation_metrics(
                    model, params, data_gen, loaded_val_data, n_samples=50,
                    gauge_patch_nx=min(patch_nx, 128),
                    gauge_patch_ny=min(patch_ny, 128),
                )
                if nse_val > best_nse:
                    best_nse = nse_val
            else:
                nse_val, rmse_val = -jnp.inf, jnp.inf

            print_epoch_stats(
                epoch, epoch, start_time, float(loss),
                float(terms.get("pde", 0.0)),
                float(terms.get("ic", 0.0)),
                float(terms.get("bc", 0.0)),
                0.0, 0.0,
                float(terms.get("neg_h", 0.0)),
                float(nse_val), float(rmse_val),
                time.time() - epoch_start
            )

            if aim_run:
                aim_run.track(float(loss), name="loss", step=epoch, context={"subset": "total"})
                for k, v in terms.items():
                    aim_run.track(float(v), name="loss", step=epoch, context={"subset": k})

            if float(loss) < float(best_loss):
                best_loss = float(loss)
                best_params = copy.deepcopy(params)

    end_time = time.time()
    final_stats = {
        "total_weighted_loss": float(loss),
        "epoch": epochs - 1,
        "time_elapsed_seconds": end_time - start_time,
        "nse": float(best_nse),
        "rmse": float("inf"),
        "unweighted_losses": {k: float(v) for k, v in terms.items()},
    }
    # best_nse_stats needs all required keys for print_final_summary
    best_nse_stats = {
        "nse": float(best_nse),
        "epoch": epochs - 1,
        "time_elapsed_seconds": end_time - start_time,
        "rmse": float("inf"),
        "total_weighted_loss": float(best_loss),
        "unweighted_losses": {k: float(v) for k, v in terms.items()},
    }
    print_final_summary(end_time - start_time, best_nse_stats, final_stats)

    # Use best_params for saving/inference
    params_to_use = best_params

    if ask_for_confirmation():
        save_model(params_to_use, model_dir, trial_name)

        # Plotting
        print("Generating Grid Plots...")
        t_final = float(cfg["domain"]["t_final"])

        test_inp, _ = data_gen.get_full_batch(
            key, 1, t_mode="random",
            time_sampling="uniform",
            sweep_index=0
        )
        test_inp = test_inp.at[..., 2].set(t_final)

        U_out = model.apply({"params": params_to_use}, test_inp, training=False)
        h_final = np.array(U_out[0, :, :, 0])

        plt.figure(figsize=(10, 4))
        plt.imshow(h_final, origin="lower", aspect="auto", cmap="viridis")
        plt.colorbar(label="H (m)")
        plt.title(f"Predicted H at t={t_final}s")
        save_path = os.path.join(results_dir, "final_grid.png")
        plt.savefig(save_path)
        if aim_run:
            aim_run.track(Image(save_path), name="final_grid")
        print(f"Saved plot to {save_path}")

    if aim_run:
        aim_run.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
