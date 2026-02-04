# src/losses.py
import jax
import jax.numpy as jnp
import jax.nn
from flax import linen as nn
from flax.core import FrozenDict
from typing import Dict, Any, Optional, Tuple, Callable

from src.physics import SWEPhysics, h_exact
# If using the differentiable interpolator from data.py
from src.data import bathymetry_fn 
from src.config import DTYPE

# ==========================================
# 1. CORE PDE LOSS (Physics)
# ==========================================

def compute_pde_loss(model: nn.Module, params: Dict[str, Any], pde_batch: jnp.ndarray,
                     config: FrozenDict, pde_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Compute the PDE residual mean squared error (MSE) for the SWE."""
    if pde_batch.shape[-1] != 3:
        raise ValueError(f"PDE batch requires shape (N, 3), but got {pde_batch.shape}")
    if pde_mask is None:
        pde_mask = jnp.ones((pde_batch.shape[0],), dtype=bool)

    U_pred = model.apply({'params': params['params']}, pde_batch, train=True)
    
    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)

    jac_U = jax.vmap(jax.jacfwd(U_fn))(pde_batch)
    dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]

    # 1. Unpack coordinates
    x_batch = pde_batch[..., 0]
    y_batch = pde_batch[..., 1]

    # 2. Get Bathymetry Slopes (Differentiable)
    # This uses the global interpolator defined in src/data.py
    _, bed_grad_x, bed_grad_y = bathymetry_fn(x_batch, y_batch)

    eps = config["numerics"]["eps"]
    physics = SWEPhysics(U_pred, eps=eps)

    g = config["physics"]["g"]
    n_manning = config["physics"]["n_manning"]
    inflow = config["physics"]["inflow"]

    JF, JG = physics.flux_jac(g=g)
    div_F = jnp.einsum('nij,nj->ni', JF, dU_dx)
    div_G = jnp.einsum('nij,nj->ni', JG, dU_dy)

    # 3. Pass gradients to source term
    S = physics.source(g=g, n_manning=n_manning, inflow=inflow, 
                       bed_grad_x=bed_grad_x, bed_grad_y=bed_grad_y)

    residual = (dU_dt + div_F + div_G - S)

    h_mask = jnp.where(U_pred[..., 0] < eps, 0.0, 1.0)
    final_residual = residual * h_mask[..., None]* pde_mask[..., None]
    return jnp.mean(final_residual ** 2)

def compute_neg_h_loss(model: nn.Module, params: Dict[str, Any], pde_points: jnp.ndarray,
                     pde_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Compute penalty for negative water height."""
    U_pred = model.apply({'params': params['params']}, pde_points, train=False)
    h_pred = U_pred[..., 0]
    if pde_mask is not None:
        h_pred = h_pred * pde_mask
    return jnp.mean(jax.nn.relu(-h_pred) ** 2)

def compute_ic_loss(model: nn.Module, params: Dict[str, Any], ic_batch: jnp.ndarray) -> jnp.ndarray:
    """Compute initial condition loss for h=0, hu=0, hv=0 at t=0."""
    U_pred = model.apply({'params': params['params']}, ic_batch, train=False)
    return jnp.mean(U_pred**2)

# ==========================================
# 2. ATOMIC BOUNDARY LOSSES (New)
# ==========================================

def loss_boundary_dirichlet_h(model: nn.Module, params: Dict[str, Any],
                              batch: jnp.ndarray,
                              h_target: jnp.ndarray) -> jnp.ndarray:
    """
    Enforces a prescribed water level h.
    The target 'h_target' can be constant or dynamic (time-varying vector).
    """
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    h_pred = U_pred[..., 0]
    
    if h_target.ndim != h_pred.ndim:
        h_target = h_target.squeeze()
        
    return jnp.mean((h_pred - h_target)**2)


def loss_boundary_dirichlet_hu(model: nn.Module, params: Dict[str, Any],
                               batch: jnp.ndarray,
                               hu_target: jnp.ndarray) -> jnp.ndarray:
    """
    Enforces a prescribed momentum hu.
    The target 'hu_target' can be constant or dynamic.
    """
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    hu_pred = U_pred[..., 1]
    
    if hu_target.ndim != hu_pred.ndim:
        hu_target = hu_target.squeeze()
        
    return jnp.mean((hu_pred - hu_target)**2)

def loss_boundary_dirichlet_hv(model: nn.Module, params: Dict[str, Any],
                               batch: jnp.ndarray,
                               hv_target: jnp.ndarray) -> jnp.ndarray:
    """
    Enforces a prescribed momentum hv.
    The target 'hv_target' can be constant or dynamic.
    """
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    hv_pred = U_pred[..., 2]
    
    if hv_target.ndim != hv_pred.ndim:
        hv_target = hv_target.squeeze()
        
    return jnp.mean((hv_pred - hv_target)**2)


def loss_boundary_wall_slip_general(model: nn.Module, params: Dict[str, Any],
                                    wall_batch: jnp.ndarray,
                                    normal_vectors: jnp.ndarray) -> jnp.ndarray:
    """
    General 'Slip' condition for walls of ANY orientation.
    Enforces dot(velocity, normal) = 0.
    
    Args:
        wall_batch: Shape (N, 3) -> [x, y, t]
        normal_vectors: Shape (N, 2) -> [nx, ny]
    """
    U_pred = model.apply({'params': params['params']}, wall_batch, train=False)
    
    h_pred = U_pred[..., 0]
    hu_pred = U_pred[..., 1]
    hv_pred = U_pred[..., 2]
    
    # Calculate Velocity (u, v) with safety epsilon
    # We use a small epsilon to avoid division by zero in dry areas
    h_safe = jnp.maximum(h_pred, 1e-6)
    u_pred = hu_pred / h_safe
    v_pred = hv_pred / h_safe
    
    # Stack velocity vector: Shape (N, 2)
    vel_vectors = jnp.stack([u_pred, v_pred], axis=-1)
    
    # Compute dot product: v . n
    # sum along the last axis (components)
    normal_flux = jnp.sum(vel_vectors * normal_vectors, axis=-1)
    
    return jnp.mean(normal_flux**2)


def loss_boundary_wall_vertical(model: nn.Module, params: Dict[str, Any],
                                batch: jnp.ndarray) -> jnp.ndarray:
    """
    Optimization: Enforces hu = 0 (No flow through X-boundary).
    Equivalent to general slip with normal=[1,0] or [-1,0], but more stable.
    """
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    hu_pred = U_pred[..., 1]
    return jnp.mean(hu_pred**2)


def loss_boundary_wall_horizontal(model: nn.Module, params: Dict[str, Any],
                                  batch: jnp.ndarray) -> jnp.ndarray:
    """
    Optimization: Enforces hv = 0 (No flow through Y-boundary).
    Equivalent to general slip with normal=[0,1] or [0,-1], but more stable.
    """
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    hv_pred = U_pred[..., 2]
    return jnp.mean(hv_pred**2)


def loss_boundary_neumann_outflow_x(model: nn.Module, params: Dict[str, Any],
                                    batch: jnp.ndarray) -> jnp.ndarray:
    """
    Enforces Zero Gradient ONLY in the x-direction (Normal to boundary).
    Mathematically: d(U)/dx = 0.
    """
    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)
    
    # Calculate Jacobian: [N, 3 (outputs), 3 (inputs)]
    jac_U = jax.vmap(jax.jacfwd(U_fn))(batch)
    
    # Gradient w.r.t X is index 0
    dU_dx = jac_U[..., 0] 
    
    return jnp.mean(dU_dx**2)


# ==========================================
# 3. LEGACY / COMPATIBILITY WRAPPERS
# ==========================================

def compute_bc_loss(model: nn.Module, params: Dict[str, Any],
                    left_batch: jnp.ndarray, right_batch: jnp.ndarray,
                    bottom_batch: jnp.ndarray, top_batch: jnp.ndarray,
                    config: FrozenDict,
                    bc_fn: Optional[Callable] = None) -> jnp.ndarray:
    """
    Legacy wrapper that composes the atomic losses for standard rectangular domains.
    """
    # 1. Left Boundary
    if bc_fn is not None:
        # Test 1: Time-varying Height Only
        t_left = left_batch[..., 2]
        h_target = bc_fn(t_left)
        loss_left = loss_boundary_dirichlet_h(model, params, left_batch, h_target)
    else:
        # Analytical: Exact H and HU
        u_const = config["physics"]["u_const"]
        n_manning = config["physics"]["n_manning"]
        t_left = left_batch[..., 2]
        h_true = h_exact(0.0, t_left, n_manning, u_const)
        hu_true = h_true * u_const
        loss_left = (loss_boundary_dirichlet_h(model, params, left_batch, h_true) + 
                     loss_boundary_dirichlet_hu(model, params, left_batch, hu_true))

    # 2. Right Boundary (Neumann Outflow)
    loss_right = loss_boundary_neumann_outflow_x(model, params, right_batch)

    # 3. Top/Bottom (Horizontal Walls)
    loss_bottom = loss_boundary_wall_horizontal(model, params, bottom_batch)
    loss_top = loss_boundary_wall_horizontal(model, params, top_batch)

    return loss_left + loss_right + loss_bottom + loss_top

def compute_building_bc_loss(model: nn.Module, params: Dict[str, Any],
                             building_left_batch: jnp.ndarray,
                             building_right_batch: jnp.ndarray,
                             building_bottom_batch: jnp.ndarray,
                             building_top_batch: jnp.ndarray) -> jnp.ndarray:
    """
    Computes slip loss for a rectangular building obstacle.
    """
    loss_left = loss_boundary_wall_vertical(model, params, building_left_batch)
    loss_right = loss_boundary_wall_vertical(model, params, building_right_batch)
    loss_bottom = loss_boundary_wall_horizontal(model, params, building_bottom_batch)
    loss_top = loss_boundary_wall_horizontal(model, params, building_top_batch)

    return loss_left + loss_right + loss_bottom + loss_top

def compute_data_loss(model: nn.Module, params: Dict[str, Any], data_batch: jnp.ndarray, config: FrozenDict) -> jnp.ndarray:
    """Data loss for sparse observations."""
    points_batch = data_batch[:, [1, 2, 0]] # x, y, t
    h_true = data_batch[:, 3]
    u_true = data_batch[:, 4]
    v_true = data_batch[:, 5]

    U_pred = model.apply({'params': params['params']}, points_batch, train=False)
    h_pred = U_pred[..., 0]
    hu_pred = U_pred[..., 1]
    hv_pred = U_pred[..., 2]
    
    # Recalculate true momentum for consistency
    eps = config["numerics"]["eps"]
    h_true_safe = jnp.maximum(h_true, eps)
    hu_true = h_true_safe * u_true
    hv_true = h_true_safe * v_true

    return (jnp.mean((h_pred - h_true)**2) + 
            jnp.mean((hu_pred - hu_true)**2) + 
            jnp.mean((hv_pred - hv_true)**2))

def total_loss(terms: Dict[str, jnp.ndarray], weights: Dict[str, float]) -> jnp.ndarray:
    loss = 0.0
    for key in terms.keys():
        if key in weights:
             loss += weights[key] * terms[key]
    return loss