# src/losses_deeponet.py
# This file is parallel to losses.py, but adapted for DeepONet.
# Loss functions now accept branch_batch and trunk_batch.

import jax
import jax.numpy as jnp
import jax.nn
from flax import linen as nn
from flax.core import FrozenDict
from typing import Dict, Any

# Import the new physics class and the analytical solver
from src.physics_deeponet import SWEPhysics_DeepONet, h_exact
from src.config import DTYPE

def _get_params_from_batch(branch_batch: jnp.ndarray, config: FrozenDict) -> Dict[str, jnp.ndarray]:
    """Extracts parameter arrays (n_manning, u_const) from the branch batch."""
    param_names = tuple(config["physics"]["param_bounds"].keys())
    param_map = {name: i for i, name in enumerate(param_names)}
    
    n_manning_idx = param_map.get('n_manning')
    u_const_idx = param_map.get('u_const')
    
    # Get the static value from config as a fallback
    n_manning_default = config["physics"]["n_manning"]
    u_const_default = config["physics"]["u_const"]
    
    # Extract arrays from the batch if they exist, else use default
    # Ensure array has shape (N,)
    n_manning = branch_batch[..., n_manning_idx] if n_manning_idx is not None else jnp.full(branch_batch.shape[0], n_manning_default, dtype=DTYPE)
    u_const = branch_batch[..., u_const_idx] if u_const_idx is not None else jnp.full(branch_batch.shape[0], u_const_default, dtype=DTYPE)
    
    return {'n_manning': n_manning, 'u_const': u_const}

def compute_pde_loss_deeponet(model: nn.Module, params: Dict[str, Any],
                              branch_batch: jnp.ndarray,
                              trunk_batch: jnp.ndarray,
                              config: FrozenDict) -> jnp.ndarray:
    """Compute the PDE residual MSE for the DeepONet."""
    
    # 1. Define the function to differentiate: U(params, coords)
    def U_fn(b_batch, t_batch):
        return model.apply({'params': params['params']}, b_batch, t_batch, train=False)

    # 2. Get gradients w.r.t. TRUNK inputs (argnums=1)
    # jac_U will have shape (N, 3, 3) -> (batch, output_dim, input_dim)
    jac_U = jax.vmap(jax.jacfwd(U_fn, argnums=1))(branch_batch, trunk_batch)
    dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]

    # 3. Get model prediction
    U_pred = U_fn(branch_batch, trunk_batch)

    # 4. Get dynamic physics parameters
    phys_params = _get_params_from_batch(branch_batch, config)
    eps = config["numerics"]["eps"]
    g = config["physics"]["g"]
    inflow = config["physics"]["inflow"]

    # 5. Compute physics terms
    physics = SWEPhysics_DeepONet(U_pred, eps=eps)
    JF, JG = physics.flux_jac(g=g)
    
    # n_manning is now an array (N,)
    S = physics.source(g=g, n_manning=phys_params['n_manning'], inflow=inflow)

    div_F = jnp.einsum('nij,nj->ni', JF, dU_dx)
    div_G = jnp.einsum('nij,nj->ni', JG, dU_dy)
    
    residual = (dU_dt + div_F + div_G - S)
    h_mask = jnp.where(U_pred[..., 0] < eps, 0.0, 1.0)
    
    # No building mask for analytical scenario
    final_residual = residual * h_mask[..., None]

    return jnp.mean(final_residual ** 2)

def compute_neg_h_loss_deeponet(model: nn.Module, params: Dict[str, Any],
                                branch_batch: jnp.ndarray,
                                trunk_batch: jnp.ndarray,
                                config: FrozenDict) -> jnp.ndarray:
    """Compute a penalty for negative water height (h_pred < 0)."""
    U_pred = model.apply({'params': params['params']}, branch_batch, trunk_batch, train=False)
    h_pred = U_pred[..., 0]
    neg_h_penalty = jax.nn.relu(-h_pred)
    return jnp.mean(neg_h_penalty ** 2)


def compute_ic_loss_deeponet(model: nn.Module, params: Dict[str, Any],
                             branch_batch: jnp.ndarray,
                             trunk_batch: jnp.ndarray,
                             config: FrozenDict) -> jnp.ndarray:
    """Compute initial condition loss for h=0, hu=0, hv=0 at t=0."""
    U_pred = model.apply({'params': params['params']}, branch_batch, trunk_batch, train=False)
    # Target is U = [0, 0, 0]
    err = U_pred[..., 0]**2 + U_pred[..., 1]**2 + U_pred[..., 2]**2
    return jnp.mean(err)

def compute_bc_loss_deeponet(model: nn.Module, params: Dict[str, Any],
                             batches: Dict[str, jnp.ndarray],
                             config: FrozenDict) -> jnp.ndarray:
    """Compute boundary condition loss for all domain boundaries."""
    
    # Extract dynamic parameters from the branch batches
    params_left = _get_params_from_batch(batches['branch_left'], config)
    
    # Left Boundary (Dirichlet for h and hu)
    U_left = model.apply({'params': params['params']}, batches['branch_left'], batches['trunk_left'], train=False)
    h_pred_left, hu_pred_left = U_left[..., 0], U_left[..., 1]
    
    t_left = batches['trunk_left'][..., 2] # Time coordinate
    
    # Use parameter arrays from branch batch to calculate true solution
    h_true_left = h_exact(0.0, t_left, params_left['n_manning'], params_left['u_const'])
    hu_true_left = h_true_left * params_left['u_const']
    
    res_left_h = h_pred_left - h_true_left
    res_left_hu = hu_pred_left - hu_true_left

    # Right Boundary (Zero Gradient)
    def U_fn_right(b, t):
        return model.apply({'params': params['params']}, b, t, train=False)
    jac_U_right = jax.vmap(jax.jacfwd(U_fn_right, argnums=1))(batches['branch_right'], batches['trunk_right'])
    dU_dx_right = jac_U_right[..., 0]
    res_right_grad = dU_dx_right

    # Bottom Boundary (No-flux: hv = 0)
    U_bottom = model.apply({'params': params['params']}, batches['branch_bottom'], batches['trunk_bottom'], train=False)
    res_bottom_hv = U_bottom[..., 2]

    # Top Boundary (No-flux: hv = 0)
    U_top = model.apply({'params': params['params']}, batches['branch_top'], batches['trunk_top'], train=False)
    res_top_hv = U_top[..., 2]

    loss = (jnp.mean(res_left_h**2) + jnp.mean(res_left_hu**2) +
            jnp.mean(res_right_grad**2) +
            jnp.mean(res_bottom_hv**2) +
            jnp.mean(res_top_hv**2))
    return loss

# --- Building and Data loss functions are not needed for this script ---
# We keep the total_loss function simple
def total_loss(terms: Dict[str, jnp.ndarray], weights: Dict[str, float]) -> jnp.ndarray:
    """Compute the weighted sum of loss terms."""
    loss = 0.0
    for key in terms.keys():
        # Only add loss if it's in weights (i.e., it's an active term)
        loss += weights.get(key, 0.0) * terms.get(key, 0.0)
    return loss