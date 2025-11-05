# src/losses.py
import jax
import jax.numpy as jnp
import jax.nn
from flax import linen as nn
from flax.core import FrozenDict
from typing import Dict, Any

from src.physics import SWEPhysics, h_exact
from src.utils import mask_points_inside_building # <-- Import the masking function

def compute_pde_loss(model: nn.Module, params: Dict[str, Any], pde_batch: jnp.ndarray, config: FrozenDict) -> jnp.ndarray:
    """Compute the PDE residual mean squared error (MSE) for the SWE."""
    # Ensure pde_batch has the correct shape (N, 3) for model input
    if pde_batch.shape[-1] != 3:
        raise ValueError(f"PDE batch requires shape (N, 3), but got {pde_batch.shape}")
    U_pred = model.apply({'params': params['params']}, pde_batch, train=True)
    def U_fn(pts):
        # Ensure input points for jacfwd also have shape (N, 3)
        return model.apply({'params': params['params']}, pts, train=False)

    jac_U = jax.vmap(jax.jacfwd(U_fn))(pde_batch)
    dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]

    eps = config["numerics"]["eps"]
    physics = SWEPhysics(U_pred, eps=eps)

    g = config["physics"]["g"]
    n_manning = config["physics"]["n_manning"]
    inflow = config["physics"]["inflow"]

    JF, JG = physics.flux_jac(g=g)
    div_F = jnp.einsum('nij,nj->ni', JF, dU_dx)
    div_G = jnp.einsum('nij,nj->ni', JG, dU_dy)
    S = physics.source(g=g, n_manning=n_manning, inflow=inflow)

    residual = (dU_dt + div_F + div_G - S)

    # Mask for physical realism (zero residual where water depth is near zero)
    h_mask = jnp.where(U_pred[..., 0] < eps, 0.0, 1.0)

    # --- NEW: EFFICIENT MASKING INSIDE JIT-COMPILED FUNCTION ---
    # Check if a building is defined in the config
    if "building" in config:
        # Create a boolean mask (True for points OUTSIDE the building)
        # We need to reshape it to (batch_size, 1) to allow broadcasting with the residual
        building_mask = mask_points_inside_building(pde_batch, config["building"])[..., None]

        # Apply both the water depth mask and the building mask.
        # The residual for points inside the building will become zero.
        final_residual = residual * h_mask[..., None] * building_mask
    else:
        # If no building, just apply the water depth mask
        final_residual = residual * h_mask[..., None]

    return jnp.mean(final_residual ** 2)

def compute_neg_h_loss(model: nn.Module, params: Dict[str, Any], pde_points: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a penalty for negative water height (h_pred < 0).
    Uses a quadratic penalty on the negative part: mean(relu(-h_pred)^2)
    """
    # We use the pde_batch as it covers the whole domain/time
    U_pred = model.apply({'params': params['params']}, pde_points, train=False)
    h_pred = U_pred[..., 0]
    
    # jnp.relu(-h_pred) is equivalent to max(0, -h_pred)
    # This is 0 if h_pred is positive, and |-h_pred| if h_pred is negative.
    neg_h_penalty = jax.nn.relu(-h_pred)
    
    # Return the mean squared penalty
    return jnp.mean(neg_h_penalty ** 2)


def compute_ic_loss(model: nn.Module, params: Dict[str, Any], ic_batch: jnp.ndarray) -> jnp.ndarray:
    """Compute initial condition loss for h=0, hu=0, hv=0 at t=0."""
    # Ensure ic_batch has the correct shape (N, 3) for model input
    if ic_batch.shape[-1] != 3:
         raise ValueError(f"IC batch requires shape (N, 3), but got {ic_batch.shape}")
    U_pred = model.apply({'params': params['params']}, ic_batch, train=False)
    # Target is U = [0, 0, 0]
    err = U_pred[..., 0]**2 + U_pred[..., 1]**2 + U_pred[..., 2]**2
    return jnp.mean(err)

def compute_bc_loss(model: nn.Module, params: Dict[str, Any],
                    left_batch: jnp.ndarray, right_batch: jnp.ndarray,
                    bottom_batch: jnp.ndarray, top_batch: jnp.ndarray,
                    config: FrozenDict) -> jnp.ndarray:
    """Compute boundary condition loss for all domain boundaries."""
    # Ensure BC batches have the correct shape (N, 3) for model input
    for name, batch in [('left', left_batch), ('right', right_batch), ('bottom', bottom_batch), ('top', top_batch)]:
        if batch.shape[-1] != 3:
             raise ValueError(f"BC batch '{name}' requires shape (N, 3), but got {batch.shape}")

    u_const = config["physics"]["u_const"]
    n_manning = config["physics"]["n_manning"]

    # Left Boundary (Dirichlet for h and hu)
    U_left = model.apply({'params': params['params']}, left_batch, train=False)
    h_pred_left, hu_pred_left = U_left[..., 0], U_left[..., 1]
    t_left = left_batch[..., 2] # Time coordinate is the 3rd column (index 2)
    h_true_left = h_exact(0.0, t_left, n_manning, u_const) # x=0 for left boundary
    hu_true_left = h_true_left * u_const
    res_left_h = h_pred_left - h_true_left
    res_left_hu = hu_pred_left - hu_true_left

    # Right Boundary (Zero Gradient)
    def U_fn_right(pts):
        return model.apply({'params': params['params']}, pts, train=False)
    # Compute Jacobian w.r.t inputs (x, y, t)
    jac_U_right = jax.vmap(jax.jacfwd(U_fn_right))(right_batch)
    # dU/dx corresponds to the gradient along the first input dimension (index 0)
    dU_dx_right = jac_U_right[..., 0] # Shape: (batch_size, output_dim=3)
    res_right_grad = dU_dx_right # Target is zero gradient [0, 0, 0]

    # Bottom Boundary (No-flux: hv = 0)
    U_bottom = model.apply({'params': params['params']}, bottom_batch, train=False)
    res_bottom_hv = U_bottom[..., 2] # Target is hv = 0

    # Top Boundary (No-flux: hv = 0)
    U_top = model.apply({'params': params['params']}, top_batch, train=False)
    res_top_hv = U_top[..., 2] # Target is hv = 0

    loss = (jnp.mean(res_left_h**2) + jnp.mean(res_left_hu**2) +
            jnp.mean(res_right_grad**2) + # Compare all components [dh/dx, d(hu)/dx, d(hv)/dx] to zero
            jnp.mean(res_bottom_hv**2) +
            jnp.mean(res_top_hv**2))
    return loss

def compute_building_bc_loss(model: nn.Module, params: Dict[str, Any],
                             building_left_batch: jnp.ndarray,
                             building_right_batch: jnp.ndarray,
                             building_bottom_batch: jnp.ndarray,
                             building_top_batch: jnp.ndarray) -> jnp.ndarray:
    """
    Compute slip boundary condition loss for a rectangular building obstacle.
    Ensures normal velocity component is zero at each wall.
    """
    # Ensure Building BC batches have the correct shape (N, 3) for model input
    for name, batch in [('left', building_left_batch), ('right', building_right_batch), ('bottom', building_bottom_batch), ('top', building_top_batch)]:
        if batch.ndim == 0 or batch.shape[-1] != 3: # Check also if batch is potentially empty/scalar
             raise ValueError(f"Building BC batch '{name}' requires shape (N, 3), but got {batch.shape}")

    # Left wall (normal is in +x direction, so u=0 -> hu=0)
    U_left = model.apply({'params': params['params']}, building_left_batch, train=False)
    loss_left = jnp.mean(U_left[..., 1]**2)  # hu**2 should be 0

    # Right wall (normal is in -x direction, so u=0 -> hu=0)
    U_right = model.apply({'params': params['params']}, building_right_batch, train=False)
    loss_right = jnp.mean(U_right[..., 1]**2)  # hu**2 should be 0

    # Bottom wall (normal is in +y direction, so v=0 -> hv=0)
    U_bottom = model.apply({'params': params['params']}, building_bottom_batch, train=False)
    loss_bottom = jnp.mean(U_bottom[..., 2]**2)  # hv**2 should be 0

    # Top wall (normal is in -y direction, so v=0 -> hv=0)
    U_top = model.apply({'params': params['params']}, building_top_batch, train=False)
    loss_top = jnp.mean(U_top[..., 2]**2)  # hv**2 should be 0

    return loss_left + loss_right + loss_bottom + loss_top

# --- NEW: Function to compute data loss ---
def compute_data_loss(model: nn.Module, params: Dict[str, Any], data_batch: jnp.ndarray, config: FrozenDict) -> jnp.ndarray:
    """
    Compute the Mean Squared Error (MSE) loss between model predictions and provided data points.
    Assumes data_batch has shape (N, 6) with columns [t, x, y, h, u, v].
    We need to predict [h, hu, hv].
    """
    # Ensure data_batch has the correct shape (N, 6)
    if data_batch.shape[-1] != 6:
        raise ValueError(f"Data batch requires shape (N, 6), but got {data_batch.shape}")

    # Extract input points (t, x, y) - Note the order change to match model input (x, y, t)
    points_batch = data_batch[:, [1, 2, 0]] # Columns x, y, t
    # Extract true output values (h, u, v)
    h_true = data_batch[:, 3]
    u_true = data_batch[:, 4]
    v_true = data_batch[:, 5]

    # Predict U = [h, hu, hv] using the model
    U_pred = model.apply({'params': params['params']}, points_batch, train=False)
    h_pred = U_pred[..., 0]
    hu_pred = U_pred[..., 1]
    hv_pred = U_pred[..., 2]

    # Calculate true hu and hv from the data
    eps = config["numerics"]["eps"]
    # Ensure h_true is safe for division, though it's used multiplicatively here
    h_true_safe = jnp.maximum(h_true, eps)
    hu_true = h_true_safe * u_true
    hv_true = h_true_safe * v_true

    # Calculate the MSE between predicted [h, hu, hv] and true [h, hu, hv]
    loss_h = jnp.mean((h_pred - h_true)**2)
    loss_hu = jnp.mean((hu_pred - hu_true)**2)
    loss_hv = jnp.mean((hv_pred - hv_true)**2)

    # Combine the losses (could also return individual components if needed)
    total_data_loss = loss_h + loss_hu + loss_hv
    return total_data_loss
# --- END NEW ---

def total_loss(terms: Dict[str, jnp.ndarray], weights: Dict[str, float]) -> jnp.ndarray:
    """Compute the weighted sum of loss terms."""
    loss = (weights.get('pde', 0.0) * terms.get('pde', 0.0) +
            weights.get('ic', 0.0) * terms.get('ic', 0.0) +
            weights.get('bc', 0.0) * terms.get('bc', 0.0))

    # Conditionally add building loss if it exists in both terms and weights
    if 'building_bc' in terms and 'building_bc' in weights:
        loss += weights['building_bc'] * terms.get('building_bc', 0.0)

    # --- NEW: Conditionally add data loss ---
    if 'data' in terms and 'data' in weights:
        loss += weights['data'] * terms.get('data', 0.0)
    # --- END NEW ---

    return loss