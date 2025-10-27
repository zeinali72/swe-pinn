# src/gradnorm.py
import jax
import jax.numpy as jnp
import flax.struct
import optax
from flax.core import FrozenDict
from typing import Dict, Any, Callable, Tuple, List
import chex # For type hinting

# --- Import specific loss computation functions directly ---
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss,
    compute_building_bc_loss, compute_data_loss
)
from src.config import DTYPE

# Define which layer parameters to use for GradNorm (e.g., the final Dense layer)
SHARED_LAYER_NAME = 'output_layer'  # Adjust based on actual model architecture

@flax.struct.dataclass
class GradNormState:
    weights: chex.ArrayTree # Trainable loss weights (w_i)
    initial_losses: Dict[str, float] # L_i(0) for each loss term
    opt_state: optax.OptState # Optimizer state for the weights

def init_gradnorm(loss_keys: List[str], initial_losses: Dict[str, float], gradnorm_lr: float) -> GradNormState:
    """Initializes the GradNorm state."""
    num_losses = len(loss_keys)
    # Initialize weights to 1.0, matching the structure of loss_keys
    weights = jnp.ones(num_losses, dtype=DTYPE)

    # Ensure initial losses (provided based on loss_keys) have a minimum value
    processed_initial_losses = {k: max(initial_losses[k], 1e-8) for k in loss_keys}

    optimizer = optax.adam(learning_rate=gradnorm_lr)
    opt_state = optimizer.init(weights)

    return GradNormState(
        weights=weights,
        initial_losses=processed_initial_losses, # Store dict with keys
        opt_state=opt_state
    )

def _get_shared_layer_params(params: FrozenDict) -> chex.ArrayTree:
    """Extracts parameters of the specified shared layer."""
    # This assumes the parameters are nested under 'params'.
    # Flax names layers in lists as LayerName_index (e.g., Dense_0, Dense_1...)
    # Access the specific dense layer directly by its name
    if SHARED_LAYER_NAME not in params['params']:
        available_layers = list(params['params'].keys())
        # Provide a more helpful error message
        raise ValueError(
            f"Shared layer '{SHARED_LAYER_NAME}' not found in model parameters. "
            f"Check model depth and naming. Available top-level layers: {available_layers}"
        )
    return params['params'][SHARED_LAYER_NAME]

# Define individual loss functions compatible with jax.value_and_grad
def pde_loss_fn(params, model, batch, config):
    return compute_pde_loss(model, params, batch, config)

def ic_loss_fn(params, model, batch, config):
    # No wrapping needed - compute_ic_loss expects params directly
    return compute_ic_loss(model, params, batch)

def bc_loss_fn(params, model, batches, config):
    # No wrapping needed - compute_bc_loss expects params directly
    return compute_bc_loss(model, params, batches['left'], batches['right'], batches['bottom'], batches['top'], config)

def building_bc_loss_fn(params, model, batches, config):
    # No wrapping needed - compute_building_bc_loss expects params directly
    return compute_building_bc_loss(model, params,
                                     batches.get('left', jnp.empty((0,3), dtype=DTYPE)),
                                     batches.get('right', jnp.empty((0,3), dtype=DTYPE)),
                                     batches.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                                     batches.get('top', jnp.empty((0,3), dtype=DTYPE)))

def data_loss_fn(params, model, batch, config):
    # No wrapping needed - compute_data_loss expects params directly
    return compute_data_loss(model, params, batch, config)

# Map loss keys to their respective functions and required batch data
LOSS_FN_MAP = {
    'pde': {'func': pde_loss_fn, 'batch_key': 'pde'},
    'ic': {'func': ic_loss_fn, 'batch_key': 'ic'},
    'bc': {'func': bc_loss_fn, 'batch_key': 'bc'}, # BC needs dict of batches
    'building_bc': {'func': building_bc_loss_fn, 'batch_key': 'building_bc'}, # Bldg BC needs dict
    'data': {'func': data_loss_fn, 'batch_key': 'data'} # Keep data here, but it won't be used if excluded in train_gradnorm.py
}

# Apply JIT functionally to _compute_gradient_norm
def _compute_gradient_norm_impl(grads: chex.ArrayTree) -> jnp.ndarray:
    """Computes the L2 norm of the gradients (flattened)."""
    leaves, _ = jax.tree_util.tree_flatten(grads)
    if not leaves:
        return jnp.array(0.0, dtype=DTYPE)
    # Ensure leaves are JAX arrays before raveling
    leaves_arrays = [jnp.asarray(leaf) for leaf in leaves]
    flat_grads = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves_arrays])
    norm = jnp.linalg.norm(flat_grads)
    # Ensure numerical stability - avoid exact zeros
    return jnp.maximum(norm, 1e-12)

_compute_gradient_norm = jax.jit(_compute_gradient_norm_impl)


# This function should NOT be jitted
def update_gradnorm_weights(
    gradnorm_state: GradNormState,
    model_params: FrozenDict, # Current model parameters
    model: Any, # The model instance (e.g., FourierPINN)
    all_batches: Dict[str, Any], # Dict containing batches for relevant loss terms
    config: FrozenDict,
    alpha: float,
    gradnorm_lr: float # Learning rate for GradNorm weight updates
) -> Tuple[GradNormState, Dict[str, float]]:
    """
    Computes individual loss gradients and updates GradNorm weights.
    Returns the updated state and the dictionary of new weights.
    """
    loss_keys = list(gradnorm_state.initial_losses.keys()) # Keys used for this GradNorm instance
    num_losses = len(loss_keys)
    current_losses = {}
    grads_wrt_shared_layer = {}

    # --- 1. Calculate current losses and gradients w.r.t. shared layer ---
    # Get the structure without unnecessary copying
    shared_layer_params = _get_shared_layer_params(model_params)

    for key in loss_keys: # Iterate only over relevant keys
        if key not in LOSS_FN_MAP:
            print(f"Internal Warning: Loss key '{key}' from GradNorm state not found in LOSS_FN_MAP. Skipping.")
            continue

        loss_info = LOSS_FN_MAP[key]
        loss_func = loss_info['func']
        batch_key = loss_info['batch_key']
        batch_data = all_batches.get(batch_key)

        # Handle cases where batch data might be missing or empty
        is_empty_batch = False
        if batch_data is None: is_empty_batch = True
        elif isinstance(batch_data, jnp.ndarray) and batch_data.shape[0] == 0: is_empty_batch = True
        elif isinstance(batch_data, dict) and not any(isinstance(b, jnp.ndarray) and b.shape[0] > 0 for b in batch_data.values()): is_empty_batch = True

        if is_empty_batch:
            current_losses[key] = 0.0
            grads_wrt_shared_layer[key] = jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params)
            print(f"GradNorm Warning: Empty or missing batch for '{key}'. Using zero loss/grads.")
            continue

        # Compute loss and gradient for the specific task w.r.t. model_params
        grad_fn = jax.value_and_grad(loss_func, argnums=0, has_aux=False)

        try:
            # Pass necessary arguments to the loss function
            loss_val, full_grads = grad_fn(model_params, model, batch_data, config)

            current_losses[key] = float(loss_val) # Store as float
            # Extract gradients only for the shared layer
            grads_wrt_shared_layer[key] = _get_shared_layer_params(full_grads)

        except Exception as e:
            print(f"Error computing loss/gradient for '{key}': {e}. Using zero loss/grads.")
            current_losses[key] = 0.0
            grads_wrt_shared_layer[key] = jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params)


    # Convert dict values to a jnp array in the order defined by loss_keys
    current_losses_array = jnp.array([current_losses.get(k, 0.0) for k in loss_keys], dtype=DTYPE)
    initial_losses_array = jnp.array([gradnorm_state.initial_losses[k] for k in loss_keys], dtype=DTYPE) # Already max(val, 1e-8)

    # --- 2. Calculate GradNorm Loss (L_grad) ---
    def gradnorm_loss_calculation(current_weights_array):
        # Ensure weights are positive and non-zero
        w = jnp.maximum(jnp.abs(current_weights_array), 1e-8)

        # L_i(t) / L_i(0)
        loss_ratios = current_losses_array / initial_losses_array
        loss_ratios = jnp.nan_to_num(loss_ratios, nan=1.0, posinf=1.0, neginf=1.0) # Handle all edge cases

        # r_i(t) = relative inverse training rate
        mean_loss_ratio = jnp.mean(loss_ratios)
        mean_loss_ratio = jnp.maximum(mean_loss_ratio, 1e-8) # Avoid division by zero
        relative_inverse_rates = loss_ratios / mean_loss_ratio

        # G_i = || grad(w_i * L_i) ||_W = w_i * || grad(L_i) ||_W
        grad_norms = jnp.array([_compute_gradient_norm(grads_wrt_shared_layer[k]) for k in loss_keys], dtype=DTYPE)
        weighted_grad_norms = w * grad_norms

        # G_avg = mean(G_j)
        mean_weighted_grad_norm = jnp.mean(weighted_grad_norms)
        mean_weighted_grad_norm = jnp.maximum(mean_weighted_grad_norm, 1e-8) # Avoid potential zero mean

        # Target gradient norms: G_avg * (r_i)^alpha
        target_grad_norms = mean_weighted_grad_norm * (relative_inverse_rates ** alpha)

        # L_grad = sum | G_i - target_grad_norms | (L1 loss)
        gradnorm_loss = jnp.sum(jnp.abs(weighted_grad_norms - target_grad_norms))

        return gradnorm_loss

    # --- 3. Compute gradient of L_grad w.r.t current_weights and update ---
    gradnorm_loss_val, gradnorm_grads = jax.value_and_grad(gradnorm_loss_calculation)(gradnorm_state.weights)
    optimizer = optax.adam(learning_rate=gradnorm_lr) # Optimizer for weights
    updates, new_opt_state = optimizer.update(gradnorm_grads, gradnorm_state.opt_state, gradnorm_state.weights)
    new_weights_unnormalized = optax.apply_updates(gradnorm_state.weights, updates)

    # --- 4. Renormalize weights ---
    new_weights_positive = jnp.maximum(new_weights_unnormalized, 1e-8) # Ensure positivity with small minimum
    weight_sum = jnp.sum(new_weights_positive)
    weight_sum = jnp.maximum(weight_sum, 1e-8) # Avoid division by zero
    new_weights_normalized = num_losses * new_weights_positive / weight_sum

    # --- 5. Update state ---
    updated_state = GradNormState(
        weights=new_weights_normalized,
        initial_losses=gradnorm_state.initial_losses, # Keep initial losses constant
        opt_state=new_opt_state
    )

    # Create dict for easy use in train_step
    new_weights_dict = {key: float(w) for key, w in zip(loss_keys, new_weights_normalized)}

    # Removed per-step weight printing

    return updated_state, new_weights_dict