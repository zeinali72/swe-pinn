# src/gradnorm.py
import jax
import jax.numpy as jnp
import flax.struct
import optax
from flax.core import FrozenDict
from typing import Dict, Any, Callable, Tuple, List
import chex # For type hinting

from src.losses import compute_pde_loss, compute_ic_loss, compute_bc_loss, compute_building_bc_loss, compute_data_loss
from src.config import DTYPE

# Define which layer parameters to use for GradNorm (e.g., the final Dense layer)
# This might need adjustment based on your specific model naming conventions.
SHARED_LAYER_NAME = 'output_layer'

@flax.struct.dataclass
class GradNormState:
    weights: chex.ArrayTree # Trainable loss weights (w_i)
    initial_losses: Dict[str, float] # L_i(0) for each loss term
    opt_state: optax.OptState # Optimizer state for the weights

def init_gradnorm(loss_keys: List[str], initial_losses: Dict[str, float], gradnorm_lr: float) -> GradNormState:
    """Initializes the GradNorm state."""
    num_losses = len(loss_keys)
    # Initialize weights to 1.0
    weights = jnp.ones(num_losses, dtype=DTYPE)

    # Ensure initial losses have a minimum value to avoid division by zero
    processed_initial_losses = {k: max(v, 1e-8) for k, v in initial_losses.items()}

    optimizer = optax.adam(learning_rate=gradnorm_lr)
    opt_state = optimizer.init(weights)

    return GradNormState(
        weights=weights,
        initial_losses=processed_initial_losses,
        opt_state=opt_state
    )

def _get_shared_layer_params(params: FrozenDict) -> chex.ArrayTree:
    """Extracts parameters of the specified shared layer."""
    # This assumes the parameters are nested under 'params'. Adjust if needed.
    if SHARED_LAYER_NAME not in params['params']:
        raise ValueError(f"Shared layer '{SHARED_LAYER_NAME}' not found in model parameters. Available layers: {list(params['params'].keys())}")
    return params['params'][SHARED_LAYER_NAME]

# Define individual loss functions compatible with jax.value_and_grad
# These need to accept model, params, batch, config etc. as required by original functions
# They should return a single scalar loss value.

def pde_loss_fn(params, model, batch, config):
    return compute_pde_loss(model, params, batch, config)

def ic_loss_fn(params, model, batch, config):
    return compute_ic_loss(model, params, batch) # Config not needed? Check src/losses.py

def bc_loss_fn(params, model, batches, config):
    return compute_bc_loss(model, params, batches['left'], batches['right'], batches['bottom'], batches['top'], config)

def building_bc_loss_fn(params, model, batches, config):
    # Need to handle potentially empty batches inside compute_building_bc_loss or here
    return compute_building_bc_loss(model, params,
                                     batches.get('left', jnp.empty((0,3), dtype=DTYPE)),
                                     batches.get('right', jnp.empty((0,3), dtype=DTYPE)),
                                     batches.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                                     batches.get('top', jnp.empty((0,3), dtype=DTYPE)))

def data_loss_fn(params, model, batch, config):
    return compute_data_loss(model, params, batch, config)

# Map loss keys to their respective functions and required batch data
LOSS_FN_MAP = {
    'pde': {'func': pde_loss_fn, 'batch_key': 'pde'},
    'ic': {'func': ic_loss_fn, 'batch_key': 'ic'},
    'bc': {'func': bc_loss_fn, 'batch_key': 'bc'}, # BC needs dict of batches
    'building_bc': {'func': building_bc_loss_fn, 'batch_key': 'building_bc'}, # Bldg BC needs dict
    'data': {'func': data_loss_fn, 'batch_key': 'data'}
}

@jax.jit
def _compute_gradient_norm(grads: chex.ArrayTree) -> jnp.ndarray:
    """Computes the L2 norm of the gradients (flattened)."""
    # Flatten the gradients PyTree and concatenate into a single vector
    leaves, _ = jax.tree_util.tree_flatten(grads)
    if not leaves:
        return jnp.array(0.0, dtype=DTYPE)
    flat_grads = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
    return jnp.linalg.norm(flat_grads)


# This function should NOT be jitted as it involves loops and potentially dynamic logic
def update_gradnorm_weights(
    gradnorm_state: GradNormState,
    model_params: FrozenDict, # Current model parameters
    model: Any, # The model instance (e.g., FourierPINN)
    all_batches: Dict[str, Any], # Dict containing batches for all loss terms
    config: FrozenDict,
    alpha: float,
    gradnorm_lr: float # Learning rate for GradNorm weight updates
) -> Tuple[GradNormState, Dict[str, float]]:
    """
    Computes individual loss gradients and updates GradNorm weights.
    Returns the updated state and the dictionary of new weights.
    """
    loss_keys = list(gradnorm_state.initial_losses.keys())
    num_losses = len(loss_keys)
    current_losses = {}
    grads_wrt_shared_layer = {}

    # --- 1. Calculate current losses and gradients w.r.t. shared layer ---
    shared_layer_params_structure = jax.tree_util.tree_map(lambda x: x, _get_shared_layer_params(model_params))

    for key in loss_keys:
        if key not in LOSS_FN_MAP:
            print(f"Warning: Loss key '{key}' from initial losses not found in LOSS_FN_MAP. Skipping.")
            continue

        loss_info = LOSS_FN_MAP[key]
        loss_func = loss_info['func']
        batch_key = loss_info['batch_key']
        batch_data = all_batches.get(batch_key)

        # Handle cases where batch data might be missing or empty
        is_empty_batch = False
        if batch_data is None:
            is_empty_batch = True
        elif isinstance(batch_data, jnp.ndarray) and batch_data.shape[0] == 0:
            is_empty_batch = True
        elif isinstance(batch_data, dict) and not any(b.shape[0] > 0 for b in batch_data.values() if isinstance(b, jnp.ndarray)):
             is_empty_batch = True # For dict batches like BC, Bldg BC

        if is_empty_batch:
            current_losses[key] = 0.0
            # Set gradients to zero structure
            grads_wrt_shared_layer[key] = jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params_structure)
            print(f"Warning: Empty or missing batch for '{key}'. Loss and gradients set to zero.")
            continue

        # Compute loss and gradient for the specific task w.r.t. the shared layer ONLY
        # We need `value_and_grad` based on the specific loss function
        # The `argnums=0` assumes the loss function takes `params` as the first argument
        # We need has_aux=False as loss functions return scalar
        grad_fn = jax.value_and_grad(loss_func, argnums=0, has_aux=False)

        # Pass necessary arguments to the loss function
        # Need to handle different signatures (e.g., bc needs a dict)
        if key in ['bc', 'building_bc']:
            loss_val, full_grads = grad_fn(model_params, model, batch_data, config)
        else:
            loss_val, full_grads = grad_fn(model_params, model, batch_data, config)

        current_losses[key] = float(loss_val) # Store as float
        # Extract gradients only for the shared layer
        grads_wrt_shared_layer[key] = _get_shared_layer_params(full_grads)

    # Convert current_losses dict values to a jnp array in correct order
    current_losses_array = jnp.array([current_losses.get(k, 0.0) for k in loss_keys], dtype=DTYPE)
    initial_losses_array = jnp.array([gradnorm_state.initial_losses.get(k, 1e-8) for k in loss_keys], dtype=DTYPE)


    # --- 2. Calculate GradNorm Loss (L_grad) ---
    def gradnorm_loss_calculation(current_weights):
        # Ensure weights are positive
        w = jnp.abs(current_weights)

        # L_i(t) / L_i(0)
        loss_ratios = current_losses_array / initial_losses_array
        # Handle potential NaNs if initial loss was near zero despite clipping
        loss_ratios = jnp.nan_to_num(loss_ratios, nan=1.0) # Assume ratio is 1 if initial loss was ~0

        # r_i(t) = relative inverse training rate
        mean_loss_ratio = jnp.mean(loss_ratios)
        # Avoid division by zero if all loss ratios are zero
        mean_loss_ratio = jnp.maximum(mean_loss_ratio, 1e-8)
        relative_inverse_rates = loss_ratios / mean_loss_ratio

        # G_i = || grad(w_i * L_i) || = w_i * || grad(L_i) || (since w_i is scalar)
        grad_norms = jnp.array([_compute_gradient_norm(grads_wrt_shared_layer[k]) for k in loss_keys], dtype=DTYPE)
        weighted_grad_norms = w * grad_norms

        # G_avg = mean(G_j)
        mean_weighted_grad_norm = jnp.mean(weighted_grad_norms)

        # Target gradient norms: G_avg * (r_i)^alpha
        target_grad_norms = mean_weighted_grad_norm * (relative_inverse_rates ** alpha)

        # L_grad = sum | G_i - target_grad_norms | (L1 loss)
        gradnorm_loss = jnp.sum(jnp.abs(weighted_grad_norms - target_grad_norms))

        return gradnorm_loss

    # --- 3. Compute gradient of L_grad w.r.t current_weights and update ---
    gradnorm_loss_val, gradnorm_grads = jax.value_and_grad(gradnorm_loss_calculation)(gradnorm_state.weights)
    optimizer = optax.adam(learning_rate=gradnorm_lr) # Recreate optimizer with potentially new LR
    updates, new_opt_state = optimizer.update(gradnorm_grads, gradnorm_state.opt_state, gradnorm_state.weights)
    new_weights_unnormalized = optax.apply_updates(gradnorm_state.weights, updates)

    # --- 4. Renormalize weights ---
    # Ensure weights remain positive and sum to num_losses
    new_weights_positive = jnp.maximum(new_weights_unnormalized, 0) # Ensure positivity
    weight_sum = jnp.sum(new_weights_positive)
    # Avoid division by zero if all weights become zero
    weight_sum = jnp.maximum(weight_sum, 1e-8)
    new_weights_normalized = num_losses * new_weights_positive / weight_sum

    # --- 5. Update state ---
    updated_state = GradNormState(
        weights=new_weights_normalized,
        initial_losses=gradnorm_state.initial_losses, # Keep initial losses constant
        opt_state=new_opt_state
    )

    # Create dict for easy use in train_step
    new_weights_dict = {key: float(w) for key, w in zip(loss_keys, new_weights_normalized)}

    # Optional: Log GradNorm specific values if needed
    # print(f"GradNorm Update: L_grad={gradnorm_loss_val:.4e}, Raw W={new_weights_unnormalized}, Norm W={new_weights_normalized}")

    return updated_state, new_weights_dict