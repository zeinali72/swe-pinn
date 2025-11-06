# src/gradnorm.py
import jax
import jax.numpy as jnp
import flax.struct
import optax
from flax.core import FrozenDict
from typing import Dict, Any, Callable, Tuple, List
import chex # For type hinting
import importlib

# --- Import specific loss computation functions directly ---
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss,
    compute_building_bc_loss, compute_data_loss,
    compute_neg_h_loss
)
from src.config import DTYPE

@flax.struct.dataclass
class GradNormState:
    weights: chex.ArrayTree
    initial_losses: Dict[str, float]
    opt_state: optax.OptState

def init_gradnorm(loss_keys: List[str], initial_losses: Dict[str, float], gradnorm_lr: float) -> GradNormState:
    """Initializes the GradNorm state."""
    num_losses = len(loss_keys)
    weights = jnp.ones(num_losses, dtype=DTYPE)
    processed_initial_losses = {k: max(initial_losses.get(k, 1e-8), 1e-8) for k in loss_keys}
    optimizer = optax.adam(learning_rate=gradnorm_lr)
    opt_state = optimizer.init(weights)

    return GradNormState(
        weights=weights,
        initial_losses=processed_initial_losses,
        opt_state=opt_state
    )

def _get_shared_layer_name(model: Any) -> str:
    """
    Returns the standardized name of the shared output layer used for GradNorm.
    All models (MLP, FourierPINN, DGMNetwork) must name this layer 'output_layer'.
    """
    return 'output_layer'

def _get_shared_layer_params(params: FrozenDict, shared_layer_name: str) -> chex.ArrayTree:
    """Extracts parameters of the specified shared layer."""
    if 'params' not in params:
        params = FrozenDict({'params': params})
    if shared_layer_name not in params['params']:
        available_layers = list(params['params'].keys())
        raise ValueError(
            f"Shared layer '{shared_layer_name}' not found in model parameters. "
            f"Available top-level layers: {available_layers}"
        )
    return params['params'][shared_layer_name]


# --- Define individual loss functions compatible with jax.value_and_grad ---
def pde_loss_fn(params, model, batch, config):
    return compute_pde_loss(model, params, batch, config)

def ic_loss_fn(params, model, batch, config):
    return compute_ic_loss(model, params, batch)

def bc_loss_fn(params, model, batches, config):
    return compute_bc_loss(model, params, batches['left'], batches['right'], batches['bottom'], batches['top'], config)

def building_bc_loss_fn(params, model, batches, config):
    return compute_building_bc_loss(model, params,
                                     batches.get('left', jnp.empty((0,3), dtype=DTYPE)),
                                     batches.get('right', jnp.empty((0,3), dtype=DTYPE)),
                                     batches.get('bottom', jnp.empty((0,3), dtype=DTYPE)),
                                     batches.get('top', jnp.empty((0,3), dtype=DTYPE)))

def data_loss_fn(params, model, batch, config):
    return compute_data_loss(model, params, batch, config)

# <<<--- 2. ADD A WRAPPER FOR YOUR NEW LOSS ---
def neg_h_loss_fn(params, model, batch, config):
    """Wrapper for the negative height loss."""
    # It uses the PDE batch
    return compute_neg_h_loss(model, params, batch)

# --- 3. ADD YOUR NEW LOSS TO THE 'LOSS_FN_MAP' DICTIONARY ---
# This map is the central registry for all GradNorm-compatible losses.
LOSS_FN_MAP = {
    'pde': {'func': pde_loss_fn, 'batch_key': 'pde'},
    'ic': {'func': ic_loss_fn, 'batch_key': 'ic'},
    'bc': {'func': bc_loss_fn, 'batch_key': 'bc'}, 
    'building_bc': {'func': building_bc_loss_fn, 'batch_key': 'building_bc'},
    'data': {'func': data_loss_fn, 'batch_key': 'data'},
    'neg_h': {'func': neg_h_loss_fn, 'batch_key': 'pde'} # <<<--- NEW LINE
}

# (This function is JITted)
def _compute_gradient_norm_impl(grads: chex.ArrayTree) -> jnp.ndarray:
    """Computes the L2 norm of the gradients (flattened)."""
    leaves, _ = jax.tree_util.tree_flatten(grads)
    if not leaves:
        return jnp.array(0.0, dtype=DTYPE)
    leaves_arrays = [jnp.asarray(leaf) for leaf in leaves]
    flat_grads = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves_arrays])
    norm = jnp.linalg.norm(flat_grads)
    return jnp.maximum(norm, 1e-12)

_compute_gradient_norm = jax.jit(_compute_gradient_norm_impl)


# This function already uses the LOSS_FN_MAP, so it's scalable.
def update_gradnorm_weights(
    gradnorm_state: GradNormState,
    model_params: FrozenDict, 
    model: Any, 
    all_batches: Dict[str, Any], 
    config: FrozenDict,
    alpha: float,
    gradnorm_lr: float
) -> Tuple[GradNormState, Dict[str, float]]:
    """
    Computes individual loss gradients and updates GradNorm weights.
    Returns the updated state and the dictionary of new weights.
    """
    loss_keys = list(gradnorm_state.initial_losses.keys())
    num_losses = len(loss_keys)
    current_losses = {}
    grads_wrt_shared_layer = {}

    shared_layer_name = _get_shared_layer_name(model)
    shared_layer_params = _get_shared_layer_params(model_params, shared_layer_name)

    for key in loss_keys: 
        if key not in LOSS_FN_MAP:
            print(f"Internal Warning: Loss key '{key}' from GradNorm state not found in LOSS_FN_MAP. Skipping.")
            continue

        loss_info = LOSS_FN_MAP[key]
        loss_func = loss_info['func']
        batch_key = loss_info['batch_key']
        batch_data = all_batches.get(batch_key)

        is_empty_batch = False
        if batch_data is None: is_empty_batch = True
        elif isinstance(batch_data, jnp.ndarray) and batch_data.shape[0] == 0: is_empty_batch = True
        elif isinstance(batch_data, dict) and not any(isinstance(b, jnp.ndarray) and b.shape[0] > 0 for b in batch_data.values()): is_empty_batch = True

        if is_empty_batch:
            current_losses[key] = 0.0
            grads_wrt_shared_layer[key] = jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params)
            continue

        grad_fn = jax.value_and_grad(loss_func, argnums=0, has_aux=False)

        try:
            loss_val, full_grads = grad_fn(model_params, model, batch_data, config)
            current_losses[key] = float(loss_val)
            grads_wrt_shared_layer[key] = _get_shared_layer_params(full_grads, shared_layer_name)
        except Exception as e:
            print(f"Error computing loss/gradient for '{key}': {e}.")
            import traceback
            traceback.print_exc()
            current_losses[key] = 0.0
            grads_wrt_shared_layer[key] = jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params)

    current_losses_array = jnp.array([current_losses.get(k, 0.0) for k in loss_keys], dtype=DTYPE)
    initial_losses_array = jnp.array([gradnorm_state.initial_losses[k] for k in loss_keys], dtype=DTYPE) 

    def gradnorm_loss_calculation(current_weights_array):
        w = jnp.maximum(jnp.abs(current_weights_array), 1e-8)
        loss_ratios = current_losses_array / initial_losses_array
        loss_ratios = jnp.nan_to_num(loss_ratios, nan=1.0, posinf=1.0, neginf=1.0) 
        mean_loss_ratio = jnp.mean(loss_ratios)
        mean_loss_ratio = jnp.maximum(mean_loss_ratio, 1e-8)
        relative_inverse_rates = loss_ratios / mean_loss_ratio
        grad_norms = jnp.array([_compute_gradient_norm(grads_wrt_shared_layer.get(k, jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params))) for k in loss_keys], dtype=DTYPE)
        weighted_grad_norms = w * grad_norms
        mean_weighted_grad_norm = jnp.mean(weighted_grad_norms)
        mean_weighted_grad_norm = jnp.maximum(mean_weighted_grad_norm, 1e-8)
        target_grad_norms = mean_weighted_grad_norm * (relative_inverse_rates ** alpha)
        gradnorm_loss = jnp.sum(jnp.abs(weighted_grad_norms - target_grad_norms))
        return gradnorm_loss

    gradnorm_loss_val, gradnorm_grads = jax.value_and_grad(gradnorm_loss_calculation)(gradnorm_state.weights)
    optimizer = optax.adam(learning_rate=gradnorm_lr)
    updates, new_opt_state = optimizer.update(gradnorm_grads, gradnorm_state.opt_state, gradnorm_state.weights)
    new_weights_unnormalized = optax.apply_updates(gradnorm_state.weights, updates)

    new_weights_positive = jnp.maximum(new_weights_unnormalized, 1e-8)
    weight_sum = jnp.sum(new_weights_positive)
    weight_sum = jnp.maximum(weight_sum, 1e-8)
    new_weights_normalized = num_losses * new_weights_positive / weight_sum

    updated_state = GradNormState(
        weights=new_weights_normalized,
        initial_losses=gradnorm_state.initial_losses,
        opt_state=new_opt_state
    )

    new_weights_dict = {key: float(w) for key, w in zip(loss_keys, new_weights_normalized)}
    return updated_state, new_weights_dict


# --- 4. REFACTOR get_initial_losses TO USE THE MAP ---
# This function is now much simpler and automatically handles any new loss
# added to LOSS_FN_MAP.
def get_initial_losses(model: Any, params: FrozenDict, all_batches: Dict[str, Any], config: FrozenDict) -> Dict[str, float]:
    """Computes the initial value for each loss term (L_i(0)) using LOSS_FN_MAP."""
    initial_losses = {}
    # Get the list of active loss keys from the *batches* provided.
    # This ensures we only compute losses we have data for.
    active_loss_keys = list(all_batches.keys())
    
    print("Calculating initial losses (L_i(0))...")
    for loss_key in active_loss_keys:
        if loss_key not in LOSS_FN_MAP:
            if loss_key in config.get('loss_weights', {}): # Check if it's a known weight
                print(f"Warning: Loss key '{loss_key}' has a weight but is not in LOSS_FN_MAP. Skipping initial loss calculation.")
            continue

        loss_info = LOSS_FN_MAP[loss_key]
        loss_func = loss_info['func']
        batch_key = loss_info['batch_key']
        
        # Get the correct batch. Note: batch_key might be different from loss_key (e.g., 'neg_h' uses 'pde' batch)
        batch_data = all_batches.get(batch_key) 

        # Check if the required batch is present and valid
        is_empty_batch = False
        if batch_data is None: is_empty_batch = True
        elif isinstance(batch_data, jnp.ndarray) and batch_data.shape[0] == 0: is_empty_batch = True
        elif isinstance(batch_data, dict) and not any(isinstance(b, jnp.ndarray) and b.shape[0] > 0 for b in batch_data.values() if isinstance(b, jnp.ndarray)): is_empty_batch = True

        if is_empty_batch:
            # This can happen if, e.g., 'neg_h' is active but 'pde' is not
            print(f"  Warning: Cannot compute initial loss for '{loss_key}', required batch '{batch_key}' is empty/missing. Setting to 1e-8.")
            initial_losses[loss_key] = 1e-8
            continue

        # Compute the loss value
        try:
            # Call the registered function (pde_loss_fn, ic_loss_fn, etc.)
            loss_val = loss_func(params, model, batch_data, config)
            initial_losses[loss_key] = max(float(loss_val), 1e-8)
            print(f"  Initial loss for {loss_key:<12}: {initial_losses[loss_key]:.4e}")
        except Exception as e:
            print(f"  Error calculating initial loss for {loss_key}: {e}. Setting to 1e-8.")
            initial_losses[loss_key] = 1e-8

    # Filter to ensure we only return losses that were actually active
    final_initial_losses = {k: v for k, v in initial_losses.items() if k in all_batches}
    
    print(f"Final Initial Losses for GradNorm: {final_initial_losses}")
    return final_initial_losses