# src/gradnorm.py
import jax
import jax.numpy as jnp
import flax.struct
import optax
from flax.core import FrozenDict
from typing import Dict, Any, Callable, Tuple, List
import chex
import importlib

# --- Import standard PINN loss functions ---
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss,
    compute_building_bc_loss, compute_data_loss,
    compute_neg_h_loss
)
# --- Import OperatorNet loss functions ---
from src.operator_learning.losses_op import (
    compute_pde_loss_op, compute_ic_loss_op, compute_bc_loss_op,
    compute_neg_h_loss_op
)
from src.config import DTYPE

@flax.struct.dataclass
class GradNormState:
    weights: chex.Array
    initial_losses: chex.Array
    opt_state: optax.OptState
    loss_keys: Tuple[str, ...] = flax.struct.field(pytree_node=False)

def init_gradnorm(loss_keys: List[str], initial_losses: Dict[str, float], gradnorm_lr: float) -> GradNormState:
    """Initializes the GradNorm state."""
    num_losses = len(loss_keys)
    weights = jnp.ones(num_losses, dtype=DTYPE)
    
    # Convert initial_losses dict to array, ensuring order matches loss_keys
    init_loss_values = [max(initial_losses.get(k, 1e-8), 1e-8) for k in loss_keys]
    initial_losses_arr = jnp.array(init_loss_values, dtype=DTYPE)
    
    optimizer = optax.adam(learning_rate=gradnorm_lr)
    opt_state = optimizer.init(weights)

    return GradNormState(
        weights=weights,
        initial_losses=initial_losses_arr,
        opt_state=opt_state,
        loss_keys=tuple(loss_keys)
    )

def _get_shared_layer_name(model: Any) -> str:
    """
    Returns the standardized name of the shared output layer used for GradNorm.
    All models must name this layer 'output_layer'.
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

# ==============================================================================
# --- Standard PINN GradNorm Functions ---
# ==============================================================================

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
def neg_h_loss_fn(params, model, batch, config):
    return compute_neg_h_loss(model, params, batch)

# Map for standard PINN losses
LOSS_FN_MAP = {
    'pde': {'func': pde_loss_fn, 'batch_key': 'pde'},
    'ic': {'func': ic_loss_fn, 'batch_key': 'ic'},
    'bc': {'func': bc_loss_fn, 'batch_key': 'bc'}, 
    'building_bc': {'func': building_bc_loss_fn, 'batch_key': 'building_bc'},
    'data': {'func': data_loss_fn, 'batch_key': 'data'},
    'neg_h': {'func': neg_h_loss_fn, 'batch_key': 'pde'}
}

# (This function is JITted)
def _compute_gradient_norm_impl(grads: chex.ArrayTree) -> jnp.ndarray:
    """Computes the L2 norm of the gradients (flattened)."""
    leaves = jax.tree_util.tree_leaves(grads)
    if not leaves:
        return jnp.array(0.0, dtype=DTYPE)
    # Compute norm squared for each leaf and sum, then sqrt
    sq_norms = [jnp.sum(jnp.square(leaf)) for leaf in leaves]
    total_sq_norm = jnp.sum(jnp.array(sq_norms))
    norm = jnp.sqrt(total_sq_norm)
    return jnp.maximum(norm, 1e-12)

_compute_gradient_norm = jax.jit(_compute_gradient_norm_impl)

def _is_batch_empty_runtime(batch: Any) -> jnp.ndarray:
    """Checks if a batch is empty (size 0) at runtime. Returns boolean scalar."""
    if isinstance(batch, (dict, FrozenDict)):
        leaves = jax.tree_util.tree_leaves(batch)
        if not leaves:
            return jnp.array(True)
        # Check the first leaf
        return leaves[0].shape[0] == 0
    return batch.shape[0] == 0

def update_gradnorm_weights(
    gradnorm_state: GradNormState,
    model_params: FrozenDict, 
    model: Any, 
    all_batches: Dict[str, Any], 
    config: FrozenDict,
    alpha: float,
    gradnorm_lr: float
) -> Tuple[GradNormState, Dict[str, Any]]:
    """
    Computes individual loss gradients and updates GradNorm weights for PINNs.
    Optimized for GPU execution using jax.lax.cond and scan.
    """
    loss_keys = gradnorm_state.loss_keys
    num_losses = len(loss_keys)
    
    shared_layer_name = _get_shared_layer_name(model)
    shared_layer_params_template = _get_shared_layer_params(model_params, shared_layer_name)

    loss_vals_list = []
    shared_grads_list = []

    for key in loss_keys:
        if key not in LOSS_FN_MAP:
            loss_vals_list.append(jnp.array(0.0, dtype=DTYPE))
            shared_grads_list.append(jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params_template))
            continue

        loss_info = LOSS_FN_MAP[key]
        loss_func = loss_info['func']
        batch_key = loss_info['batch_key']
        batch_data = all_batches.get(batch_key)

        # Static check for None
        if batch_data is None:
            loss_vals_list.append(jnp.array(0.0, dtype=DTYPE))
            shared_grads_list.append(jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params_template))
            continue

        def compute_loss_and_grad(b):
            val, grads = jax.value_and_grad(loss_func, argnums=0)(model_params, model, b, config)
            shared_grads = _get_shared_layer_params(grads, shared_layer_name)
            return val, shared_grads

        def empty_batch_fallback(b):
            return jnp.array(0.0, dtype=DTYPE), jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params_template)

        is_empty = _is_batch_empty_runtime(batch_data)
        l_val, s_grads = jax.lax.cond(is_empty, empty_batch_fallback, compute_loss_and_grad, batch_data)
        
        loss_vals_list.append(l_val)
        shared_grads_list.append(s_grads)

    current_losses_array = jnp.stack(loss_vals_list)
    stacked_shared_grads = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *shared_grads_list)
    initial_losses_array = gradnorm_state.initial_losses

    def gradnorm_loss_calculation(current_weights_array):
        w = jnp.maximum(jnp.abs(current_weights_array), 1e-8)
        loss_ratios = current_losses_array / initial_losses_array
        loss_ratios = jnp.nan_to_num(loss_ratios, nan=1.0, posinf=1.0, neginf=1.0) 
        mean_loss_ratio = jnp.mean(loss_ratios)
        mean_loss_ratio = jnp.maximum(mean_loss_ratio, 1e-8)
        relative_inverse_rates = loss_ratios / mean_loss_ratio
        
        def scan_norm_fn(carry, grads_slice):
            n = _compute_gradient_norm(grads_slice)
            return carry, n
        _, grad_norms = jax.lax.scan(scan_norm_fn, None, stacked_shared_grads)
        
        weighted_grad_norms = w * grad_norms
        mean_weighted_grad_norm = jnp.mean(weighted_grad_norms)
        mean_weighted_grad_norm = jnp.maximum(mean_weighted_grad_norm, 1e-8)
        target_grad_norms = mean_weighted_grad_norm * (relative_inverse_rates ** alpha)
        gradnorm_loss = jnp.sum(jnp.abs(weighted_grad_norms - jax.lax.stop_gradient(target_grad_norms)))
        return gradnorm_loss

    gradnorm_loss_val, gradnorm_grads = jax.value_and_grad(gradnorm_loss_calculation)(gradnorm_state.weights)
    optimizer = optax.adam(learning_rate=gradnorm_lr)
    updates, new_opt_state = optimizer.update(gradnorm_grads, gradnorm_state.opt_state, gradnorm_state.weights)
    new_weights_unnormalized = optax.apply_updates(gradnorm_state.weights, updates)

    new_weights_positive = jnp.maximum(new_weights_unnormalized, 1e-8)
    weight_sum = jnp.sum(new_weights_positive)
    weight_sum = jnp.maximum(weight_sum, 1e-8)
    new_weights_normalized = num_losses * new_weights_positive / weight_sum

    updated_state = gradnorm_state.replace(
        weights=new_weights_normalized,
        opt_state=new_opt_state
    )

    new_weights_dict = {key: new_weights_normalized[i] for i, key in enumerate(loss_keys)}
    return updated_state, new_weights_dict


def get_initial_losses(model: Any, params: FrozenDict, all_batches: Dict[str, Any], config: FrozenDict) -> Dict[str, float]:
    """Computes the initial value for each loss term (L_i(0)) for PINNs."""
    initial_losses = {}
    
    print("Calculating initial losses (L_i(0))...")
    for loss_key, loss_info in LOSS_FN_MAP.items():
        batch_key = loss_info['batch_key']
        if batch_key not in all_batches:
            continue
            
        batch_data = all_batches[batch_key]
        # Simple check for empty batch (not inside JIT here)
        is_empty = False
        if batch_data is None: is_empty = True
        elif isinstance(batch_data, (dict, FrozenDict)):
             if not jax.tree_util.tree_leaves(batch_data): is_empty = True
             elif jax.tree_util.tree_leaves(batch_data)[0].shape[0] == 0: is_empty = True
        elif batch_data.shape[0] == 0: is_empty = True

        if is_empty:
             initial_losses[loss_key] = 1e-8
             continue
             
        loss_func = loss_info['func']
        try:
            # Mark model (1) and config (3) as static
            loss_val = jax.jit(loss_func, static_argnums=(1, 3))(params, model, batch_data, config)
            initial_losses[loss_key] = float(loss_val)
            print(f"  Initial loss for {loss_key:<12}: {initial_losses[loss_key]:.4e}")
        except Exception as e:
            print(f"  Error calculating initial loss for {loss_key}: {e}. Setting to 1e-8.")
            initial_losses[loss_key] = 1e-8

    return initial_losses

# ==============================================================================
# --- OperatorNet (DeepONet) GradNorm Functions ---
# ==============================================================================

# --- Wrappers for OperatorNet losses ---
def pde_loss_fn_op(params, model, batch, config):
    return compute_pde_loss_op(model, params, batch['branch'], batch['trunk'], config)
def ic_loss_fn_op(params, model, batch, config):
    return compute_ic_loss_op(model, params, batch['branch'], batch['trunk'], config)
def bc_loss_fn_op(params, model, batch, config):
    return compute_bc_loss_op(model, params, batch, config) # batch is already a dict
def neg_h_loss_fn_op(params, model, batch, config):
    return compute_neg_h_loss_op(model, params, batch['branch'], batch['trunk'], config)

# Map for OperatorNet losses
OPERATOR_LOSS_FN_MAP = {
    'pde': {'func': pde_loss_fn_op, 'batch_key': 'pde'},
    'ic': {'func': ic_loss_fn_op, 'batch_key': 'ic'},
    'bc': {'func': bc_loss_fn_op, 'batch_key': 'bc'}, 
    'neg_h': {'func': neg_h_loss_fn_op, 'batch_key': 'pde'}
}

def get_initial_losses_operator(model: Any, params: FrozenDict, all_batches: Dict[str, Any], config: FrozenDict) -> Dict[str, float]:
    """Computes the initial value for each loss term (L_i(0)) for OperatorNet."""
    initial_losses = {}
    
    print("Calculating initial losses (L_i(0)) for OperatorNet...")
    for loss_key, loss_info in OPERATOR_LOSS_FN_MAP.items():
        batch_key = loss_info['batch_key']
        if batch_key not in all_batches: continue
        batch_data = all_batches[batch_key]
        
        is_empty = False
        if batch_data is None: is_empty = True
        elif isinstance(batch_data, (dict, FrozenDict)):
             if not jax.tree_util.tree_leaves(batch_data): is_empty = True
             elif jax.tree_util.tree_leaves(batch_data)[0].shape[0] == 0: is_empty = True
        elif batch_data.shape[0] == 0: is_empty = True

        if is_empty:
             initial_losses[loss_key] = 1e-8
             continue
        loss_func = loss_info['func']
        try:
            # Mark model (1) and config (3) as static
            loss_val = jax.jit(loss_func, static_argnums=(1, 3))(params, model, batch_data, config)
            initial_losses[loss_key] = float(loss_val)
            print(f"  Initial loss for {loss_key:<12}: {initial_losses[loss_key]:.4e}")
        except Exception as e:
            print(f"  Error calculating initial loss for {loss_key}: {e}. Setting to 1e-8.")
            initial_losses[loss_key] = 1e-8
            
    return initial_losses


def update_gradnorm_weights_operatornet(
    gradnorm_state: GradNormState,
    model_params: FrozenDict, 
    model: Any, 
    all_batches: Dict[str, Any], 
    config: FrozenDict,
    alpha: float,
    gradnorm_lr: float
) -> Tuple[GradNormState, Dict[str, Any]]:
    """
    Computes individual loss gradients and updates GradNorm weights for OperatorNet.
    """
    loss_keys = gradnorm_state.loss_keys
    num_losses = len(loss_keys)
    shared_layer_name = _get_shared_layer_name(model)
    shared_layer_params_template = _get_shared_layer_params(model_params, shared_layer_name)

    loss_vals_list = []
    shared_grads_list = []

    for key in loss_keys: 
        if key not in OPERATOR_LOSS_FN_MAP:
            loss_vals_list.append(jnp.array(0.0, dtype=DTYPE))
            shared_grads_list.append(jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params_template))
            continue

        loss_info = OPERATOR_LOSS_FN_MAP[key]
        loss_func = loss_info['func']
        batch_key = loss_info['batch_key']
        batch_data = all_batches.get(batch_key)

        if batch_data is None:
            loss_vals_list.append(jnp.array(0.0, dtype=DTYPE))
            shared_grads_list.append(jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params_template))
            continue

        def compute_loss_and_grad(b):
            val, grads = jax.value_and_grad(loss_func, argnums=0)(model_params, model, b, config)
            shared_grads = _get_shared_layer_params(grads, shared_layer_name)
            return val, shared_grads

        def empty_batch_fallback(b):
            return jnp.array(0.0, dtype=DTYPE), jax.tree_util.tree_map(jnp.zeros_like, shared_layer_params_template)

        is_empty = _is_batch_empty_runtime(batch_data)
        l_val, s_grads = jax.lax.cond(is_empty, empty_batch_fallback, compute_loss_and_grad, batch_data)
        
        loss_vals_list.append(l_val)
        shared_grads_list.append(s_grads)

    current_losses_array = jnp.stack(loss_vals_list)
    stacked_shared_grads = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *shared_grads_list)
    initial_losses_array = gradnorm_state.initial_losses

    def gradnorm_loss_calculation(current_weights_array):
        w = jnp.maximum(jnp.abs(current_weights_array), 1e-8)
        loss_ratios = current_losses_array / initial_losses_array
        loss_ratios = jnp.nan_to_num(loss_ratios, nan=1.0, posinf=1.0, neginf=1.0) 
        mean_loss_ratio = jnp.mean(loss_ratios)
        mean_loss_ratio = jnp.maximum(mean_loss_ratio, 1e-8)
        relative_inverse_rates = loss_ratios / mean_loss_ratio
        
        def scan_norm_fn(carry, grads_slice):
            n = _compute_gradient_norm(grads_slice)
            return carry, n
        _, grad_norms = jax.lax.scan(scan_norm_fn, None, stacked_shared_grads)
        
        weighted_grad_norms = w * grad_norms
        mean_weighted_grad_norm = jnp.mean(weighted_grad_norms)
        mean_weighted_grad_norm = jnp.maximum(mean_weighted_grad_norm, 1e-8)
        target_grad_norms = mean_weighted_grad_norm * (relative_inverse_rates ** alpha)
        gradnorm_loss = jnp.sum(jnp.abs(weighted_grad_norms - jax.lax.stop_gradient(target_grad_norms)))
        return gradnorm_loss

    gradnorm_loss_val, gradnorm_grads = jax.value_and_grad(gradnorm_loss_calculation)(gradnorm_state.weights)
    optimizer = optax.adam(learning_rate=gradnorm_lr)
    updates, new_opt_state = optimizer.update(gradnorm_grads, gradnorm_state.opt_state, gradnorm_state.weights)
    new_weights_unnormalized = optax.apply_updates(gradnorm_state.weights, updates)

    new_weights_positive = jnp.maximum(new_weights_unnormalized, 1e-8)
    weight_sum = jnp.sum(new_weights_positive)
    weight_sum = jnp.maximum(weight_sum, 1e-8)
    new_weights_normalized = num_losses * new_weights_positive / weight_sum

    updated_state = gradnorm_state.replace(
        weights=new_weights_normalized,
        opt_state=new_opt_state
    )

    new_weights_dict = {key: new_weights_normalized[i] for i, key in enumerate(loss_keys)}
    return updated_state, new_weights_dict