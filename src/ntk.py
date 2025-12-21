# src/ntk.py
import jax
import jax.numpy as jnp
from typing import Dict, Any, List

def compute_ntk_traces(model: Any, params: Any, batch: Dict[str, Any], config: Any, active_keys: List[str]) -> Dict[str, jnp.ndarray]:
    """
    Computes the trace of the NTK for each loss term on the GPU.
    Higher trace = "Easier" to learn. Lower trace = "Stiffer" term.
    """
    from src.losses import (
        compute_pde_loss, compute_ic_loss, compute_bc_loss, compute_data_loss, compute_neg_h_loss
    )

    def get_trace(loss_fn_wrapper):
        # We compute the gradient of the scalar loss w.r.t parameters.
        # The trace of the NTK is the squared norm of this gradient.
        grads = jax.grad(loss_fn_wrapper)(params)
        flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
        return jnp.sum(jnp.square(flat_grads))

    # Wrapper functions to match the loss signatures
    loss_fns = {
        'pde': lambda p: compute_pde_loss(model, p, batch['pde'], config),
        'ic': lambda p: compute_ic_loss(model, p, batch['ic']),
        'bc': lambda p: compute_bc_loss(
            model, p, 
            batch['bc']['left'], batch['bc']['right'], 
            batch['bc']['bottom'], batch['bc']['top'], config
        ),
        'data': lambda p: compute_data_loss(model, p, batch['data'], config),
        'neg_h': lambda p: compute_neg_h_loss(model, p, batch['pde'])
    }

    traces = {}
    for key in active_keys:
        if key in loss_fns:
            traces[key] = get_trace(loss_fns[key])
        else:
            traces[key] = jnp.array(1.0) # Fallback
            
    return traces

@jax.jit
def update_ntk_weights(traces: Dict[str, jnp.ndarray], current_weights: Dict[str, jnp.ndarray], ema_alpha: float = 0.1) -> Dict[str, jnp.ndarray]:
    """
    Calculates new weights: weight_i = Tr(K_pde) / Tr(K_i).
    Applies an Exponential Moving Average (EMA) to keep training stable.
    """
    pde_trace = traces.get('pde', jnp.array(1.0))
    new_weights = {}
    
    for key, trace in traces.items():
        # Lambda_i = Trace(PDE) / Trace(i)
        target_weight = pde_trace / jnp.maximum(trace, 1e-12)
        
        # Apply EMA: weight = (1-alpha)*old + alpha*new
        new_weights[key] = (1.0 - ema_alpha) * current_weights.get(key, 1.0) + ema_alpha * target_weight
        
    return new_weights