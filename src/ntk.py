# src/ntk.py
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from src.physics import SWEPhysics

def compute_ntk_traces(model: Any, params: Any, batch: Dict[str, Any], config: Any, active_keys: Tuple[str, ...]) -> Dict[str, jnp.ndarray]:
    """
    Computes the Trace of the NTK for each loss term efficiently.
    Uses point-by-point VJPs to avoid the memory-intensive full Jacobian.
    """
    
    def get_single_res(p, x, term_type):
        """Computes residual for a single point x."""
        def u_fn(pt): return model.apply({'params': p['params']}, pt, train=False)
        
        if term_type == 'pde':
            u_val = u_fn(x)
            # Spatial Jacobian [3, 3]
            jac_u = jax.jacfwd(u_fn)(x)
            du_dx, du_dy, du_dt = jac_u[:, 0], jac_u[:, 1], jac_u[:, 2]
            
            physics = SWEPhysics(u_val, eps=config["numerics"]["eps"])
            jf, jg = physics.flux_jac(g=config["physics"]["g"])
            div_f = jf @ du_dx
            div_g = jg @ du_dy
            s = physics.source(g=config["physics"]["g"], n_manning=config["physics"]["n_manning"], inflow=config["physics"]["inflow"])
            return du_dt + div_f + div_g - s # [3]
            
        elif term_type == 'neg_h':
            h = u_fn(x)[0]
            return jnp.array([jax.nn.relu(-h)]) # [1]
            
        else: # ic or bc
            return u_fn(x) # [3]

    def compute_trace_efficient(data, term_type):
        if data is None or data.shape[0] == 0:
            return jnp.array(0.0)
            
        def point_grad_norm_sq(x):
            """Computes ||grad r(x)||^2 for a single point x."""
            res_val, vjp_fn = jax.vjp(lambda p: get_single_res(p, x, term_type), params)
            num_outputs = res_val.shape[0]
            
            def comp_norm(i):
                v = jnp.zeros((num_outputs,)).at[i].set(1.0)
                grad_tuple = vjp_fn(v)[0]
                flat_grad, _ = jax.flatten_util.ravel_pytree(grad_tuple)
                return jnp.sum(jnp.square(flat_grad))
                
            return jnp.sum(jax.vmap(comp_norm)(jnp.arange(num_outputs)))

        # MEMORY FIX: Use lax.scan to iterate over points sequentially instead of vmap
        def scan_body(carry, x):
            return carry + point_grad_norm_sq(x), None
        
        total_trace, _ = jax.lax.scan(scan_body, 0.0, data)
        return total_trace

def update_ntk_weights_algo1(traces: Dict[str, jnp.ndarray], current_weights: Dict[str, jnp.ndarray], ema_alpha: float = 0.1):
    """
    Implements Algorithm 1 (Wang et al. 2020): lambda_i = Tr(K_total) / Tr(K_i).
    Includes a protection epsilon and moving average for stability.
    """
    # Tr(K) is the sum of all individual traces [cite: 256, 352]
    total_trace = jnp.sum(jnp.array(list(traces.values())))
    
    new_weights = {}
    for key, trace in traces.items():
        # Formula: lambda = Tr(K_total) / Tr(K_i) 
        # We add epsilon to prevent division by zero or explosive weights
        target_weight = total_trace / jnp.maximum(trace, 1e-8)
        
        # Clamp to avoid extreme values found in your previous run
        target_weight = jnp.clip(target_weight, 1e-2, 1e3)
        
        # Exponential Moving Average for training stability
        new_weights[key] = (1.0 - ema_alpha) * current_weights.get(key, 1.0) + ema_alpha * target_weight
        
    return new_weights
