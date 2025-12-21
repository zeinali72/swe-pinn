import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from src.physics import SWEPhysics

def compute_ntk_traces(model: Any, params: Any, batch: Dict[str, Any], config: Any, active_keys: Tuple[str, ...]) -> Dict[str, jnp.ndarray]:
    """Computes the Trace of the NTK for each loss term using lax.scan for memory efficiency."""
    
    def get_single_res(p, x, term_type):
        """Computes residual for a single point x."""
        def u_fn(pt): return model.apply({'params': p['params']}, pt, train=False)
        
        if term_type == 'pde':
            u_val = u_fn(x)
            jac_u = jax.jacfwd(u_fn)(x)
            du_dx, du_dy, du_dt = jac_u[:, 0], jac_u[:, 1], jac_u[:, 2]
            physics = SWEPhysics(u_val, eps=config["numerics"]["eps"])
            jf, jg = physics.flux_jac(g=config["physics"]["g"])
            s = physics.source(g=config["physics"]["g"], n_manning=config["physics"]["n_manning"], inflow=config["physics"]["inflow"])
            return du_dt + jf @ du_dx + jg @ du_dy - s
        elif term_type == 'neg_h':
            h = u_fn(x)[0]
            return jnp.array([jax.nn.relu(-h)])
        else:
            return u_fn(x)

    def compute_trace_efficient(data, term_type):
        if data is None or data.shape[0] == 0:
            return jnp.array(0.0)
            
        def point_grad_norm_sq(x):
            res_val, vjp_fn = jax.vjp(lambda p: get_single_res(p, x, term_type), params)
            num_outputs = res_val.shape[0]
            
            def comp_norm(i):
                v = jnp.zeros((num_outputs,)).at[i].set(1.0)
                grad_tuple = vjp_fn(v)[0]
                flat_grad, _ = jax.flatten_util.ravel_pytree(grad_tuple)
                return jnp.sum(jnp.square(flat_grad))
                
            return jnp.sum(jax.vmap(comp_norm)(jnp.arange(num_outputs)))

        # Use lax.scan to iterate over the batch without expanding memory
        def scan_body(carry, x):
            return carry + point_grad_norm_sq(x), None
        
        total_trace, _ = jax.lax.scan(scan_body, 0.0, data)
        return total_trace

    # Build the dictionary for all active loss terms
    traces = {}
    if 'pde' in active_keys:
        traces['pde'] = compute_trace_efficient(batch['pde'], 'pde')
    if 'ic' in active_keys:
        traces['ic'] = compute_trace_efficient(batch['ic'], 'ic')
    if 'bc' in active_keys:
        bc_trace = 0.0
        for wall in ['left', 'right', 'bottom', 'top']:
            pts = batch['bc'].get(wall)
            if pts is not None:
                bc_trace += compute_trace_efficient(pts, 'bc')
        traces['bc'] = bc_trace
    if 'neg_h' in active_keys:
        traces['neg_h'] = compute_trace_efficient(batch['pde'], 'neg_h')
    
    # CRITICAL: Return the dictionary so update_ntk_weights_algo1 doesn't get None
    return traces

def update_ntk_weights_algo1(traces: Dict[str, jnp.ndarray], current_weights: Dict[str, jnp.ndarray], ema_alpha: float = 0.1):
    """Implements Algorithm 1 (Wang et al. 2020): lambda_i = Tr(K_total) / Tr(K_i)."""
    # The sum of eigenvalues is equivalent to the trace of the kernel [cite: 263, 337, 352]
    total_trace = jnp.sum(jnp.array(list(traces.values())))
    
    new_weights = {}
    for key, trace in traces.items():
        # Calibrate weights to balance convergence rates [cite: 334, 352, 362]
        target_weight = total_trace / jnp.maximum(trace, 1e-8)
        target_weight = jnp.clip(target_weight, 1e-2, 1e6) 
        new_weights[key] = (1.0 - ema_alpha) * current_weights.get(key, 1.0) + ema_alpha * target_weight
        
    return new_weights