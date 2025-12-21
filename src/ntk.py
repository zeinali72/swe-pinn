# src/ntk.py
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, List
from src.physics import SWEPhysics

def compute_ntk_traces(model: Any, params: Any, batch: Dict[str, Any], config: Any, active_keys: Tuple[str, ...]) -> Dict[str, jnp.ndarray]:
    """
    Computes the Trace of the NTK for each loss term efficiently using VJP.
    Tr(K) = sum_{j=1}^N || grad_theta r(x_j, theta) ||^2.
    """
    
    # 1. Pointwise Residual Functions
    def get_pde_res(p, x):
        # x is a single point [x, y, t]
        def u_fn(pt):
            return model.apply({'params': p['params']}, pt, train=False)
            
        # Pointwise Jacobian w.r.t inputs (x, y, t)
        jac_u = jax.jacfwd(u_fn)(x) # [3, 3]
        u_val = u_fn(x) # [3]
        
        du_dx, du_dy, du_dt = jac_u[:, 0], jac_u[:, 1], jac_u[:, 2]
        
        physics = SWEPhysics(u_val, eps=config["numerics"]["eps"])
        jf, jg = physics.flux_jac(g=config["physics"]["g"])
        
        div_f = jf @ du_dx
        div_g = jg @ du_dy
        s = physics.source(g=config["physics"]["g"], n_manning=config["physics"]["n_manning"], inflow=config["physics"]["inflow"])
        
        return du_dt + div_f + div_g - s # [3]

    def get_ic_res(p, x):
        return model.apply({'params': p['params']}, x, train=False)

    def get_bc_res(p, x):
        return model.apply({'params': p['params']}, x, train=False)

    def get_neg_h_res(p, x):
        h = model.apply({'params': p['params']}, x, train=False)[0]
        return jnp.array([jax.nn.relu(-h)]) # 1 component

    # 2. Efficient Trace Calculation via VJP
    def get_point_trace(p, x, res_fn, num_components):
        # Trace(K) for one point = sum of squared norms of grads of each component
        _, vjp_fn = jax.vjp(lambda param: res_fn(param, x), p)
        
        def get_comp_grad_norm_sq(k):
            e_k = jnp.zeros((num_components,)).at[k].set(1.0)
            g_tuple = vjp_fn(e_k)
            flat_g, _ = jax.flatten_util.ravel_pytree(g_tuple[0])
            return jnp.sum(jnp.square(flat_g))
            
        return jnp.sum(jax.vmap(get_comp_grad_norm_sq)(jnp.arange(num_components)))

    def compute_trace_for_term(data, res_fn, num_components):
        if data is None or data.shape[0] == 0:
            return jnp.array(0.0)
        # Vectorize over the batch of points
        point_traces = jax.vmap(lambda x: get_point_trace(params, x, res_fn, num_components))(data)
        return jnp.sum(point_traces)

    traces = {}
    if 'pde' in active_keys:
        traces['pde'] = compute_trace_for_term(batch['pde'], get_pde_res, 3)
    if 'ic' in active_keys:
        traces['ic'] = compute_trace_for_term(batch['ic'], get_ic_res, 3)
    if 'bc' in active_keys:
        bc_trace = 0.0
        for wall in ['left', 'right', 'bottom', 'top']:
            pts = batch['bc'].get(wall)
            if pts is not None and pts.shape[0] > 0:
                bc_trace += compute_trace_for_term(pts, get_bc_res, 3)
        traces['bc'] = bc_trace
    if 'neg_h' in active_keys:
        traces['neg_h'] = compute_trace_for_term(batch['pde'], get_neg_h_res, 1)
    
    return traces

def update_ntk_weights_stable(traces: Dict[str, jnp.ndarray], current_weights: Dict[str, jnp.ndarray], ema_alpha: float = 0.1):
    """Calculates lambda_i = Tr(K_pde) / Tr(K_i) with EMA."""
    pde_trace = traces.get('pde', jnp.array(1.0))
    new_weights = {}
    for key, trace in traces.items():
        # Avoid division by zero
        target_weight = pde_trace / jnp.maximum(trace, 1e-12)
        new_weights[key] = (1.0 - ema_alpha) * current_weights.get(key, 1.0) + ema_alpha * target_weight
    return new_weights