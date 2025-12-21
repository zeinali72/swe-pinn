# src/ntk.py
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from src.physics import SWEPhysics

def compute_ntk_traces(model: Any, params: Any, batch: Dict[str, Any], config: Any, active_keys: Tuple[str, ...]) -> Dict[str, jnp.ndarray]:
    """
    Computes the Trace of the NTK for each loss term as defined in Wang et al. (2020).
    Tr(K) = sum_{j=1}^N || grad_theta r(x_j, theta) ||^2.
    """
    
    # 1. Define Residual Functions (Returning flattened vectors for Jacobian)
    def pde_res_vec(p, pts):
        def U_fn(x): return model.apply({'params': p['params']}, x, train=False)
        U_pred = U_fn(pts)
        # Pointwise Jacobian w.r.t inputs for PDE terms
        jac_U = jax.vmap(jax.jacfwd(U_fn))(pts)
        dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]
        
        physics = SWEPhysics(U_pred, eps=config["numerics"]["eps"])
        JF, JG = physics.flux_jac(g=config["physics"]["g"])
        div_F = jnp.einsum('nij,nj->ni', JF, dU_dx)
        div_G = jnp.einsum('nij,nj->ni', JG, dU_dy)
        S = physics.source(g=config["physics"]["g"], n_manning=config["physics"]["n_manning"], inflow=config["physics"]["inflow"])
        return (dU_dt + div_F + div_G - S).ravel()

    def ic_res_vec(p, pts):
        return model.apply({'params': p['params']}, pts, train=False).ravel()

    def bc_res_vec(p, bc_batches):
        res_list = []
        for wall in ['left', 'right', 'bottom', 'top']:
            pts = bc_batches.get(wall)
            if pts is not None and pts.shape[0] > 0:
                res_list.append(model.apply({'params': p['params']}, pts, train=False).ravel())
        return jnp.concatenate(res_list) if res_list else jnp.array([0.0])

    def neg_h_res_vec(p, pts):
        h = model.apply({'params': p['params']}, pts, train=False)[..., 0]
        return jax.nn.relu(-h).ravel()

    # 2. Trace Calculation Logic (Frobenius Norm of Jacobian)
    def get_trace(residual_func, data):
        # Tr(K) is the sum of squared entries of the Jacobian matrix 
        jac = jax.jacfwd(residual_func)(params, data)
        flat_jac, _ = jax.flatten_util.ravel_pytree(jac)
        return jnp.sum(jnp.square(flat_jac))

    traces = {}
    if 'pde' in active_keys:
        traces['pde'] = get_trace(pde_res_vec, batch['pde'])
    if 'ic' in active_keys:
        traces['ic'] = get_trace(ic_res_vec, batch['ic'])
    if 'bc' in active_keys:
        traces['bc'] = get_trace(bc_res_vec, batch['bc'])
    if 'neg_h' in active_keys:
        traces['neg_h'] = get_trace(neg_h_res_vec, batch['pde'])
    
    return traces

def update_ntk_weights_algo1(traces: Dict[str, jnp.ndarray], current_weights: Dict[str, jnp.ndarray], ema_alpha: float = 0.1):
    """
    Implements the adaptive weighting Algorithm 1 from Wang et al. (2020).
    lambda_i = Tr(K_total) / Tr(K_i) 
    """
    # Tr(K_total) = sum of all active component traces [cite: 256, 352]
    total_trace = jnp.sum(jnp.array([traces.get(k, 0.0) for k in traces.keys()]))
    
    new_weights = {}
    for key in traces.keys():
        trace = traces[key]
        # Target weight per Algorithm 1: Tr(K) / Tr(K_i) 
        target_weight = total_trace / jnp.maximum(trace, 1e-12)
        
        # Apply EMA to stabilize training 
        new_weights[key] = (1.0 - ema_alpha) * current_weights.get(key, 1.0) + ema_alpha * target_weight
        
    return new_weights