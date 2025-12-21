# src/ntk.py
import jax
import jax.numpy as jnp
from typing import Dict, Any, List
from src.losses import SWEPhysics

def compute_ntk_traces_original(model: Any, params: Any, batch: Dict[str, Any], config: Any, active_keys: List[str]) -> Dict[str, jnp.ndarray]:
    """
    Computes the TRUE Trace of the NTK for each loss term as per Wang et al. (2020).
    Tr(K) = sum_{points} || grad_theta Residual(point) ||^2.
    """
    
    # 1. Define Residual Functions (Vector-valued, one entry per point/constraint)
    def pde_res_vec(p, pts):
        U_pred = model.apply({'params': p['params']}, pts, train=False)
        def U_fn(x): return model.apply({'params': p['params']}, x, train=False)
        jac_U = jax.vmap(jax.jacfwd(U_fn))(pts)
        dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]
        
        physics = SWEPhysics(U_pred, eps=config["numerics"]["eps"])
        JF, JG = physics.flux_jac(g=config["physics"]["g"])
        div_F = jnp.einsum('nij,nj->ni', JF, dU_dx)
        div_G = jnp.einsum('nij,nj->ni', JG, dU_dy)
        S = physics.source(g=config["physics"]["g"], n_manning=config["physics"]["n_manning"], inflow=config["physics"]["inflow"])
        
        # Flatten residuals (N, 3) -> (N*3,)
        return (dU_dt + div_F + div_G - S).ravel()

    def ic_res_vec(p, pts):
        # Initial condition residual: Predicted U - [0, 0, 0]
        return model.apply({'params': p['params']}, pts, train=False).ravel()

    def bc_res_vec(p, bc_batches):
        # Combine all boundary residuals into one vector
        res_list = []
        for wall in ['left', 'right', 'bottom', 'top']:
            pts = bc_batches.get(wall)
            if pts is not None and pts.shape[0] > 0:
                res_list.append(model.apply({'params': p['params']}, pts, train=False).ravel())
        return jnp.concatenate(res_list) if res_list else jnp.array([0.0])

    # 2. Trace Calculation Logic
    def get_trace(residual_func, data):
        # Tr(K) is the squared Frobenius norm of the Jacobian of the residual vector
        # || J ||_F^2 = sum( (grad Residual)^2 )
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
    
    return traces

def update_ntk_weights_stable(traces: Dict[str, jnp.ndarray], current_weights: Dict[str, jnp.ndarray], ema_alpha: float = 0.1):
    """lambda_i = Tr(K_pde) / Tr(K_i) with EMA stability."""
    pde_trace = traces.get('pde', jnp.array(1.0))
    new_weights = {}
    for key, trace in traces.items():
        target_weight = pde_trace / jnp.maximum(trace, 1e-12)
        new_weights[key] = (1.0 - ema_alpha) * current_weights.get(key, 1.0) + ema_alpha * target_weight
    return new_weights