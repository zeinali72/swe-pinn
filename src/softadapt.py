# src/softadapt.py
import jax.numpy as jnp
from typing import Dict

def softadapt_update(
    current_losses: Dict[str, float],
    prev_losses: Dict[str, float],
    epsilon: float = 1e-8
) -> Dict[str, float]:
    """
    Calculates new loss weights using the SoftAdapt formula with the softmax trick.
    
    w_j(i) = exp(L_j(i) / L_j(i-1) - mu) / sum(exp(L_k(i) / L_k(i-1) - mu))
    where mu = max(L_j(i) / L_j(i-1))
    """
    # 1. Identify common keys involved in dynamic weighting
    # We only update weights for terms present in both current and prev stats
    active_keys = [k for k in current_losses.keys() if k in prev_losses]
    
    if not active_keys:
        return {}

    # 2. Calculate rates of change: L(i) / L(i-1)
    rates = []
    for k in active_keys:
        # Add epsilon to denominator to prevent division by zero
        # Ensure losses are standard floats (not JAX arrays) for safety
        l_curr = float(current_losses[k])
        l_prev = float(prev_losses[k])
        rate = l_curr / (l_prev + epsilon)
        rates.append(rate)
    
    rates = jnp.array(rates)

    # 3. Apply Softmax Trick
    # mu = max(rates)
    mu = jnp.max(rates)
    
    # exp(rate - mu)
    exp_rates = jnp.exp(rates - mu)
    
    # Normalize to get weights sum = 1
    sum_exp = jnp.sum(exp_rates)
    new_weights_vals = exp_rates / sum_exp
    
    # 4. Map back to keys
    # Often in PINNs, we want weights to scale nicely. 
    # SoftAdapt normalizes them to sum to 1. 
    # If you prefer them to sum to the number of terms (keeping scale ~1.0), 
    # multiply by len(active_keys).
    # For now, we return the strict formula (sum=1).
    scale_factor = 1.0  # Change to len(active_keys) if you want larger gradients
    
    new_weights_dict = {
        k: float(w * scale_factor) for k, w in zip(active_keys, new_weights_vals)
    }
    
    return new_weights_dict