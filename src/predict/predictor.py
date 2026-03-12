"""JIT-compiled Predictor for batched forward passes.

Used by:
- Training loop: periodic validation on eval grid
- Inference script: full post-training evaluation
"""
import time

import jax
import jax.numpy as jnp


def _apply_min_depth(predictions: jnp.ndarray, min_depth: float) -> jnp.ndarray:
    """Zero out predictions where water depth h falls below min_depth.

    When h < min_depth the cell is considered dry: h, hu, and hv are all set
    to zero.  When min_depth <= 0 the function is a no-op.
    """
    if min_depth <= 0.0:
        return predictions
    h = predictions[..., 0]
    mask = jnp.where(h >= min_depth, 1.0, 0.0)
    return predictions * mask[..., None]


class Predictor:
    """JIT-compiled forward pass for PINN models.

    Args:
        model: A Flax ``nn.Module`` instance.
        batch_size: Maximum number of points to evaluate in one JIT call.
        min_depth: Minimum water depth threshold. Predictions with h below
            this value are zeroed out (dry-cell masking). Default 0.0 (no masking).
    """

    def __init__(self, model, batch_size: int = 50_000, min_depth: float = 0.0):
        self.model = model
        self.batch_size = batch_size
        self.min_depth = min_depth

        @jax.jit
        def _predict(params, coords):
            return model.apply(params, coords, train=False)

        self._predict = _predict

    def predict_full(self, params: dict, coords_flat: jnp.ndarray) -> jnp.ndarray:
        """Run batched prediction over an arbitrarily large coordinate array.

        Args:
            params: Model parameters dict (must contain 'params' key).
            coords_flat: (N, 3) array of (x, y, t) coordinates.

        Returns:
            (N, 3) predictions [h, hu, hv] with dry-cell masking applied.
        """
        flax_params = {'params': params['params']}
        predictions = []
        for i in range(0, len(coords_flat), self.batch_size):
            batch = coords_flat[i:i + self.batch_size]
            pred = self._predict(flax_params, batch)
            predictions.append(pred)
        result = jnp.concatenate(predictions, axis=0)
        return _apply_min_depth(result, self.min_depth)

    def predict_timed(self, params: dict, coords_flat: jnp.ndarray):
        """Forward pass with wall-clock timing.

        Returns:
            (predictions, elapsed_seconds)
        """
        flax_params = {'params': params['params']}
        # Warm up JIT
        _ = self._predict(flax_params, coords_flat[:1]).block_until_ready()
        # Timed pass
        start = time.perf_counter()
        preds = self.predict_full(params, coords_flat)
        preds.block_until_ready()
        elapsed = time.perf_counter() - start
        return preds, elapsed
