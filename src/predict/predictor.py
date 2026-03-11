"""JIT-compiled Predictor for batched forward passes.

Used by:
- Training loop: periodic validation on eval grid
- Inference script: full post-training evaluation
"""
import time

import jax
import jax.numpy as jnp


class Predictor:
    """JIT-compiled forward pass for PINN models.

    Args:
        model: A Flax ``nn.Module`` instance.
        batch_size: Maximum number of points to evaluate in one JIT call.
    """

    def __init__(self, model, batch_size: int = 50_000):
        self.model = model
        self.batch_size = batch_size

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
            (N, 3) predictions [h, hu, hv].
        """
        flax_params = {'params': params['params']}
        predictions = []
        for i in range(0, len(coords_flat), self.batch_size):
            batch = coords_flat[i:i + self.batch_size]
            pred = self._predict(flax_params, batch)
            predictions.append(pred)
        return jnp.concatenate(predictions, axis=0)

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
