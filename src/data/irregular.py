"""Samplers for irregular (non-rectangular) domains using triangulated meshes."""
import os
from functools import partial
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict

from src.config import DTYPE
from src.data.sampling import sample_domain


@partial(jax.jit, static_argnames=['n_points', 't_bounds'])
def _sample_interior_core(key, n_points, t_bounds, tri_coords, tri_cdf):
    k_tri, k_bary, k_t = random.split(key, 3)
    probs = jnp.diff(tri_cdf, prepend=0.0)
    tri_indices = random.choice(k_tri, tri_coords.shape[0], shape=(n_points,), p=probs)
    chosen = tri_coords[tri_indices]

    r = random.uniform(k_bary, (n_points, 2), dtype=DTYPE)
    u, v = r[:, 0:1], r[:, 1:2]
    is_outside = (u + v) > 1.0
    u = jnp.where(is_outside, 1.0 - u, u)
    v = jnp.where(is_outside, 1.0 - v, v)

    xy = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0]) + v * (chosen[:, 2] - chosen[:, 0])

    if t_bounds is not None:
        t = random.uniform(k_t, (n_points, 1), minval=t_bounds[0], maxval=t_bounds[1], dtype=DTYPE)
    else:
        t = jnp.zeros((n_points, 1), dtype=DTYPE)

    return jnp.hstack([xy, t])


@partial(jax.jit, static_argnames=['n_points', 't_bounds'])
def _sample_boundary_core(key, n_points, t_bounds, starts, vecs, cdf):
    """Sample boundary points and calculate unit normal vectors.

    Returns: [x, y, t, nx, ny]
    """
    k_seg, k_pos, k_t = random.split(key, 3)

    probs = jnp.diff(cdf, prepend=0.0)
    seg_indices = random.choice(k_seg, starts.shape[0], shape=(n_points,), p=probs)

    pos = random.uniform(k_pos, (n_points, 1), dtype=DTYPE)
    xy = starts[seg_indices] + pos * vecs[seg_indices]

    if t_bounds is not None:
        t = random.uniform(k_t, (n_points, 1), minval=t_bounds[0], maxval=t_bounds[1], dtype=DTYPE)
    else:
        t = jnp.zeros((n_points, 1), dtype=DTYPE)

    selected_vecs = vecs[seg_indices]
    vx = selected_vecs[:, 0:1]
    vy = selected_vecs[:, 1:2]

    nx = -vy
    ny = vx

    norm = jnp.sqrt(nx**2 + ny**2 + 1e-10)
    nx = nx / norm
    ny = ny / norm

    return jnp.hstack([xy, t, nx, ny])


class IrregularDomainSampler:
    """Mesh-based sampler for non-rectangular domains (Experiments 7/8)."""

    def __init__(self, artifacts_path: str):
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Artifacts not found at {artifacts_path}")

        print(f"Loading sampler artifacts from: {artifacts_path}")
        try:
            data = np.load(artifacts_path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load .npz file: {e}")

        try:
            self.tri_coords = jnp.array(data['tri_coords'], dtype=DTYPE)
            self.tri_cdf = jnp.array(data['tri_cdf'], dtype=DTYPE)
        except KeyError as e:
            raise KeyError(f"Missing mandatory interior mesh data: {e}")

        self.boundaries = {}
        all_keys = list(data.files)

        for key in all_keys:
            if key.startswith("bc_") and key.endswith("_starts"):
                label = key[3:-7]
                key_s = f"bc_{label}_starts"
                key_v = f"bc_{label}_vectors"
                key_c = f"bc_{label}_cdf"

                if key_v in all_keys and key_c in all_keys:
                    starts = jnp.array(data[key_s], dtype=DTYPE)
                    vecs = jnp.array(data[key_v], dtype=DTYPE)
                    cdf = jnp.array(data[key_c], dtype=DTYPE)
                    self.boundaries[label] = (starts, vecs, cdf)
                    print(f"  - Registered boundary '{label}': {starts.shape[0]} segments")

    def sample_interior(self, key, n_points: int, t_bounds: tuple = None) -> jnp.ndarray:
        return _sample_interior_core(
            key, n_points, t_bounds,
            self.tri_coords, self.tri_cdf
        )

    def sample_boundary(self, key, n_points: int, t_bounds: tuple, boundary_type: str = 'wall') -> jnp.ndarray:
        if boundary_type not in self.boundaries:
            return jnp.zeros((0, 5), dtype=DTYPE)

        starts, vecs, cdf = self.boundaries[boundary_type]
        return _sample_boundary_core(
            key, n_points, t_bounds,
            starts, vecs, cdf
        )


class DeepONetParametricSampler:
    """Unified sampler for DeepONet training.

    Handles both physical parameter sampling (branch input)
    and coordinate sampling (trunk input).
    """

    def __init__(self, config: FrozenDict):
        self.config = config
        self.physics_cfg = config["physics"]
        self.domain_cfg = config["domain"]
        self.param_bounds = self.physics_cfg.get("param_bounds", {})
        self.param_names = tuple(sorted(self.param_bounds.keys()))
        self.n_params = len(self.param_names)

    def sample_parameters(self, key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """Sample just the parameters."""
        if self.n_params == 0:
            return jnp.zeros((n_samples, 0), dtype=DTYPE)

        keys = random.split(key, self.n_params)
        samples = []
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[name]
            p_sample = random.uniform(keys[i], (n_samples, 1), minval=min_val, maxval=max_val, dtype=DTYPE)
            samples.append(p_sample)

        return jnp.hstack(samples)

    def sample_batch(self, key: jax.random.PRNGKey, n_samples: int,
                     mode: str = 'pde',
                     x_bounds: Tuple[float, float] = None,
                     y_bounds: Tuple[float, float] = None,
                     t_bounds: Tuple[float, float] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a batch of (branch_input, trunk_input)."""
        k1, k2 = random.split(key)

        branch_batch = self.sample_parameters(k1, n_samples)

        if x_bounds is None: x_bounds = (0., self.domain_cfg["lx"])
        if y_bounds is None: y_bounds = (0., self.domain_cfg["ly"])

        if t_bounds is None:
            if mode == 'ic':
                t_bounds = (0., 0.)
            else:
                t_bounds = (0., self.domain_cfg["t_final"])

        trunk_batch = sample_domain(k2, n_samples, x_bounds, y_bounds, t_bounds)

        return branch_batch, trunk_batch
