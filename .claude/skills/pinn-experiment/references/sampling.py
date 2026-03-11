"""
Collocation point sampling strategies for PINN training.

Provides LHS for rectangular domains, area-weighted triangle sampling for
irregular domains, and CDF-weighted segment sampling for boundaries.

Depends on: jax, jax.numpy, jax.random
"""

import jax
import jax.numpy as jnp
import jax.random as random
from typing import Optional, Tuple


# =============================================================================
# Latin Hypercube Sampling (Experiments 1-6)
# =============================================================================

def latin_hypercube_sample(
    key: random.PRNGKey,
    n_points: int,
    bounds: list,
) -> jnp.ndarray:
    """Generate Latin Hypercube samples in arbitrary dimensions.

    Parameters
    ----------
    key : PRNGKey
    n_points : int
        Number of points to sample.
    bounds : list of (low, high) tuples
        Bounds for each dimension. E.g., [(0, lx), (0, ly), (0, t_final)].

    Returns
    -------
    jnp.ndarray, shape (n_points, n_dims)
    """
    n_dims = len(bounds)
    keys = random.split(key, n_dims)

    samples = []
    for i, (lo, hi) in enumerate(bounds):
        # Create evenly spaced intervals, then perturb within each
        intervals = jnp.linspace(lo, hi, n_points + 1)
        u = random.uniform(keys[i], (n_points,))
        points = intervals[:-1] + u * (intervals[1:] - intervals[:-1])

        # Random permutation
        perm_key = random.fold_in(keys[i], 1)
        perm = random.permutation(perm_key, n_points)
        points = points[perm]

        samples.append(points)

    return jnp.stack(samples, axis=-1)


def lhs_with_building_mask(
    key: random.PRNGKey,
    n_points: int,
    bounds: list,
    building_mask_fn: callable,
    oversample_factor: float = 2.0,
) -> jnp.ndarray:
    """LHS with rejection sampling to exclude building interiors (Experiment 2).

    Parameters
    ----------
    key : PRNGKey
    n_points : int
        Desired number of points (after masking).
    bounds : list of (low, high) tuples
    building_mask_fn : callable
        Function (x, y) -> bool array, True = inside building (excluded).
    oversample_factor : float
        Generate this many extra points to account for rejection.

    Returns
    -------
    jnp.ndarray, shape (n_points, n_dims)
    """
    n_oversample = int(n_points * oversample_factor)
    key1, key2 = random.split(key)

    points = latin_hypercube_sample(key1, n_oversample, bounds)

    # Reject points inside buildings
    inside = building_mask_fn(points[:, 0], points[:, 1])
    valid = points[~inside]

    # If not enough points, recursively oversample more
    if valid.shape[0] < n_points:
        extra = lhs_with_building_mask(
            key2, n_points - valid.shape[0], bounds,
            building_mask_fn, oversample_factor * 1.5,
        )
        valid = jnp.concatenate([valid, extra], axis=0)

    return valid[:n_points]


# =============================================================================
# Area-Weighted Triangle Sampling (Experiments 7-8)
# =============================================================================

def triangle_areas(
    vertices: jnp.ndarray,
    triangles: jnp.ndarray,
) -> jnp.ndarray:
    """Compute areas of triangles in a mesh.

    Parameters
    ----------
    vertices : jnp.ndarray, shape (V, 2)
        Vertex coordinates.
    triangles : jnp.ndarray, shape (T, 3)
        Triangle vertex indices.

    Returns
    -------
    jnp.ndarray, shape (T,)
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Cross product magnitude / 2
    areas = 0.5 * jnp.abs(
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )
    return areas


def sample_triangular_mesh(
    key: random.PRNGKey,
    n_points: int,
    vertices: jnp.ndarray,
    triangles: jnp.ndarray,
    t_range: Tuple[float, float] = (0.0, 1.0),
) -> jnp.ndarray:
    """Sample points uniformly from a triangulated mesh, area-weighted.

    Each triangle is selected with probability proportional to its area,
    then a point is sampled uniformly within the selected triangle.

    Parameters
    ----------
    key : PRNGKey
    n_points : int
    vertices : jnp.ndarray, shape (V, 2)
    triangles : jnp.ndarray, shape (T, 3)
    t_range : tuple (t_min, t_max)
        Temporal range for the third coordinate.

    Returns
    -------
    jnp.ndarray, shape (n_points, 3)
        Sampled points (x, y, t).
    """
    key1, key2, key3 = random.split(key, 3)

    areas = triangle_areas(vertices, triangles)
    probs = areas / areas.sum()

    # Select triangles
    tri_indices = random.choice(key1, triangles.shape[0], shape=(n_points,), p=probs)

    # Sample within each triangle using barycentric coordinates
    r1 = random.uniform(key2, (n_points,))
    r2 = random.uniform(key3, (n_points,))

    # Ensure point is inside triangle: if r1+r2 > 1, reflect
    mask = r1 + r2 > 1.0
    r1 = jnp.where(mask, 1.0 - r1, r1)
    r2 = jnp.where(mask, 1.0 - r2, r2)

    v0 = vertices[triangles[tri_indices, 0]]
    v1 = vertices[triangles[tri_indices, 1]]
    v2 = vertices[triangles[tri_indices, 2]]

    # Barycentric: P = (1-r1-r2)*v0 + r1*v1 + r2*v2
    xy = (
        (1.0 - r1[:, None] - r2[:, None]) * v0
        + r1[:, None] * v1
        + r2[:, None] * v2
    )

    # Sample time uniformly
    key_t = random.fold_in(key1, 42)
    t = random.uniform(key_t, (n_points, 1), minval=t_range[0], maxval=t_range[1])

    return jnp.concatenate([xy, t], axis=-1)


# =============================================================================
# CDF-Weighted Segment Sampling (Boundary, Experiments 7-8)
# =============================================================================

def sample_boundary_segments(
    key: random.PRNGKey,
    n_points: int,
    segments: jnp.ndarray,
    t_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample points along boundary segments, length-weighted.

    Parameters
    ----------
    key : PRNGKey
    n_points : int
    segments : jnp.ndarray, shape (S, 2, 2)
        Each segment defined by two endpoints: segments[i] = [[x0,y0], [x1,y1]].
    t_range : tuple

    Returns
    -------
    points : jnp.ndarray, shape (n_points, 3)
        Sampled boundary points (x, y, t).
    normals : jnp.ndarray, shape (n_points, 2)
        Outward unit normals at each sampled point.
    """
    key1, key2, key3 = random.split(key, 3)

    # Segment lengths
    diffs = segments[:, 1] - segments[:, 0]  # (S, 2)
    lengths = jnp.sqrt(jnp.sum(diffs**2, axis=-1))  # (S,)
    probs = lengths / lengths.sum()

    # Select segments proportional to length
    seg_indices = random.choice(key1, segments.shape[0], shape=(n_points,), p=probs)

    # Sample position along each segment
    alpha = random.uniform(key2, (n_points, 1))
    p0 = segments[seg_indices, 0]  # (n_points, 2)
    p1 = segments[seg_indices, 1]  # (n_points, 2)
    xy = p0 + alpha * (p1 - p0)

    # Sample time
    t = random.uniform(key3, (n_points, 1), minval=t_range[0], maxval=t_range[1])
    points = jnp.concatenate([xy, t], axis=-1)

    # Compute outward normals (rotate tangent 90 degrees clockwise)
    tangents = diffs[seg_indices]  # (n_points, 2)
    tangent_lengths = jnp.sqrt(jnp.sum(tangents**2, axis=-1, keepdims=True))
    tangents_unit = tangents / (tangent_lengths + 1e-10)

    # Normal = (dy, -dx) for clockwise rotation (outward for CCW-wound boundary)
    normals = jnp.stack([tangents_unit[:, 1], -tangents_unit[:, 0]], axis=-1)

    return points, normals


# =============================================================================
# IC Sampling (all experiments)
# =============================================================================

def sample_initial_condition(
    key: random.PRNGKey,
    n_points: int,
    spatial_bounds: list,
    method: str = "lhs",
    vertices: Optional[jnp.ndarray] = None,
    triangles: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Sample initial condition points (t=0).

    Parameters
    ----------
    key : PRNGKey
    n_points : int
    spatial_bounds : list of (lo, hi) for x and y
    method : 'lhs' or 'mesh'
    vertices, triangles : optional mesh data for method='mesh'

    Returns
    -------
    jnp.ndarray, shape (n_points, 3) with t=0
    """
    if method == "lhs":
        bounds_with_t = spatial_bounds + [(0.0, 0.0)]
        points = latin_hypercube_sample(key, n_points, spatial_bounds)
        t_col = jnp.zeros((n_points, 1))
        return jnp.concatenate([points, t_col], axis=-1)
    elif method == "mesh":
        if vertices is None or triangles is None:
            raise ValueError("vertices and triangles required for mesh sampling")
        points = sample_triangular_mesh(
            key, n_points, vertices, triangles, t_range=(0.0, 0.0)
        )
        return points.at[:, 2].set(0.0)
    else:
        raise ValueError(f"Unknown IC sampling method: {method}")
