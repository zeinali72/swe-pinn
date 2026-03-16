# optimisation/utils.py
"""Shared utility helpers for Optuna optimisation scripts."""
import os

import numpy as np
import jax.numpy as jnp


def sanitize_for_yaml(data):
    """Recursively convert JAX/numpy types to plain Python for YAML serialisation."""
    if isinstance(data, (jnp.ndarray, np.ndarray)):
        return data.tolist()
    if isinstance(data, (jnp.float32, jnp.float64, np.float32, np.float64)):
        return float(data)
    if isinstance(data, dict):
        return {k: sanitize_for_yaml(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_yaml(item) for item in data]
    return data


def setup_study_storage(args_storage, project_root):
    """Resolve the Optuna storage URL, creating the database directory if needed.

    Parameters
    ----------
    args_storage : str or None
        The ``--storage`` CLI argument.  When *None*, a default SQLite path
        under ``optimisation/database/`` is used.
    project_root : str
        Absolute path to the repository root.

    Returns
    -------
    str
        A ``sqlite:///`` storage URL suitable for ``optuna.create_study``.
    """
    if args_storage is None:
        db_dir = os.path.join(project_root, "optimisation", "database")
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, "all_my_studies.db")
        return f"sqlite:///{db_file}"

    storage_path = args_storage
    if storage_path.startswith("sqlite:///"):
        db_file = storage_path.split("sqlite:///")[-1]
        if not os.path.isabs(db_file):
            db_file = os.path.join(project_root, db_file)
            storage_path = f"sqlite:///{db_file}"
        db_dir = os.path.dirname(db_file)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    return storage_path
