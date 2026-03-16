# optimisation/utils.py
"""Shared utility helpers for Optuna optimisation scripts."""
import os

import numpy as np
import jax.numpy as jnp
import optuna


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

    Supports SQLite (default) and PostgreSQL backends.  When ``args_storage``
    is *None*, the ``OPTUNA_STORAGE`` environment variable is checked before
    falling back to a local SQLite database.

    Parameters
    ----------
    args_storage : str or None
        The ``--storage`` CLI argument.  When *None*, falls back to the
        ``OPTUNA_STORAGE`` env var, then to a default SQLite path under
        ``optimisation/database/``.
    project_root : str
        Absolute path to the repository root.

    Returns
    -------
    str
        A storage URL suitable for ``optuna.create_study``.

    Examples
    --------
    >>> setup_study_storage(None, "/repo")               # default SQLite
    >>> setup_study_storage("sqlite:///my.db", "/repo")   # explicit SQLite
    >>> setup_study_storage("postgresql://u:p@host/db", "/repo")  # PostgreSQL
    """
    if args_storage is None:
        args_storage = os.environ.get("OPTUNA_STORAGE")

    if args_storage is None:
        db_dir = os.path.join(project_root, "optimisation", "database")
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, "all_my_studies.db")
        return f"sqlite:///{db_file}"

    if args_storage.startswith("postgresql://") or args_storage.startswith("postgres://"):
        return args_storage

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

    raise ValueError(
        f"Unsupported storage URL: {args_storage}. "
        "Use sqlite:/// or postgresql://"
    )


def _is_remote_storage(storage_url):
    """Return True if *storage_url* points to a remote (PostgreSQL) backend."""
    return storage_url.startswith("postgresql://") or storage_url.startswith("postgres://")


def create_storage(storage_url):
    """Create an ``optuna.storages.RDBStorage`` with appropriate settings.

    For remote backends (PostgreSQL), enables heartbeat and retry so that
    preempted trials (e.g. on Google Colab) are detected and retried.
    For SQLite, returns the URL string directly (Optuna handles it).
    """
    if _is_remote_storage(storage_url):
        return optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=120,
            grace_period=600,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(
                max_retry=1
            ),
        )
    return storage_url
