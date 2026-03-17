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


def _load_env_file(project_root):
    """Load key=value pairs from the repo-level ``.env`` file.

    Returns a dict of the parsed variables.  Lines starting with ``#`` and
    blank lines are skipped.  Values may optionally be quoted.
    """
    env_path = os.path.join(project_root, ".env")
    env_vars = {}
    if not os.path.isfile(env_path):
        return env_vars
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            env_vars[key] = value
    return env_vars


def _build_remote_url(project_root):
    """Build a PostgreSQL connection URL from the ``.env`` file.

    Required keys: ``OPTUNA_DB_USER``, ``OPTUNA_DB_PASSWORD``,
    ``OPTUNA_DB_HOST``, ``OPTUNA_DB_NAME``.
    Optional: ``OPTUNA_DB_PORT`` (default 5432), ``OPTUNA_DB_SSLMODE``
    (default ``require``), ``OPTUNA_DB_OPTIONS`` (extra query params,
    e.g. ``channel_binding=require`` for Neon).

    Raises
    ------
    FileNotFoundError
        If ``.env`` does not exist.
    KeyError
        If a required key is missing.
    """
    env_path = os.path.join(project_root, ".env")
    if not os.path.isfile(env_path):
        raise FileNotFoundError(
            f"Remote storage requested but credentials file not found: {env_path}\n"
            "Copy .env.example to .env and fill in your database credentials."
        )
    env_vars = _load_env_file(project_root)
    required = ["OPTUNA_DB_USER", "OPTUNA_DB_PASSWORD", "OPTUNA_DB_HOST", "OPTUNA_DB_NAME"]
    missing = [k for k in required if k not in env_vars]
    if missing:
        raise KeyError(
            f"Missing required credentials in .env: {', '.join(missing)}"
        )
    user = env_vars["OPTUNA_DB_USER"]
    password = env_vars["OPTUNA_DB_PASSWORD"]
    host = env_vars["OPTUNA_DB_HOST"]
    dbname = env_vars["OPTUNA_DB_NAME"]
    port = env_vars.get("OPTUNA_DB_PORT", "5432")
    sslmode = env_vars.get("OPTUNA_DB_SSLMODE", "require")
    url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode={sslmode}"
    # Extra query params (e.g. channel_binding=require for Neon)
    options = env_vars.get("OPTUNA_DB_OPTIONS", "")
    if options:
        url += f"&{options}"
    return url


def setup_study_storage(storage_backend, project_root, cli_storage=None):
    """Resolve the Optuna storage URL.

    Parameters
    ----------
    storage_backend : str
        ``"local"`` for SQLite, ``"remote"`` for PostgreSQL.
        Comes from the HPO config ``hpo_settings.storage_backend``.
    project_root : str
        Absolute path to the repository root.
    cli_storage : str or None
        The ``--storage`` CLI argument.  When provided, overrides everything.

    Returns
    -------
    str
        A storage URL suitable for ``optuna.create_study``.
    """
    # CLI --storage always wins (escape hatch)
    if cli_storage is not None:
        return _resolve_explicit_url(cli_storage, project_root)

    if storage_backend == "remote":
        return _build_remote_url(project_root)

    # Default: local SQLite
    db_dir = os.path.join(project_root, "optimisation", "database")
    os.makedirs(db_dir, exist_ok=True)
    db_file = os.path.join(db_dir, "all_my_studies.db")
    return f"sqlite:///{db_file}"


def _resolve_explicit_url(url, project_root):
    """Resolve an explicit storage URL (from CLI --storage).

    Handles PostgreSQL passthrough and SQLite relative-path resolution.
    """
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return url

    if url.startswith("sqlite:///"):
        db_file = url.split("sqlite:///")[-1]
        if not os.path.isabs(db_file):
            db_file = os.path.join(project_root, db_file)
            url = f"sqlite:///{db_file}"
        db_dir = os.path.dirname(db_file)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        return url

    raise ValueError(
        f"Unsupported storage URL: {url}. "
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
