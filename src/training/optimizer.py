"""Optimizer creation — shared across all experiments."""
import warnings

import optax


def create_optimizer(cfg, num_batches=None):
    """Build the standard optax optimizer chain (clip + adam + reduce_on_plateau).

    Parameters
    ----------
    cfg : FrozenDict
        Full experiment config.
    num_batches : int, optional
        Batches per epoch.  Only used as a fallback multiplier when the
        deprecated ``accumulation_factor`` key is present instead of the
        canonical ``accumulation_size``.

    Returns
    -------
    optax.GradientTransformation
    """
    rop_cfg = cfg.get("training", {}).get("reduce_on_plateau", {})

    # Resolve accumulation_size — canonical key takes priority.
    if "accumulation_size" in rop_cfg:
        accum = int(rop_cfg["accumulation_size"])
    elif "accumulation_factor" in rop_cfg:
        warnings.warn(
            "Config key 'accumulation_factor' is deprecated; "
            "use 'accumulation_size' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        factor = int(rop_cfg["accumulation_factor"])
        accum = num_batches * factor if num_batches is not None else factor
    else:
        accum = 1

    return optax.chain(
        optax.clip_by_global_norm(cfg.get("training", {}).get("clip_norm", 1.0)),
        optax.adam(learning_rate=cfg["training"]["learning_rate"]),
        optax.contrib.reduce_on_plateau(
            factor=float(rop_cfg.get("factor", 0.5)),
            patience=int(rop_cfg.get("patience", 5)),
            rtol=float(rop_cfg.get("rtol", 1e-4)),
            atol=float(rop_cfg.get("atol", 0.0)),
            cooldown=int(rop_cfg.get("cooldown", 1)),
            accumulation_size=accum,
            min_scale=float(rop_cfg.get("min_scale", 1e-6)),
        ),
    )
