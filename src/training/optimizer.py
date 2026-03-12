"""Optimizer creation — shared across all experiments."""
import optax


def create_optimizer(cfg, num_batches=None):
    """Build the standard optax optimizer chain (clip + adam + reduce_on_plateau).

    Parameters
    ----------
    cfg : FrozenDict
        Full experiment config.
    num_batches : int, optional
        Batches per epoch.  Used for ``accumulation_size`` when
        ``accumulation_factor`` is present in config instead of a fixed
        ``accumulation_size``.

    Returns
    -------
    optax.GradientTransformation
    """
    rop_cfg = cfg.get("training", {}).get("reduce_on_plateau", {})

    # Resolve accumulation_size: either explicit or num_batches * factor
    if "accumulation_size" in rop_cfg:
        accum = int(rop_cfg["accumulation_size"])
    elif num_batches is not None:
        accum = num_batches * int(rop_cfg.get("accumulation_factor", 1))
    else:
        accum = int(rop_cfg.get("accumulation_factor", 1))

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
