"""Registry mapping scenario names to experiment-specific HPO functions."""

from experiments.experiment_1.train import (
    compute_losses as _compute_losses_exp1,
    make_generate_epoch_data as _make_gen_exp1,
    make_validation_fn as _make_val_exp1,
)
from experiments.experiment_2.train import (
    compute_losses as _compute_losses_exp2,
    make_generate_epoch_data as _make_gen_exp2,
    make_validation_fn as _make_val_exp2,
)

REGISTRY = {
    "experiment_1": {
        "compute_losses": _compute_losses_exp1,
        "make_generate_epoch_data": _make_gen_exp1,
        "make_validation_fn": _make_val_exp1,
    },
    "experiment_2": {
        "compute_losses": _compute_losses_exp2,
        "make_generate_epoch_data": _make_gen_exp2,
        "make_validation_fn": _make_val_exp2,
    },
}


def get_experiment_fns(scenario_name: str) -> dict:
    """Get experiment-specific functions by scenario name.

    Raises KeyError with helpful message if scenario not registered.
    """
    if scenario_name not in REGISTRY:
        registered = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(
            f"Scenario '{scenario_name}' not registered for HPO. "
            f"Available: {registered}. "
            f"To add a new experiment, register it in optimisation/experiment_registry.py."
        )
    return REGISTRY[scenario_name]
