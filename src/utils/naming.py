"""Trial naming utilities."""
import datetime


def generate_trial_name(experiment_name: str, arch_name: str = "", variant: str = "") -> str:
    """Generate a unique trial name: ``YYYY-MM-DD_HH-MM-SS_<experiment>_<arch>[_<variant>]``.

    Parameters
    ----------
    experiment_name:
        Logical experiment label, e.g. ``"experiment_1"``.
    arch_name:
        Model architecture, e.g. ``"MLP"``, ``"FourierPINN"``.  Lowercased
        automatically.  Omitted from the name when empty.
    variant:
        Optional suffix for special variants, e.g. ``"IS"``, ``"relobralo"``.
        Omitted when empty.
    """
    now = datetime.datetime.now()
    parts = [now.strftime('%Y-%m-%d_%H-%M-%S'), experiment_name]
    if arch_name:
        parts.append(arch_name.lower())
    if variant:
        parts.append(variant)
    return "_".join(parts)
