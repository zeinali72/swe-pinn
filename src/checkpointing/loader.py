"""Load saved checkpoints."""
import pickle
from pathlib import Path
from typing import Optional, Tuple

import yaml


def load_checkpoint(path: str) -> Tuple[Optional[dict], Optional[dict]]:
    """Load a checkpoint directory and return (params, metadata).

    Args:
        path: Path to checkpoint directory (e.g. ``experiments/exp1/checkpoints/best_nse``).

    Returns:
        A tuple of (params_dict, metadata_dict). Either may be None if the
        corresponding file is missing.
    """
    ckpt_dir = Path(path)
    params = None
    metadata = None

    model_path = ckpt_dir / 'model.pkl'
    if model_path.exists():
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        params = data.get('params', data)

    meta_path = ckpt_dir / 'metadata.yaml'
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = yaml.safe_load(f)

    return params, metadata
