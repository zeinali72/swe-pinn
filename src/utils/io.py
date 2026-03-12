"""Model I/O utilities."""
import os
import pickle
from typing import Dict, Any


def save_model(params: Dict[str, Any], save_dir: str, trial_name: str) -> str:
    """Save model parameters to a pickle file and return the path."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{trial_name}_params.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(params, f)
    return model_path
