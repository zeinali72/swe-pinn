"""Dual-checkpoint manager: best_nse, best_loss, and final.

Each checkpoint stores model parameters, optimiser state, metadata,
and (for best_nse / best_loss) the validation outputs on the eval grid
so that the best predictions can be inspected without re-running inference.
"""
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, List

import yaml
import jax.numpy as jnp


class CheckpointManager:
    """Manages three checkpoints: best_nse, best_loss, final.

    Args:
        experiment_dir: Root directory for this experiment's outputs.
        model:          Flax model (used for forward pass when saving outputs).
        eval_coords:    Flattened evaluation grid coordinates (N, 3).
                        If None, validation_outputs.npz will not be saved.
        reference:      Reference solution on the same grid (N, 3).
                        If None, validation_outputs.npz will not be saved.
    """

    def __init__(
        self,
        experiment_dir: str,
        model=None,
        eval_coords: Optional[jnp.ndarray] = None,
        reference: Optional[jnp.ndarray] = None,
    ):
        self.experiment_dir = Path(experiment_dir)
        self.model = model
        self.eval_coords = eval_coords
        self.reference = reference

        self.best_nse_h: float = -float('inf')
        self.best_nse_epoch: int = -1
        self.best_loss: float = float('inf')
        self.best_loss_epoch: int = -1

        for subdir in ['best_nse', 'best_loss', 'final']:
            (self.experiment_dir / 'checkpoints' / subdir).mkdir(
                parents=True, exist_ok=True,
            )

    def update(
        self,
        epoch: int,
        params: dict,
        opt_state,
        val_metrics: Dict[str, float],
        losses: Dict[str, float],
        total_loss: float,
        config: dict,
        neg_depth: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Check both criteria and save if improved. Returns list of events."""
        saved = []
        nse_h = val_metrics.get('selection_metric', val_metrics.get('nse_h', -float('inf')))

        if nse_h > self.best_nse_h:
            prev_nse = self.best_nse_h
            prev_epoch = self.best_nse_epoch
            self.best_nse_h = nse_h
            self.best_nse_epoch = epoch
            self._save_checkpoint(
                'best_nse', epoch, params, opt_state,
                val_metrics, losses, total_loss, config, neg_depth,
            )
            self._save_validation_outputs('best_nse', params, epoch,
                                          val_metrics, losses)
            saved.append(('best_nse', nse_h, epoch, prev_nse, prev_epoch))

        if total_loss < self.best_loss:
            prev_loss = self.best_loss
            prev_epoch = self.best_loss_epoch
            self.best_loss = total_loss
            self.best_loss_epoch = epoch
            self._save_checkpoint(
                'best_loss', epoch, params, opt_state,
                val_metrics, losses, total_loss, config, neg_depth,
            )
            self._save_validation_outputs('best_loss', params, epoch,
                                          val_metrics, losses)
            saved.append(('best_loss', total_loss, epoch, prev_loss, prev_epoch))

        return saved

    def save_final(self, epoch: int, params: dict, opt_state,
                   val_metrics: Dict[str, float], losses: Dict[str, float],
                   total_loss: float, config: dict,
                   neg_depth: Optional[Dict[str, float]] = None):
        """Save the final-epoch checkpoint (no validation outputs)."""
        self._save_checkpoint(
            'final', epoch, params, opt_state,
            val_metrics, losses, total_loss, config, neg_depth,
        )

    def get_best_nse_stats(self) -> dict:
        """Load metadata for the best-NSE checkpoint."""
        return self._load_metadata('best_nse')

    def get_best_loss_stats(self) -> dict:
        """Load metadata for the best-loss checkpoint."""
        return self._load_metadata('best_loss')

    def get_best_nse_params(self) -> Optional[dict]:
        """Load parameters for the best-NSE checkpoint."""
        return self._load_params('best_nse')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _save_checkpoint(self, name, epoch, params, opt_state,
                         val_metrics, losses, total_loss, config, neg_depth):
        ckpt_dir = self.experiment_dir / 'checkpoints' / name
        with open(ckpt_dir / 'model.pkl', 'wb') as f:
            pickle.dump({'params': params, 'opt_state': opt_state}, f)
        metadata = {
            'epoch': int(epoch),
            'total_loss': float(total_loss),
            'validation_metrics': {k: float(v) for k, v in val_metrics.items()},
            'losses': {k: float(v) for k, v in losses.items()},
        }
        if neg_depth:
            metadata['neg_depth'] = {k: float(v) for k, v in neg_depth.items()}
        with open(ckpt_dir / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

    def _save_validation_outputs(self, name, params, epoch,
                                 val_metrics, losses):
        if self.eval_coords is None or self.reference is None:
            return
        if self.model is None:
            return
        ckpt_dir = self.experiment_dir / 'checkpoints' / name
        predictions = self.model.apply(
            {'params': params['params']}, self.eval_coords, train=False,
        )
        jnp.savez(
            str(ckpt_dir / 'validation_outputs.npz'),
            coords=self.eval_coords,
            predictions=predictions,
            reference=self.reference,
            epoch=epoch,
        )

    def _load_metadata(self, name) -> dict:
        path = self.experiment_dir / 'checkpoints' / name / 'metadata.yaml'
        if path.exists():
            with open(path) as f:
                metadata = yaml.safe_load(f) or {}
            if 'total_weighted_loss' not in metadata and 'total_loss' in metadata:
                metadata['total_weighted_loss'] = metadata['total_loss']
            return metadata
        return {}

    def _load_params(self, name) -> Optional[dict]:
        path = self.experiment_dir / 'checkpoints' / name / 'model.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data.get('params', data)
        return None
