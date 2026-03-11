"""Formatted console output for PINN training.

Provides structured logging: training header, per-epoch status lines,
checkpoint events, learning-rate events, and a completion summary.
"""
import datetime
import time
from typing import Dict, Optional


class ConsoleLogger:
    """Manages formatted console output throughout a training run."""

    def __init__(self, config: dict):
        self.config = config
        self._n_lr_reductions = 0

    # ------------------------------------------------------------------
    # G.1  Header
    # ------------------------------------------------------------------
    def print_header(self):
        cfg = self.config
        model_cfg = cfg.get('model', {})
        domain_cfg = cfg.get('domain', {})
        physics_cfg = cfg.get('physics', {})
        training_cfg = cfg.get('training', {})
        loss_w = cfg.get('loss_weights', {})
        gradnorm_cfg = cfg.get('gradnorm', {})

        scenario = cfg.get('scenario', 'unknown')
        arch = model_cfg.get('name', 'unknown')
        has_building = 'building' in cfg
        precision = cfg.get('device', {}).get('dtype', 'float32')

        try:
            import jax
            jax_version = jax.__version__
            devices = jax.devices()
            device_str = str(devices[0]) if devices else 'unknown'
        except Exception:
            jax_version = 'unknown'
            device_str = 'unknown'

        print("=" * 72)
        print(f"EXPERIMENT: {scenario}")
        print(f"ARCHITECTURE: {arch}")
        print(f"DATE: {datetime.datetime.now().isoformat(timespec='seconds')}")
        print(f"DEVICE: {device_str}, JAX {jax_version}")
        print(f"PRECISION: {precision}")
        print("=" * 72)

        print(
            f"Domain: {domain_cfg.get('lx', '?')}m x "
            f"{domain_cfg.get('ly', '?')}m, "
            f"T={domain_cfg.get('t_final', '?')}s"
        )
        print(
            f"Manning n: {physics_cfg.get('n_manning', '?')} | "
            f"Buildings: {has_building}"
        )

        grid = cfg.get('grid', {})
        ic_bc = cfg.get('ic_bc_grid', {})
        n_pde = grid.get('nx', 0) * grid.get('ny', 0) * grid.get('nt', 0)
        n_ic = ic_bc.get('nx_ic', 0) * ic_bc.get('ny_ic', 0)
        print(f"Collocation: PDE={n_pde} IC={n_ic}")

        w_parts = []
        for k, v in loss_w.items():
            short = k.replace('_weight', '').upper()
            w_parts.append(f"{short}={v}")
        print(f"Loss weights: {' '.join(w_parts)}")

        lr = training_cfg.get('learning_rate', '?')
        epochs = training_cfg.get('epochs', '?')
        clip = training_cfg.get('clip_norm', 1.0)
        gn = "GradNorm" if gradnorm_cfg.get('enable', False) else "static"
        print(f"LR: {lr} | Max epochs: {epochs} | Grad clip: {clip}")
        print(f"Weighting: {gn}")
        print("Checkpoints: best_nse + best_loss + final")
        print("=" * 72)

    # ------------------------------------------------------------------
    # G.2  Per-epoch log line
    # ------------------------------------------------------------------
    def print_epoch(
        self,
        epoch: int,
        max_epochs: int,
        losses: Dict[str, float],
        total_loss: float,
        lr: float,
        grad_norm: float,
        val_metrics: Dict[str, float],
        neg_h_frac: float,
        epoch_time: float,
        weights: Optional[Dict[str, float]] = None,
    ):
        key_map = {
            'pde': 'PDE', 'ic': 'IC', 'bc': 'BC',
            'building_bc': 'Bld', 'data': 'Dat', 'neg_h': 'Phy',
        }
        parts = []
        for key in ['pde', 'ic', 'bc', 'building_bc', 'data', 'neg_h']:
            val = losses.get(key, 0.0)
            if abs(float(val)) > 1e-12:
                label = key_map.get(key, key.upper())
                parts.append(f"{label}:{float(val):.4e}")
        parts_str = ' '.join(parts)

        nse_h = val_metrics.get('nse_h', float('nan'))

        print(
            f"Epoch {epoch + 1:>6d}/{max_epochs} | "
            f"Loss: {total_loss:.6e} [{parts_str}] | "
            f"LR: {lr:.2e} | "
            f"\u2207: {grad_norm:.2e} | "
            f"NSE(h): {nse_h:.4f} | "
            f"-h: {neg_h_frac:.1%} | "
            f"{epoch_time:.1f}s"
        )
        if weights is not None:
            print(
                f"  Weights: "
                f"{ {k: f'{v:.2e}' for k, v in weights.items()} }"
            )

    # ------------------------------------------------------------------
    # G.3  Checkpoint events
    # ------------------------------------------------------------------
    def print_checkpoint_nse(self, nse_h: float, epoch: int,
                             prev_nse: float, prev_epoch: int):
        print(
            f">>> [best_nse] New best NSE(h) = {nse_h:.4f} at epoch "
            f"{epoch + 1} (prev: {prev_nse:.4f} @ {prev_epoch + 1})"
        )
        print(
            "    Saved: model.pkl + validation_outputs.npz "
            "\u2192 checkpoints/best_nse/"
        )

    def print_checkpoint_loss(self, loss: float, epoch: int,
                              prev_loss: float, prev_epoch: int):
        print(
            f">>> [best_loss] New best loss = {loss:.6e} at epoch "
            f"{epoch + 1} (prev: {prev_loss:.6e} @ {prev_epoch + 1})"
        )
        print(
            "    Saved: model.pkl + validation_outputs.npz "
            "\u2192 checkpoints/best_loss/"
        )

    # ------------------------------------------------------------------
    # G.4  Learning rate event
    # ------------------------------------------------------------------
    def print_lr_event(self, old_lr: float, new_lr: float,
                       epoch: int, stalled_epochs: int):
        print(
            f">>> LR reduced: {old_lr:.2e} \u2192 {new_lr:.2e} at epoch "
            f"{epoch + 1} (stalled for {stalled_epochs} epochs)"
        )
        self._n_lr_reductions += 1

    # ------------------------------------------------------------------
    # G.5  Completion summary
    # ------------------------------------------------------------------
    def print_completion_summary(
        self,
        total_time: float,
        final_epoch: int,
        best_nse_stats: dict,
        best_loss_stats: dict,
        final_losses: Dict[str, float],
        final_val_metrics: Dict[str, float],
        neg_depth_final: dict,
        neg_depth_best_nse: dict,
        neg_depth_best_loss: dict,
        final_lr: float,
        converged: bool = False,
    ):
        scenario = self.config.get('scenario', 'unknown')
        h = int(total_time // 3600)
        m = int((total_time % 3600) // 60)
        s = int(total_time % 60)

        print("\n" + "=" * 72)
        print(f"TRAINING COMPLETE \u2014 {scenario}")
        print("=" * 72)
        print(f"Outcome: {'Converged' if converged else 'Max epochs reached'}")
        print(f"Final epoch: {final_epoch + 1}")
        print(f"Total training time: {h}h {m}m {s}s")

        # --- Best NSE checkpoint ---
        if best_nse_stats:
            vm = best_nse_stats.get('validation_metrics', {})
            print(
                f"\nBEST NSE CHECKPOINT (epoch "
                f"{best_nse_stats.get('epoch', 0) + 1}):"
            )
            print(
                f"  NSE:  h={vm.get('nse_h', -999):.4f}  "
                f"hu={vm.get('nse_hu', -999):.4f}  "
                f"hv={vm.get('nse_hv', -999):.4f}"
            )
            print(
                f"  RMSE: h={vm.get('rmse_h', 999):.6f}m  "
                f"hu={vm.get('rmse_hu', 999):.6f}  "
                f"hv={vm.get('rmse_hv', 999):.6f}"
            )
            print(
                f"  L2:   h={vm.get('rel_l2_h', 999):.6f}  "
                f"hu={vm.get('rel_l2_hu', 999):.6f}  "
                f"hv={vm.get('rel_l2_hv', 999):.6f}"
            )
            print(
                f"  Loss at that epoch: "
                f"{best_nse_stats.get('total_loss', float('inf')):.6e}"
            )

        # --- Best loss checkpoint ---
        if best_loss_stats:
            vm = best_loss_stats.get('validation_metrics', {})
            print(
                f"\nBEST LOSS CHECKPOINT (epoch "
                f"{best_loss_stats.get('epoch', 0) + 1}):"
            )
            print(
                f"  Loss: "
                f"{best_loss_stats.get('total_loss', float('inf')):.6e}"
            )
            print(
                f"  NSE:  h={vm.get('nse_h', -999):.4f}  "
                f"hu={vm.get('nse_hu', -999):.4f}  "
                f"hv={vm.get('nse_hv', -999):.4f}"
            )
            print(
                f"  RMSE: h={vm.get('rmse_h', 999):.6f}m  "
                f"hu={vm.get('rmse_hu', 999):.6f}  "
                f"hv={vm.get('rmse_hv', 999):.6f}"
            )
            print(
                f"  L2:   h={vm.get('rel_l2_h', 999):.6f}  "
                f"hu={vm.get('rel_l2_hu', 999):.6f}  "
                f"hv={vm.get('rel_l2_hv', 999):.6f}"
            )

        # --- Final epoch ---
        if final_losses:
            total = sum(float(v) for v in final_losses.values())
            parts = ' '.join(
                f"{k.upper()}:{float(v):.6e}"
                for k, v in final_losses.items()
                if abs(float(v)) > 1e-12
            )
            print(f"\nFINAL EPOCH ({final_epoch + 1}):")
            print(f"  Loss: {total:.6e} [{parts}]")

        if final_val_metrics:
            print(
                f"  NSE:  h={final_val_metrics.get('nse_h', -999):.4f}  "
                f"hu={final_val_metrics.get('nse_hu', -999):.4f}  "
                f"hv={final_val_metrics.get('nse_hv', -999):.4f}"
            )

        # --- Negative depth ---
        def _neg_str(nd):
            if not nd:
                return "N/A"
            return f"{nd.get('fraction', 0):.2%} of points, min(h) = {nd.get('min', 0):.6e} m"

        print(f"\nNEGATIVE DEPTH:")
        print(f"  Final epoch: {_neg_str(neg_depth_final)}")
        print(f"  Best NSE epoch: {_neg_str(neg_depth_best_nse)}")
        print(f"  Best loss epoch: {_neg_str(neg_depth_best_loss)}")

        print(
            f"\nLR: final={final_lr:.2e}, "
            f"reduced {self._n_lr_reductions} times"
        )
        print("=" * 72)
