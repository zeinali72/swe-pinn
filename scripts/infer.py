#!/usr/bin/env python
"""CLI entry point for the inference pipeline.

Usage:
    python scripts/infer.py --config <yaml> --checkpoint <dir> [options]

Examples:
    # Single checkpoint
    python scripts/infer.py \\
        --config configs/experiment_1_fourier.yaml \\
        --checkpoint models/experiment_1/trial_001/checkpoints/best_nse

    # All checkpoints in a trial
    python scripts/infer.py \\
        --config configs/experiment_3.yaml \\
        --checkpoint models/experiment_3/trial_001/checkpoints \\
        --checkpoints all
"""
import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(
        description="Run post-training inference and validation on a trained SWE-PINN model.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint directory (or parent directory when using --checkpoints all).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for reports. Auto-generated if not set.",
    )
    parser.add_argument(
        "--checkpoints", default="best_nse",
        help=(
            "Which checkpoint(s) to evaluate: best_nse, best_loss, final, "
            "or 'all' to evaluate all three. Default: best_nse."
        ),
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--skip-conservation", action="store_true",
        help="Skip expensive autodiff conservation metrics.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50_000,
        help="Predictor batch size (default: 50000).",
    )

    args = parser.parse_args()

    from src.inference.runner import run_inference, run_inference_multi

    common_kwargs = dict(
        skip_plots=args.skip_plots,
        skip_conservation=args.skip_conservation,
        batch_size=args.batch_size,
    )

    if args.checkpoints in ("best_nse", "best_loss", "final"):
        results = run_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            checkpoint_name=args.checkpoints,
            **common_kwargs,
        )
    else:
        results = run_inference_multi(
            config_path=args.config,
            checkpoint_dir=args.checkpoint,
            output_dir=args.output,
            checkpoints=args.checkpoints,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
