# CLAUDE.md

SWE-PINN: Physics-Informed Neural Network framework for urban flood prediction, solving 2D Shallow Water Equations with JAX/Flax. See @docs/experimental_programme_reference.md for the authoritative experiment spec.

## Reporting Requirement (applies to every task)

IMPORTANT: After finishing ANY task — implementation, refactor, bugfix, config change, experiment run, doc edit — always end with a **Full Implementation Report** containing:

1. **Summary** — one-paragraph description of what the task accomplished.
2. **Files changed** — every file created, modified, or deleted, with a one-line reason per file.
3. **Key code changes** — function/class-level description of the substantive edits (not a re-print of the diff).
4. **Configuration / hyperparameter changes** — any YAML, CLI flag, or environment variable change, with old → new values.
5. **Commands run** — shell commands executed (training, tests, installs), their exit status, and where their outputs live.
6. **Verification** — tests run, manual checks performed, W&B run URLs if a training job was launched, and the observed result.
7. **Open items / follow-ups** — anything left unfinished, deferred, or that needs user review.

This rule is global across all branches and overrides any default "be concise" behaviour for end-of-task summaries. Brevity is still preferred *during* the task; the full report is only at the end.

## Key Commands

```bash
# Training (all experiments follow this pattern)
python -m experiments.experiment_<N>.train --config configs/<config>.yaml

# Example: experiment 1 with variants
python -m experiments.experiment_1.train --config configs/experiment_1/experiment_1.yaml
python -m experiments.experiment_1.train_imp_samp --config configs/experiment_1/experiment_1_imp_samp.yaml
python -m experiments.experiment_1.train_relobralo --config configs/experiment_1/experiment_1_relobralo.yaml

# HPO-optimised configs live in configs/train/
python -m experiments.experiment_2.train --config configs/train/experiment_2_fourier_final.yaml

# Tests
python -m unittest discover test          # all tests
python -m unittest test.test_losses       # single file

# HPO
python optimisation/run_optimization.py --config <hpo_config> --n_trials 100

# Inference
python scripts/infer.py --config <config>.yaml --checkpoint models/<experiment>/<trial>/checkpoints/best_nse --checkpoints best_nse

# Dependencies
pip install -r .devcontainer/requirements.txt
```

## Code Style

- IMPORTANT: Always load configs via `src.config.load_config()` — never parse YAML directly
- Use `jax.numpy as jnp` (not `numpy`) inside JIT-compiled functions
- No linter or pre-commit hooks are configured

## Testing

- Tests force CPU: `os.environ["JAX_PLATFORM_NAME"] = "cpu"` at top of test files
- Use `unittest.mock` for file I/O and data; setUp/tearDown for temp directories
- Test both data-free and data-driven modes
- **Manual training smoke-tests: use 200 epochs**

## Common Pitfalls

- **JIT side effects**: No Python side effects inside `@jax.jit` functions — first call traces, subsequent calls replay the trace
- **Float precision**: Some physics computations require `float64`. Set via `config.device.dtype`
- **Large data files**: Use `np.load(..., mmap_mode='r')` for multi-GB validation datasets
- **Water depth masking**: `h >= eps` is required for numerical stability in SWE computations

## Experimental Programme

IMPORTANT: Always consult `docs/experimental_programme_reference.md` when:
- Implementing or modifying any evaluation metric or inference plot
- Setting up W&B tracking (what to log, at what frequency)
- Building or modifying any module in `src/metrics/` or `src/plots/`

Key conventions:
- Accuracy metrics (NSE, RMSE, MAE, Rel L2) reported separately for h, hu, hv
- Colour palette: Exeter (Deep Green #003C3C, Teal #007D69, Mint #00C896) + Blue Heart (Navy #0D2B45, Ocean #1B5E8A, Sky #4FA3D1). Arial font. 300 DPI.

---

## Git Workflow

- **Feature branches target `main` directly.** PR base is `main` (not a claude-cleanup branch — that rule was relaxed 2026-04-20).
- **Squash-merge** via `gh pr merge <N> --squash --delete-branch`.
- **Branch naming**: `feat/<area>-<short>` for features, `refactor/<short>` for refactors, `fix/<short>` for bugfixes.
- One PR per coherent scope. If a PR grows past ~500 lines across unrelated concerns, split it.
- Never force-push `main`.
- Tag headline result commits (`git tag exp1-ablation-v1`) so any thesis figure can be reproduced from `git checkout <tag>`.

## Repo Organization (what goes where)

| Dir | Purpose |
|---|---|
| `src/` | Reusable library code (physics, losses, training loop, scaling, monitoring). No experiment-specific logic. |
| `experiments/experiment_N/` | Experiment drivers. Entry-points that import from `src/` and produce a trained model. |
| `configs/experiment_N/` | YAML hyperparameter files. One per run variant. No Python. |
| `scripts/` | Automation around experiments — orchestrators, run launchers, aggregators, figure generators. Not experiments themselves. |
| `test/` | Unit tests for `src/` and smoke tests for experiments. |
| `docs/` | Design docs, session notes, workflow recipes. |
| `models/`, `results/`, `wandb/` | Generated artifacts (gitignored). Produced by training runs. |

**Rule of thumb**: if it produces a model → `experiments/`. If it consumes models/results to produce something else → `scripts/`. If it's reusable pure logic → `src/`.

## Artifact Organization (models/, archive/)

- **Active runs** land under `models/<timestamped-trial>/` as before (unchanged behaviour).
- **Archived runs** live under `models/archive/` and are intentionally excluded from the rotation / hygiene sweeps. Move a run there when it produced a headline figure or thesis result you must keep.
- **Local `wandb/` and `results/` are disposable.** W&B server is the authoritative log; local dirs can be deleted after `wandb sync`.
- **Reproducibility**: every headline run is recoverable from (a) the git tag/SHA at the time of training, (b) the W&B run URL, (c) the saved config snapshot in `models/<trial>/`. If any of the three is missing, the run is not thesis-quality.
