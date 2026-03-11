# Cleanup Issues Index

**Milestone:** [repo-cleanup](https://github.com/zeinali72/swe-pinn/milestone/1)
**Branch:** `claude-cleanup`
**Generated:** 2026-03-11

## Already resolved (prior cleanup on `claude-cleanup`)

The following original plan items were resolved by deleting the affected files before issue creation:

| Original Plan Item | Resolution |
|---|---|
| Guard broken `operator_learning` import in `gradnorm.py` | `src/gradnorm.py` deleted |
| Add missing `lx`, `ly` to `experiment_5.yaml` | Values already added |
| Create `experiment_6.yaml` config | Experiment 6 removed entirely (script + config deleted) |
| Remove `softadapt.py` | `src/softadapt.py` deleted |
| Merge duplicated GradNorm update functions | `src/gradnorm.py` deleted |

---

## Active Issues

| Issue | Title | Milestone | Labels | Status |
|-------|-------|-----------|--------|--------|
| [#4](https://github.com/zeinali72/swe-pinn/issues/4) | [M1] Remove broken DeepONet scripts and their configs | M1 — Critical Fixes | cleanup, critical | Open |
| [#5](https://github.com/zeinali72/swe-pinn/issues/5) | [M1] Add missing lx, ly to experiment_7.yaml | M1 — Critical Fixes | cleanup, critical | Open |
| [#6](https://github.com/zeinali72/swe-pinn/issues/6) | [M2] Rename all config files to experiment_\<N\>_\<arch\>_\<variant\>.yaml | M2 — Naming | cleanup, naming | Open |
| [#7](https://github.com/zeinali72/swe-pinn/issues/7) | [M2] Update all scenario: keys to match experiment_N data folder names | M2 — Naming | cleanup, naming | Open |
| [#8](https://github.com/zeinali72/swe-pinn/issues/8) | [M2] Fix sensivity_analysis_output typo | M2 — Naming | cleanup, naming | Open |
| [#9](https://github.com/zeinali72/swe-pinn/issues/9) | [M2] Standardise HPO config casing | M2 — Naming | cleanup, naming | Open |
| [#10](https://github.com/zeinali72/swe-pinn/issues/10) | [M3] Remove pix2pix_experiment_3.py if not part of thesis | M3 — Dead Code | cleanup, dead-code | Open |
| [#11](https://github.com/zeinali72/swe-pinn/issues/11) | [M4] Extract shared load_validation_data() utility | M4 — Refactor | cleanup, refactor | Open |
| [#12](https://github.com/zeinali72/swe-pinn/issues/12) | [M4] Standardise loss_slip_wall_generalized signature | M4 — Refactor | cleanup, refactor | Open |
| [#13](https://github.com/zeinali72/swe-pinn/issues/13) | [M4] Add bathymetry load warning instead of silent zero fallback | M4 — Refactor | cleanup, refactor | Open |
| [#14](https://github.com/zeinali72/swe-pinn/issues/14) | [M4] Add shape assertions on data load | M4 — Refactor | cleanup, refactor | Open |
| [#15](https://github.com/zeinali72/swe-pinn/issues/15) | [M5] Replace hardcoded absolute paths with relative paths | M5 — Portability | cleanup, portability | Open |
| [#16](https://github.com/zeinali72/swe-pinn/issues/16) | [M5] Add CLI arguments to render_video.py | M5 — Portability | cleanup, portability | Open |
| [#17](https://github.com/zeinali72/swe-pinn/issues/17) | [M5] Add __init__.py to all experiment directories | M5 — Portability | cleanup, portability | Open |
| [#18](https://github.com/zeinali72/swe-pinn/issues/18) | [M6] Add docstrings to all public functions in core modules | M6 — Docs | cleanup, docs | Open |
| [#19](https://github.com/zeinali72/swe-pinn/issues/19) | [M6] Add module-level docstrings to each experiment script | M6 — Docs | cleanup, docs | Open |
| [#20](https://github.com/zeinali72/swe-pinn/issues/20) | [M6] Update README.md and CLAUDE.md to reflect post-cleanup repo structure | M6 — Docs | cleanup, docs | Open |

---

## Summary by Milestone

| Milestone | Issues | Priority |
|-----------|--------|----------|
| M1 — Critical Fixes | #4, #5 | **Immediate** — unblocks training |
| M2 — Naming Standardisation | #6, #7, #8, #9 | **High** — reduces confusion |
| M3 — Dead Code Removal | #10 | **Medium** — reduces maintenance |
| M4 — Code Deduplication & Safety | #11, #12, #13, #14 | **Medium** — improves reliability |
| M5 — Portability | #15, #16, #17 | **Medium** — enables collaboration |
| M6 — Documentation | #18, #19, #20 | **Low** — improves onboarding |

**Total: 17 active issues across 6 milestones.**
