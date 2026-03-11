# SWE-PINN Cleanup Plan

**Created:** 2026-03-11
**Source:** [docs/repo_audit.md](repo_audit.md)

Each issue below is independently executable. Issues are grouped into milestones ordered by priority.

---

## Milestone 1 — Critical Fixes (Unblock Training)

These issues block `src/train.py` or individual experiment scripts from running at all.

---

### Issue #1: Guard broken `operator_learning` import in `gradnorm.py`

**Description:**
`gradnorm.py` line 18 unconditionally imports from `src.operator_learning.losses_op`, a module that does not exist. Because `src/train.py` imports `gradnorm.py` at the top level, this breaks the entire unified training entry point — `python src/train.py <config>` crashes immediately with `ImportError`.

**Affected files:**
- `src/gradnorm.py` (line 18–21)
- `src/train.py` (line 37 — imports gradnorm)

**Action items:**
1. Wrap the `from src.operator_learning.losses_op import ...` block in `gradnorm.py` with a `try/except ImportError` guard that sets the imported names to `None` or raises a clear error only when they are actually called.
2. If `update_gradnorm_weights_operatornet()` references these imports, add a runtime guard at the top of that function that raises `ImportError("src.operator_learning is not installed")` if the imports are `None`.
3. Verify `python src/train.py configs/experiment_3.yaml` starts without `ImportError`.

**Experiment scope:** All experiments (blocks unified training entry point).

---

### Issue #2: Guard broken imports in `experiment_1_deeponet.py`

**Description:**
`experiment_1_deeponet.py` (line 33–35) imports `compute_operator_pde_loss`, `compute_operator_ic_loss`, and `compute_operator_data_loss` from `src.losses`. These functions do not exist in `src/losses.py`. The script crashes on import with `ImportError`.

**Affected files:**
- `src/scenarios/experiment_1/experiment_1_deeponet.py` (lines 33–35)
- `src/losses.py` (missing functions)

**Action items:**
1. Add a `try/except ImportError` guard around the `compute_operator_*` imports.
2. Add a clear error message at the top of the script's `main()` function if the imports failed: `raise ImportError("DeepONet operator loss functions are not implemented in src/losses.py")`.
3. Alternatively, if DeepONet is being removed (see Milestone 3, Issue #16), skip this fix and remove the script instead.

**Experiment scope:** Experiment 1 (DeepONet variant only).

---

### Issue #3: Guard broken imports in `train_deeponet.py`

**Description:**
`train_deeponet.py` imports from `src.operator_learning.losses_op` (line 37) and `src.operator_learning.physics_op` (line 41), neither of which exist. Additionally, line 671 references `val_param_names` which is undefined (should be `param_names`, defined at line 143).

**Affected files:**
- `src/scenarios/experiment_1/train_deeponet.py` (lines 37, 41, 671)

**Action items:**
1. Wrap `src.operator_learning.*` imports in `try/except ImportError` guards.
2. Add a runtime guard at the top of the training function.
3. Fix `val_param_names` → `param_names` at line 671.
4. Alternatively, if DeepONet is being removed (see Milestone 3, Issue #16), skip this fix and remove the script instead.

**Experiment scope:** Experiment 1 (DeepONet variant only).

---

### Issue #4: Add missing `lx`, `ly` to `experiment_5.yaml`

**Description:**
`experiment_5.yaml` is missing `domain.lx` and `domain.ly` fields. Any experiment script that accesses `config["domain"]["lx"]` will crash with `KeyError` at runtime.

**Affected files:**
- `configs/experiment_5.yaml`

**Action items:**
1. Determine correct `lx`, `ly` values for Experiment 5 (benchmark_test_3 domain). Check the corresponding data files or `preprocess_irregular.py` output for spatial extents.
2. Add `lx` and `ly` under the `domain:` section in `experiment_5.yaml`.
3. Verify `python -m src.scenarios.experiment_5.experiment_5 configs/experiment_5.yaml` starts without `KeyError`.

**Experiment scope:** Experiment 5.

---

### Issue #5: Add missing `lx`, `ly` to `experiment_7.yaml`

**Description:**
`experiment_7.yaml` is missing `domain.lx` and `domain.ly` fields. Same issue as #4 but for the irregular-boundary experiment.

**Affected files:**
- `configs/experiment_7.yaml`

**Action items:**
1. Determine correct `lx`, `ly` values for Experiment 7 (benchmark_test_5 domain). Check the mesh artifacts or `domain_artifacts.npz` for spatial extents.
2. Add `lx` and `ly` under the `domain:` section in `experiment_7.yaml`.
3. Verify `python -m src.scenarios.experiment_7.experiment_7 configs/experiment_7.yaml` starts without `KeyError`.

**Experiment scope:** Experiment 7.

---

### Issue #6: Create `experiment_6.yaml` config

**Description:**
Experiment 6 has a training script (`src/scenarios/experiment_6/experiment_6.py`) and a data folder (`data/experiment_6/`) but no config file. `render_video.py` also references `configs/experiment_6.yaml` and will fail. This experiment cannot be run.

**Affected files:**
- `configs/experiment_6.yaml` (to be created)
- `scripts/render_video.py` (references this config)
- `src/scenarios/experiment_6/experiment_6.py` (needs a config to run)

**Action items:**
1. Create `configs/experiment_6.yaml` by cloning `experiment_5.yaml` (same experiment phase, closest in complexity).
2. Set `scenario: experiment_6` (or the correct data folder name — see Milestone 2).
3. Determine correct `domain.lx`, `domain.ly`, `domain.t_final`, and physics parameters for this scenario.
4. Verify the script can load the config without errors.

**Experiment scope:** Experiment 6.

---

## Milestone 2 — Naming Standardisation

These issues address the six different naming conventions, mismatched scenario keys, and typos that make the repo hard to navigate.

---

### Issue #7: Rename all config files to `experiment_<N>_<arch>_<variant>.yaml`

**Description:**
Config files use six different naming patterns (see audit Section 9.2). A single convention — `experiment_<N>_<arch>_<variant>.yaml` — should be adopted. The `configs/train/` subdirectory should be flattened into `configs/`.

**Affected files:**
- `configs/train/mlp_experiment_1.yaml` → `configs/experiment_1_mlp.yaml`
- `configs/train/fourier_experiment_1.yaml` → `configs/experiment_1_fourier_hpo.yaml`
- `configs/train/fourier_experiment_2.yaml` → `configs/experiment_2_fourier.yaml`
- `configs/train/DGM_no_experiment_1.yaml` → `configs/experiment_1_dgm.yaml`
- `configs/train/DGM_experiment_2.yaml` → `configs/experiment_2_dgm.yaml`
- `configs/dgm_datafree_static_experiment_1.yaml` → `configs/experiment_1_dgm_datafree_static.yaml`
- `configs/dgm_datafree_gradnorm_experiment_1.yaml` → `configs/experiment_1_dgm_datafree_gradnorm.yaml`
- `configs/config_operatornet_experiment_1.yaml` → `configs/experiment_1_operatornet.yaml`
- `configs/experiment_1_ntk_config.yaml` → `configs/experiment_1_ntk.yaml`
- `configs/pix2pix_experiment_3.yaml` → `configs/experiment_3_pix2pix.yaml`
- Any scripts, job files, or documentation that reference old config names.

**Action items:**
1. Rename each config file using `git mv` to preserve history.
2. Search all `.py`, `.sh`, and `.md` files for references to old filenames and update them.
3. Remove the `configs/train/` directory once empty.
4. Verify no broken references remain with `grep -r` for old filenames.

**Experiment scope:** All experiments.

---

### Issue #8: Update all `scenario:` keys to match `experiment_N` data folder names

**Description:**
Every data-dependent config has a `scenario:` key that doesn't match the actual data folder name (see audit Section 9.4). The code constructs paths as `data/{scenario}/`, so either the configs or the data folders must change. Since data folders already use `experiment_N/`, the configs should be updated.

**Affected files:**
- `configs/experiment_3.yaml`: `scenario: benchmark_test_1` → `scenario: experiment_3`
- `configs/pix2pix_experiment_3.yaml`: `scenario: benchmark_test_1` → `scenario: experiment_3`
- `configs/experiment_4.yaml`: `scenario: benchmark_test_2` → `scenario: experiment_4`
- `configs/experiment_5.yaml`: `scenario: benchmark_test_3` → `scenario: experiment_5`
- `configs/experiment_7.yaml`: `scenario: benchmark_test_5` → `scenario: experiment_7`
- `configs/experiment_8.yaml`: `scenario: benchmark_test_4` → `scenario: experiment_8`
- HPO configs in `optimisation/configs/` that reference `one_building_DEM_zero` or `no_building_scenario` — update to `experiment_1` or `experiment_2` as appropriate.
- Any experiment scripts that hardcode scenario names.

**Action items:**
1. Update each config file's `scenario:` key to match its `data/experiment_N/` folder.
2. Verify that `data/experiment_N/` folders exist and contain the expected `.npy` files.
3. If data folders still use legacy names (e.g. `data/benchmark_test_1/`), rename them with `git mv` to `data/experiment_3/` etc.
4. Search all scripts for hardcoded scenario strings and update.

**Experiment scope:** Experiments 2–8.

---

### Issue #9: Fix `sensivity_analysis_output` typo

**Description:**
The directory `optimisation/sensivity_analysis_output/` has a typo — it should be `sensitivity_analysis_output/`.

**Affected files:**
- `optimisation/sensivity_analysis_output/` (directory)
- `optimisation/extract_best_params.py` (line 8, references this path)
- Any other scripts or documentation referencing this directory.

**Action items:**
1. Rename the directory with `git mv optimisation/sensivity_analysis_output optimisation/sensitivity_analysis_output`.
2. Update all references in `extract_best_params.py` and any other files.
3. Grep the entire repo for `sensivity` to catch any remaining references.

**Experiment scope:** HPO pipeline.

---

### Issue #10: Standardise HPO config casing

**Description:**
HPO configs use inconsistent casing: exploration configs use `BUILDING`/`NOBUILDING` (uppercase) while exploitation configs use `building`/`nobuilding` (lowercase). This inconsistency makes it harder to script over all HPO configs.

**Affected files:**
- `optimisation/configs/exploration/*.yaml` (uppercase naming)
- `optimisation/configs/exploitation/*.yaml` (lowercase naming)
- Any shell scripts that glob over these configs.

**Action items:**
1. Choose a single casing convention (lowercase recommended: `building`, `nobuilding`).
2. Rename exploration configs from `*_BUILDING.yaml` → `*_building.yaml` and `*_NOBUILDING.yaml` → `*_nobuilding.yaml` using `git mv`.
3. Update any references in `run_all_hpo.sh`, `run_all_exploitations.sh`, and job scripts.
4. Optionally adopt the unified prefix: `hpo_exploration_<arch>_<scenario>.yaml` / `hpo_exploitation_<arch>_<scenario>.yaml`.

**Experiment scope:** HPO pipeline.

---

## Milestone 3 — Dead Code Removal

Remove code that is unused, broken beyond repair, or not part of the thesis.

---

### Issue #11: Remove `softadapt.py`

**Description:**
`src/softadapt.py` (58 lines) is never imported anywhere in the codebase. `grep -r "softadapt" --include="*.py"` returns zero matches outside the file itself. It is dead code with no integration point.

**Affected files:**
- `src/softadapt.py` (remove)
- `CLAUDE.md` (remove references to SoftAdapt)
- Any documentation mentioning SoftAdapt as a supported strategy.

**Action items:**
1. Delete `src/softadapt.py`.
2. Remove any references to SoftAdapt in `CLAUDE.md` and `README.md`.
3. Verify no import breaks with `python -c "import src"`.

**Experiment scope:** None (unused code).

---

### Issue #12: Remove or quarantine broken DeepONet scripts

**Description:**
Two DeepONet scripts are broken due to missing `src/operator_learning/` module:
- `src/scenarios/experiment_1/experiment_1_deeponet.py` — imports nonexistent `compute_operator_*` from `src.losses`
- `src/scenarios/experiment_1/train_deeponet.py` — imports nonexistent `src.operator_learning.*`; also has undefined variable `val_param_names`

These scripts cannot run. If DeepONet operator learning is not part of the thesis, they should be removed. If it is planned for future work, they should be quarantined in a separate branch.

**Affected files:**
- `src/scenarios/experiment_1/experiment_1_deeponet.py` (remove or quarantine)
- `src/scenarios/experiment_1/train_deeponet.py` (remove or quarantine)
- `configs/experiment_1_deeponet.yaml` (remove if scripts are removed)
- `configs/config_operatornet_experiment_1.yaml` (remove if scripts are removed)

**Action items:**
1. Confirm with the project owner whether DeepONet operator learning is needed.
2. If **not needed**: delete both scripts and their configs. Remove the `DeepONet`-related import guard added in Issue #1 (it becomes unnecessary).
3. If **needed for future work**: move to a `wip/deeponet` branch and remove from `main`.
4. Update `CLAUDE.md` to reflect the removal.

**Experiment scope:** Experiment 1 (DeepONet variants only).

---

### Issue #13: Remove `pix2pix_experiment_3.py` if not part of thesis

**Description:**
`src/scenarios/experiment_3/pix2pix_experiment_3.py` implements a Pix2Pix GAN surrogate model, which is architecturally distinct from the PINN framework. If this is an abandoned exploration and not part of the thesis, it should be removed to reduce maintenance burden.

**Affected files:**
- `src/scenarios/experiment_3/pix2pix_experiment_3.py` (remove if not needed)
- `configs/pix2pix_experiment_3.yaml` (remove if script is removed)

**Action items:**
1. Confirm with the project owner whether the Pix2Pix approach is part of the thesis.
2. If **not needed**: delete the script and its config.
3. If **needed**: keep and ensure it is documented in the experiment phase description.

**Experiment scope:** Experiment 3 (Pix2Pix variant only).

---

## Milestone 4 — Code Deduplication and Safety

Reduce duplicated logic and add safety checks to prevent silent failures.

---

### Issue #14: Extract shared `load_validation_data()` utility

**Description:**
Every experiment script (Exp 2–8) repeats the same data loading pattern: `np.load(path)` followed by column reordering `data[:, [1, 2, 0]]` to convert from `[t, x, y, ...]` to `[x, y, t]`. This is duplicated in 7+ locations and is error-prone — if the column order changes, every script must be updated.

**Affected files:**
- `src/data.py` (add utility function)
- `src/scenarios/experiment_2/experiment_2.py`
- `src/scenarios/experiment_3/experiment_3.py`
- `src/scenarios/experiment_4/experiment_4.py`
- `src/scenarios/experiment_5/experiment_5.py`
- `src/scenarios/experiment_6/experiment_6.py`
- `src/scenarios/experiment_7/experiment_7.py`
- `src/scenarios/experiment_8/experiment_8.py`
- `src/scenarios/experiment_8/experiment_8_imp_samp.py`

**Action items:**
1. Add a `load_validation_data(path: str) -> tuple[np.ndarray, np.ndarray]` function to `src/data.py` that:
   - Loads the `.npy` file
   - Reorders columns to `[x, y, t]` for inputs and extracts `[h, u, v]` for targets
   - Returns `(inputs, targets)` tuple
2. Replace all inline `data[:, [1, 2, 0]]` patterns in experiment scripts with calls to this utility.
3. Run existing tests to verify no regressions.

**Experiment scope:** Experiments 2–8.

---

### Issue #15: Merge duplicated GradNorm update functions

**Description:**
`src/gradnorm.py` contains two nearly identical functions: `update_gradnorm_weights()` (lines 107–193) and `update_gradnorm_weights_operatornet()` (lines 300–389). They are ~87 lines each and 95% identical, differing only in which loss functions they call.

**Affected files:**
- `src/gradnorm.py`

**Action items:**
1. Refactor into a single `update_gradnorm_weights()` function that accepts a loss function map (dict of name → callable) as a parameter.
2. Callers pass the appropriate loss functions for standard PINN or operator learning modes.
3. Remove `update_gradnorm_weights_operatornet()`.
4. Update all call sites (if any remain after Issue #12).
5. Run `python -m unittest test.test_train_gradnorm` to verify.

**Experiment scope:** All experiments using GradNorm.

---

### Issue #16: Standardise `loss_slip_wall_generalized` signature

**Description:**
`loss_slip_wall_generalized()` in `src/losses.py` (line 136) uses `model.apply(params, coords, train=True)` while all other loss functions use `model.apply({'params': params['params']}, batch, train=False)`. This inconsistency means callers must handle this function differently, breaking any generic loss iteration (e.g., GradNorm).

**Affected files:**
- `src/losses.py` (line 128–160, `loss_slip_wall_generalized`)
- `src/scenarios/experiment_7/experiment_7.py` (caller)
- `src/scenarios/experiment_8/experiment_8.py` (caller)
- `src/scenarios/experiment_8/experiment_8_imp_samp.py` (caller)

**Action items:**
1. Update `loss_slip_wall_generalized` to accept `params` in the same format as other loss functions and call `model.apply({'params': params['params']}, ...)`.
2. Update all callers in Exp 7 and Exp 8 scripts to pass `params` in the standard format.
3. Verify that the function produces identical results after the change.

**Experiment scope:** Experiments 7–8.

---

### Issue #17: Add bathymetry load warning instead of silent zero fallback

**Description:**
In `src/data.py` (lines 228–231), if `load_bathymetry()` has not been called, `bathymetry_fn(x, y)` silently returns `(0, 0, 0)` — treating the domain as perfectly flat. This means a wrong bathymetry file path or a missing `load_bathymetry()` call produces no error, and the model trains on incorrect physics without any indication.

**Affected files:**
- `src/data.py` (lines 228–231, `bathymetry_fn`)

**Action items:**
1. Add a `warnings.warn("Bathymetry not loaded — using flat domain (z=0). Call load_bathymetry() first if terrain is expected.", stacklevel=2)` on the first call when `_BATHYMETRY_FN is None`.
2. Use a module-level flag to emit the warning only once per session.
3. Optionally add a `require_bathymetry=True` parameter to experiment configs that, when set, raises an error instead of warning.

**Experiment scope:** Experiments 3–8 (all terrain-dependent experiments).

---

### Issue #18: Add shape assertions on data load

**Description:**
No shape validation exists when loading `.npy` data files. If a data file has the wrong number of columns or unexpected shape, errors are silent or produce cryptic downstream failures (e.g., wrong predictions, dimension mismatches deep in the training loop).

**Affected files:**
- `src/data.py` (add validation to the new `load_validation_data()` from Issue #14, or as a standalone utility)
- All experiment scripts that load `.npy` files.

**Action items:**
1. In the `load_validation_data()` function (Issue #14), add assertions:
   - `assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D"`
   - `assert data.shape[1] >= 6, f"Expected >=6 columns [t,x,y,h,u,v], got {data.shape[1]}"`
2. Add similar assertions for training data loading.
3. Include the file path in error messages for debuggability.

**Experiment scope:** Experiments 2–8.

---

## Milestone 5 — Portability

Remove machine-specific assumptions so the repo works on any clone location.

---

### Issue #19: Replace hardcoded absolute paths with relative paths

**Description:**
`optimisation/extract_best_params.py` (lines 7–8) uses hardcoded absolute paths:
```python
db_dir = Path("/workspaces/swe-pinn/optimisation/database/exploration")
output_base_dir = Path("/workspaces/swe-pinn/optimisation/sensivity_analysis_output/best_parameters")
```
These break on any machine where the repo is not cloned to `/workspaces/swe-pinn`.

**Affected files:**
- `optimisation/extract_best_params.py` (lines 7–8)
- Any other scripts with hardcoded absolute paths (search with `grep -rn "/workspaces/swe-pinn"` across all `.py` and `.sh` files).

**Action items:**
1. Replace absolute paths with `Path(__file__).resolve().parent / "database/exploration"` etc.
2. Search the entire repo for `/workspaces/swe-pinn` and fix all occurrences.
3. Verify the script runs correctly from the repo root.

**Experiment scope:** HPO pipeline.

---

### Issue #20: Add CLI arguments to `render_video.py`

**Description:**
`scripts/render_video.py` has three hardcoded values (lines 32–35):
```python
TRIAL_DIR_NAME = "2026-02-10_17-41_experiment_6"
CONFIG_NAME = "experiment_6.yaml"
SHAPEFILE_PATH = "data/experiment_6/2D Zones.shp"
```
The script only works for one specific trial and will fail because `configs/experiment_6.yaml` doesn't exist (see Issue #6).

**Affected files:**
- `scripts/render_video.py`

**Action items:**
1. Replace hardcoded values with `argparse` CLI arguments:
   - `--trial-dir` (required): path to the trial directory
   - `--config` (required): path to the config YAML file
   - `--shapefile` (optional): path to the shapefile for domain overlay
2. Keep the current values as defaults or examples in the `--help` text.
3. Verify the script runs with explicit arguments.

**Experiment scope:** Visualisation (all experiments).

---

### Issue #21: Add `__init__.py` to all experiment directories

**Description:**
None of the 8 `src/scenarios/experiment_N/` directories contain `__init__.py`. Running experiments as `python -m src.scenarios.experiment_1.experiment_1` may fail depending on Python version and import resolution, because Python needs `__init__.py` files to recognise directories as packages (unless using namespace packages, which is not the convention in this repo).

**Affected files:**
- `src/scenarios/experiment_1/__init__.py` (create)
- `src/scenarios/experiment_2/__init__.py` (create)
- `src/scenarios/experiment_3/__init__.py` (create)
- `src/scenarios/experiment_4/__init__.py` (create)
- `src/scenarios/experiment_5/__init__.py` (create)
- `src/scenarios/experiment_6/__init__.py` (create)
- `src/scenarios/experiment_7/__init__.py` (create)
- `src/scenarios/experiment_8/__init__.py` (create)

**Action items:**
1. Create empty `__init__.py` files in each `src/scenarios/experiment_N/` directory.
2. Verify `python -m src.scenarios.experiment_1.experiment_1 --help` (or equivalent) works.

**Experiment scope:** All experiments.

---

## Milestone 6 — Documentation

Improve code discoverability and onboarding.

---

### Issue #22: Add docstrings to all public functions in core modules

**Description:**
Core modules (`src/models.py`, `src/losses.py`, `src/physics.py`, `src/data.py`, `src/gradnorm.py`, `src/config.py`, `src/utils.py`, `src/reporting.py`, `src/ntk.py`) contain public functions without docstrings. Adding docstrings improves IDE support, `help()` output, and onboarding for new contributors.

**Affected files:**
- `src/models.py`
- `src/losses.py`
- `src/physics.py`
- `src/data.py`
- `src/gradnorm.py`
- `src/config.py`
- `src/utils.py`
- `src/reporting.py`
- `src/ntk.py`

**Action items:**
1. Add Google-style docstrings to every public function and class, documenting parameters, return values, and any side effects (e.g., `load_bathymetry` sets a global).
2. Do not add docstrings to private/internal helpers unless their behaviour is non-obvious.
3. Keep docstrings concise — one-liner for simple functions, multi-line for complex ones.

**Experiment scope:** All experiments (core infrastructure).

---

### Issue #23: Add inline comments to each experiment script explaining purpose and phase

**Description:**
Experiment scripts lack a header comment explaining which research phase they belong to, what scenario they implement, and how they differ from adjacent experiments. A reader must cross-reference `CLAUDE.md` to understand the purpose of each script.

**Affected files:**
- `src/scenarios/experiment_1/experiment_1.py`
- `src/scenarios/experiment_1/analytical_ntk.py`
- `src/scenarios/experiment_1/experiment_1_lbfgs_finetune.py`
- `src/scenarios/experiment_2/experiment_2.py`
- `src/scenarios/experiment_3/experiment_3.py`
- `src/scenarios/experiment_4/experiment_4.py`
- `src/scenarios/experiment_5/experiment_5.py`
- `src/scenarios/experiment_6/experiment_6.py`
- `src/scenarios/experiment_7/experiment_7.py`
- `src/scenarios/experiment_8/experiment_8.py`
- `src/scenarios/experiment_8/experiment_8_imp_samp.py`

**Action items:**
1. Add a module-level docstring to each experiment script with:
   - Research phase (1, 2, or 3)
   - One-sentence description of what the experiment tests
   - Key differences from the previous experiment
   - Required data files and config
2. Example format:
   ```python
   """Experiment 3 — X-direction terrain slope (Phase 2).

   Introduces bathymetry via bi-linear interpolation into the SWE momentum
   equations. Establishes data sampling ratio methodology for cases where
   physics-only training is insufficient.

   Requires: configs/experiment_3.yaml, data/experiment_3/
   Builds on: Experiment 2 (adds terrain; removes building obstacle).
   """
   ```

**Experiment scope:** All experiments.

---

### Issue #24: Update `README.md` to reflect final repo structure

**Description:**
After completing Milestones 1–5, the repository structure will have changed: configs will be renamed, dead code removed, new utilities added, and directories renamed. `README.md` must be updated to reflect the final state.

**Affected files:**
- `README.md`
- `CLAUDE.md` (repository structure section)

**Action items:**
1. Update the directory tree in `README.md` to match the post-cleanup structure.
2. Update any config file references to use new names.
3. Remove references to deleted files (softadapt.py, DeepONet scripts if removed, pix2pix if removed).
4. Add a section mapping experiment numbers to research phases.
5. Update `CLAUDE.md` repository structure section to match.
6. Verify all code examples in README still work with the new file names.

**Experiment scope:** All (documentation).

---

## Summary

| Milestone | Issues | Priority |
|-----------|--------|----------|
| 1 — Critical Fixes | #1–#6 | **Immediate** — unblocks training |
| 2 — Naming Standardisation | #7–#10 | **High** — reduces confusion |
| 3 — Dead Code Removal | #11–#13 | **Medium** — reduces maintenance |
| 4 — Code Deduplication & Safety | #14–#18 | **Medium** — improves reliability |
| 5 — Portability | #19–#21 | **Medium** — enables collaboration |
| 6 — Documentation | #22–#24 | **Low** — improves onboarding |

**Total: 24 issues across 6 milestones.**
