# SWE-PINN Codebase Analysis

Comprehensive code review covering sanity, logic, redundancy, and performance across
`src/`, `experiments/`, `scripts/`, `optimisation/`, `test/`, and `configs/`.

---

## Table of Contents

1. [Critical Bugs](#1-critical-bugs)
2. [Logic Errors](#2-logic-errors)
3. [Sanity Issues](#3-sanity-issues)
4. [Redundancy](#4-redundancy)
5. [Performance & Bottlenecks](#5-performance--bottlenecks)
6. [Testing Gaps](#6-testing-gaps)
7. [Configuration Issues](#7-configuration-issues)

---

## 1. Critical Bugs

### `optimisation/optimization_train_loop.py`: PRNG Key Reuse Destroys Epoch Randomisation

* **What should be changed**: Line 402 resets `train_key = key`, where `key` is the
  original seed from line 122 (`key = random.PRNGKey(trial_cfg["training"]["seed"])`).
  This means every epoch starts with the **same PRNG state**, producing identical
  collocation point samples across all epochs.
* **Why**: The entire purpose of re-sampling each epoch is to expose the network to
  diverse collocation points. Reusing the same key makes HPO trials train on identical
  data every epoch, severely degrading sample diversity and final model quality. Results
  from any HPO study using this code are compromised.
* **How**: Delete line 402. The split on line 356 (`train_key, epoch_key = random.split(train_key)`)
  already correctly advances the PRNG chain:
  ```python
  # Line 402 — DELETE this line:
  train_key = key  # <-- BUG: resets to original seed
  ```

---

### `experiments/experiment_7/train.py`: Missing `FrozenDict` Import

* **What should be changed**: Line 138 uses `FrozenDict(cfg_dict)` but the module never
  imports `FrozenDict` from `flax.core`.
* **Why**: The script will crash with a `NameError` at runtime before training begins.
* **How**: Add the import at the top of the file alongside other flax imports:
  ```python
  from flax.core import FrozenDict
  ```

### `experiments/experiment_8/train.py`: Missing `FrozenDict` Import

* **What should be changed**: Same issue as experiment 7 — line 142 uses `FrozenDict(cfg_dict)` without importing it.
* **Why**: Runtime `NameError` crash.
* **How**: Add `from flax.core import FrozenDict` to the imports.

---

### `experiments/experiment_6/train.py`: Axis Mismatch in Boundary Concatenation

* **What should be changed**: Line 193 concatenates two batched wall segments along `axis=1`:
  ```python
  bc_left_wall = jnp.concatenate([bc_left_wall_bottom, bc_left_wall_above], axis=1)
  ```
  Both inputs have shape `(num_batches, batch_size, 3)`, producing
  `(num_batches, 2*batch_size, 3)`. Downstream loss functions expect uniform
  `batch_size` across all boundary terms.
* **Why**: The left wall batches will have double the points compared to other walls,
  causing an implicit over-weighting of left wall BC losses. Worse, if any downstream
  code indexes into these batches expecting `batch_size` rows, it will silently use
  only half the data or produce shape errors inside `lax.scan`.
* **How**: Either (a) halve `n_bc_per_wall` for left-wall sub-segments so the concatenated
  result matches `batch_size`, or (b) concatenate along `axis=0` to double the number
  of batches (then adjust `num_batches` accordingly), or (c) interleave sampling within
  the `generate_epoch_data` function.

---

## 2. Logic Errors

### `src/losses/pde.py`: Inconsistent `train` Flag in PDE Loss

* **What should be changed**: Lines 21 and 24 use conflicting `train` flags:
  ```python
  U_pred = model.apply({'params': params['params']}, pde_batch, train=True)   # L21
  def U_fn(pts):
      return model.apply({'params': params['params']}, pts, train=False)      # L24
  ```
  `U_pred` (used for physics terms like `SWEPhysics`) is computed with `train=True`,
  but the Jacobian `jac_U` (used for spatial/temporal derivatives) is computed through
  `U_fn` with `train=False`. These are **two independent forward passes** on the same
  data with potentially different outputs.
* **Why**: If any architecture uses stochastic layers (dropout, stochastic depth) or
  stateful layers (batch norm), the two forward passes will produce inconsistent
  values. The PDE residual `dU/dt + div(F) + div(G) - S` will be evaluated with
  `U_pred` from one pass and derivatives from another, introducing a systematic bias.
  Even without stochastic layers, this is a latent bug waiting to trigger on
  architecture changes.
* **How**: Use a single forward pass. Compute derivatives via `jacfwd` on `U_fn` with
  a consistent `train` flag, and reuse the same prediction for `SWEPhysics`:
  ```python
  def U_fn(pts):
      return model.apply({'params': params['params']}, pts, train=False)

  jac_U = jax.vmap(jax.jacfwd(U_fn))(pde_batch)
  U_pred = U_fn(pde_batch)  # Same function, same train flag
  ```

---

### `src/losses/pde.py`: Negative Depth Loss Diluted by Masking

* **What should be changed**: In `compute_neg_h_loss` (lines 57-64), the mask is
  multiplied into `h_pred` before computing the penalty:
  ```python
  h_pred = h_pred * pde_mask          # masked points become 0
  return jnp.mean(jax.nn.relu(-h_pred) ** 2)  # mean over ALL points
  ```
  While masked points correctly contribute 0 penalty, the `jnp.mean` denominator
  includes them, diluting the loss by the fraction of masked points.
* **Why**: If 50% of points are masked, the effective negative-depth penalty is halved
  compared to the unmasked case. This creates an implicit coupling between domain
  geometry (mask fraction) and loss magnitude, making loss weight tuning
  geometry-dependent.
* **How**: Apply mask after computing per-point losses and use a masked mean:
  ```python
  per_point_loss = jax.nn.relu(-h_pred) ** 2
  if pde_mask is not None:
      per_point_loss = per_point_loss * pde_mask
      return jnp.sum(per_point_loss) / jnp.maximum(jnp.sum(pde_mask), 1.0)
  return jnp.mean(per_point_loss)
  ```

---

### `src/physics/swe.py`: Redundant Dead Assignment in Source Term

* **What should be changed**: Lines 37-39 initialise `sox` and `soy` to `0.0` then
  immediately overwrite them:
  ```python
  sox = soy = 0.0                                      # L37 — dead
  sox = -bed_grad_x if bed_grad_x is not None else 0.0  # L38
  soy = -bed_grad_y if bed_grad_y is not None else 0.0  # L39
  ```
* **Why**: The first assignment on L37 is dead code. While not a runtime bug, it
  obscures intent and can mislead reviewers into thinking a default persists.
* **How**: Remove line 37 entirely.

---

### `src/config.py`: Global Mutable State for `DTYPE` and `EPS`

* **What should be changed**: `load_config` (lines 31-33) sets module-level globals
  `DTYPE` and `EPS` as a side effect. Every module that does
  `from src.config import DTYPE` captures the value at import time, but the global
  itself changes on each `load_config` call.
* **Why**: (1) Calling `load_config` with different configs in the same process
  (e.g. HPO with varying dtypes) silently mutates shared state. (2) Modules that
  import `DTYPE` at the top level get the value from the **first** `load_config` call
  only. (3) Tests that load different configs can leak state between test cases.
* **How**: Pass `dtype` and `eps` through the config dict and access them at point of
  use. As a transitional step, expose them as functions:
  ```python
  def get_dtype():
      if DTYPE is None:
          raise RuntimeError("load_config() has not been called yet")
      return DTYPE
  ```

---

### `src/losses/boundary.py`: Unsafe `squeeze()` for Shape Alignment

* **What should be changed**: Lines 14-15 (and analogous lines 25-26, 36-37) use
  `.squeeze()` to align target and prediction dimensions:
  ```python
  if h_target.ndim != h_pred.ndim:
      h_target = h_target.squeeze()
  ```
* **Why**: `squeeze()` removes **all** singleton dimensions. If `h_target` has shape
  `(1, N, 1)`, it becomes `(N,)` — silently dropping a batch dimension. This can
  cause broadcasting errors or incorrect loss values.
* **How**: Use explicit reshape to match the prediction shape:
  ```python
  if h_target.ndim != h_pred.ndim:
      h_target = h_target.reshape(h_pred.shape)
  ```

---

### `optimisation/objective_function.py`: Lambda Closure Variable Capture

* **What should be changed**: Line 121-122 defines a lambda inside a loop that
  captures loop variables `w`, `min_w`, `max_w` by reference:
  ```python
  for w in ["pde_weight", "ic_weight", "bc_weight", "neg_h_weight", "building_bc_weight"]:
      min_w, max_w = (1.0, 1e6) if w == "pde_weight" else (1e-2, 1e3)
      trial_params["loss_weights"][w] = suggest(w, weights_cfg,
          lambda: trial.suggest_float(w, min_w, max_w, log=True))
  ```
* **Why**: In Python, closures capture variables by reference. If `suggest()` defers
  lambda evaluation (which `get_hpo_value` does when `config_val is None`), all
  lambdas will use the final loop iteration's values for `w`, `min_w`, `max_w`.
  Currently safe because `get_hpo_value` calls the lambda immediately, but any
  refactoring to lazy evaluation will break.
* **How**: Bind loop variables via default arguments:
  ```python
  lambda w=w, min_w=min_w, max_w=max_w: trial.suggest_float(w, min_w, max_w, log=True)
  ```

---

## 3. Sanity Issues

### `test/test_train.py`: Broken Module Import

* **What should be changed**: Line 12 imports from `experiments.experiment_1.train`,
  but the `experiments/` directory uses a flat script layout, not a package. The
  actual entry point would need a different import path, and the test's mock setup
  may not patch the correct targets.
* **Why**: The integration test cannot run, giving zero coverage of the experiment_1
  training flow through the test suite.
* **How**: Update the import to match the actual module path:
  ```python
  from experiments.experiment_1.train import main as train_main
  ```
  Also verify the `experiments/` directory has proper `__init__.py` files (it does
  have a root one, but check subdirectories), or adjust `sys.path` accordingly.

---

### `test/test_train.py`: Redundant Test Config (Dead Code)

* **What should be changed**: `test/test_assets/test_config.yaml` exists but is never
  loaded. The test instead generates a config programmatically in
  `create_test_config()` (lines 37-116) with **different values** (e.g.
  `learning_rate: 0.001` vs the YAML's `0.0001`).
* **Why**: Maintenance burden — two competing config definitions that can drift.
  Developers may update the YAML thinking it's used, with no effect on tests.
* **How**: Either delete the YAML file and keep the programmatic config, or load the
  YAML file and remove the programmatic one. One source of truth.

---

### `scripts/render_video.py`: Unused Imports

* **What should be changed**: Lines 12-15 import `concurrent.futures`, `subprocess`,
  `time`, and `multiprocessing`, none of which are used in the script.
* **Why**: Dead imports increase load time and confuse readers about the script's
  capabilities.
* **How**: Remove the four unused imports.

---

### `src/data/bathymetry.py`: Global Mutable State for Bathymetry Function

* **What should be changed**: Lines 12-14 use module-level mutable state
  (`_BATHYMETRY_FN`, `_BATHYMETRY_WARNING_EMITTED`) to store a JIT-compiled
  function pointer.
* **Why**: (1) Not thread-safe if multiple experiments run in the same process.
  (2) Prevents testing with different DEMs without process restart. (3) JAX's
  functional paradigm discourages hidden mutable state.
* **How**: Return the bathymetry function from `load_bathymetry()` and pass it
  explicitly to callers (loss functions, data generators). Store it in the config
  or a context object rather than module globals.

---

### `experiments/experiment_3/train.py`, `experiment_5/train.py`: Unused `matplotlib` Import

* **What should be changed**: Both files import `matplotlib.pyplot as plt` but never
  call any plotting functions.
* **Why**: Unnecessary dependency loading. On headless servers, matplotlib import can
  trigger backend warnings.
* **How**: Remove `import matplotlib.pyplot as plt` from both files.

---

### `experiments/experiment_1/train.py`: Unused `val_key` Variable

* **What should be changed**: Line 105 assigns `val_key = setup["val_key"]` but it is
  never used in the function body.
* **Why**: Dead variable, suggests incomplete refactoring.
* **How**: Remove the assignment or use it for validation point sampling.

---

## 4. Redundancy

### `experiments/experiment_3-8`: Massive Code Duplication Across Experiment Scripts

* **What should be changed**: The following code blocks are duplicated near-verbatim
  across 5-6 experiment files:

  | Block | Lines (approx.) | Duplicated In |
  |-------|:---:|---|
  | Asset loading (DEM + BC CSV) | ~26 lines | exp 3, 4, 5, 6, 7 |
  | Validation data loading | ~15 lines | exp 3, 4, 5, 6, 7 |
  | Sampling config extraction | ~10 lines | exp 3, 4, 5, 6, 7, 8 |
  | `run_training_loop()` call | ~20 lines | exp 3, 4, 5, 6, 7 |
  | `plot_gauge()` nested function | ~25 lines | exp 4, 6, 7, 8 |
  | `make_compute_losses()` boilerplate | ~30 lines (common parts) | exp 3-8 |

  Total duplicated code: **~600+ lines** across experiment files.

* **Why**: Any bug fix or improvement must be applied in 5-6 places. The FrozenDict
  import bug (Section 1) is a direct consequence — experiments 1 and 2 were fixed
  but 7 and 8 were not. This is a maintenance time bomb.

* **How**: Extract shared scaffolding into `src/training/` or a shared experiment base:

  1. **Asset loading**: Create `src/training/setup.py::load_experiment_assets(cfg)` that
     loads DEM and BC CSV, returning a context dict.
  2. **Gauge plotting**: Move `plot_gauge()` to `src/utils/plotting.py::plot_gauge_timeseries()`.
  3. **Loss factory pattern**: Each experiment already defines a `make_compute_losses()`
     closure. Factor the common PDE/IC/neg_h/data terms into a base, and let
     experiments provide only their BC-specific terms via a callback.
  4. **Experiment runner**: Create a `run_experiment(cfg, compute_losses_factory)` function
     that handles the full setup-train-save-plot pipeline, with experiment-specific
     behavior injected via the loss factory and optional plot callbacks.

---

### `src/losses/boundary.py`: Three Nearly Identical Dirichlet Loss Functions -- RESOLVED

* **Status**: Fixed. The three thin wrappers (`loss_boundary_dirichlet_h`,
  `loss_boundary_dirichlet_hu`, `loss_boundary_dirichlet_hv`) have been removed.
  All callers now use the consolidated `loss_boundary_dirichlet(model, params, batch, target, var_idx)` directly.

---

### `src/models/pinn.py`: Repeated `Normalize` Construction

* **What should be changed**: All three architectures (`FourierPINN`, `MLP`,
  `DGMNetwork`) create identical `Normalize` layers with the same parameters from
  `domain_cfg`. `FourierPINN` creates it in `setup()`, but `MLP` and `DGMNetwork`
  create it inline in `__call__`.
* **Why**: Inconsistent initialization patterns. In `MLP` and `DGMNetwork`, the
  `Normalize` layer is re-created on every `__call__` invocation (Flax handles this
  via `@nn.compact`, but it's still unnecessary boilerplate).
* **How**: Standardise: either all use `setup()` (like `FourierPINN`) or all use
  `@nn.compact` inline. Prefer `setup()` for clarity.

---

### `optimisation/run_optimization.py` & `run_sensitivity_analysis.py`: Duplicated `sanitize_for_yaml`

* **What should be changed**: Both files contain identical `sanitize_for_yaml()`
  functions (~6 lines each) at lines 133-138 and 137-142 respectively.
* **Why**: Any fix to YAML serialisation must be applied in two places.
* **How**: Extract to a shared utility, e.g. `optimisation/utils.py::sanitize_for_yaml()`.

---

## 5. Performance & Bottlenecks

### `src/training/loop.py`: `copy.deepcopy(params)` on Every Best-Model Update

* **What should be changed**: Lines 165 and 175 deep-copy the full parameter tree
  whenever a new best NSE or best loss is found:
  ```python
  best_params_nse = copy.deepcopy(params)
  best_params_loss = copy.deepcopy(params)
  ```
* **Why**: `copy.deepcopy` on JAX arrays is expensive — it copies device memory to
  host, duplicates, and copies back. For large models (512-wide, 6 layers deep),
  this can add 10-50ms per checkpoint update, and it happens twice (NSE + loss).
  Over thousands of epochs, this accumulates.
* **How**: Use `jax.tree.map(jnp.copy, params)` which stays on-device:
  ```python
  best_params_nse = jax.tree.map(jnp.copy, params)
  ```

---

### `src/data/batching.py`: Modulo-Based Batch Cycling Instead of Reshuffling

* **What should be changed**: `get_batches_tensor` (lines 19-31) cycles through
  available batches using modulo indexing:
  ```python
  indices = jnp.arange(total_batches) % n_batches_avail
  return data[indices]
  ```
  When `total_batches > n_batches_avail`, later batches are exact duplicates of
  earlier ones within the same epoch.
* **Why**: The network sees repeated data within a single epoch, reducing effective
  sample diversity. The sampling function already generates fresh points each epoch,
  so this is only a within-epoch issue, but it still wastes gradient steps on
  duplicate data.
* **How**: If more batches are needed than available, re-permute and tile:
  ```python
  if total_batches > n_batches_avail:
      reps = (total_batches + n_batches_avail - 1) // n_batches_avail
      data = jnp.tile(data, (reps, 1, 1))[:total_batches]
      data = random.permutation(key, data, axis=0)
  ```
  Or better, sample enough points to fill `total_batches * batch_size` directly.

---

### `src/losses/pde.py`: Bathymetry Recomputed Every PDE Loss Call

* **What should be changed**: Line 34 calls `bathymetry_fn(x_batch, y_batch)` on
  every PDE loss evaluation. Since bathymetry is static (terrain doesn't change
  during training), this recomputes elevation gradients for potentially overlapping
  spatial coordinates across epochs.
* **Why**: The `bathymetry_fn` involves a JIT-compiled `vmap` of `map_coordinates` +
  `grad`, which is not free. For high PDE sample counts (50k+ points), this adds
  measurable overhead.
* **How**: JIT caching mitigates this for same-shaped inputs, so this is a low-priority
  optimisation. For larger-scale runs, consider pre-computing bed gradients on a
  fixed grid and using interpolation lookups instead of AD-through-interpolation.

---

### `experiments/experiment_8/train_imp_samp.py`: Double Gradient Computation

* **What should be changed**: The importance sampling variant computes PDE gradients
  twice per update cycle:
  1. `compute_pde_residual_vector()` at line 86 — for importance weights
  2. `compute_weighted_pde_loss()` at line 139 — for the actual training step

  Both call `jax.vmap(jax.jacfwd(U_fn))(pde_batch)` independently.
* **Why**: Jacobian computation via `jacfwd` is the most expensive part of each step.
  Computing it twice doubles the dominant cost of each iteration.
* **How**: Restructure to compute the Jacobian once and pass it to both the residual
  vector computation and the loss function. Or combine both operations into a single
  function that returns both the loss value and the per-point residual magnitudes.

---

### `experiments/experiment_8/train_imp_samp.py`: Blocking `block_until_ready()` in Loop

* **What should be changed**: Lines 532-536 use `block_until_ready()` inside a
  batch loop:
  ```python
  batch_errs_gpu = get_residuals_jitted(model, params, batch_pts_gpu, cfg)
  batch_errs_cpu = np.array(batch_errs_gpu.block_until_ready())
  ```
* **Why**: `block_until_ready()` forces CPU to wait for GPU completion on each batch,
  preventing pipelining of CPU-side batch preparation with GPU computation. This
  serialises what could be overlapped work.
* **How**: Enqueue all batches first, then collect results:
  ```python
  futures = [get_residuals_jitted(model, params, jax.device_put(batch), cfg)
             for batch in batches]
  results = [np.array(f.block_until_ready()) for f in futures]
  ```

---

### `scripts/extract_gauge_timeseries.py`: Full Array Load Without Memory Mapping

* **What should be changed**: Line 51 loads the entire data array into RAM:
  ```python
  data = np.load(input_path)
  ```
  Other scripts (e.g. `generate_training_data.py`) correctly use `mmap_mode='r'`.
* **Why**: For multi-GB validation tensors, this can cause OOM on memory-constrained
  systems.
* **How**: Use `np.load(input_path, mmap_mode='r')`.

---

## 6. Testing Gaps

### Zero Unit Tests for 5 Critical Modules

* **What should be changed**: The following modules have no unit tests whatsoever:

  | Module | Files | Lines | Risk |
  |--------|:---:|:---:|---|
  | `src/losses/` | 4 | ~350 | PDE physics correctness |
  | `src/models/` | 4 | ~310 | Architecture forward-pass correctness |
  | `src/physics/` | 1 | ~45 | SWE Jacobian/source terms |
  | `src/checkpointing/` | 3 | ~200 | Save/load round-trip integrity |
  | `src/monitoring/` | 3 | ~200 | Logging accuracy |

* **Why**: The physics implementation and loss functions are the mathematical core of
  the project. Bugs in these modules (like the `train` flag inconsistency in Section 2)
  can silently degrade model quality without raising exceptions.
* **How**: Priority test additions:
  1. **Physics**: Test `SWEPhysics.flux_jac()` and `.source()` against hand-computed
     values for known states (e.g. quiescent water: h=1, u=v=0).
  2. **Losses**: Test `compute_pde_loss` returns 0 for an exact solution. Test
     `compute_neg_h_loss` is 0 for positive h and positive for negative h.
  3. **Models**: Test forward pass shape correctness for each architecture.
  4. **Checkpointing**: Test save-then-load round-trip preserves parameter values.

---

### `test/test_train.py`: Weak Assertions

* **What should be changed**: The sole integration test (lines 119-156) only checks
  that output directories and files exist. It does not verify:
  - Checkpoint parameters are loadable and non-zero
  - Loss decreased over the 2 training epochs
  - Validation NSE is finite
  - Model predictions have correct output shape
* **Why**: The test passes even if training produces NaN losses or zero-valued
  parameters.
* **How**: Add substantive assertions:
  ```python
  import pickle
  with open(os.path.join(final_ckpt_dir, "model.pkl"), "rb") as f:
      loaded_params = pickle.load(f)
  self.assertIsNotNone(loaded_params)
  # Check at least one parameter array has non-zero values
  leaves = jax.tree.leaves(loaded_params)
  self.assertTrue(any(jnp.any(l != 0) for l in leaves))
  ```

---

## 7. Configuration Issues

### `configs/`: Inconsistent `accumulation_factor` vs `accumulation_size` Field Names

* **What should be changed**: Base configs use `accumulation_factor` (a multiplier
  of `num_batches`), while HPO-optimised configs use `accumulation_size` (an absolute
  count):
  ```yaml
  # configs/experiment_1.yaml
  accumulation_factor: 1

  # configs/train/experiment_1_mlp_final.yaml
  accumulation_size: 235
  ```
  The code in `src/training/optimizer.py` (lines 23-29) silently handles both,
  but with different semantics.
* **Why**: Ambiguous configuration leads to subtle behavioural differences that are
  hard to debug. A user copying a base config and modifying it may not realise the
  two fields have different units.
* **How**: Standardise on `accumulation_size` (absolute) everywhere. Add a deprecation
  warning in the optimizer setup if `accumulation_factor` is encountered. Document
  the semantics clearly in the config schema.

---

### `configs/`: Stale `CONFIG_PATH` References

* **What should be changed**: Several config files contain hardcoded `CONFIG_PATH`
  values that reference non-existent files:
  ```yaml
  # configs/experiment_3.yaml, line 73
  CONFIG_PATH: configs/train/test1_fourier_final.yaml  # Does not exist
  ```
* **Why**: `load_config()` overrides `CONFIG_PATH` on line 29 with the actual path,
  so these stale values are harmless but misleading.
* **How**: Remove hardcoded `CONFIG_PATH` from all YAML files since `load_config()`
  sets it automatically.

---

### `configs/`: Inconsistent `min_depth` Thresholds

* **What should be changed**: The dry-cell clamping threshold varies 10x across configs
  without documented rationale:
  ```
  experiment_1.yaml:                  min_depth: 0.05   (50 mm)
  experiment_1_fourier.yaml:          min_depth: 0.005  (5 mm)
  train/experiment_1_mlp_final.yaml:  min_depth: 0.0    (disabled)
  ```
* **Why**: Inconsistent thresholds make cross-experiment comparisons unreliable. A
  model trained with `min_depth: 0.05` will report different validation metrics than
  one with `min_depth: 0.0` even with identical predictions.
* **How**: Standardise `min_depth` per experiment (not per architecture). Document the
  chosen value and rationale in each experiment config.

---

### `src/config.py`: No Config Schema Validation

* **What should be changed**: `load_config()` (lines 20-35) loads YAML without
  validating that required keys exist or have correct types. Missing keys only surface
  as `KeyError` deep in the training loop.
* **Why**: Fail-fast behaviour at config load time would save debugging time. Currently,
  a missing `numerics.eps` key causes a crash inside the JIT-compiled loss function,
  where the traceback is unhelpful.
* **How**: Add validation after loading:
  ```python
  REQUIRED_KEYS = [
      ("training", "learning_rate"), ("training", "epochs"),
      ("model", "name"), ("domain", "lx"), ("device", "dtype"),
      ("numerics", "eps"),
  ]
  for *path, key in REQUIRED_KEYS:
      section = config
      for p in path:
          section = section.get(p, {})
      if key not in section:
          raise ValueError(f"Missing required config key: {'.'.join(path + [key])}")
  ```

---

## Summary

| Category | Count | Critical | High | Medium | Low |
|----------|:---:|:---:|:---:|:---:|:---:|
| Critical Bugs | 4 | 4 | — | — | — |
| Logic Errors | 5 | — | 3 | 2 | — |
| Sanity Issues | 6 | — | 1 | 3 | 2 |
| Redundancy | 4 | — | — | 3 | 1 |
| Performance | 6 | — | 1 | 3 | 2 |
| Testing Gaps | 2 | — | 2 | — | — |
| Configuration | 4 | — | 1 | 2 | 1 |
| **Total** | **31** | **4** | **8** | **13** | **6** |

### Recommended Fix Order

1. **Immediate** (Critical): PRNG key reuse in HPO loop, missing FrozenDict imports (exp 7 & 8), axis mismatch in exp 6
2. **High Priority**: PDE loss `train` flag inconsistency, config schema validation, test coverage for physics/losses
3. **Medium Priority**: Experiment code deduplication, `copy.deepcopy` replacement, config field standardisation
4. **Low Priority**: Dead imports cleanup, `squeeze()` hardening, batch cycling improvement
