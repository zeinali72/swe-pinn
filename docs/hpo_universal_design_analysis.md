# Analysis: Universal HPO Pipeline for All Experiments

> **Status**: Design analysis
> **Date**: 2026-03-16
> **Question**: Can we restructure `optimisation/` so that each experiment's `train.py` does the heavy lifting, while HPO just feeds modified configs and collects metrics?

## Executive Summary

**Yes. The key insight is that every experiment's `main()` already has the same shape:**

```
main(config_path):
    1. Setup    →  config, model, terrain, closures     (~100 lines, experiment-specific)
    2. Train    →  run_training_loop(...)                (~1 line, shared)
    3. Save     →  post_training_save(...)               (~20 lines, HPO doesn't need)
    4. Return   →  best_nse                              (already returns a metric!)
```

The proposal is simple: **split `main()` into `setup(cfg)` and the rest**, then HPO calls `setup()` with a trial-modified config and runs its own lighter loop. No registry, no separate factory functions, no parallel code paths — just reuse what each experiment already has.

---

## 1. Current State: What Each Experiment's `main()` Does

Every experiment follows the same pattern. Here is experiment 3 annotated:

```python
def main(config_path):
    # ─── SETUP PHASE (reusable by HPO) ───────────────────────
    setup = setup_experiment(config_path)                        # shared
    cfg, cfg_dict, model, params, train_key = ...               # unpack

    experiment_paths = resolve_experiment_paths(cfg, ...)        # shared
    terrain = load_terrain_assets(cfg, base_data_path, ...)      # shared
    bc_fn_static = terrain["bc_fn"]                              # experiment needs this

    data_free, _ = resolve_data_mode(cfg)                        # shared
    data_points_full, ... = load_training_data(...)              # shared
    validation = load_validation_from_file(...)                  # shared

    n_pde = get_sampling_count_from_config(cfg, "n_points_pde")  # shared
    num_batches = calculate_num_batches(...)                     # shared
    optimiser = create_optimizer(cfg, num_batches)               # shared

    generate_epoch_data_jit = jax.jit(generate_epoch_data)       # EXPERIMENT-SPECIFIC closure
    compute_losses_fn = make_compute_losses(bc_fn_static)        # EXPERIMENT-SPECIFIC closure
    scan_body = make_scan_body(..., compute_losses_fn=...)       # shared (wraps experiment fn)
    validation_fn = ...                                          # EXPERIMENT-SPECIFIC (exp 8)

    # ─── TRAIN PHASE (HPO replaces this) ─────────────────────
    loop_result = run_training_loop(...)                         # shared, full production loop

    # ─── SAVE PHASE (HPO skips this) ─────────────────────────
    post_training_save(...)                                      # plotting, Aim, checkpoints
    return loop_result["best_nse_stats"]["nse"]
```

**The setup phase produces exactly what HPO needs**: `model`, `params`, `generate_epoch_data_jit`, `scan_body`, `validation_fn`, `num_batches`. The train phase is a single shared function call. The save phase is irrelevant for HPO.

---

## 2. Proposed Design: `setup_trial()` Extracted from `main()`

### 2.1 The Refactor

Split each experiment's `main()` into two functions:

```python
# experiments/experiment_3/train.py

def setup_trial(cfg_dict: dict) -> dict:
    """Setup everything needed for one training run.

    This is the SAME code that main() already executes, just extracted
    into a reusable function. Both main() and HPO call this.
    """
    cfg = FrozenDict(cfg_dict)
    model, params, train_key, val_key = init_model_from_config(cfg)

    experiment_paths = resolve_experiment_paths(cfg, ...)
    terrain = load_terrain_assets(cfg, ...)
    bc_fn_static = terrain["bc_fn"]

    data_free, _ = resolve_data_mode(cfg)
    data_points_full, ... = load_training_data(...)
    validation = load_validation_from_file(...)

    n_pde = get_sampling_count_from_config(cfg, "n_points_pde")
    num_batches = calculate_num_batches(...)
    optimiser = create_optimizer(cfg, num_batches)
    opt_state = optimiser.init(params)

    generate_epoch_data_jit = jax.jit(generate_epoch_data)  # experiment-specific closure
    compute_losses_fn = make_compute_losses(bc_fn_static)
    scan_body = make_scan_body(..., compute_losses_fn=...)

    return {
        "cfg": cfg,
        "model": model, "params": params,
        "train_key": train_key,
        "optimiser": optimiser, "opt_state": opt_state,
        "generate_epoch_data_jit": generate_epoch_data_jit,
        "scan_body": scan_body,
        "num_batches": num_batches,
        "validation_fn": make_validation_fn(...),   # or inline like exp 8
        "data_free": data_free,
        # production-only extras (HPO ignores these):
        "validation_data_loaded": ...,
        "val_points_all": ..., "h_true_val_all": ...,
        "compute_all_losses_fn": ...,
    }


def main(config_path: str):
    """Production training — calls setup_trial() then the full loop."""
    cfg_dict = load_config(config_path)
    ctx = setup_trial(cfg_dict)

    trial_name, results_dir, model_dir = create_output_dirs(...)

    loop_result = run_training_loop(
        cfg=ctx["cfg"], model=ctx["model"], params=ctx["params"],
        generate_epoch_data_jit=ctx["generate_epoch_data_jit"],
        scan_body=ctx["scan_body"], num_batches=ctx["num_batches"],
        validation_fn=ctx["validation_fn"],
        ...  # Aim tracking, plotting, checkpoints
    )

    post_training_save(...)
    return loop_result["best_nse_stats"]["nse"]
```

**`main()` becomes a thin wrapper around `setup_trial()` + `run_training_loop()` + `post_training_save()`.** No logic is duplicated.

### 2.2 The HPO Wrapper (One File, All Experiments)

```python
# optimisation/optimization_train_loop.py (rewritten, ~80 lines)

def run_training_trial(trial, trial_cfg_dict, scenario):
    """Generic HPO trial — works for ANY experiment."""

    # 1. Import the experiment's setup_trial dynamically
    mod = importlib.import_module(f"experiments.{scenario}.train")
    ctx = mod.setup_trial(trial_cfg_dict)

    model     = ctx["model"]
    params    = ctx["params"]
    opt_state = ctx["opt_state"]
    train_key = ctx["train_key"]
    cfg       = ctx["cfg"]

    # 2. Read HPO settings from config
    objective_key = trial_cfg_dict.get("hpo_settings", {}).get("objective_key", "nse_h")
    hpo_patience  = trial_cfg_dict.get("training", {}).get("hpo_patience", 300)
    epochs        = trial_cfg_dict["training"]["epochs"]

    # 3. Generic training loop with Optuna integration
    best_metric = -jnp.inf
    last_improvement = -hpo_patience

    for epoch in range(epochs):
        train_key, epoch_key = random.split(train_key)
        scan_inputs = ctx["generate_epoch_data_jit"](epoch_key)
        (params, opt_state), (batch_terms, batch_totals) = lax.scan(
            ctx["scan_body"], (params, opt_state), scan_inputs,
        )

        metrics = ctx["validation_fn"](model, params)
        current = metrics.get(objective_key, -jnp.inf)

        if current > best_metric:
            best_metric = current
            last_improvement = epoch

        if epoch - last_improvement > hpo_patience:
            break

        trial.report(best_metric, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(best_metric)
```

**That is the entire HPO training wrapper.** No sampling count logic, no validation loading, no terrain handling, no model init — all of that lives inside `setup_trial()` where it belongs.

### 2.3 The YAML Config (Unchanged + Minor Additions)

```yaml
# optimisation/configs/experiment_7_fourier.yaml

scenario: experiment_7                    # → imports experiments.experiment_7.train

hpo_settings:
  data_free: true
  objective_key: "nse_h"                  # which key from validation_fn output
  direction: "maximize"
  opt_epochs: 2000
  production_epochs: 20000

hpo_hyperparameters:
  learning_rate: {min: 1e-6, max: 0.01, log: true}
  batch_size: [256, 512, 1024]
  model_width: [128, 256, 512, 1024]
  model_depth: {min: 3, max: 6}
  sampling:
    n_points_pde: {min: 10000, max: 120000, log: true}
    n_points_ic: {min: 1000, max: 20000, log: true}
    n_points_bc_inflow: {min: 500, max: 5000, log: true}
    n_points_bc_domain: {min: 2000, max: 20000, log: true}
  loss_weights:
    pde_weight: {min: 1.0, max: 1e6, log: true}
    ic_weight: {min: 0.01, max: 1000, log: true}
    bc_weight: {min: 0.01, max: 1000, log: true}

# Everything below is the normal training config — passed through to setup_trial()
model:
  name: FourierPINN
  output_dim: 3
domain:
  t_final: 300.0
physics:
  g: 9.81
  n_manning: 0.03
# ... rest of experiment config
```

### 2.4 The Flow

```
run_optimization.py
  │  creates Optuna study
  │
  ▼
objective_function.py
  │  for each trial:
  │    1. read hpo_hyperparameters from YAML
  │    2. suggest values via Optuna
  │    3. overlay onto base config dict
  │
  ▼
optimization_train_loop.py
  │  1. import experiments.{scenario}.train
  │  2. call setup_trial(modified_cfg_dict)      ← experiment does ALL its own setup
  │  3. run generic epoch loop with pruning       ← ~30 lines, same for all experiments
  │  4. return best metric to Optuna
  │
  ▼
experiments/experiment_N/train.py::setup_trial()
     does exactly what main() already does:
     config → terrain → model → closures → return context dict
```

---

## 3. What Changes Per Experiment

### 3.1 Effort Per Experiment

The refactor for each experiment is mechanical — cut lines from `main()`, paste into `setup_trial()`, have `main()` call `setup_trial()`:

| Experiment | Setup Complexity | Special Handling | Lines to Move |
|-----------|-----------------|-----------------|---------------|
| Exp 1 | Simple (flat, analytical) | Analytical validation | ~60 |
| Exp 2 | Simple + building | Building masking | ~70 |
| Exp 3 | Terrain + DEM | `load_terrain_assets()` | ~80 |
| Exp 4 | Terrain + split inflow | Inflow width calc | ~85 |
| Exp 5 | Terrain + single inflow | Similar to 4 | ~80 |
| Exp 6 | Terrain + split inflow | Similar to 4 | ~85 |
| Exp 7 | Irregular mesh | `IrregularDomainSampler`, late model init | ~100 |
| Exp 8 | Irregular + buildings | Combined NSE validation, `upstream_discharge_width` | ~120 |

**None of this is new code.** It is reorganising existing code in `main()`.

### 3.2 Experiment 8 Example (Hardest Case)

Experiment 8 is the most complex. Its `setup_trial()` would be:

```python
def setup_trial(cfg_dict: dict) -> dict:
    experiment_name = get_experiment_name(cfg_dict, "experiment_8")
    experiment_paths = resolve_experiment_paths(cfg_dict, experiment_name)
    base_data_path = experiment_paths["base_data_path"]
    scenario_name = experiment_paths["scenario_name"]

    # Irregular domain — must happen BEFORE model init
    artifacts_path = resolve_configured_asset_path(cfg_dict, base_data_path, scenario_name, "domain_artifacts")
    domain_sampler = IrregularDomainSampler(artifacts_path)
    apply_irregular_domain_bounds(cfg_dict, domain_sampler)
    apply_output_scales(cfg_dict, (1.0, 1.0, 1.0))

    # Derive discharge width from mesh geometry
    bc_cfg = cfg_dict.setdefault("boundary_conditions", {})
    if 'upstream' in domain_sampler.boundaries:
        bc_cfg["upstream_discharge_width"] = domain_sampler.boundary_length('upstream')

    # NOW init model (domain bounds are populated)
    cfg = FrozenDict(cfg_dict)
    model, params, train_key, val_key = init_model_from_config(cfg)

    # Loss weights, terrain, data
    static_weights_dict, current_weights_dict = extract_loss_weights(cfg)
    dem_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "dem")
    load_bathymetry(dem_path)
    bc_csv_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "boundary_condition")
    bc_fn_static = load_boundary_condition(bc_csv_path)

    data_free, has_data_loss = resolve_data_mode(cfg)
    data_points_full, has_data_loss, data_free = load_training_data(...)
    validation = load_validation_from_file(...)

    # Sampling, optimizer, closures — exactly as in current main()
    n_pde = get_sampling_count_from_config(cfg, "n_points_pde")
    ...
    generate_epoch_data_jit = jax.jit(generate_epoch_data)
    compute_losses_fn = make_compute_losses(bc_fn_static)
    scan_body = make_scan_body(...)

    # Experiment 8's custom combined-NSE validation
    def validation_fn(model, params):
        ...  # same code as current main(), returns {'selection_metric': ..., 'nse_h': ...}

    return {
        "cfg": cfg, "model": model, "params": params,
        "train_key": train_key,
        "optimiser": optimiser, "opt_state": opt_state,
        "generate_epoch_data_jit": generate_epoch_data_jit,
        "scan_body": scan_body,
        "num_batches": num_batches,
        "validation_fn": validation_fn,
        ...
    }
```

**This is literally the first ~160 lines of the current `main()`, moved into a function and returning a dict instead of flowing into `run_training_loop()`.** Zero new logic.

---

## 4. What Stays Unchanged

| Component | Changes? | Why |
|-----------|----------|-----|
| `run_optimization.py` | No | Study creation / Optuna orchestration is generic |
| `objective_function.py` | Minor | Add `objective_key` reading from config (1 line) |
| `experiment_registry.py` | **Deleted** | No longer needed — `importlib.import_module(scenario)` replaces static registry |
| `run_sensitivity_analysis.py` | No | Same Optuna wrapper, different sampler |
| `extract_best_params.py` | No | Post-hoc analysis, doesn't touch training |
| `analyze_importance.py` | No | Post-hoc analysis |
| `utils.py` | No | YAML sanitization, storage setup |
| HPO YAML configs | Add `objective_key` | Trivial addition |
| `src/training/` modules | No | Shared infrastructure stays shared |

---

## 5. Critical Assessment

### 5.1 What Makes This Approach Better

**It follows the grain of the codebase instead of fighting it.**

The previous analysis (my first attempt) proposed building a universal HPO wrapper that could handle all experiments' quirks internally. That fights the fact that experiments are genuinely different. This approach accepts that and says: **let each experiment handle its own setup, and keep the HPO loop dumb.**

The result:
- `optimization_train_loop.py` drops from ~200 lines to ~80 lines
- `experiment_registry.py` is deleted entirely
- Each experiment gains one function (`setup_trial`) that is **not new code** — it is existing code moved from `main()`
- `main()` gets simpler too (becomes `setup_trial()` + `run_training_loop()` + `post_training_save()`)

### 5.2 The One Real Risk

**`setup_trial()` must be callable with a modified config dict (not a file path).**

Currently `main()` receives a file path and calls `load_config(path)`. For HPO, the config is built in memory by `objective_function.py`. This means `setup_trial()` must accept a dict, not a path. This is straightforward — `setup_experiment()` already accepts dicts — but it requires each experiment to be tested with both entry points:
1. `main("path/to/config.yaml")` — production
2. `setup_trial({"training": {...}, "model": {...}, ...})` — HPO

Both paths converge at the same code, so the risk is low.

### 5.3 Validation Frequency in HPO

One thing the current `optimization_train_loop.py` does that `run_training_loop()` does not: it validates at a configurable frequency and reports to Optuna. The production loop validates every epoch and logs to Aim.

For HPO, we need the lighter validation-every-N-epochs pattern with Optuna reporting. This is the one piece of logic that genuinely differs and must stay in the HPO wrapper. It is ~30 lines of the epoch loop.

### 5.4 What About `run_training_loop()` Itself?

An even more aggressive approach: could HPO just call `run_training_loop()` directly with HPO-specific hooks?

**No, and here is why.** `run_training_loop()` does:
- Aim experiment tracking (creates tracker, logs metrics)
- Console banner printing
- Checkpoint saving (best_nse, best_loss, final)
- Early stopping based on production patience
- `compute_all_losses_fn` at end
- Returns a rich `loop_result` dict

HPO needs none of this. It needs:
- Optuna `trial.report()` + `trial.should_prune()`
- Intra-trial patience (different from production patience)
- Return a single float

Trying to make `run_training_loop()` serve both use cases would add `if is_hpo:` branches everywhere. **Two clean loops sharing the same setup is better than one tangled loop.**

### 5.5 Dynamic Import vs. Registry

The proposal uses `importlib.import_module(f"experiments.{scenario}.train")` instead of a static registry. This is simpler (no registry file to maintain) but has a trade-off:
- **Pro**: Adding a new experiment requires zero changes to `optimisation/` — just write the experiment and a YAML config
- **Con**: Typos in `scenario` produce an `ImportError` at runtime instead of a clear `KeyError` with available options

Either approach works. The dynamic import is cleaner for the "any experiment, zero code changes" goal.

---

## 6. Implementation Roadmap

```
Step 1: Extract setup_trial() from experiments 1 and 2
├── Move setup code from main() into setup_trial(cfg_dict) -> dict
├── Have main() call setup_trial() (no behaviour change)
├── Verify: python experiments/experiment_1/train.py --config ... (unchanged output)
└── Verify: existing HPO tests still pass

Step 2: Rewrite optimization_train_loop.py to use setup_trial()
├── Replace current 200-line wrapper with ~80-line version
├── Dynamic import via scenario name
├── Read objective_key from hpo_settings
├── Delete experiment_registry.py
└── Verify: existing exp 1-2 HPO configs still work

Step 3: Extract setup_trial() from experiments 3–8
├── Same mechanical refactor as step 1
├── Write one HPO config per experiment (start with exploration)
└── Integration test: 1 trial, 10 epochs, CPU

Step 4: Run real HPO (GPU)
├── Pick 2 experiments (e.g., exp 3 + exp 7) for pilot
├── 50 trials each, exploitation phase
└── Compare best NSE vs. current hand-tuned configs
```

**Steps 1–3 are purely structural refactoring with no logic changes.** Step 4 is where the value is realised.

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Can we reuse experiment `train.py` for HPO? | **Yes — extract `setup_trial()` from `main()`** |
| How much new code? | **~80 lines** (rewritten HPO loop). Everything else is moved, not written. |
| How much code deleted? | **~120 lines** (current `optimization_train_loop.py`) + **40 lines** (`experiment_registry.py`) |
| Does this work for irregular domains (exp 7–8)? | **Yes — each experiment handles its own domain setup** |
| Does this work for custom objectives (exp 8 combined NSE)? | **Yes — `objective_key` in YAML selects from `validation_fn` output** |
| What breaks? | **Nothing — `main()` calls `setup_trial()` internally, same behaviour** |
| Can we add experiment 9 in the future? | **Yes — write `train.py` with `setup_trial()`, write YAML config, done** |
