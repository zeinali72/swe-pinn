# Legacy HPO Artifacts

This directory contains archived hyperparameter optimisation artifacts that are
no longer actively used but are preserved for reference. These represent early
exploration runs that were superseded by later, better-tuned studies.

## Why These Are Legacy

The configs in this directory use a deprecated config schema (pre-refactor naming
conventions, missing fields). They were used in initial exploration phases and
have been replaced by the configs in `../configs/exploration/` and
`../configs/exploitation/`.

## Archived Configs

| File | Architecture | Loss Weighting | Notes |
|------|-------------|----------------|-------|
| `configs/hpo_DGM_datafree_gradnorm.yaml` | DGMNetwork | GradNorm | Early DGM exploration with adaptive weighting |
| `configs/hpo_DGM_datafree_static.yaml` | DGMNetwork | Static | Early DGM exploration with static weights |
| `configs/hpo_fourier_datafree_static.yaml` | FourierPINN | Static | Early Fourier exploration with static weights |

## Archived Databases (Not in Git)

These databases are stored locally and are NOT tracked in git (see `../.gitignore`).
To organise local archives, create the subdirectories first:

```bash
mkdir -p optimisation/legacy/{databases,logs,results}
```

Then move local archive databases here.

| File | Study Name | Trials | Best NSE | Notes |
|------|-----------|--------|----------|-------|
| `databases/archive_exploitation_fourier_building_v2.db` | `hpo-exploitation-fourier-building-v2` | 59 (41c, 17p, 1f) | 0.6053 | Superseded by exploitation version |
| `databases/archive_exploitation_fourier_building_v1.db` | `hpo-exploitation-fourier-building` | 50 (37c, 13p) | 0.3809 | **DUPLICATE name** with `exploitation/` version — this is the older, lower-performing run |
| `databases/archive_sensitivity_mlp_building.db` | `hpo-sensitivity-mlp-building` | 100 (100c) | 0.9158 | **DUPLICATE name** with `exploration/` version — separate run with different results |

### Duplicate Study Name Resolution

1. **`hpo-exploitation-fourier-building`**: The authoritative version is in
   `../database/exploitation/` (100 trials, NSE=0.9679). The archive version
   (50 trials, NSE=0.3809) is an earlier, superseded run.

2. **`hpo-sensitivity-mlp-building`**: The authoritative version is in
   `../database/exploration/` (100 trials, NSE=0.8644). The archive version
   (100 trials, NSE=0.9158) is a separate earlier run.

## Querying Legacy Databases

```python
import optuna

# Load a legacy database
study = optuna.load_study(
    study_name="hpo-exploitation-fourier-building-v2",
    storage="sqlite:///optimisation/legacy/databases/archive_exploitation_fourier_building_v2.db"
)
print(f"Best trial: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")
```

## Active Databases (Reference)

The following databases are actively used and remain in `../database/`:

**Exploration (6 studies, QMCSampler sensitivity):**

| Study Name | Trials | Best NSE |
|-----------|--------|----------|
| `hpo-sensitivity-dgm-building` | 102 (100c, 2f) | 0.9846 |
| `hpo-sensitivity-dgm-nobuilding` | 100 (100c) | 0.9967 |
| `hpo-sensitivity-fourier-building` | 100 (100c) | 0.6812 |
| `hpo-sensitivity-fourier-nobuilding` | 101 (98c, 3f) | 0.8278 |
| `hpo-sensitivity-mlp-building` | 100 (100c) | 0.8644 |
| `hpo-sensitivity-mlp-nobuilding` | 100 (100c) | 0.9831 |

**Exploitation (6 studies, TPE):**

| Study Name | Trials | Best NSE |
|-----------|--------|----------|
| `hpo-exploitation-dgm-building` | 50 (38c, 12p) | 0.7818 |
| `hpo-exploitation-dgm-nobuilding` | 50 (28c, 22p) | 0.9936 |
| `hpo-exploitation-fourier-building` | 100 (71c, 29p) | 0.9679 |
| `hpo-exploitation-fourier-nobuilding` | 50 (35c, 15p) | 0.9406 |
| `hpo-exploitation-mlp-building` | 100 (39c, 61p) | 0.9727 |
| `hpo-exploitation-mlp-nobuilding` | 50 (35c, 15p) | 0.9730 |

## Local Housekeeping (Not in Git)

The following tasks apply to local workstation files only (databases, logs, and
HTML plots are not tracked in git):

- Move `logs/archive/*.log` to `legacy/logs/`
- Move root-level sensitivity logs (`logs/hpo-sensitivity-*.log`) into
  `logs/exploration/`
- Move `results/archive/` to `legacy/results/`
- Extract missing DGM sensitivity best configs from databases using
  `extract_best_params.py`
