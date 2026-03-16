# Remote HPO Storage (PostgreSQL / Neon DB)

By default, Optuna stores trials in a local SQLite database. For cloud
environments (Google Colab, Codespaces) or multi-machine parallelism, you can
use PostgreSQL instead.

## Setup

1. Create a Neon DB project at https://neon.tech (or any PostgreSQL instance).
2. Get the connection string: `postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require`
3. Install the driver:
   ```bash
   pip install psycopg2-binary
   # or install with the optional dependency group:
   pip install -e ".[postgres]"
   ```

## Usage

### Via environment variable (recommended for Colab)

```python
import os
os.environ["OPTUNA_STORAGE"] = "postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require"
```

Then run HPO without `--storage`:

```bash
python optimisation/run_optimization.py --config configs/my_config.yaml --n_trials 50
```

### Via CLI argument

```bash
python optimisation/run_optimization.py \
    --config configs/my_config.yaml \
    --storage "postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require" \
    --n_trials 50
```

### Priority order

1. `--storage` CLI argument (highest)
2. `OPTUNA_STORAGE` environment variable
3. Local SQLite at `optimisation/database/all_my_studies.db` (default)

## Latency considerations

Remote databases add ~50-200ms per Optuna operation. Without mitigation, every
validation epoch triggers a `trial.report()` + `trial.should_prune()` round-trip.

**Required for remote storage:** Set `training.hpo_report_interval` in your HPO
config to reduce DB calls. Without this, training will be bottlenecked by network
latency:

```yaml
training:
  hpo_report_interval: 10  # report every 10th validation, not every one
```

The heartbeat is configured automatically (120s interval, 600s grace period)
for Colab preemption recovery.

## Multi-machine parallelism

Multiple machines can share the same PostgreSQL study. Each machine runs:

```bash
python optimisation/run_optimization.py \
    --config configs/my_config.yaml \
    --storage "postgresql://..." \
    --study_name "shared-study" \
    --n_trials 25
```

Optuna handles trial coordination via the database.
