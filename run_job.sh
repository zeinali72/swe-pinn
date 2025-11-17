#!/bin/bash
set -e # Exit immediately if any command fails

echo "--- Preparing log and database directories in /workspace/mnt ---"
# Create directories in the persistent mount
mkdir -p /workspace/mnt/logs
mkdir -p /workspace/mnt/database

echo "--- [3/6] Starting Sensitivity Analysis: FourierPINN (Without Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_fourier_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-fourier-nobuilding" \
  --storage "sqlite:////workspace/mnt/database/hpo-sensitivity-fourier-nobuilding.db" | tee /workspace/mnt/logs/hpo-sensitivity-fourier-nobuilding.log

echo "--- [4/6] Starting Sensitivity Analysis: FourierPINN (With Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_fourier_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-fourier-building" \
  --storage "sqlite:////workspace/mnt/database/hpo-sensitivity-fourier-building.db" | tee /workspace/mnt/logs/hpo-sensitivity-fourier-building.log

echo "--- [5/6] Starting Sensitivity Analysis: DGM (Without Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-nobuilding" \
  --storage "sqlite:////workspace/mnt/database/hpo-sensitivity-dgm-nobuilding.db" | tee /workspace/mnt/logs/hpo-sensitivity-dgm-nobuilding.log

echo "--- [6/6] Starting Sensitivity Analysis: DGM (With Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-building" \
  --storage "sqlite:////workspace/mnt/database/hpo-sensitivity-dgm-building.db" | tee /workspace/mnt/logs/hpo-sensitivity-dgm-building.log

echo "--- All sensitivity analysis runs complete. ---"