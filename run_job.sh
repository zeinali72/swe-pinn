#!/bin/bash
set -e # Exit immediately if any command fails

echo "--- [1/6] Starting Sensitivity Analysis: MLP (Without Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_mlp_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-mlp-nobuilding" \
  --storage "sqlite:///optimisation/database/hpo-sensitivity-mlp-nobuilding.db" | tee optimisation/logs/hpo-sensitivity-mlp-nobuilding.log

echo "--- [2/6] Starting Sensitivity Analysis: MLP (With Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_mlp_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-mlp-building" \
  --storage "sqlite:///optimisation/database/hpo-sensitivity-mlp-building.db" | tee optimisation/logs/hpo-sensitivity-mlp-building.log

echo "--- [3/6] Starting Sensitivity Analysis: FourierPINN (Without Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_fourier_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-fourier-nobuilding" \
  --storage "sqlite:///optimisation/database/hpo-sensitivity-fourier-nobuilding.db" | tee optimisation/logs/hpo-sensitivity-fourier-nobuilding.log

echo "--- [4/6] Starting Sensitivity Analysis: FourierPINN (With Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_fourier_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-fourier-building" \
  --storage "sqlite:///optimisation/database/hpo-sensitivity-fourier-building.db" | tee optimisation/logs/hpo-sensitivity-fourier-building.log

echo "--- [5/6] Starting Sensitivity Analysis: DGM (Without Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-nobuilding" \
  --storage "sqlite:///optimisation/database/hpo-sensitivity-dgm-nobuilding.db" | tee optimisation/logs/hpo-sensitivity-dgm-nobuilding.log

echo "--- [6/6] Starting Sensitivity Analysis: DGM (With Building) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-building" \
  --storage "sqlite:///optimisation/database/hpo-sensitivity-dgm-building.db" | tee optimisation/logs/hpo-sensitivity-dgm-building.log

echo "--- All sensitivity analysis runs complete. ---"