#!/bin/bash
set -e # Exit immediately if any command fails

echo "--- [1/6] Starting: MLP (Without Building) ---"
python3 -u -m optimisation.run_optimization --config optimisation/configs/hpo_mlp_datafree_static_NOBUILDING.yaml --n_trials 100 --study_name "hpo-mlp-datafree-nobuilding" | tee optimisation/logs/hpo-mlp-datafree-nobuilding.log

echo "--- [2/6] Starting: MLP (With Building) ---"
python3 -u -m optimisation.run_optimization --config optimisation/configs/hpo_mlp_datafree_static_BUILDING.yaml --n_trials 100 --study_name "hpo-mlp-datafree-building" | tee optimisation/logs/hpo-mlp-datafree-building.log

echo "--- [3/6] Starting: FourierPINN (Without Building) ---"
python3 -u -m optimisation.run_optimization --config optimisation/configs/hpo_fourier_datafree_static_NOBUILDING.yaml --n_trials 100 --study_name "hpo-fourier-datafree-nobuilding" | tee optimisation/logs/hpo-fourier-datafree-nobuilding.log

echo "--- [4/6] Starting: FourierPINN (With Building) ---"
python3 -u -m optimisation.run_optimization --config optimisation/configs/hpo_fourier_datafree_static_BUILDING.yaml --n_trials 100 --study_name "hpo-fourier-datafree-building" | tee optimisation/logs/hpo-fourier-datafree-building.log

echo "--- [5/6] Starting: DGM (Without Building) ---"
python3 -u -m optimisation.run_optimization --config optimisation/configs/hpo_dgm_datafree_static_NOBUILDING.yaml --n_trials 100 --study_name "hpo-dgm-datafree-nobuilding" | tee optimisation/logs/hpo-dgm-datafree-nobuilding.log

echo "--- [6/6] Starting: DGM (With Building) ---"
python3 -u -m optimisation.run_optimization --config optimisation/configs/hpo_dgm_datafree_static_BUILDING.yaml --n_trials 100 --study_name "hpo-dgm-datafree-building" | tee optimisation/logs/hpo-dgm-datafree-building.log

echo "--- All HPO runs complete. ---"