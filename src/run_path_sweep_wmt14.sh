#!/bin/bash
# Sequential path-correlation sweep on WMT14.
# All runs use pre-trained ELF-B-de-en backbone (no Stage 1).
# Order: rho3_rbf (primary) → rho2_rbf → rho1_rbf → rho3_exp
set -e

cd /home.na1/ad.wsu.edu/khiem.tran/pvt/CorrFlow/src
PYTHON=/home.na1/ad.wsu.edu/khiem.tran/pvt/miniconda3/envs/corrflow/bin/python

CONFIGS=(
  "configs/training_configs/train_corrflow_stage2_wmt14_path_rho3_rbf.yml"
  "configs/training_configs/train_corrflow_stage2_wmt14_path_rho2_rbf.yml"
  "configs/training_configs/train_corrflow_stage2_wmt14_path_rho1_rbf.yml"
  "configs/training_configs/train_corrflow_stage2_wmt14_path_rho3_exp.yml"
)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yml)
  echo "=========================================="
  echo "Starting: $name"
  echo "Time: $(date)"
  echo "=========================================="
  $PYTHON -u train_stage2.py --config "$cfg" 2>&1 | tee "logs/${name}.log"
  echo "Finished: $name at $(date)"
done

echo "All 4 path-correlation runs complete."
