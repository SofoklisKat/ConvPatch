#!/usr/bin/env bash
# Run CIFAR-100 sweep for seeds 0, 1, 2 (paper protocol).
# Usage: DEVICE=cuda:0 ./scripts/run_cifar100_all_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"

for SEED in 0 1 2; do
  echo "========== CIFAR-100 seed ${SEED} =========="
  DATASET=cifar100 DEVICE="$DEVICE" SEED="$SEED" EPOCHS="$EPOCHS" \
    NUM_WORKERS="$NUM_WORKERS" ./scripts/run_experiments.sh
done

echo "========== All seeds done =========="
.venv/bin/python scripts/summarize_runs.py
