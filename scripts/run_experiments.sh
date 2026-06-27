#!/usr/bin/env bash
# Run ConvPatch ViT-Tiny sweep (all 5 patch-embed variants).
# Skips runs that already reached EPOCHS. Safe to re-run / resume.
#
# Usage:
#   ./scripts/run_experiments.sh                      # CIFAR-10 (default)
#   DATASET=cifar100 ./scripts/run_experiments.sh     # CIFAR-100
#   DEVICE=cuda:1 SEED=0 EPOCHS=200 ./scripts/run_experiments.sh
#   DATASET=cifar100 SEED=1 ./scripts/run_experiments.sh
#   ./scripts/run_experiments.sh --only conv_stem hierarchical
#
# Environment:
#   DATASET      cifar10 | cifar100 (default: cifar10)
#   DEVICE       GPU id (default: cuda:0)
#   SEED         random seed (default: 0)
#   EPOCHS       training epochs per run (default: 200)
#   NUM_WORKERS  dataloader workers (default: 4)
#   PYTHON       python binary (default: .venv/bin/python)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET="${DATASET:-cifar10}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" runs

case "$DATASET" in
  cifar10)
    CONFIGS=(
      configs/cifar10_vit_tiny_linear.yaml
      configs/cifar10_vit_tiny_conv_stem.yaml
      configs/cifar10_vit_tiny_overlapping.yaml
      configs/cifar10_vit_tiny_hierarchical.yaml
      configs/cifar10_vit_tiny_dwsep_conv_stem.yaml
    )
    ;;
  cifar100)
    CONFIGS=(
      configs/cifar100_vit_tiny_linear.yaml
      configs/cifar100_vit_tiny_conv_stem.yaml
      configs/cifar100_vit_tiny_overlapping.yaml
      configs/cifar100_vit_tiny_hierarchical.yaml
      configs/cifar100_vit_tiny_dwsep_conv_stem.yaml
    )
    ;;
  *)
    echo "Unknown DATASET=${DATASET} (use cifar10 or cifar100)" >&2
    exit 1
    ;;
esac

ONLY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ONLY+=("$1")
        shift
      done
      ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run_name_for() {
  "$PYTHON" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$1"
}

patch_embed_for() {
  "$PYTHON" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model']['patch_embed'])" "$1"
}

is_done() {
  local run_dir="runs/${1}_seed${SEED}"
  local log="${run_dir}/log.csv"
  [[ -f "$log" ]] || return 1
  local n
  n=$(tail -n +2 "$log" | wc -l)
  [[ "$n" -ge "$EPOCHS" ]]
}

wait_for_gpu() {
  while pgrep -f "^${PYTHON} -u -m convpatch[.]train" >/dev/null 2>&1; do
    echo "[$(date '+%H:%M:%S')] Waiting for another training job..."
    sleep 30
  done
}

should_run() {
  local cfg="$1"
  if [[ ${#ONLY[@]} -eq 0 ]]; then
    return 0
  fi
  local pe
  pe=$(patch_embed_for "$cfg")
  local x
  for x in "${ONLY[@]}"; do
    [[ "$x" == "$pe" ]] && return 0
  done
  return 1
}

echo "=== ConvPatch experiments | dataset=${DATASET} ==="
echo "device=${DEVICE} seed=${SEED} epochs=${EPOCHS} workers=${NUM_WORKERS}"
"$PYTHON" -c "import torch; print('cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

for cfg in "${CONFIGS[@]}"; do
  should_run "$cfg" || continue

  name=$(run_name_for "$cfg")
  logfile="${LOG_DIR}/${name}_seed${SEED}.log"

  if is_done "$name"; then
    echo "[skip] ${name} already completed (${EPOCHS} epochs, seed ${SEED})"
    continue
  fi

  wait_for_gpu
  echo "[$(date '+%H:%M:%S')] Starting ${name} (${cfg})"
  "$PYTHON" -u -m convpatch.train \
    --config "$cfg" \
    --set \
      "seed=${SEED}" \
      "train.epochs=${EPOCHS}" \
      "train.device=${DEVICE}" \
      "data.num_workers=${NUM_WORKERS}" \
    2>&1 | tee "$logfile"
  echo "[$(date '+%H:%M:%S')] Finished ${name}"
done

echo "=== Done ==="
"$PYTHON" scripts/summarize_runs.py 2>/dev/null || true
