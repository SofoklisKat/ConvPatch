#!/usr/bin/env bash
# Create venv and install ConvPatch on a new GPU machine.
#
# Usage:
#   ./scripts/setup_remote.sh
#   CUDA_INDEX=https://download.pytorch.org/whl/cu124 ./scripts/setup_remote.sh
#
# CUDA wheel index (pick one that matches your driver):
#   cu118  - older GPUs / CUDA 11.8
#   cu121  - CUDA 12.1 (works with driver 535+)
#   cu124  - CUDA 12.4
#   cpu    - no GPU
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
PYTHON="${PYTHON_BIN:-python3}"

echo "=== ConvPatch remote setup ==="
echo "Python: $($PYTHON --version)"
echo "CUDA index: ${CUDA_INDEX}"

if [[ ! -d .venv ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYTHON" .venv
  else
    "$PYTHON" -m venv .venv
  fi
fi

PIP=".venv/bin/pip"
if command -v uv >/dev/null 2>&1; then
  PIP="uv pip"
fi

if [[ "$CUDA_INDEX" == "cpu" ]]; then
  $PIP install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
else
  $PIP install -r requirements.txt --extra-index-url "$CUDA_INDEX"
fi
$PIP install -e .

echo "=== Verify ==="
.venv/bin/python -c "
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  cuda:{i} {p.name} ({p.total_memory/1e9:.1f} GB)')
"

echo "Done. Run experiments with: ./scripts/run_experiments.sh"
