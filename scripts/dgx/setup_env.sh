#!/usr/bin/env bash
# One-time Python environment for MIPO-NDD on a DGX A100 (DGX OS / Ubuntu).
# Run from the project root on a machine WITH internet access (a login node is
# fine; no GPU is needed to install):   bash scripts/dgx/setup_env.sh
#
# Overrides:  MIPO_ENV=/path/to/venv  PYTHON=python3.11  TORCH_INDEX=https://download.pytorch.org/whl/cu121
#             HF_HOME=/path/to/hf_cache  (ESM weights are stored here; must be visible to GPU nodes)
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_DIR="${MIPO_ENV:-$PWD/.venv}"
PY="${PYTHON:-python3}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "Python >= 3.10 is required; $PY is $("$PY" -V 2>&1). DGX OS 5 ships 3.8. Install Miniforge, then:" >&2
  echo "  conda create -y -n mipo python=3.11 && conda activate mipo && PYTHON=python bash scripts/dgx/setup_env.sh" >&2
  exit 1
fi

# CUDA 12 wheels need an R525+ driver; older DGX drivers (R450/R470) need CUDA 11.8 wheels.
if [ -z "${TORCH_INDEX:-}" ]; then
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 || true)"
  if [ -n "$DRIVER" ] && [ "$DRIVER" -lt 525 ]; then
    TORCH_INDEX=https://download.pytorch.org/whl/cu118
  else
    TORCH_INDEX=https://download.pytorch.org/whl/cu121
  fi
  echo "NVIDIA driver major version: ${DRIVER:-unknown (no nvidia-smi here)}; using $TORCH_INDEX"
fi

if ! "$PY" -m venv "$ENV_DIR"; then
  echo "Could not create $ENV_DIR. On Ubuntu, venv needs the python3-venv package (admin: apt install python3-venv)." >&2
  echo "Without admin rights, install Miniforge in your home or /raid folder, then:" >&2
  echo "  conda create -y -n mipo python=3.11 && conda activate mipo && PYTHON=python bash scripts/dgx/setup_env.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install "torch>=2.4,<3" --index-url "$TORCH_INDEX"
python -m pip install -e '.[test]'

# Pre-download both ESM2 encoders so GPU nodes can run with HF_HUB_OFFLINE=1.
python - <<'EOF'
from transformers import AutoTokenizer, EsmForMaskedLM
for name in ["facebook/esm2_t33_650M_UR50D", "facebook/esm2_t30_150M_UR50D"]:
    AutoTokenizer.from_pretrained(name)
    EsmForMaskedLM.from_pretrained(name)
    print("cached", name)
EOF

python -m pytest -q
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU visible here:', torch.cuda.is_available())"
echo "Environment ready: source $ENV_DIR/bin/activate ; export HF_HOME=$HF_HOME"
