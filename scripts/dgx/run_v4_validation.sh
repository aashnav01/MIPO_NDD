#!/usr/bin/env bash
#SBATCH --job-name=mipo-ndd-v4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=mipo-ndd-v4-%j.out
# Runs scripts/dgx_v4_validation.py with the project environment.
#
# Slurm:  cd <project root> && sbatch scripts/dgx/run_v4_validation.sh [launcher options]
#         (#SBATCH lines above are defaults; adjust --gres/--time/--partition for your cluster)
# Direct: cd <project root> && bash scripts/dgx/run_v4_validation.sh --gpus 0,1,2,3 [options]
#         (inside tmux/screen so an SSH disconnect does not stop it)
# Resubmitting the same command resumes: finished runs are skipped, interrupted ones
# restart from their last completed epoch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
[ -f scripts/dgx_v4_validation.py ] || { echo "Run from the project root" >&2; exit 1; }
# shellcheck disable=SC1091
source "${MIPO_ENV:-$PWD/.venv}/bin/activate"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights were cached by setup_env.sh
# DataLoader workers pass batches through file descriptors and shared memory.
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
python scripts/dgx_v4_validation.py --root "${MIPO_ROOT:-$PWD/outputs}" "$@"
