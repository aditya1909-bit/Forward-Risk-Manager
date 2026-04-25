#!/bin/bash
#SBATCH --job-name=frm-bench-econ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/local/scratch/%u/forward-risk-manager/logs/slurm-%j.out

set -euo pipefail

NETID="${NETID:-$USER}"
SCRATCH_ROOT="/local/scratch/${NETID}/forward-risk-manager"
REPO_ROOT="${SCRATCH_ROOT}/repo"
RUNTIME_CFG="${SCRATCH_ROOT}/runs/runtime_turing_econ.toml"

cd "${REPO_ROOT}"
source "/local/scratch/$USER/venv/bin/activate"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"

python scripts/prepare_turing_run.py \
  --base-config configs/default.toml \
  --cluster-config configs/turing_econ.toml \
  --runtime-config "${RUNTIME_CFG}" \
  --netid "${NETID}"

python scripts/benchmark_training.py \
  --config "${RUNTIME_CFG}" \
  --modes ff_financial,backprop_supervised_return
