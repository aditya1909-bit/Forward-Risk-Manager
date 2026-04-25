#!/bin/bash
#SBATCH --job-name=frm-smoke-econ-lg
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/local/scratch/%u/forward-risk-manager/logs/slurm-%j.out

set -euo pipefail

NETID="${NETID:-$USER}"
SCRATCH_ROOT="/local/scratch/${NETID}/forward-risk-manager"
REPO_ROOT="${SCRATCH_ROOT}/repo"
RUNTIME_CFG="${SCRATCH_ROOT}/runs/runtime_turing_econ_smoke_large.toml"

cd "${REPO_ROOT}"
if [[ -f "${SCRATCH_ROOT}/venv/bin/activate" ]]; then
  source "${SCRATCH_ROOT}/venv/bin/activate"
else
  source "/local/scratch/${NETID}/venv/bin/activate"
fi
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"

python scripts/prepare_turing_run.py \
  --base-config configs/default.toml \
  --cluster-config configs/turing_econ_smoke_large.toml \
  --runtime-config "${RUNTIME_CFG}" \
  --netid "${NETID}"

python scripts/benchmark_training.py \
  --config "${RUNTIME_CFG}" \
  --modes ff_financial,backprop_supervised_return
