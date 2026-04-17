#!/bin/bash
#SBATCH --job-name=frm-sweep
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/local/scratch/%u/forward-risk-manager/logs/slurm-%j.out

set -euo pipefail

NETID="${NETID:-$USER}"
SCRATCH_ROOT="/local/scratch/${NETID}/forward-risk-manager"
REPO_ROOT="${SCRATCH_ROOT}/repo"
RUNTIME_CFG="${SCRATCH_ROOT}/runs/runtime_turing_sweep.toml"

cd "${REPO_ROOT}"
source "${SCRATCH_ROOT}/venv/bin/activate"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"

python scripts/prepare_turing_run.py \
  --base-config configs/default.toml \
  --cluster-config configs/turing.toml \
  --runtime-config "${RUNTIME_CFG}" \
  --netid "${NETID}"

torchrun --standalone --nproc_per_node=2 scripts/ff_sweep.py \
  --config "${RUNTIME_CFG}" \
  --distributed \
  --backend nccl
