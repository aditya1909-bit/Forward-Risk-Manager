#!/bin/bash
#SBATCH --job-name=frm-smoke
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/local/scratch/%u/forward-risk-manager/logs/slurm-%j.out

set -euo pipefail

NETID="${NETID:-$USER}"
SCRATCH_ROOT="/local/scratch/${NETID}/forward-risk-manager"
REPO_ROOT="${SCRATCH_ROOT}/repo"
RUNTIME_CFG="${SCRATCH_ROOT}/runs/runtime_turing_smoke.toml"

cd "${REPO_ROOT}"
source "${SCRATCH_ROOT}/venv/bin/activate"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"

python scripts/prepare_turing_run.py \
  --base-config configs/default.toml \
  --cluster-config configs/turing.toml \
  --runtime-config "${RUNTIME_CFG}" \
  --netid "${NETID}"

torchrun --standalone --nproc_per_node=2 scripts/train_ff_gnn.py \
  --config "${RUNTIME_CFG}" \
  --distributed \
  --backend nccl \
  --epochs 1 \
  --graph-limit 128 \
  --batch-size 8 \
  --no-torch-compile

torchrun --standalone --nproc_per_node=2 scripts/benchmark_training.py \
  --config "${RUNTIME_CFG}" \
  --distributed \
  --backend nccl
