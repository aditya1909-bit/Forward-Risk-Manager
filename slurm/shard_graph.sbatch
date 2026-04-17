#!/bin/bash
#SBATCH --job-name=frm-shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/local/scratch/%u/forward-risk-manager/logs/slurm-%j.out

set -euo pipefail

NETID="${NETID:-$USER}"
SCRATCH_ROOT="/local/scratch/${NETID}/forward-risk-manager"
REPO_ROOT="${SCRATCH_ROOT}/repo"

cd "${REPO_ROOT}"
source "${SCRATCH_ROOT}/venv/bin/activate"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"

python scripts/shard_graph_artifact.py \
  --graphs "${SCRATCH_ROOT}/data/processed/graphs_master_ff_rich.pt" \
  --out "${SCRATCH_ROOT}/data/processed/graphs_master_ff_rich.pt.sharded" \
  --shard-size 256 \
  --torch-num-threads 8
