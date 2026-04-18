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
source "/local/scratch/$USER/venv/bin/activate"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"

python scripts/prepare_turing_run.py \
  --base-config configs/default.toml \
  --cluster-config configs/turing.toml \
  --runtime-config "${RUNTIME_CFG}" \
  --netid "${NETID}"

python - <<'CFG'
from pathlib import Path
p = Path(r"/local/scratch/asdutta/forward-risk-manager/runs/runtime_turing_smoke.toml")
t = p.read_text()
marker = "[train]\n"
insert = '[train]\namp = false\namp_dtype = "float32"\n'
if marker in t:
    head, rest = t.split(marker, 1)
    t = head + insert + rest
p.write_text(t)
CFG

torchrun --standalone --nproc_per_node=2 scripts/train_ff_gnn.py \
  --config "${RUNTIME_CFG}" \
  --distributed \
  --backend nccl \
  --epochs 1 \
  --graph-limit 16 \
  --batch-size 2 \
  --no-torch-compile --no-auto-tune-batch --loader-workers 0

torchrun --standalone --nproc_per_node=2 scripts/benchmark_training.py \
  --config "${RUNTIME_CFG}" \
  --distributed \
  --backend nccl
