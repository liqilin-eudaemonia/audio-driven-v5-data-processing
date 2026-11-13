#!/usr/bin/env bash
set -euo pipefail
ulimit -n 65536
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_VISIBLE_DEVICES:?Set explicitly to the GPUs allocated for training}"
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/experiments/glare_current500h_nonprosody_cache_20261110/runtime_deps:$PWD/compare/.envs/glare-deps:$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec compare/.envs/avatarforcing/bin/python -m torch.distributed.run \
  --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29671}" \
  --nproc_per_node="${NPROC_PER_NODE:-1}" scripts/train_glare_paper_settings_v7.py "$@"
