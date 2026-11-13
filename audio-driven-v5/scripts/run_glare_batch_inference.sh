#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_VISIBLE_DEVICES:?Set the allocated inference GPU explicitly}"
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/experiments/glare_current500h_nonprosody_cache_20261110/runtime_deps:$PWD/compare/.envs/glare-deps:$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec compare/.envs/avatarforcing/bin/python scripts/infer_glare_batch.py "$@"
