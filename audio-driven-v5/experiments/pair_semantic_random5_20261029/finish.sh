#!/usr/bin/env bash
set -euo pipefail
cd /private/liqi_workspace/project_code/audio-driven-v5
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python -u experiments/pair_semantic_random5_20261029/build_baselines.py > experiments/pair_semantic_random5_20261029/media.log 2>&1
python -u experiments/pair_semantic_random5_20261029/publish_baselines.py > experiments/pair_semantic_random5_20261029/publish.log 2>&1
python -u experiments/pair_semantic_random5_20261029/export.py > experiments/pair_semantic_random5_20261029/export.log 2>&1
python -u experiments/pair_boundary_real_audit_20261029/publish_review.py >> experiments/pair_semantic_random5_20261029/publish.log 2>&1
