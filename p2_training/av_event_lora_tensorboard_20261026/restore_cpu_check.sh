#!/usr/bin/env bash
set -euo pipefail
P2_BACKUP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
P2_PYTHON="${P2_PYTHON:-python}"
P2_VALIDATION_DIR="${P2_VALIDATION_DIR:-$P2_BACKUP_ROOT/validation/recheck}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$P2_BACKUP_ROOT/snapshot/audio-driven-v5/src:$P2_BACKUP_ROOT/snapshot/audio-driven-v4/src:$P2_BACKUP_ROOT/snapshot/Omni-LTX-stage2/packages/ltx-core/src:$P2_BACKUP_ROOT/snapshot/Omni-LTX-stage2/packages/ltx-pipelines/src:$P2_BACKUP_ROOT/snapshot/Omni-LTX-stage2/packages/ltx-trainer/src"
mkdir -p "$P2_VALIDATION_DIR"
"$P2_PYTHON" "$P2_BACKUP_ROOT/verify_backup.py"
"$P2_PYTHON" "$P2_BACKUP_ROOT/snapshot/audio-driven-v5/scripts/check_p2_listen_code.py" --output "$P2_VALIDATION_DIR/unit_tests.json"
P2_DDP_METRICS_DIR="$(mktemp -d "$P2_VALIDATION_DIR/ddp_metrics.XXXXXX")"
"$P2_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$P2_BACKUP_ROOT/snapshot/audio-driven-v5/scripts/check_p2_listen_tensorboard_ddp.py" --output "$P2_DDP_METRICS_DIR"
"$P2_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$P2_BACKUP_ROOT/snapshot/audio-driven-v5/scripts/check_p2_listen_cpu_ddp.py" \
  --config "$P2_BACKUP_ROOT/snapshot/audio-driven-v5/configs/p2_listen_av_event_text_lora_h200_v1.json" \
  --output "$P2_VALIDATION_DIR/training_ddp.json"
