#!/usr/bin/env bash
set -euo pipefail
BACKUP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
P2_PYTHON=${P2_PYTHON:-python}
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$BACKUP_ROOT/snapshot/audio-driven-v5/src:$BACKUP_ROOT/snapshot/audio-driven-v5/tests:$BACKUP_ROOT/snapshot/audio-driven-v5/scripts:$BACKUP_ROOT/snapshot/audio-driven-v4/src:$BACKUP_ROOT/snapshot/Omni-LTX-stage2/packages/ltx-core/src:$BACKUP_ROOT/snapshot/Omni-LTX-stage2/packages/ltx-trainer/src:$BACKUP_ROOT/snapshot/Omni-LTX-stage2/packages/ltx-pipelines/src"
"$P2_PYTHON" "$BACKUP_ROOT/verify_backup.py"
cd "$BACKUP_ROOT/snapshot/audio-driven-v5"
"$P2_PYTHON" - <<'PY'
from pathlib import Path
import importlib
root=Path.cwd().parent
for name in ['listen_benchmark.open_verbatim_subset','listen_benchmark.open_subset_reader','listen_audio_full','listen_audio_postprocess','p2_listen','p3_listen','omni_av','ltx_core']:
    p=Path(importlib.import_module(name).__file__).resolve()
    assert p.is_relative_to(root),(name,p)
    print('snapshot import:',name)
PY
"$P2_PYTHON" -m pytest -q -p no:cacheprovider \
 tests/test_listen_full_raw.py tests/test_listen_full_fusion.py \
 tests/test_listen_existing_fusion.py tests/test_listen_integration_pilot.py \
 tests/test_listen_official_audio_gate.py tests/test_listen_official_event_alignment.py \
 tests/test_listen_chinese_asr_filter.py tests/test_listener_audio_virtual.py tests/test_listener_audio_bulk.py \
 tests/listen_audio_full/test_full_pipeline.py tests/test_listen_backchannel_face_stats.py \
 tests/test_listen_train_subset.py tests/test_postprocessed_subset.py tests/test_open_verbatim_subset.py \
 tests/test_processed_test_panel.py tests/test_p2_full_feature_pipeline.py tests/test_p2_listen_av_only.py tests/test_inference_isolation.py
