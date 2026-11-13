#!/usr/bin/env bash
set -euo pipefail
BACKUP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
P2_PYTHON=${P2_PYTHON:-python}
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$BACKUP_ROOT/snapshot/audio-driven-v5/src:$BACKUP_ROOT/snapshot/audio-driven-v5/tests:$BACKUP_ROOT/snapshot/audio-driven-v4/src:$BACKUP_ROOT/snapshot/Omni-LTX-stage2/packages/ltx-core/src:$BACKUP_ROOT/snapshot/Omni-LTX-stage2/packages/ltx-trainer/src:$BACKUP_ROOT/snapshot/Omni-LTX-stage2/packages/ltx-pipelines/src"
"$P2_PYTHON" "$BACKUP_ROOT/verify_backup.py"
cd "$BACKUP_ROOT/snapshot/audio-driven-v5"
"$P2_PYTHON" - <<'PY'
from pathlib import Path
import importlib
root=Path.cwd().parent
for name in ['listen_audio_postprocess','p2_listen','p3_listen','omni_av','ltx_core']:
 module=importlib.import_module(name);path=Path(module.__file__).resolve()
 assert path.is_relative_to(root),(name,path)
 print(name,path)
PY
"$P2_PYTHON" -m unittest -v test_listener_audio_virtual test_listener_audio_bulk test_listen_official_audio_gate test_listen_official_event_alignment test_listen_chinese_asr_filter test_p2_listen_av_only test_inference_isolation
