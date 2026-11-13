#!/usr/bin/env bash
set -euo pipefail
cd /private/liqi_workspace/project_code/audio-driven-v5
LTX_LAUNCH_LOGS=/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_ltx_hidden_clip_events_current500h_20261106/launch_node12_8_4_11_$(date -u +%Y%m%dT%H%M%SZ)_$$
exec /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python /private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_ltx_events.py launch --config /private/liqi_workspace/project_code/audio-driven-v5/configs/p2_ltx_hidden_clip_events_current500h_20261106.json --cluster /private/liqi_workspace/project_code/audio-driven-v5/configs/p2_ltx_hidden_clip_32gpu_node12_8_4_11_20261106.json --logs "$LTX_LAUNCH_LOGS" "$@"
