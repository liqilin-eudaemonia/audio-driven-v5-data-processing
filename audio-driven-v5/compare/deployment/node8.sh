#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="$PWD"
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
case "${1:-}" in
 dystream-ui)
  shift; export PATH="$BASE/.envs/dystream/bin:$PATH"; cd "$BASE/DyStream"
  exec "$BASE/.envs/dystream/bin/python" "$BASE/deployment/dystream_ui.py" "$@" ;;
 dystream-batch)
  shift; export PATH="$BASE/.envs/dystream/bin:$PATH"; cd "$BASE/DyStream"
  exec "$BASE/.envs/dystream/bin/python" main.py --config configs/motion_gen/sample.yaml --override exp_name=compare_gpu7 model.module_name=model.motion_generation.motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder resume_ckpt=checkpoints/last.ckpt "$@" ;;
 l2l)
  shift; cd "$BASE/learning2listen/src"; export L2L_PATH="$PWD"
  exec "$BASE/.envs/l2l/bin/python" test_vq_decoder.py "$@" ;;
 reactface)
  shift; export PATH="$BASE/.envs/reactface/bin:$PATH"; cd "$BASE/ReactFace"
  exec "$BASE/.envs/reactface/bin/python" dyadic_reaction_inference.py "$@" ;;
 mm2022)
  shift; cd "$BASE/deployment/workspaces/mm2022/vico"
  exec "$BASE/.envs/vico/bin/python" eval.py "$@" ;;
 speakervid-luminance)
  shift; cd "$BASE/SpeakerVid-5M-Code"
  exec "$BASE/.envs/speakervid/bin/python" cal_luminance/filter_light.py "$@" ;;
 *) echo 'Usage: node8.sh {dystream-ui|dystream-batch|l2l|reactface|mm2022|speakervid-luminance} [official arguments]' >&2; exit 2 ;;
esac
