# Source this before running lm-listener evaluation.
#   source /private/liqi_workspace/project_code/audio-driven-v5/evaluation/lm-listener/env.sh

ROOT="/private/liqi_workspace/project_code/audio-driven-v5/evaluation/lm-listener"
export DECA_PATH="${ROOT}/third_party/DECA"
export PYTHONPATH="${ROOT}/third_party/emoca:${PYTHONPATH}"
export LM_LISTENER_ROOT="${ROOT}"

if [ -f /private/software/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /private/software/conda/etc/profile.d/conda.sh
  conda activate lm-listener
fi
