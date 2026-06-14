#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/mglmm}"
VIS_SAVE_DIR="${VIS_SAVE_DIR:-./output/vis_output}"
MASK_SAVE_DIR="${MASK_SAVE_DIR:-./output/masks}"

python inference.py \
  --version "${MODEL_PATH}" \
  --vis_save_dir "${VIS_SAVE_DIR}" \
  --mask_save_dir "${MASK_SAVE_DIR}" \
  "$@"
