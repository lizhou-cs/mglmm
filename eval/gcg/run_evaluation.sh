#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"
MASTER_PORT="${MASTER_PORT:-24999}"
NUM_GPUS="${NUM_GPUS:-1}"

CKPT_PATH=$1
RESULT_PATH=$2
DATASET_DIR="${DATASET_DIR:-./data}"

# Path to the GranD-f evaluation dataset images directoryå
IMAGE_DIR="${IMAGE_DIR:-${DATASET_DIR}/GranDf/GranDf_HA_images/val_test}"

# Path to the GranD-f evaluation dataset ground-truths directory
GT_DIR="${GT_DIR:-${DATASET_DIR}/GranDf/annotations/val_test}"

# Path to the BERT model
BERT_MODEL_PATH="${BERT_MODEL_PATH:-bert-base-uncased}"
PREDICTION_DIR="${RESULT_PATH}/$(basename "$CKPT_PATH")"

# Run Inference
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval/gcg/infer.py \
  --version "$CKPT_PATH" \
  --dataset_dir "$IMAGE_DIR" \
  --results_dir "$RESULT_PATH" \
  --gt_dir "$GT_DIR" \
  --bert_model "$BERT_MODEL_PATH" \
  --world_size "$NUM_GPUS"

# Evaluate
python eval/gcg/evaluate.py --prediction_dir_path "$PREDICTION_DIR" --gt_dir_path "$GT_DIR" --split "val" --bert_model "$BERT_MODEL_PATH"
python eval/gcg/evaluate.py --prediction_dir_path "$PREDICTION_DIR" --gt_dir_path "$GT_DIR" --split "test" --bert_model "$BERT_MODEL_PATH"
