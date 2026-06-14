#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="./:${PYTHONPATH:-}"

TASK="${TASK:-seg}"
MASTER_PORT="${MASTER_PORT:-24999}"
NUM_GPUS="${NUM_GPUS:-1}"

VERSION="${VERSION:-./checkpoints/llava-llama-2-13b-chat-lightning-preview}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
VISION_PRETRAINED="${VISION_PRETRAINED:-./checkpoints/sam_vit_h_4b8939.pth}"
DATASET_DIR="${DATASET_DIR:-./data}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-./output/checkpoints}"
LOG_BASE_DIR="${LOG_BASE_DIR:-./output/logs}"
EXP_NAME="${EXP_NAME:-mglmm_${TASK}}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
GRAD_STEPS="${GRAD_STEPS:-1}"
LR="${LR:-2e-4}"
WORKERS="${WORKERS:-8}"

ARGS=(
  --version "${VERSION}"
  --vision-tower "${VISION_TOWER}"
  --vision_pretrained "${VISION_PRETRAINED}"
  --dataset_dir "${DATASET_DIR}"
  --ckpt_base_dir "${CKPT_BASE_DIR}"
  --log_base_dir "${LOG_BASE_DIR}"
  --batch_size "${BATCH_SIZE}"
  --val_batch_size 1
  --workers "${WORKERS}"
  --exp_name "${EXP_NAME}"
  --lora_r 8
  --lr "${LR}"
  --pretrained
  --epochs "${EPOCHS}"
  --steps_per_epoch "${STEPS_PER_EPOCH}"
  --grad_accumulation_steps "${GRAD_STEPS}"
)

case "${TASK}" in
  cap)
    ARGS+=(
      --use_cap_data
      --cap_dataset "CocoCap"
      --cap_sample_rates "1"
      --val_dataset "CocoCapVal"
    )
    ;;
  seg)
    ARGS+=(
      --use_segm_data
      --segm_dataset "Refer_Segm"
      --segm_sample_rates "1"
      --val_dataset "RefCOCOgSegmVal"
      --mask_validation
    )
    ;;
  gcg)
    ARGS+=(
      --use_gcg_data
      --gcg_dataset "GranDf_GCG"
      --gcg_sample_rates "1"
      --model_max_length 2048
      --val_dataset "FlickrGCGVal|RefCocoGCGVal|PsgGCGVal"
    )
    ;;
  *)
    echo "Unsupported TASK='${TASK}'. Use one of: cap, seg, gcg." >&2
    exit 1
    ;;
esac

deepspeed --num_gpus "${NUM_GPUS}" --master_port "${MASTER_PORT}" train.py "${ARGS[@]}"
