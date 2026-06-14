#!/usr/bin/env bash

export PYTHONPATH="./:$PYTHONPATH"

EXP_NAME=$1
# Adjust if needed
DATASET="${DATASET:-refcocog##flickr##nocaps##vg}"
DATA_DIR="${DATA_DIR:-./data}"
RESULT_DIR="${2:-./output/results/caption/${EXP_NAME}}"

# split string by "##"
IFS='##' read -ra ADDR <<< "$DATASET"
for ds in "${ADDR[@]}"; do
  echo "Evaluating $ds"
  if [ "$ds" == "refcocog" ]; then
    ANNOTATION_FILE="${DATA_DIR}/RefCoco_Reg/mdetr_annotations/finetune_refcocog_val_captions.json"
    IMAGE_DIR="${DATA_DIR}/coco/train2014"
  elif [ "$ds" == "flickr" ]; then
    ANNOTATION_FILE="${DATA_DIR}/flickr_30k/mdetr_annotations/final_flickr_mergedGT_test_caption.json"
    IMAGE_DIR="${DATA_DIR}/flickr_30k/images"
  elif [ "$ds" == "nocaps" ]; then
    ANNOTATION_FILE="${DATA_DIR}/nocaps/annotations/nocaps_val_4500_captions.json"
    IMAGE_DIR="${DATA_DIR}/nocaps/images"
  elif [ "$ds" == "vg" ]; then
    ANNOTATION_FILE="${DATA_DIR}/visual_genome/test_caption.json"
    IMAGE_DIR="${DATA_DIR}/visual_genome/images"
  elif [ "$ds" == '' ]; then
    continue
  else
    echo "Unknown dataset: $ds"
    continue
  fi
  RESULT_PATH="${RESULT_DIR}/${ds}"
  # Evaluate
  python eval/region_captioning/evaluate.py --annotation_file "$ANNOTATION_FILE" --results_dir "$RESULT_PATH"
done


## args for torchrun
# CKPT_PATH=''
# MASTER_PORT=24999
# NUM_GPUS=8  # Adjust it as per the available #GPU
# BATCH_SIZE=16

## Run Inference
# torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval/region_captioning/infer.py \
#     --version "$CKPT_PATH" \
#     --annotation_file "$ANNOTATION_FILE" \
#     --image_dir "$IMAGE_DIR" \
#     --dataset "$DATASET" \
#     --results_dir "$RESULT_PATH" \
#     --batch_size_per_gpu "$BATCH_SIZE" \
#     --model_max_length 128 \
#     --world_size "$NUM_GPUS"
