#!/bin/sh

## USAGE

## bash eval/referring_seg/run_evaluation.sh <path to the HF checkpoints path> <path to the directory to save the evaluation results>

## USAGE


# Adjust the environment variable if you have multiple gpus available, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 if you have 4 GPUs available
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=24999

# Positional arguments for the bash scripts
CKPT_PATH=$1
RESULT_PATH=$2
BATCH_SIZE="${BATCH_SIZE:-16}"
DATASET_DIR="${DATASET_DIR:-./data}"
# val_dataset=("refcoco|val" "refcoco|testA" "refcoco|testB" "refcoco+|val" "refcoco+|testA" "refcoco+|testB" "refcocog|val" "refcocog|test")
# val_dataset=("ReasonSeg|val" "MUSE|val")
val_dataset=('refcoco|val##refcoco|testA##refcoco|testB##refcoco+|val##refcoco+|testA##refcoco+|testB##refcocog|val##refcocog|test')


for data in "${val_dataset[@]}"
do
  echo "Evaluating $data ..."
  deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py \
  --version "$CKPT_PATH" \
  --val_dataset "$data" \
  --dataset_dir "$DATASET_DIR" \
  --results_dir "$RESULT_PATH" \
  --val_batch_size "$BATCH_SIZE"
done

# # RefCOCO
# deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --val_dataset "refcoco|val" --results_dir "$RESULT_PATH"
# deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --val_dataset "refcoco|testA" --results_dir "$RESULT_PATH"
# deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --val_dataset "refcoco|testB" --results_dir "$RESULT_PATH"

# # RefCOCO+
# deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --val_dataset "refcoco+|val" --results_dir "$RESULT_PATH"
# deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --val_dataset "refcoco+|testA" --results_dir "$RESULT_PATH"
# deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --val_dataset "refcoco+|testB" --results_dir "$RESULT_PATH"

# # RefCOCOg
# deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --val_dataset "refcocog|val" --results_dir "$RESULT_PATH"
# deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --val_dataset "refcocog|test" --results_dir "$RESULT_PATH"
