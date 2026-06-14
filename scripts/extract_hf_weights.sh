export PYTHONPATH="./:$PYTHONPATH"

exp_nam=$1

CKPT_BASE_DIR="${CKPT_BASE_DIR:-./output/checkpoints}"

ckpt_path="${CKPT_BASE_DIR}/${exp_nam}"
config_file="${ckpt_path}/config.yaml"
checkpoint="${ckpt_path}/ckpt_model_best"
script_name="${checkpoint}/zero_to_fp32.py"

work_dir=$(pwd)
output_dir="${work_dir}/output/pretrained"
target_bin_file="${output_dir}/${exp_nam}_pytorch_model.bin"
exp_save_path="output/pretrained/${exp_nam}"

mkdir -p ${output_dir}

echo "Extracting fp32 consolidated weights from a zero 1, 2 and 3 DeepSpeed checkpoints ..."
python ${script_name} ${checkpoint} ${target_bin_file}

echo "Merging lora weights ..."
python "${work_dir}/scripts/merge_lora_weights.py" \
  --weight ${target_bin_file} \
  --config ${config_file} \
  --save_path ${exp_save_path} \
  --local-rank -1

if [ $? -eq 0 ]; then
  echo "Merged weights saved to ${exp_save_path}."
  echo "Intermediate fp32 file saved to ${target_bin_file}."
else
  echo "Failed to merge lora weights."
fi
