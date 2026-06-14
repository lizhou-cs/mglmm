import os
import yaml
import torch
import argparse
from peft import get_peft_model
from train import setup_tokenizer_and_special_tokens, initialize_model, initialize_modules, setup_lora_config


def parse_args():
    parser = argparse.ArgumentParser(description="MGLMM: Merge lora weights and save model in hf format")

    parser.add_argument("--weight", required=True, type=str, help="Path to the .bin model "
                                                                  "(generated using the script zero_to_fp32.py)")
    parser.add_argument("--config", required=True, type=str, help="Path to the config file")
    parser.add_argument("--save_path", required=True, type=str, help="Path to save the hf model.")
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")

    return parser.parse_args()


def load_model_args(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)


def main():
    args = parse_args()
    model_args = load_model_args(args.config)

    # Create output directory if not exists already
    os.makedirs(args.save_path, exist_ok=True)

    # Initialize the tokenizer and model
    tokenizer = setup_tokenizer_and_special_tokens(model_args)
    model = initialize_model(model_args, tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    initialize_modules(model, tokenizer, model_args)

    if model_args.lora_r > 0:
        lora_config = setup_lora_config(model, model_args)
        model = get_peft_model(model, lora_config)

    # Load the state-dict from --weights
    state_dict = torch.load(args.weight, map_location="cpu")
    updated_state_dict = {}

    for key in state_dict.keys():
        if "vision_tower" in key:
            continue
        updated_key = f"base_model.model.{key}"
        updated_state_dict[updated_key] = state_dict[key]
    model.load_state_dict(updated_state_dict, strict=False)

    # Merge and save
    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
