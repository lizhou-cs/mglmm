import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')))

import json
import tqdm
import torch
import argparse
from functools import partial
from torch.utils.data import ConcatDataset

from dataset.dataset import custom_collate_fn
from dataset.segm_datasets.Panoptic_Segm_ds import PanopticSegmDataset
from tools.utils import (dict_to_cuda, set_random_seed, IMAGE_TOKEN_INDEX)
from eval.utils.initialize import process_args, setup_tokenizer, initialize_model, prepare_for_inference
from eval.utils.utils import mask_to_rle_pytorch, coco_encode_rle
from eval.utils.ddp import init_distributed_mode


def parse_args(args):
    parser = argparse.ArgumentParser(description="MGLMM Model Evaluation")

    # Model-specific settings
    parser.add_argument("--version", required=True, help="Path to the pretrained model for evaluation.")
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--val_dataset", default="panoptic|val", type=str)
    parser.add_argument("--results_dir", default=None, type=str)

    # Evaluation settings
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    return parser.parse_args(args)


def initialize_datasets_and_loaders(dataset_name, split, tokenizer, args):
    # Dataset settings for ReferSegDataset
    common_ds_args = {
        "dataset_dir": args.dataset_dir,
        "tokenizer": tokenizer,
        "global_image_encoder": args.vision_tower,
        "precision": args.precision,
        "image_size": args.image_size
    }

    # Validation datasets
    if dataset_name == "panoptic":
        val_datasets = [
            PanopticSegmDataset(**common_ds_args, validation=True, inference=True)
        ]
    else:
        assert False, f"Unknown dataset: {dataset_name}"

    return val_datasets


def setup_data_loaders(val_datasets, tokenizer, args):
    sampler_args = {"shuffle": False, "drop_last": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.num_workers, "pin_memory": False}

    collate_fn_args_val = partial(custom_collate_fn, tokenizer=tokenizer,
                                  mm_use_im_start_end=args.mm_use_im_start_end, mm_use_prompt_start_end=args.mm_use_prompt_start_end,
                                  inference=True)
    # Validation loader
    combined_val_datasets = ConcatDataset(val_datasets)
    val_loader = torch.utils.data.DataLoader(
        combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
        sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args)
        )

    return val_loader


def inference(val_loader, model_engine, tokenizer):
    results = []
    model_engine.eval()
    for data_batch in tqdm.tqdm(val_loader):
        # Prepare data and convert relevant tensors to the appropriate type
        data_batch = dict_to_cuda(data_batch)
        for key in ["global_enc_images", "grounding_enc_images"]:
            data_batch[key] = data_batch[key].to(dtype=torch.bfloat16, device=args.local_rank)

        orig_sizes = []
        for label in data_batch["label_list"]:
            orig_sizes.append(label.shape[-2:])
        data_batch["orig_sizes"] = orig_sizes

        torch.cuda.empty_cache()
        output_ids, pred_masks = model_engine.evaluate(**data_batch)
        assert len(output_ids) == 1

        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)

        categories, category_ids = parser_category_id(text_output)

        pred_masks_tensor = pred_masks[0].cpu()
        binary_pred_masks = pred_masks_tensor > 0
        uncompressed_mask_rles = mask_to_rle_pytorch(binary_pred_masks)

        assert len(category_ids) == len(uncompressed_mask_rles)
        for category, category_id, m in zip(categories, category_ids, uncompressed_mask_rles):
            results.append({
                'file_name': data_batch['image_paths'][0],
                'category': category,
                'category_id': category_id,
                'segmentation': coco_encode_rle(m)
            })
    return results

def load_category_info(args):
    category_name_to_id = {}
    category_file = os.path.join(args.dataset_dir, "cocopanoptic/annotations/panoptic_coco_categories.json")
    categories = json.load(open(category_file, "r"))
    for i, category in enumerate(categories):
        category_name = category["name"]
        category_name = PanopticSegmDataset.CATEGORY_REMAP.get(category_name, category_name)
        category_name = category_name.split('-')[0]
        if category_name not in category_name_to_id:
            category_name_to_id[category_name] = i
        else:
            print(f"Duplicate category name: {category_name}")
    return category_name_to_id


def parser_category_id(text_output):
    categories, category_ids = [], []
    while "[SEG]" in text_output:
        seg_pos = text_output.find('[SEG]')        
        tmp_text = text_output[:seg_pos]
        text_output = text_output[seg_pos + 5:]

        p_start, p_end = tmp_text.find('<p>'), tmp_text.find('</p>')
        if p_start == -1 or p_end == -1:
            continue

        category = tmp_text[p_start + 3: p_end]
        category = category.split('-')[0].strip()
        try:
            category_id = g_category_name_to_id[category]
        except KeyError:
            category_id = 255
        categories.append(category)
        category_ids.append(category_id)

    return categories, category_ids


def save_results(results):
    # Save results
    results_path = os.path.join(args.results_dir, os.path.basename(args.version))
    os.makedirs(results_path, exist_ok=True)
    results_file = f"{results_path}/results_{args.rank}.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    torch.distributed.barrier()

    if args.rank == 0:
        # Merge and load the results files
        merged_file_path = f"{results_path}/merged_results.json"
        merged_results = []
        for result_file in os.listdir(results_path):
            if result_file.endswith(".json"):
                merged_results += json.load(open(f"{results_path}/{result_file}", "r"))

        with open(merged_file_path, 'w') as f:
            json.dump(merged_results, f)

        # remove old results files
        for result_file in os.listdir(results_path):
            if result_file.endswith(".json") and result_file != "merged_results.json":
                os.remove(f"{results_path}/{result_file}")

    torch.distributed.barrier()

def main(args):
    tokenizer = setup_tokenizer(args)
    model = initialize_model(args, tokenizer)
    model = prepare_for_inference(model, tokenizer, args)

    # set args with model config
    args.vision_tower = model.config.vision_tower
    args.mm_use_im_start_end = model.config.mm_use_im_start_end
    args.mm_use_prompt_start_end = model.config.mm_use_prompt_start_end

    model_engine = model.cuda()

    val_dataset_list = args.val_dataset.split('##')
    for val_dataset_name in val_dataset_list:
        print(f"Evaluating {val_dataset_name} ...")
        dataset_name, split = val_dataset_name.split('|')

        val_datasets = initialize_datasets_and_loaders(dataset_name, split, tokenizer, args)
        val_loader = setup_data_loaders(val_datasets, tokenizer, args)
        results = inference(val_loader, model_engine, tokenizer)
        save_results(results)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = process_args(args)
    args.rank = int(os.getenv('RANK', 0))

    g_category_name_to_id = load_category_info(args)

    set_random_seed()
    init_distributed_mode(args)
    main(args)
