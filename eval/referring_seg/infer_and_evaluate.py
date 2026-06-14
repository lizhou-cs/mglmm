import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')))

import tqdm
import torch
import argparse
from functools import partial
from torch.utils.data import ConcatDataset

from dataset.dataset import custom_collate_fn
from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset, GReferCOCOValDataset
from dataset.segm_datasets.Panoptic_Segm_ds import PanopticSegmDataset
from dataset.reason_datasets.reason_seg import ReasonSegDataset
from dataset.reason_datasets.multi_reason_seg import MultiReasonSegDataset
from tools.utils import (AverageMeter, Summary, intersectionAndUnionGPU, dict_to_cuda, set_random_seed)
from eval.utils.initialize import process_args, setup_tokenizer, initialize_model, prepare_for_inference
from eval.utils.utils import save_metrics_to_json
from eval.utils.ddp import init_distributed_mode
from eval.utils.matcher import match_masks
from model.llava.mm_utils import tokenizer_image_token


def parse_args(args):
    parser = argparse.ArgumentParser(description="MGLMM Model Evaluation")

    # Model-specific settings
    parser.add_argument("--version", required=True, help="Path to the pretrained model for evaluation.")
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--val_dataset", default="refcocog|val", type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--explanatory", action="store_true", default=False)

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
    if dataset_name in ["refcoco", "refcoco+", "refcocog"]:
        num_classes_per_sample = 1 if dataset_name in datasets_need_merge else 3
        val_datasets = [
            ReferSegmDataset(**common_ds_args, refer_segm_data=dataset_name, split=split, validation=True, inference=True, 
                             explanatory=args.explanatory, num_classes_per_sample=num_classes_per_sample)
            ]
        for dataset in val_datasets:
            dataset._set_len(len(dataset.refer_segm_data[dataset_name]['images']))
    elif dataset_name == "grefcoco":
        val_datasets = [
            GReferCOCOValDataset(**common_ds_args, refer_segm_data=dataset_name, split=split, validation=True, 
                                 inference=True, explanatory=args.explanatory)
        ]
    elif dataset_name == "ReasonSeg":
        val_datasets = [
            ReasonSegDataset(**common_ds_args, reason_seg_data=dataset_name, split=split, validation=True)
        ]
    elif dataset_name == "MUSE":
        val_datasets = [
            MultiReasonSegDataset(**common_ds_args, split=split, validation=True, inference=True, use_expand_question_list=True),
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


def prepare_data_batch(data_batch, args, tokenizer, evaluation_mode=False):
    data_batch = dict_to_cuda(data_batch)
    for key in ["global_enc_images", "grounding_enc_images"]:
        data_batch[key] = data_batch[key].to(dtype=torch.bfloat16, device=args.local_rank)

    if evaluation_mode:
        # add the original sizes of images
        orig_sizes = []
        for label in data_batch["label_list"]:
            orig_sizes.append(label.shape[-2:])
        data_batch["orig_sizes"] = orig_sizes

        # remove the answer from the input_ids
        input_ids = []
        conversation_list = data_batch["conversation_list"]
        for conversation in conversation_list:
            prompt = conversation.split("ASSISTANT:")[0] + "ASSISTANT:"
            input_ids.append(tokenizer_image_token(prompt, tokenizer, return_tensors="pt"))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        input_ids = input_ids.to(device=args.local_rank)
        data_batch["input_ids"] = input_ids

    return data_batch

def evaluate_model_performance(val_loader, model_engine, args, tokenizer, enable_match=False, enable_merge=False):
    # enable_match: whether to match the predicted masks with the ground truth masks
    # enable_merge: whether to merge all channels into one for both predicted and ground truth masks
    assert not (enable_match and enable_merge), "enable_match and enable_merge are mutually exclusive"

    # Trackers for metrics
    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }

    model_engine.eval()
    for data_batch in tqdm.tqdm(val_loader):

        data_batch = prepare_data_batch(data_batch, args, tokenizer, evaluation_mode=(enable_match or enable_merge))
        torch.cuda.empty_cache()

        # Model inference without gradient tracking
        if enable_match:
            _, pred_masks = model_engine.evaluate(**data_batch)
            gt_masks = data_batch["masks_list"]

            new_pred_masks, new_gt_masks = [], []
            indices = match_masks(pred_masks, gt_masks)
            for idx in indices:
                new_pred_masks.append(pred_masks[idx[0]])
                new_gt_masks.append(gt_masks[idx[1]])
            pred_masks, gt_masks = new_pred_masks, new_gt_masks
        elif enable_merge:
            _, pred_masks = model_engine.evaluate(**data_batch)
            gt_masks = data_batch["masks_list"]

            new_pred_masks, new_gt_masks = [], []
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                if gt_mask == []:
                    gt_mask = torch.zeros_like(pred_mask, dtype=pred_mask.dtype, device=pred_mask.device)
                # merge all channels into one
                pred_mask = (pred_mask > 0).int()
                pred_mask = pred_mask.sum(dim=0, keepdim=True)
                gt_mask = gt_mask.sum(dim=0, keepdim=True)

                # assign 1 to all non-zero values
                pred_mask = (pred_mask > 0).int()
                gt_mask = (gt_mask > 0).int()
                
                new_pred_masks.append(pred_mask)
                new_gt_masks.append(gt_mask)
            pred_masks, gt_masks = new_pred_masks, new_gt_masks
        else:
            with torch.no_grad():
                results = model_engine(**data_batch)
            pred_masks, gt_masks = results["pred_masks"], results["gt_masks"]

        assert len(pred_masks) == 1
        pred_masks = (pred_masks[0] > 0).int()
        gt_masks = gt_masks[0].int()
        
        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        for target, prediction in zip(gt_masks, pred_masks):
            intersect, union_, _ = intersectionAndUnionGPU(
                prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
            )
            intersection += intersect
            union += union_
            accuracy_iou += intersect / (union_ + 1e-5)
            # handles no-object targets
            accuracy_iou[union_ == 0] += 1.0
        
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        accuracy_iou = accuracy_iou.cpu().numpy() / gt_masks.shape[0]
        trackers["intersection"].update(intersection)
        trackers["union"].update(union)
        trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])

    for meter in trackers.values():
        meter.all_reduce()

    iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
    class_iou = iou_per_class[1]
    global_iou = trackers["gIoU"].avg[1]

    return global_iou, class_iou


def save_results(val_dataset_name, giou, ciou, args):
    torch.distributed.barrier()

    if args.rank == 0:
        # Update and save the results
        results_dir = os.path.join(args.results_dir, os.path.basename(args.version))
        results_file = os.path.join(results_dir, f"stats_batch_size={args.val_batch_size}.json")
        result_dict = {"model": results_dir, "dataset": val_dataset_name, "giou": str(giou), "ciou": str(ciou)}

        os.makedirs(results_dir, exist_ok=True)
        save_metrics_to_json(result_dict, results_file)

        # Print all the results
        print(result_dict)


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
        enable_match = dataset_name in dataset_need_match
        enable_merge = dataset_name in datasets_need_merge 

        val_datasets = initialize_datasets_and_loaders(dataset_name, split, tokenizer, args)
        val_loader = setup_data_loaders(val_datasets, tokenizer, args)
        giou, ciou = evaluate_model_performance(val_loader, model_engine, args, tokenizer,
                                                enable_match=enable_match, enable_merge=enable_merge)
        save_results(val_dataset_name, giou, ciou, args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = process_args(args)

    set_random_seed()
    init_distributed_mode(args)

    dataset_need_match = []
    datasets_need_merge = ["grefcoco", "ReasonSeg"]
    main(args)
