import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')))

import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from transformers import CLIPImageProcessor
from torch.utils.data import DataLoader, DistributedSampler
from eval.utils.ddp import CapDDP, RegionCapDDP, init_distributed_mode
from eval.utils.utils import preprocess_instruction, preprocess_image, postprocess_text
from eval.utils.initialize import process_args, setup_tokenizer, initialize_model, prepare_for_inference
from eval.region_captioning.evaluate import calculate_metrics
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import IMAGE_TOKEN_INDEX, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="MGLMM Inference - Region Captioning")

    parser.add_argument("--version", required=True, help="The model path in huggingface format.")
    parser.add_argument("--annotation_file",
                        default="data/RefCoco_Reg/mdetr_annotations/finetune_refcocog_val_captions.json", type=str,
                        help="Replace with 'data/visual_genome/test_caption.json' for VG.")
    parser.add_argument("--image_dir", default="data/coco_2014/train2014", type=str,
                        help="Replace with 'data/visual_genome/images' for VG")
    parser.add_argument("--dataset", default="flickr##nocaps##vg##refcocog", type=str, help="Dataset name, options are 'flickr', 'nocaps', 'vg', 'refcocog'.")
    parser.add_argument("--results_dir", default="results", type=str, help="The path to save the results.")

    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--with_region", action="store_true", default=False)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"], )
    parser.add_argument("--precision", default="bf16", type=str, choices=["bf16", "fp32"], help="Options are 'bf16', 'fp32'")

    # DDP Related parameters
    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers for dataloader.")
    parser.add_argument("--val_batch_size", required=False, default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--sample_num', default=0, type=int, help='number of samples to print')

    return parser.parse_args()


def preprocess_batch(batch, instruction=None, clip_image_processor=None, transform=None, tokenizer=None, image_dir=None, args=None):
    image_id_list, filename_list, bbox_list, gt_list = [], [], [], []
    for (image_id, filename, bbox, gt) in batch:
        image_id_list.append(image_id)
        filename_list.append(filename)
        bbox_list.append(bbox)
        gt_list.append(gt)
        
    image_clip_list, image_list, original_size_list, resize_list = [], [], [], []
    image_path_list = [os.path.join(image_dir, filename) for filename in filename_list]

    for image_path, bbox in zip(image_path_list, bbox_list):
        image_clip, image, original_size, resize = preprocess_image(image_path, clip_image_processor, transform, bbox)
        image_clip_list.append(image_clip)
        image_list.append(image)
        original_size_list.append(original_size)
        resize_list.append(resize)

    if isinstance(instruction, str):
        instruction = [instruction] * len(image_path_list)
        
    instruction = [inst.replace('&lt;', '<').replace('&gt;', '>') for inst in instruction]
    prompts = [preprocess_instruction(inst, args) for inst in instruction]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in prompts],
        batch_first=True, padding_value=tokenizer.pad_token_id
    )

    return {
        'prompts': prompts,
        'image_id': image_id_list,
        'image_path': image_path_list,
        'input_ids': input_ids,
        'image_clip': torch.cat(image_clip_list, dim=0),
        'image': torch.cat(image_list, dim=0),
        'original_size': original_size_list,
        'resize': resize_list,
        'boxes': bbox_list,
        'gt': gt_list,
    }


def inference(batch, model, tokenizer, with_region=False):
    # Prepare inputs for inference
    image_clip = batch['image_clip']
    image = batch['image']
    input_ids = batch['input_ids']
    original_size_list = batch['original_size']
    resize_list = batch['resize']

    input_ids = input_ids.cuda()
    image_clip = image_clip.cuda().to(args.torch_dtype)
    image = image.cuda().to(args.torch_dtype)

    if with_region:
        bbox_img = batch['boxes']
        height, width = original_size_list[0]  # Original Image Dimensions

        # Rescaling BBox to 336*336
        x_scale, y_scale = 336 / width, 336 / height
        bboxes_scaled = [[bbox[0] * x_scale, bbox[1] * y_scale,
                          bbox[2] * x_scale, bbox[3] * y_scale] for bbox in bbox_img]
        ori_bboxes = np.array(bboxes_scaled, dtype=np.float64)
        height_sc, width_sc = (336, 336)  # To normalize the Image
        norm_bboxes = ori_bboxes / np.array([width_sc, height_sc, width_sc, height_sc])
        bboxes = [torch.tensor(norm_bboxes).cuda().half().to(args.torch_dtype)]
    else:
        image = None
        bboxes = None

    # Generate output
    output_ids, _ = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                            max_tokens_new=args.model_max_length, bboxes=bboxes)

    decoded_text = []
    for output_id in output_ids:
        output_id = output_id[output_id != IMAGE_TOKEN_INDEX]
        decoded_text.append(tokenizer.decode(output_id, skip_special_tokens=False))

    # Post-processing
    cleaned_text_list = []
    for text in decoded_text:
        _, cleaned_text = postprocess_text(text)
        cleaned_text_list.append(cleaned_text)

    return cleaned_text_list


def eval_model_performance(dataloader, model, tokenizer, args):
    # Iterate over all the samples, perform inference and save results
    results = []
    for _, batch in enumerate(tqdm(dataloader)):
        prompts_list = batch["prompts"]
        image_id_list = batch["image_id"]
        image_path_list = batch["image_path"]
        gt_list = batch["gt"]
        captions = inference(batch, model, tokenizer, args.with_region)  # Perform inference

        for idx, (image_id, image_path, gt, caption, prompt) in enumerate(zip(image_id_list, image_path_list, gt_list, captions, prompts_list)):
            result_dict = {}
            result_dict["image_id"] = image_id
            result_dict["image_path"] = image_path
            result_dict["gt"] = gt
            result_dict["caption"] = caption
            results.append(result_dict)
            
            if idx < args.sample_num:
                print(f"image_path: {image_path}\nprompt: {[prompt]}\ngt: {gt}\ncaption: {caption}\n")
    
    return results

def main(args):
    # Initialize the tokenizer
    tokenizer = setup_tokenizer(args)    
    # Initialize the model
    model = initialize_model(args, tokenizer)
    model = prepare_for_inference(model, tokenizer, args)

    # set args with model config
    args.vision_tower = model.config.vision_tower
    args.mm_use_im_start_end = model.config.mm_use_im_start_end
    args.mm_use_prompt_start_end = model.config.mm_use_prompt_start_end

    # Transfer the model to GPU
    # Replace with model = model.float().cuda() for 32 bit inference
    model = model.cuda()
    model.eval()

    # Initialize Image Processor for GLobal Image Encoder (CLIP)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    # Initialize the instruction
    if args.with_region:
        instruction = "Can you provide me with a detailed description of the region in the picture marked by <bbox>?"
    else:
        instruction = 'Could you please give me a detailed description of the image?'

    dataset_name_list = args.dataset.split("##")
    annotation_file_list = args.annotation_file.split("##")
    image_dir_list = args.image_dir.split("##")
    assert len(dataset_name_list) == len(annotation_file_list) == len(image_dir_list), "Length of dataset, annotation_file and results_dir should be same."

    # Iterate over all the datasets
    for dataset_name, annotation_file, image_dir in zip(dataset_name_list, annotation_file_list, image_dir_list):
        print(f"Evaluating on {dataset_name} dataset.")
        print(f"Annotation File: {annotation_file}")
        print(f"Image Directory: {image_dir}")

        if dataset_name in ['flickr', 'nocaps']:
            dataset = CapDDP(annotation_file)
        elif dataset_name in ['refcocog', 'vg']:
            dataset = RegionCapDDP(annotation_file)
        else:
            assert False, "Invalid dataset name"

        custom_collate_fn = partial(preprocess_batch, instruction=instruction, clip_image_processor=clip_image_processor, transform=transform, 
                                    tokenizer=tokenizer, image_dir=image_dir, args=args)

        distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False) if args.world_size > 1 else None
        dataloader = DataLoader(dataset, batch_size=args.val_batch_size, num_workers=args.num_workers,
                                sampler=distributed_sampler, collate_fn=custom_collate_fn)
        
        results = eval_model_performance(dataloader, model, tokenizer, args)

        results_path = os.path.join(args.results_dir, os.path.basename(args.version), dataset_name)
        os.makedirs(results_path, exist_ok=True)
        with open(f"{results_path}/{dataset_name}_{args.rank}.json", 'w') as json_file:
            json.dump(results, json_file, indent=2)

        torch.distributed.barrier()
        # evaluate the results
        if args.rank == 0:
            calculate_metrics(annotation_file, results_path)
        torch.distributed.barrier()


if __name__ == "__main__":
    args = parse_args()
    args = process_args(args)
    
    # Initialize Distributed Mode
    if args.world_size > 1:
        init_distributed_mode(args)
    else:
        print("Running on single GPU.")
        args.rank = 0

    set_random_seed()
    main(args)
