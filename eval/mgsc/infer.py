import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')))

import re
import json
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import CLIPImageProcessor

from tools.utils import IMAGE_TOKEN_INDEX, set_random_seed
from eval.utils.utils import preprocess_instruction, preprocess_image, mask_to_rle_pytorch, coco_encode_rle, postprocess_text
from eval.utils.ddp import GCGEvalDDP, init_distributed_mode
from eval.utils.initialize import process_args, setup_tokenizer, initialize_model, prepare_for_inference
from eval.mgsc.evaluate import calculate_metrics
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide


def parse_args():
    parser = argparse.ArgumentParser(description="MGLMM Inference - GCG")

    parser.add_argument("--version", required=True, help="Path to the pretrained model for evaluation.")
    parser.add_argument("--bert_model", default="bert-base-uncased", help="BERT model to use for text similarity computation.")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--gt_dir", default=None, type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    # DDP Related parameters
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def inference(instructions, image_path, clip_image_processor, transform, model, tokenizer, args):
    # Filter out special chars
    instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')

    # preprocess the instruction
    prompt = preprocess_instruction(instructions, args)

    # preprocess the image (CLIP)
    image_clip, image, original_size_list, resize_list = preprocess_image(image_path, clip_image_processor, transform)

    # Prepare inputs for inference
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    image_clip = image_clip.cuda().to(args.torch_dtype)
    image = image.cuda().to(args.torch_dtype)
    bboxes = None  # No box/region is input in GCG task

    # Generate output
    output_ids, pred_masks = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                            max_tokens_new=512, bboxes=bboxes)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    decoded_text = tokenizer.decode(output_ids, skip_special_tokens=False)

    # Post-processing
    text_output, cleaned_str = postprocess_text(decoded_text)

    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    return cleaned_str, pred_masks, phrases


def get_prompt(prompt_type='root'):
    if prompt_type == 'root':
        prompt = "Please provide a detailed description of all the objects present in this image in a comprehensive format."
    elif 'subtree:' in prompt_type:
        label = prompt_type.split(':')[1]
        prompt = "Can you provide a detailed description in a comprehensive format for {label} in the image.".format(label=label)
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    prompt = prompt + " Please respond with interleaved segmentation masks for the corresponding parts of the answer."
    return prompt


def load_annotation_info(anno_dir):
    root_anno_info = json.load(open(os.path.join(anno_dir, 'root_mgsc_caption_gt.json')))
    sub_anno_info = json.load(open(os.path.join(anno_dir, 'subtree_mgsc_caption_gt.json')))
    images = root_anno_info['images'] + sub_anno_info['images']

    annotation_info = {}
    for image in images:
        image_id = image['id']
        annotation_info[image_id] = image

    return annotation_info

def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    image_path = [item[1] for item in batch]

    return image_id, image_path


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
    
    # load annotation info
    annotation_info = load_annotation_info(args.gt_dir)

    # Create DDP Dataset
    dataset = GCGEvalDDP(args.dataset_dir)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False) if args.world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=args.val_batch_size, num_workers=2,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)

    results_dir = os.path.join(args.results_dir, os.path.basename(args.version))
    # Create output directory if not exists already
    os.makedirs(results_dir, exist_ok=True)

    # Iterate over all the images, run inference and save results
    for (image_id, image_path) in tqdm(dataloader):
        image_id, image_path = image_id[0], image_path[0]
        results_path = f"{results_dir}/{image_id[:-4]}.json"

        image_type = annotation_info[image_id[:-4]]['type']
        instruction = get_prompt(image_type)

        result_caption, pred_masks, phrases = inference(instruction, image_path, clip_image_processor, 
                                                        transform, model, tokenizer, args)

        # Convert the predicted masks into RLE format
        pred_masks_tensor = pred_masks[0].cpu()
        binary_pred_masks = pred_masks_tensor > 0
        uncompressed_mask_rles = mask_to_rle_pytorch(binary_pred_masks)
        rle_masks = []
        for m in uncompressed_mask_rles:
            rle_masks.append(coco_encode_rle(m))

        # Create results dictionary
        result_dict = {
            "image_id": image_id[:-4],
            "caption": result_caption,
            "phrases": phrases,
            "pred_masks": rle_masks
        }

        # Save the inference results
        with open(results_path, 'w') as f:
            json.dump(result_dict, f)
    
    torch.distributed.barrier()
    if args.rank == 0:
        splits = ['root', 'subtree']
        for split in splits:
            calculate_metrics(args.bert_model, args.gt_dir, results_dir, split)


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
