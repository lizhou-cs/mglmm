import os
import sys
import cv2
import random
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, \
    DEFAULT_PROMPT_END_TOKEN, DEFAULT_PROMPT_START_TOKEN
from tools.markdown_utils import process_markdown, draw_bbox, colors
from eval.utils.initialize import process_args, setup_tokenizer, initialize_model, prepare_for_inference
from eval.utils.ddp import init_distributed_mode, ImageDDP


def parse_args(args):
    parser = argparse.ArgumentParser(description="MGLMM Model Demo")
    parser.add_argument("--version", default="./checkpoints/mglmm")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--vis_save_dir", default="./output/vis_output", type=str)
    parser.add_argument('--mask_save_dir', default='./output/masks', type=str)
    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--infer_mode", choices=['root', 'subnode'], default='root', type=str, help="Inference mode")
    parser.add_argument("--markdown", action="store_true", help="Markdown output")
    parser.add_argument("--mgsc_annotation_dir", default="./data/MGLMM_gcg_new/annotations", type=str)
    parser.add_argument("--mgsc_image_dir", default="./data/SegmentAnything/imgs", type=str)
    parser.add_argument("--grandf_image_dir", default="./data/GranDf/annotations/val_test", type=str)

    return parser.parse_args(args)


def grounding_enc_processor(x: torch.Tensor) -> torch.Tensor:
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    x = (x - IMG_MEAN) / IMG_STD
    h, w = x.shape[-2:]
    x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return x


def region_enc_processor(orig_size, post_size, bbox_img):
    orig_h, orig_w = orig_size
    post_h, post_w = post_size
    y_scale = post_h / orig_h
    x_scale = post_w / orig_w

    bboxes_scaled = [[bbox[0] * x_scale, bbox[1] * y_scale, bbox[2] * x_scale, bbox[3] * y_scale] for bbox in bbox_img]

    tensor_list = []
    for box_element in bboxes_scaled:
        ori_bboxes = np.array([box_element], dtype=np.float64)
        # Normalizing the bounding boxes
        norm_bboxes = ori_bboxes / np.array([post_w, post_h, post_w, post_h])
        # Converting to tensor, handling device and data type as in the original code
        tensor_list.append(torch.tensor(norm_bboxes, device='cuda').half().to(torch.bfloat16))

    if len(tensor_list) > 1:
        bboxes = torch.stack(tensor_list, dim=1)
        bboxes = [bboxes.squeeze()]
    else:
        bboxes = tensor_list
    return bboxes


def prepare_mask(input_image, image_np, pred_masks, text_output, color_history):
    save_img = None
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue
        pred_mask = pred_mask.detach().cpu().numpy()
        mask_list = [pred_mask[i] for i in range(pred_mask.shape[0])]
        
        i = 0
        if len(mask_list) > 0:
            save_img = image_np.copy()
            colors_temp = deepcopy(colors)
            seg_count = text_output.count("[SEG]")
            mask_list = mask_list[-seg_count:]
            for curr_mask in mask_list:
                if g_color_index is not None:
                    color = colors_temp[g_color_index[i%len(g_color_index)]]
                    i += 1
                else:
                    color = random.choice(colors_temp)
                    colors_temp.remove(color)
                    if len(colors_temp) == 0:
                        colors_temp = deepcopy(colors)
                color_history.append(color)
                curr_mask = curr_mask > 0
                save_img[curr_mask] = (image_np * 0.5 + curr_mask[:, :, None].astype(np.uint8) * np.array(color) * 0.5)[
                    curr_mask]
    seg_mask = np.zeros((curr_mask.shape[0], curr_mask.shape[1], 3), dtype=np.uint8)
    seg_mask[curr_mask] = [255, 255, 255]  # white for True values
    seg_mask[~curr_mask] = [0, 0, 0]  # black for False values
    seg_mask = Image.fromarray(seg_mask)
    mask_path = os.path.join(args.mask_save_dir, input_image.split('/')[-1])
    seg_mask.save(mask_path)

    return save_img

def _inference(input_str, all_inputs, follow_up):
    global conv, conv_history
    
    bbox_img = all_inputs['boxes']
    input_image = all_inputs['image']

    input_str = input_str.replace('&lt;', '<').replace('&gt;', '>')
    prompt = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n" + input_str

    if model.config.mm_use_im_start_end:
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        if model.config.mm_use_prompt_start_end:
            replace_token += (DEFAULT_PROMPT_START_TOKEN + DEFAULT_PROMPT_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    if not follow_up or conv is None:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        conv_history = {'user': [], 'model': []}
        conv_history["user"].append(input_str)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
    else:
        conv.append_message(conv.roles[0], input_str)
        conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(input_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_np.shape[:2]
    original_size_list = [image_np.shape[:2]]

    # Prepare input for Global Image Encoder
    global_enc_image = global_enc_processor.preprocess(
        image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
    global_enc_image = global_enc_image.bfloat16()

    # Prepare input for Grounding Image Encoder
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    grounding_enc_image = (grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).
                                                   contiguous()).unsqueeze(0).cuda())
    grounding_enc_image = grounding_enc_image.bfloat16()

    # Prepare input for Region Image Encoder
    post_h, post_w = global_enc_image.shape[1:3]
    bboxes = None
    if len(bbox_img) > 0:
        bboxes = region_enc_processor((orig_h, orig_w), (post_h, post_w), bbox_img)

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    # Pass prepared inputs to model
    output_ids, pred_masks = model.evaluate(
        global_enc_image, grounding_enc_image, input_ids, resize_list, original_size_list, max_tokens_new=512,
        bboxes=bboxes)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    # For multi-turn conversation
    conv.messages.pop()
    conv.append_message(conv.roles[1], text_output)
    conv_history["model"].append(text_output)

    color_history = []
    save_img = None
    if "[SEG]" in text_output:
        save_img = prepare_mask(input_image, image_np, pred_masks, text_output, color_history)

    output_str = text_output
    if save_img is not None:
        output_image = save_img
    else:
        if len(bbox_img) > 0:
            output_image = draw_bbox(image_np.copy(), bbox_img)
        else:
            output_image = image_np

    if args.markdown:
        markdown_str = process_markdown(output_str, color_history)
    else:
        markdown_str = None

    return output_image, output_str, markdown_str


def inference(input_str, input_image, follow_up=False, save_suffix=None):
    try:
        output_image, output_str, markdown_str = _inference(input_str, {'image': input_image, 'boxes': []}, follow_up=follow_up)
        output_image = Image.fromarray(output_image)
        image_name = input_image.split('/')[-1]
        image_name = image_name.replace('.jpg', f'_{save_suffix}.jpg') if save_suffix else image_name
        output_image_path = os.path.join(args.vis_save_dir, image_name)
        output_image.save(output_image_path)
        
        if args.interactive:
            print(output_str + '\n')

        if markdown_str is not None:
            with open(output_image_path.replace('.jpg', '.md'), 'w') as f:
                f.write(output_str + '\n')
                f.write(markdown_str + '\n')
        
        return output_str
    except Exception as e:
        print(e)


def interactive_mode():
    while True:
        input_str = input("Enter Text Instruction: ")
        if input_str == "exit" or input_str == "quit":
            break
        input_image = input("Enter Image Path: ")
        folow_up = input("Is this a follow-up? (y/n): ")
        follow_up = True if folow_up == 'y' else False
        color_index = input("Enter color index (optional): (r for random, 0,1,2 for specific colors): ")
        
        global g_color_index
        if color_index == 'r':
            g_color_index = None
        elif color_index != '':
            try:
                g_color_index = color_index.split(',')
                g_color_index = [int(i) for i in g_color_index if int(i) in range(len(colors))]
            except:
                g_color_index = range(len(colors))
        
        print("input_str: ", input_str, "input_image: ", input_image)
        inference(input_str, input_image, follow_up)


def custom_collate_fn(batch):
    return {'image': batch}


def inference_mode(parallel=False):
    """
    Inference mode for GranDf and MGSC datasets
    """
    def _get_prompt(prompt_type='root'):
        if prompt_type == 'root':
            # prompt = "Please provide a detailed description of all the objects present in this image in a comprehensive format."
            prompt = "Could you please give me a detailed description of the image?"
        elif prompt_type == 'subnode':
            prompt = "Can you provide a detailed description in a comprehensive format for {label} in the image."
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        
        prompt = prompt + " Please respond with interleaved segmentation masks for the corresponding parts of the answer."
        return prompt
    
    if parallel:
        print("Using MGSC dataset")
        anno_dir = args.mgsc_annotation_dir
        image_dir = args.mgsc_image_dir
        image_files = os.listdir(anno_dir)
        image_files = [file.replace('mglmm_', '').replace('.json', '.jpg') for file in image_files]
        image_files = [os.path.join(image_dir, file) for file in image_files]
        dataset = ImageDDP('', image_files)
    else:
        print("Using GranDf dataset")
        image_dir = args.grandf_image_dir
        dataset = ImageDDP(image_dir)
    
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False) if parallel else None
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, sampler=distributed_sampler, collate_fn=custom_collate_fn)

    for batch in tqdm(dataloader):
        input_image = batch['image']
        assert len(input_image) == 1
        input_image = input_image[0]
        output_str = inference(_get_prompt(prompt_type='root'), input_image)
        
        if args.infer_mode == 'subnode':
            count = 0
            root_output = output_str
            while '<p>' in root_output and '</p>' in root_output:
                p_start, p_end = root_output.find('<p>'), root_output.find('</p>')
                label = root_output[p_start+3:p_end]
                prompt = _get_prompt(prompt_type='subnode').format(label=label)
                inference(prompt, input_image, save_suffix='subnode_{}'.format(count))
                
                count += 1
                root_output = root_output[p_end+4:]


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = process_args(args)
    os.makedirs(args.vis_save_dir, exist_ok=True)
    os.makedirs(args.mask_save_dir, exist_ok=True)

    tokenizer = setup_tokenizer(args)
    model = initialize_model(args, tokenizer)
    model = prepare_for_inference(model, tokenizer, args)

    # Transfer the model to GPU
    # Replace with model = model.float().cuda() for 32 bit inference
    model = model.cuda()
    model.eval()

    global_enc_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    conv = None
    # Only to Display output
    conv_history = {'user': [], 'model': []}

    if args.interactive:
        g_color_index = None
        interactive_mode()
    else:
        g_color_index = None
        args.markdown = True
        if args.world_size > 1:
            init_distributed_mode(args)
            inference_mode(parallel=True)
        else:
            inference_mode(parallel=False)
