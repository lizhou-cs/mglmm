import os
import re
import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from model.llava import conversation as conversation_lib
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, \
                DEFAULT_PROMPT_SPLIT_TOKEN, DEFAULT_PROMPT_END_TOKEN, DEFAULT_PROMPT_START_TOKEN


def grounding_image_ecoder_preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                                      pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
                                      img_size=1024) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""

    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))

    return x


def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out


def mask_to_rle_numpy(mask: np.ndarray):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    h, w = mask.shape

    # Put in fortran order and flatten h,w
    mask = np.transpose(mask).flatten()

    # Compute change indices
    diff = mask[1:] ^ mask[:-1]
    change_indices = np.where(diff)[0]

    # Encode run length
    cur_idxs = np.concatenate(
        ([0], change_indices + 1, [h * w])
    )
    btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
    counts = [] if mask[0] == 0 else [0]
    counts.extend(btw_idxs.tolist())

    return {"size": [h, w], "counts": counts}


def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    return iou


def bbox_to_x1y1x2y2(bbox):
    x1, y1, w, h = bbox
    bbox = [x1, y1, x1 + w, y1 + h]

    return bbox


def preprocess_instruction(instruction, args):
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
    prompt = begin_str + instruction
    if args.mm_use_im_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if args.mm_use_prompt_start_end:
            replace_token += (DEFAULT_PROMPT_START_TOKEN + DEFAULT_PROMPT_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    return prompt


def preprocess_image(image_path, clip_processpr, transform, bbox=None):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        image_np = image_np[y1:y2, x1:x2]

    original_size_list = [image_np.shape[:2]]
    image_clip = (clip_processpr.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0))

    # Preprocess the image (Grounding image encoder)
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (grounding_image_ecoder_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0))

    return image_clip, image, original_size_list, resize_list


def postprocess_text(text):
    text = text.replace("\n", "").replace("  ", " ")
    ans = text.split("ASSISTANT: ")[-1]

    # Remove tags
    cleaned_text = re.sub(r'<.*?>', '', ans)

    # Remove the [SEG] token
    cleaned_text = cleaned_text.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_text = ' '.join(cleaned_text.split()).strip("'")
    cleaned_text = cleaned_text.strip()

    return ans, cleaned_text


def save_metrics_to_json(metrics, output_file_path):
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as json_file:
            result_list = json.load(json_file)
    else:
        result_list = []
    result_list.append(metrics)

    with open(output_file_path, 'w') as json_file:
        json.dump(result_list, json_file, indent=2)
