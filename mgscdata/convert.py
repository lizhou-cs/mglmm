"""
Convert MGSCData to GCG evaluation format
"""

import io
import os
import re
import json
import copy
import base64
import argparse
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as maskUtils


OSS_ENV_KEYS = ("OSS_ACCESS_ID", "OSS_ACCESS_KEY", "OSS_BUCKET", "OSS_ENDPOINT")


def _load_oss2():
    try:
        import oss2
    except ImportError as exc:
        raise ImportError("Install optional dependency 'oss2' to read MGSC images from Aliyun OSS.") from exc
    return oss2


def _get_oss_bucket():
    missing_keys = [key for key in OSS_ENV_KEYS if not os.environ.get(key)]
    if missing_keys:
        raise RuntimeError("Missing OSS environment variables: {}".format(", ".join(missing_keys)))
    oss2 = _load_oss2()
    auth = oss2.Auth(os.environ['OSS_ACCESS_ID'], os.environ['OSS_ACCESS_KEY'])
    return oss2.Bucket(auth, os.environ['OSS_ENDPOINT'], os.environ['OSS_BUCKET'])


def download_image_from_oss(image_url, save_path):
    _get_oss_bucket().get_object_to_file(image_url, save_path)


def oss_get_file(oss_path):
    return _get_oss_bucket().get_object(oss_path)

def extract_image(data_item, cur_id):
    original_file_name = data_item['file_name']
    id = original_file_name.split('.')[0] + f"_{cur_id}"

    if args.use_oss:
        img_bytes = oss_get_file(os.path.join(args.sam_oss_dir, original_file_name)).read()
        pil_img = Image.open(io.BytesIO(img_bytes))
    else:
        pil_img = Image.open(os.path.join(args.image_source_dir, original_file_name))
    height, width = pil_img.size

    image = {
        "id": id,
        "width": width,
        "height": height,
        "type": data_item['label'],
        "file_name": f"{id}.jpg",
        "sam_file_name": original_file_name,
    }
    return image, pil_img


def extract_caption(data_item, image_id, caption_id):
    gcg_caption = data_item['gcg_caption']
    gcg_labels = data_item['gcg_labels']
    labels = [data_item[label]["label"] for label in gcg_labels]
    
    for label in gcg_labels:
        gcg_caption = gcg_caption.replace(f"<{label}>", "").replace(f"</{label}>", "")
    
    gcg_caption = re.sub(r'<.*>', '', gcg_caption)
    gcg_caption = re.sub(r'\s+', ' ', gcg_caption)

    assert '<' not in gcg_caption and '>' not in gcg_caption, f"Error: {gcg_caption} in {data_item['file_name']}"

    caption = {
        "caption": gcg_caption,
        "labels": labels,
        "image_id": image_id,
        "id": caption_id,
    }
    return caption


def extract_mask(data_item, image_id, mask_id):
    gcg_labels = data_item['gcg_labels']
    masks = []
    for label in gcg_labels:
        mask = data_item[label]["mask"]
        mask = json.loads(mask)
        mask["counts"] = base64.b64decode(mask["counts"].encode("utf-8")).decode("utf-8")
        area = int(maskUtils.area(mask))
        masks.append({
            "segmentation": mask,
            "iscrowd": 0,
            "area": area,
            "image_id": image_id,
            "category_id": 1,
            "id": mask_id,
        })
        mask_id += 1
    
    return masks


def main(data_file_list):
    data = []
    for data_file in data_file_list:
        with open(data_file, 'r') as f:
            _data = json.load(f)
        print("Loaded {} data items from {}".format(len(_data), data_file))
        data.extend(_data)
    
    root_gt = {
        "caption": {
            "images": [],
            "annotations": [],
        },
        "mask": {
            "images": [],
            "annotations": [],
            "categories": [{'id': 1, 'name': 'object'}],
        },
    }
    sub_gt = copy.deepcopy(root_gt)
    all_gt = copy.deepcopy(root_gt)

    cur_caption_id = 0
    cur_mask_id = 0
    for data_item in tqdm(data):
        image, pil_img = extract_image(data_item, cur_caption_id)
        caption = extract_caption(data_item, image_id=image['id'], caption_id=cur_caption_id)
        mask = extract_mask(data_item, image_id=image['id'], mask_id=cur_mask_id)
        
        cur_caption_id += 1
        cur_mask_id += len(mask)

        if data_item['type'] == 'root':
            root_gt['caption']['images'].append(image)
            root_gt['caption']['annotations'].append(caption)
            root_gt['mask']['images'].append(image)
            root_gt['mask']['annotations'].extend(mask)
        else:
            sub_gt['caption']['images'].append(image)
            sub_gt['caption']['annotations'].append(caption)
            sub_gt['mask']['images'].append(image)
            sub_gt['mask']['annotations'].extend(mask)
        
        all_gt['caption']['images'].append(image)
        all_gt['caption']['annotations'].append(caption)
        all_gt['mask']['images'].append(image)
        all_gt['mask']['annotations'].extend(mask)

        image_save_path = os.path.join(args.image_save_dir, image['file_name'])
        pil_img.save(image_save_path)
    
    print("Processed {} images, with root node {} and subtree {}".format(len(data), len(root_gt['caption']['images']), len(sub_gt['caption']['images'])))

    # save root_gt
    with open(f'{args.annotation_save_dir}/root_mgsc_caption_gt.json', 'w') as f:
        json.dump(root_gt['caption'], f)
    with open(f'{args.annotation_save_dir}/root_mgsc_mask_gt.json', 'w') as f:
        json.dump(root_gt['mask'], f)
    # save sub_gt
    with open(f'{args.annotation_save_dir}/subtree_mgsc_caption_gt.json', 'w') as f:
        json.dump(sub_gt['caption'], f)
    with open(f'{args.annotation_save_dir}/subtree_mgsc_mask_gt.json', 'w') as f:
        json.dump(sub_gt['mask'], f)
    # save all_gt
    with open(f'{args.annotation_save_dir}/all_mgsc_caption_gt.json', 'w') as f:
        json.dump(all_gt['caption'], f)
    with open(f'{args.annotation_save_dir}/all_mgsc_mask_gt.json', 'w') as f:
        json.dump(all_gt['mask'], f)


def parser_args():
    parser = argparse.ArgumentParser(description="Convert MGSCData to GCG evaluation format")
    parser.add_argument("--use_oss", action="store_true", help="Load source images from Aliyun OSS instead of local files")
    parser.add_argument("--sam_oss_dir", default='', type=str, help="OSS directory for SAM data")
    parser.add_argument("--image_source_dir", default='images', type=str, help="Local directory containing source images")
    parser.add_argument("--mgsc_annotations", default='annotations', type=str, help="Directory to MGSC annotations")
    parser.add_argument("--split_file", default=None, type=str, help="Path to MGLMM split file")
    parser.add_argument("--annotation_save_dir", default='evaluation/annotations', type=str, help="Directory to save annotations")
    parser.add_argument("--image_save_dir", default='evaluation/images', type=str, help="Directory to save images")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    os.makedirs(args.annotation_save_dir, exist_ok=True)
    os.makedirs(args.image_save_dir, exist_ok=True)

    split_file = args.split_file or os.path.join(args.mgsc_annotations, 'MGLMM_val_split.txt')
    with open(split_file, 'r') as f:
        val_file_list = f.read().splitlines()
        val_file_list = [os.path.join(args.mgsc_annotations, file) for file in val_file_list]

    main(val_file_list)
