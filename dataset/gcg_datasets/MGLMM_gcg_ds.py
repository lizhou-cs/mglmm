import os
import json
import base64
import random
import warnings
import torch
import numpy as np
import torch.nn.functional as F
import pycocotools.mask as maskUtils
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import GCG_QUESTIONS
from dataset.utils.oss_dataset import OSSDataset


def decode_mask(mask: str) -> np.ndarray:
    """
    mask format: str, encoded mask
    return format: np.ndarray, mask
    """
    if mask is None or mask == 'None':
        return None
    mask = json.loads(mask)
    mask['counts'] = base64.b64decode(mask['counts'].encode("utf-8"))
    mask = maskUtils.decode(mask)
    return np.array(mask)


class MGLMMDataset(OSSDataset):
    """
    Dataset Class for Grounded Conversation Generation (GCG) proposed in GLaMM.
    """
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, mglmm_data='sam_new', validation=False, random_sampling=True):
        super(MGLMMDataset, self).__init__()
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.validation = validation
        self.random_sampling = random_sampling

        self.question_templates = GCG_QUESTIONS
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""

        # Defining paths
        if mglmm_data == 'coco':
            base_dir = 'COCO_MGLMM_gcg'
            image_dir = 'coco/train2017'
        elif mglmm_data == 'sam':
            base_dir = 'MGLMM_gcg'
            image_dir = 'SegmentAnything/imgs'
        elif mglmm_data == 'sam_new':
            base_dir = 'MGLMM_gcg_new'
            image_dir = 'SegmentAnything/imgs'
        else:
            raise ValueError(f"Unsupported dataset {mglmm_data}")
        
        ann_file_name = "MGLMM_val_split.txt" if self.validation else "MGLMM_train_split.txt"

        self.base_dir = os.path.join(dataset_dir, base_dir)
        self.image_dir = os.path.join(dataset_dir, image_dir)
        self.data_infos = self._load_annotations(os.path.join(self.base_dir, ann_file_name))

        print('\033[92m' + "----MGLMM GCG-: Number of samples: {}----".format(len(self.data_infos)) + '\033[0m')

    def _load_annotations(self, ann_file_path):
        ann_file_list = [line for line in self.oss_get_file(ann_file_path).read().splitlines()]
        data_infos = []
        image_level_infos = []
        
        for ann_file in ann_file_list:
            ann_file = os.path.join(self.base_dir, "annotations", ann_file.decode())
            f = self.oss_get_file(ann_file)
            if f is not None:
                image_level_infos.append(json.loads(f.read()))

        for ann_list in image_level_infos:
            for anno in ann_list:
                if anno['gcg_caption']:
                    data_infos.append(anno)

        data_infos = data_infos[0: 1000] if self.validation else data_infos

        return data_infos

    def _parse_annotations(self, ann_info):
        annotations = {'label': ann_info['label'], 'caption': [], 'masks': [], 'file_name': ann_info['file_name'], "selected_labels": []}

        gcg_caption = ann_info['gcg_caption'].strip('"').strip()
        for label in ann_info['gcg_labels']:
            # Convert segmentation to binary mask
            annotations['masks'].append(decode_mask(ann_info[label]['mask']))
            annotations['selected_labels'].append(ann_info[label]['label'])
  
        annotations['gcg_labels'] = ann_info['gcg_labels']
        annotations['caption'] = gcg_caption
        return annotations

    def __getitem__(self, idx):
        while True:
            idx = idx if (self.validation or not self.random_sampling) else random.randint(0, self.__len__() - 1)
            data_info = self.data_infos[idx]
            # Parse annotation info
            try:
                ann = self._parse_annotations(data_info)
            except Exception as e:
                import traceback
                traceback.print_exc()
                warnings.warn(f"Error parsing annotation for image {data_info['file_name']}. Trying another image.")
                idx = random.randint(0, self.__len__() - 1)
                continue

            image_path = os.path.join(self.image_dir, ann['file_name'])
            if len(ann['label']) > 0 and len(ann['masks']) > 0:
                break
            else:
                warnings.warn(f"The image {image_path} does not have any labels. Trying another image.")
                idx = random.randint(0, self.__len__() - 1)

        data_item = {"image_path": image_path, 
                     "filename": ann['file_name'], 
                     "caption": ann['caption'], 
                     "label": ann['label'], 
                     "gcg_labels": ann['gcg_labels'], 
                     "masks": ann['masks'],
                     "selected_labels": ann['selected_labels']}
        return self.process_data(data_item)

    def __len__(self):
        return len(self.data_infos)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, caption, labels, node_name=None):
        # Prepare caption with tags
        def tag_caption(caption, labels):
            for label in labels:
                caption = caption.replace(f"<{label}>", "<p>")
                caption = caption.replace(f"</{label}>", "</p> [SEG]")
            return caption

        detailed_answer = tag_caption(caption, labels)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []

        if node_name == "root":
            question = "Please provide a detailed description of all the objects present in this image in a comprehensive format."
        else:
            label = node_name.split(":")[1].strip()
            question = "Can you provide a detailed description in a comprehensive format for {label} in the image.".format(label=label)

        question = question + " Please respond with interleaved segmentation masks for the corresponding parts of the answer."

        conv.append_message(conv.roles[0], self.begin_str + question)
        conv.append_message(conv.roles[1], detailed_answer)
        conversations.append(conv.get_prompt())
        questions = [question]
        return questions, conversations

    def process_data(self, data_item):
        data_label = data_item['label']
        masks = data_item['masks']
        caption = data_item['caption']
        gcg_labels = data_item['gcg_labels']
        image_path = data_item['image_path']
        selected_labels = data_item['selected_labels']
        
        image = self.oss_load_img_cv2(image_path)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        bboxes = None

        questions, conversations = self.create_conversations(caption, gcg_labels, node_name=data_label)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1:], dtype=torch.long) * self.IGNORE_LABEL

        return (
        image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize, questions,
        selected_labels)


def color_generator():
    """
    This is a generator that yields colors in BGR format.
    It loops through a set of predefined colors and also
    yields randomly generated colors when the predefined ones are exhausted.
    """
    # Predefined colors in BGR format
    colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [255, 0, 255],    # Magenta
        [255, 192, 203],  # Pink
        [165, 42, 42],    # Brown
        [255, 165, 0],    # Orange
        [128, 0, 128],     # Purple
        [0, 0, 128],       # Navy
        [128, 0, 0],      # Maroon
        [128, 128, 0],    # Olive
        [70, 130, 180],   # Steel Blue
        [173, 216, 230],  # Light Blue
        [255, 192, 0],    # Gold
        [255, 165, 165],  # Light Salmon
        [255, 20, 147],   # Deep Pink
    ]
    for color in itertools.cycle(colors):
        yield color

def draw_masks_on_image(image, masks, resize_factor=1, alpha=0.5):
    """
    Draw masks on an image.

    :param image: numpy array of the image
    :param masks: list of numpy arrays representing masks
    :param color: tuple with color to draw the masks, default is green
    :param alpha: float representing the transparency of the overlay, default is 0.5
    :return: image with drawn masks
    """
    # Make a copy of the original image to draw on
    draw_image = image.copy()

    # Resize the image if needed
    if resize_factor != 1:
        draw_image = cv2.resize(draw_image, (0, 0), fx=resize_factor, fy=resize_factor)
        # Update the mask coordinates
        masks = [cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor) for mask in masks]

    # Iterate over all masks and add them as an overlay
    for mask in masks:
        # Ensure the mask is boolean array
        mask = mask > 0
        color = next(color_gen)
        draw_image[mask] = (draw_image * 0.5 + mask[:, :, None].astype(np.uint8) * np.array(color) * 0.5)[mask]

    return draw_image

def show_image_with_mask(image, masks, resize_factor=1):
    image_with_masks = draw_masks_on_image(image, masks, resize_factor=resize_factor)
    cv2.imshow('Image with masks', image_with_masks)
    key = cv2.waitKeyEx(0)
    if key == ord('q'):
        exit(0)
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import cv2
    import itertools
    from dataset.utils.oss_dataset import oss_get_file

    global_enc_image = "openai/clip-vit-large-patch14-336"
    color_gen = color_generator()

    basedataset = MGLMMDataset(dataset_dir="./data", tokenizer="tokenizer",
                                   global_image_encoder=global_enc_image, random_sampling=False)
    for i, data in enumerate(basedataset):
        (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
        questions, selected_labels) = data
        image_bytes = oss_get_file(image_path).read()        
        
        print(conversations)
        answer = conversations[0]['Assistant']
        
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        show_image_with_mask(image, masks.numpy())

        for mask in masks.numpy():
            p_start, p_end = answer.find('<p>'), answer.find('</p>')
            print(answer[p_start: p_end+4])
            show_image_with_mask(image, [mask])
            answer = answer[p_end+4:]
