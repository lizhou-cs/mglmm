import os
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import ANSWER_LIST, SEG_QUESTIONS
from dataset.utils.oss_dataset import OSSDataset


class GrandReferSegmDataset(OSSDataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224, num_classes_per_sample: int = 3, 
                 max_gt_per_img=10, validation=False, random_sampling=True):
        super(GrandReferSegmDataset, self).__init__()
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.max_gt_per_img = max_gt_per_img
        self.validation = validation
        self.random_sampling = random_sampling

        self.question_templates = SEG_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""

        # Defining paths
        self.base_dir = os.path.join(dataset_dir, "GranD")
        self.anno_dir = 'referring_expression'
        self.image_folder = os.path.join(dataset_dir, 'SegmentAnything', "imgs")
        
        self.data_infos = self._load_annotations(os.path.join(self.base_dir, 'anno_files.txt'))
        print('\033[92m' + "----SEGM-: GranD Referring Segm dataset initialized----" + '\033[0m')
        print('\033[92m' + "----SEGM-: Number of samples: {}----".format(len(self.data_infos)) + '\033[0m')

    def _load_annotations(self, ann_file):
        lines = self.oss_get_file(ann_file).read().decode('utf-8').split('\n')
        data_infos = [line.strip() for line in lines if line.strip()]
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, labels):
        questions = []
        answers = []
        for i, label in enumerate(labels):
            label = label.strip()
            question_template = random.choice(self.question_templates)
            questions.append(question_template.format(class_name=label.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        return questions, conversations

    def _parse_annotations(self, ann_info):
        annotations = {'masks': [], 'labels': []}
        for ann in ann_info:
            rle = ann.get("segmentation")
            if rle:
                m = mask.decode(rle)
                m = m.astype(np.uint8)
                annotations['masks'].append(m)
                annotations['labels'].append(ann['attribute'])
        
        return annotations

    def __getitem__(self, idx):
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, self.__len__() - 1)
        json_name = self.data_infos[idx]
        image_name = json_name.replace('.json', '.jpg')
        image_path = os.path.join(self.image_folder, image_name)
        # Get the annotation
        json_file = os.path.join(self.base_dir, self.anno_dir, json_name)
        json_contents = json.load(self.oss_get_file(json_file))
        ann_info = json_contents[image_name]
        ann = self._parse_annotations(ann_info)
        data_item = {"image_path": image_path,
                     "filename": image_name,
                     "labels": ann['labels'], 
                     "masks": ann['masks']}

        return self.process_data(data_item)

    def process_data(self, data_item):
        data_labels = data_item['labels']
        data_masks = data_item['masks']

        image_path = data_item['image_path']
        image = self.oss_load_img_cv2(image_path)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # Prepare input for Segmentation module
        assert len(data_labels) > 0 

        if len(data_labels) > self.max_gt_per_img:
            shuffle_ids = torch.randperm(len(data_labels))
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]            
            selected_labels = [data_labels[i] for i in shuffle_ids]
            selected_masks = [data_masks[i] for i in shuffle_ids]
        else:
            selected_labels, selected_masks = data_labels, data_masks

        masks = np.stack(selected_masks, axis=0)
        masks = torch.from_numpy(masks)

        questions, conversations = self.create_conversations(selected_labels)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        bboxes = None

        return (
        image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
        questions, selected_labels)
