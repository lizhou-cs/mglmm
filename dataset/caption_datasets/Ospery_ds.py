import os
import re
import json
import random
import torch
import numpy as np
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from dataset.pycocotools import mask as maskUtils
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import REGION_QUESTIONS
from dataset.utils.oss_dataset import OSSDataset


class OsperyDataset(OSSDataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True,
                 image_dir='', json_path=''):
        super(OsperyDataset, self).__init__()
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

        # template for questions
        self.question_templates = None
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""

        # Defining paths
        self.base_dir = os.path.join(dataset_dir, "osprey")
        self.image_folder = os.path.join(dataset_dir, image_dir)
        annotations_file_path = os.path.join(self.base_dir, json_path)
        self.data_infos = self._load_annotations(annotations_file_path)
        print('\033[92m' + "----CAP-: Number of samples: {}----".format(len(self.data_infos)) + '\033[0m')
    
    def _load_annotations(self, ann_file):
        data_infos = json.load(self.oss_get_file(ann_file))
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def _parse_annotation(self, ann_info):
        assert len(ann_info['conversations'])%2 ==0, "annotation must be in pairs (Human vs Assistant)"
        
        questions, answers, masks, bboxes = [], [], [], []
        image_path = os.path.join(self.image_folder, ann_info['file_name'])
        
        region_num = len(ann_info['annotation'])
        str_region = ""
        for i in range(region_num):
            if i > 0:
                str_region += ','
            str_region += "region" + str(i+1) + "<mask><pos>"

            masks.append(ann_info['annotation'][i]['segmentation'])
            bboxes.append(ann_info['annotation'][i]['bbox'])

        for i in range(len(ann_info['conversations'])//2):
            question = ann_info['conversations'][i*2]['value']
            question = question.replace('<','').replace('>','')

            # prompt the number of regions in the first question
            if i == 0:
                if region_num==1:
                    region_prompt = "Ther are 1 part region in the picture: " + str_region + '. '
                else:
                    region_prompt = "Ther are {} part regions in the picture: ".format(str(region_num)) + str_region + '. '
                question = region_prompt + question

            questions.append(question + self.limit)

            answer = ann_info['conversations'][i*2+1]['value']
            answer = answer.replace('<','').replace('>','')
            answers.append(answer)

        annotations = {
            "image_path": image_path,
            "questions": questions,
            "answers": answers,
            "masks": masks,
            "bboxes": bboxes,
            "height": ann_info["height"],
            "width": ann_info["width"],
        }
        return annotations

    def _ann_to_mask(self, mask_ann, h, w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, self.__len__() - 1)
        data_info = self.data_infos[idx]
        
        annotations = self._parse_annotation(data_info)
        processed_data = self.process_data(annotations)
        return processed_data
    
    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x
    
    def create_conversations(self, questions, answers):
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

    def process_data(self, data_item):
        image_path = data_item["image_path"]
        image = self.oss_load_img_cv2(image_path)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Skip input for Grounding Image Encoder
        grounding_enc_image = None
        image_resize = None

        height, width = data_item["height"], data_item["width"]
        masks = [self._ann_to_mask(mask, height, width) for mask in data_item["masks"]]
        masks = torch.from_numpy(np.stack(masks, axis=0))
        bboxes = data_item["bboxes"]

        questions, conversations = self.create_conversations(data_item["questions"], data_item["answers"])
        label = None
        selected_labels = conversations

        return (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
                questions, selected_labels)


class OsperyConversationDataset(OsperyDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=10000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):
        
        image_dir = 'coco/train2014/'
        json_path = "less/osprey_conversation_100.json"
        self.limit = ''
        super(OsperyConversationDataset, self).__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision,
            image_size, num_classes_per_sample, validation, random_sampling,
            image_dir=image_dir, json_path=json_path)
        
        print('\033[92m' + "----CAP-: Osprey Conversation dataset initialized----" + '\033[0m')
        print(f'\033[31m----This is a test example, please replace correct json file for training ----\033[0m')


class OspreyPartLevelDataset(OsperyDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=10000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):
        
        image_dir = 'coco/train2017/'
        json_path = "less/osprey_part_level_100.json"
        self.limit = ' Answer the question using a single word or phrase.'
        super(OspreyPartLevelDataset, self).__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision,
            image_size, num_classes_per_sample, validation, random_sampling,
            image_dir=image_dir, json_path=json_path)
        
        print('\033[92m' + "----CAP-: Osprey PartLevel dataset initialized----" + '\033[0m')
        print(f'\033[31m----This is a test example, please replace correct json file for training ----\033[0m')


class OspreyLVISPosNegDataset(OsperyDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=10000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):
        
        image_dir = 'coco/train2017/'
        json_path = "less/osprey_lvis_positive_negative_100.json"
        self.limit = ''
        super(OspreyLVISPosNegDataset, self).__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision,
            image_size, num_classes_per_sample, validation, random_sampling,
            image_dir=image_dir, json_path=json_path)
        
        print('\033[92m' + "----CAP-: Osprey dataset initialized----" + '\033[0m')
        print(f'\033[31m----This is a test example, please replace correct json file for training ----\033[0m')

    def _parse_annotation(self, ann_info):
        assert len(ann_info['conversations'])%2 ==0, "annotation must be in pairs (Human vs Assistant)"
        questions, answers, masks, bboxes = [], [], [], []
        image_path = os.path.join(self.image_folder, ann_info['file_name'])

        region_num = len(ann_info['annotation'])
        for i in range(region_num):
            masks.append(ann_info['annotation'][i]['segmentation'])
            bboxes.append(ann_info['annotation'][i]['bbox'])

        for i in range(len(ann_info['conversations'])//2):
            question = ann_info['conversations'][i*2]['value']
            question = re.sub(r'<region\d+>', '<mask><pos>', question)
            questions.append(question)    
            answer = ann_info['conversations'][i*2+1]['value']
            answers.append(answer)

        annotations = {
            "image_path": image_path,
            "questions": questions,
            "answers": answers,
            "masks": masks,
            "bboxes": bboxes,
            "height": ann_info["height"],
            "width": ann_info["width"],
        }
        return annotations


class OspreyShortFormDataset(OsperyDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=10000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):
        
        image_dir = 'coco/train2017/'
        json_path = "less/osprey_short_form_100.json"
        self.limit = ' Answer the question using a single word or phrase.'
        super(OspreyShortFormDataset, self).__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision,
            image_size, num_classes_per_sample, validation, random_sampling,
            image_dir=image_dir, json_path=json_path)
        
        print('\033[92m' + "----CAP-: Osprey ShortForm dataset initialized----" + '\033[0m')
        print(f'\033[31m----This is a test example, please replace correct json file for training ----\033[0m')


class OspreyDetailedDescriptionDataset(OsperyDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=10000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):
        
        image_dir = 'coco/train2014/'
        json_path = "less/osprey_detail_description_100.json"
        self.limit = ''
        super(OspreyDetailedDescriptionDataset, self).__init__(
            dataset_dir, tokenizer, global_image_encoder, epoch_samples, precision,
            image_size, num_classes_per_sample, validation, random_sampling,
            image_dir=image_dir, json_path=json_path)
        
        print('\033[92m' + "----CAP-: Osprey dataset initialized----" + '\033[0m')
        print(f'\033[31m----This is a test example, please replace correct json file for training ----\033[0m')

    def _parse_annotation(self, ann_info):
        questions, answers, masks, bboxes = [], [], [], []
        image_path = os.path.join(self.image_folder, ann_info['file_name'])

        region_num = len(ann_info['annotation'])
        for i in range(region_num):
            masks.append(ann_info['annotation'][i]['segmentation'])
            bboxes.append(ann_info['annotation'][i]['bbox'])

            question = random.choice(REGION_QUESTIONS)
            question = question.replace('<region>', '<mask><pos>')
            questions.append(question)
            answer = re.findall(r"<.*>:\ (.*)", ann_info['description'][i])[0]
            answers.append(answer)

        annotations = {
            "image_path": image_path,
            "questions": questions,
            "answers": answers,
            "masks": masks,
            "bboxes": bboxes,
            "height": ann_info["height"],
            "width": ann_info["width"],
        }
        return annotations
