import os
import time
import json
import random
import torch
import numpy as np
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.oss_dataset import OSSDataset, oss_get_file


class PanopticApi(object):
    def __init__(self, ann_file, dataset, split) -> None:
        print("loading dataset {} into memory...".format(dataset))
        tic = time.time()

        data = json.load(oss_get_file(ann_file))
        self.data = {}
        self.data["dataset"] = dataset
        self.data["split"] = split
        self.data["images"] = data["images"]
        self.data["annotations"] = data["annotations"]
        self.data["categories"] = data["categories"]

        # create index
        self.createIndex()
        print("DONE (t=%.2fs)" % (time.time() - tic))

    def createIndex(self):
        print("creating index...")
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data["annotations"]:
            Anns[ann["id"]] = ann
            imgToAnns[ann["image_id"]] = imgToAnns.get(ann["image_id"], []) + [ann]
        for img in self.data["images"]:
            Imgs[img["id"]] = img
        for cat in self.data["categories"]:
            Cats[cat["id"]] = cat["name"]
        
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.imgToAnns = imgToAnns

        print("index created!")

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == str:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]
    
    def getAnnByImg(self, img_id):
        return self.imgToAnns.get(img_id, None)


class PanopticSegmDataset(OSSDataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    CATEGORY_REMAP = {
        'window-blind': 'blind window',
        'wall-wood': 'wooden wall',
        'wall-tile': 'tiled wall',
        'wall-stone': 'stone wall',
        'wall-brick': 'brick wall',
        'mirror-stuff': 'mirror stuff',
        'floor-wood': 'wooden floor',
        'door-stuff': 'door stuff'
    }

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=80000, num_classes_per_sample=3,
                 precision: str = "fp32", image_size: int = 224, validation=False, inference=False, random_sampling=True):
        super(PanopticSegmDataset, self).__init__()

        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.epoch_samples = epoch_samples
        self.precision = precision
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.validation = validation
        self.inference = inference
        self.random_sampling = random_sampling
        self.num_classes_per_sample = num_classes_per_sample

        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""

        self.split = 'val' if validation else 'train'
        self.panoptic_json_path = os.path.join(dataset_dir, 'cocopanoptic/annotations', f'panoptic_{self.split}2017_detection_format.json')
        self.panoptic_image_path = os.path.join(dataset_dir, f'coco/{self.split}2017')
        
        self.panoptic_api = PanopticApi(self.panoptic_json_path, 'coco', self.split)
        self.image_infos = self.panoptic_api.data['images']
        self.image_infos = self.image_infos[:1000] if (validation and not inference) else self.image_infos

        print('\033[92m' + "----SEGM-: Panoptic dataset initialized----" + '\033[0m')
        print('\033[92m' + "----SEGM-: Number of samples: {}----".format(len(self.image_infos)) + '\033[0m')

    def __len__(self):
        return len(self.image_infos)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, labels):
        question = "Can you segment all the objects in the image?"
        if len(labels) == 1:
            answer = "Sure, the image contains <p> {class_name} </p> [SEG].".format(class_name=labels[0])
        else:
            target_answers = ["<p> {class_name} </p> [SEG]".format(class_name=label) for label in labels]
            answer = "Sure, the image contains " + ', '.join(target_answers[:-1]) + ' and ' + target_answers[-1] + '.'
        
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        question = self.begin_str + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        return None, conversations
    
    def get_renamed_labels(self, category_ids):
        labels = []
        for category_id in category_ids:
            category = self.panoptic_api.loadCats(category_id)[0]
            label = self.CATEGORY_REMAP.get(category, category)
            label = label.split('-')[0]
            labels.append(label)

        # rename the labels if there are duplicates
        count = 0
        renamed_labels = []
        for i in range(len(labels)):
            if count > 0 and labels[i] == renamed_labels[-1].split('-')[0]:
                count += 1
                renamed_labels.append(labels[i] + '-' + str(count))
            elif i < len(labels) - 1 and labels[i] == labels[i + 1]:
                count = 1
                renamed_labels.append(labels[i] + '-' + str(count))
            else:
                count = 0
                renamed_labels.append(labels[i])

        return renamed_labels

    def __getitem__(self, idx):
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, self.__len__() - 1)
        image_info = self.image_infos[idx]
        anns = self.panoptic_api.getAnnByImg(image_info['id'])
        
        if anns is None:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        image_path = os.path.join(self.panoptic_image_path, image_info['file_name'])
        image = self.oss_load_img_cv2(image_path)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        selected_masks, selected_categories = [], []
        for ann in anns:
            rle = ann.get("segmentation")
            if rle:
                m = mask.decode(rle)
                m = m.astype(np.uint8)
                selected_masks.append(m)
                selected_categories.append(ann['category_id'])

        # sort the labels by their name
        shorted_idx = np.argsort(selected_categories)
        selected_categories = [selected_categories[i] for i in shorted_idx]
        selected_masks = [selected_masks[i] for i in shorted_idx]

        renamed_labels = self.get_renamed_labels(selected_categories)

        masks = np.stack(selected_masks, axis=0)
        masks = torch.from_numpy(masks)

        questions, conversations = self.create_conversations(renamed_labels)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        bboxes = None
        selected_labels = [self.panoptic_api.loadCats(category_id)[0] for category_id in selected_categories]

        return (
        image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
        questions, selected_labels)