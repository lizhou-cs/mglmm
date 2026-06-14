import os
import json
import random
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import CAPTION_QUESTIONS
from dataset.utils.oss_dataset import OSSDataset


class DenselyCaptionedDataset(OSSDataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=10000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True, extra_caption=True):
        super(DenselyCaptionedDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample
        self.validation = validation
        self.random_sampling = random_sampling
        self.extra_caption = extra_caption

        # Defining paths
        self.base_dir = os.path.join(dataset_dir, "DCI")
        self.image_folder = os.path.join(dataset_dir, "SegmentAnything/imgs")
        annotations_file_path = os.path.join(self.base_dir, "annotations")
        self.data_infos = self._load_annotations(annotations_file_path)
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        print('\033[92m' + "----CAP-: Densely Captioned Image dataset initialized----" + '\033[0m')
        print('\033[92m' + "----CAP-: Number of samples: {}----".format(len(self.data_infos)) + '\033[0m')

    def _load_annotations(self, ann_file):
        data_infos = self.oss_list_dir(ann_file, suffix=".json")
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def __getitem__(self, idx):
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, self.__len__() - 1)
        data_info = self.data_infos[idx]
        json_content = json.load(self.oss_get_file(data_info))
        data = {
                "image": json_content["image"],
                "short_caption": json_content["short_caption"],
                "extra_caption": json_content["extra_caption"],
            }
        
        processed_data = self.process_data(data)
        return processed_data
    
    def create_conversations(self, labels):
        conversations = []
        questions = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []

        question = random.choice(CAPTION_QUESTIONS).strip()
        answer = labels

        conv.append_message(conv.roles[0], self.begin_str + question)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        conversations.append(prompt)
        return questions, conversations

    def process_data(self, data_item):
        caption = data_item['extra_caption'] if self.extra_caption else data_item['short_caption']
        image_path = os.path.join(self.image_folder, data_item['image'])
        image = self.oss_load_img_cv2(image_path)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Skip input for Grounding Image Encoder
        grounding_enc_image = None
        image_resize = None

        masks, bboxes = None, None

        questions, conversations = self.create_conversations(caption)
        label = None
        selected_labels = [caption]

        return (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize,
                questions, selected_labels)
