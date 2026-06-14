import os
import cv2
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from dataset.utils.oss_dataset import OSSDataset, oss_get_file
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import ANSWER_LIST, SHORT_QUESTION_LIST, LONG_QUESTION_LIST, EXPLANATORY_QUESTION_LIST


def get_mask_from_json(json_path, img):
    anno = json.load(oss_get_file(json_path))

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence


class ReasonSegDataset(OSSDataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10, precision: str = "fp32", 
                 image_size: int = 224, num_classes_per_sample: int = 3, reason_seg_data="ReasonSeg", 
                 split='train', explanatory=-1, seg_token_num=1, validation=False, random_sampling=True):
        super(ReasonSegDataset, self).__init__()
        self.epoch_samples = epoch_samples
        self.explanatory = explanatory if not validation else -1
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.validation = validation
        self.random_sampling = random_sampling
        
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST 
        self.seg_token_num = seg_token_num
        
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""

        split = 'val' if split == 'train' and validation else split
        images = self.oss_list_dir(os.path.join(dataset_dir, "reason_seg", reason_seg_data, split), suffix='jpg')
        jsons = [path.replace(".jpg", ".json") for path in images]
        self.data_infos = (images, jsons)

        print(f'\033[92m----Reason SEG-{split}: Loaded {reason_seg_data} dataset ----\033[0m')
        print(f'\033[92m----Reason SEG-{split}: Number of samples is {len(images)} ----\033[0m')

        if self.explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            items = json.load(self.oss_get_file(os.path.join(dataset_dir, "reason_seg", reason_seg_data, "explanatory", "train.json")))
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print(f'\033[92m----Reason SEG-{split}: number of img_to_explanation samples is {len(self.img_to_explanation)} ----\033[0m')

    def __len__(self):
        return len(self.data_infos[0])
    
    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def __getitem__(self, idx):
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, self.__len__() - 1)
        images, jsons = self.data_infos
        image_path, json_path = images[idx], jsons[idx]

        image = self.oss_load_img_cv2(image_path) # (H, W, C)
        ori_size = image.shape[:2]

        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [(mask == 1).astype(np.float32) for _ in range(len(sampled_inds))]

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2 # vanilla text answer
            else:
                choice = random.randint(0, 1) # [SEG] or [SEG] token + text answer

        # preprocess image for clip
        global_enc_img = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0] # (C, H, W)

        # preprocess  image for grounding
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_img = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous()) # (C, IMG_SIZE, IMG_SIZE)
        
        questions = []
        answers = []
        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token)
        # print("______self.seg_token_num: {}, seg_token: {}__________".format(self.seg_token_num, seg_token))
        for text in sampled_sents:
            text = f'sentence:{text}' if is_sentence else f'class:{text.lower()}'
            questions.append(text)

            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] 
                    answer_temp = random.choice(self.answer_list) if self.seg_token_num == 1 else random.choice(self.answer_list).replace('[SEG]', seg_token)
                    answers.append(answer_temp)
                elif choice == 1:  # [SEG] token + text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer_temp = random.choice(self.answer_list) if self.seg_token_num == 1 else random.choice(self.answer_list).replace('[SEG]', seg_token)
                    answer = answer_temp + " {}".format(answer)
                    questions[-1] = text + " {}".format(random.choice(self.explanatory_question_list))
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = text
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
                # answer = random.choice(self.answer_list) if self.seg_token_num == 1 else random.choice(self.answer_list).replace('[SEG]', seg_token)
                answer = "Sure, [SEG]."
                answers.append(answer)

        questions, conversations = self.create_conversations(questions, answers)
        
        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation and choice == 2:
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.IGNORE_LABEL
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        
        bboxes = None

        return (image_path, global_enc_img, grounding_enc_img, bboxes, conversations, masks, label, 
                image_resize, questions, sampled_sents)

    def create_conversations(self, questions, answers):
        new_questions = []
        for question in questions:
            split, sent = question.split(":", 1)

            if split == "sentence":
                question = sent
            elif split == "class":
                question_template = random.choice(self.short_question_list)
                question = question_template.format(class_name=sent)
            else:
                raise ValueError("Not implemented yet.")
            
            question = question + " " + "Please respond with only one segmentation mask."
            new_questions.append(question)

        questions = new_questions
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