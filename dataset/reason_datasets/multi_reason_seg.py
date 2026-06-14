import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from dataset.pycocotools import mask
from transformers import CLIPImageProcessor
from model.SAM.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
from dataset.utils.oss_dataset import OSSDataset
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import LONG_QUESTION_LIST


def decode_mask(rle_mask, height, width):
    rle = mask.frPyObjects(rle_mask, height, width)
    m = mask.decode(rle)
    if len(m.shape) > 2:
        # merge multiple masks into one
        m = np.sum(m, axis=2)
    m = m.astype(np.uint8)
    return m

class MultiReasonSegDataset(OSSDataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32", 
                 image_size=224, num_classes_per_sample=3, split='train', 
                 validation=False, inference=False, random_sampling=True, explanatory=True):
        super(MultiReasonSegDataset, self).__init__()
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.split = split
        self.validation = validation
        self.inference = inference
        self.random_sampling = random_sampling
        self.explanatory = explanatory
        
        self.long_question_list = LONG_QUESTION_LIST
        
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        
        self.split = 'val' if (self.validation and self.split == 'train') else self.split
        ann_file = os.path.join(dataset_dir, 'reason_seg/MUSE', f'MUSE_{self.split}.json')
        self.data_infos = self._load_annotations(ann_file)

        print(f'\033[92m----Reason SEG: MultiReasonSeg dataset initialized----\033[0m')
        print(f'\033[92m----Reason SEG-: number of MUSE dataset is {len(self.data_infos)} ----\033[0m')
        
    def _load_annotations(self, ann_file):
        data_infos = json.load(self.oss_get_file(ann_file))
        data_infos = data_infos[0: 1000] if self.validation and not self.inference else data_infos
        return data_infos

    def __len__(self):
        return len(self.data_infos)
    

    def __getitem__(self, idx):
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, self.__len__() - 1)
        data_info = self.data_infos[idx]
        if 'file_name' in data_info:
            file_name = data_info['file_name']
            image_root = "coco/train2014"
        else:
            file_name = data_info['coco_url'].split('/')[-1]
            image_root = "coco/train2017" if 'train2017' in data_info['coco_url'] else "coco/val2017"
        image_path = os.path.join(self.dataset_dir, image_root, file_name)

        anns = data_info['ann_list']
        if len(anns) == 0:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        sampled_questions = data_info['questions'] if 'questions' in data_info else None
        sampled_answers = data_info['answers'] if 'answers' in data_info else None
        sampled_text_answers = data_info['text_answers'] if 'text_answers' in data_info else [None] * len(sampled_answers)

        if len(sampled_questions) > self.num_classes_per_sample:
            sample_idx = random.sample(range(len(sampled_questions)), self.num_classes_per_sample)
            sampled_questions = [sampled_questions[i] for i in sample_idx]
            sampled_answers = [sampled_answers[i] for i in sample_idx]
            sampled_text_answers = [sampled_text_answers[i] for i in sample_idx]

        questions, answers, masks = [], [], []
        for question, answer_list, text_answer in zip(sampled_questions, sampled_answers, sampled_text_answers):
            # skip questions with bbox infomation, such as [12, 23, 34, 45]
            if '[' in question and ']' in question:
                continue

            questions.append(question)
            # there are multiple [SEG] for one question
            for answer in answer_list:
                masks.append(decode_mask(answer["segmentation"], data_info["height"], data_info["width"]))

            category_target_list = [self._preproce_category_name(answer['category_name']) for answer in answer_list]
            if text_answer is not None:
                # replace {{ with { and }} with } and remove </s>
                text_answer = text_answer.replace('{{', '{').replace('}}', '}').replace('</s>', '')
                text_answer = self._process_answer_for_explanatory(text_answer, category_target_list)
                
                if text_answer is not None:
                    answers.append(text_answer)
                else:
                    return self.__getitem__(random.randint(0, self.__len__() - 1))
            else:
                # if no text_answer, use category_name or rephrased_name to fill answer_template
                rephrased_target_list = []
                for i, answer in enumerate(answer_list):
                    rephrased_target_list.append(answer['rephrased_name'].strip() if 'rephrased_name' in answer else category_target_list[i])
                rephrased_target_list = self._process_phrase_for_explanatory(rephrased_target_list, category_target_list)
                category_target_list = self._process_category_for_explanatory(category_target_list)
                answers.append([rephrased_target_list, category_target_list])

        if len(masks) == 0:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        
        # preprocess image
        image = self.oss_load_img_cv2(image_path)
        global_enc_img = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_img = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        conversations = self.create_conversations(questions, answers)
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL

        bboxes = None
        selected_labels = None

        return (image_path, global_enc_img, grounding_enc_img, bboxes, conversations, masks, label, 
                image_resize, questions, selected_labels)
        
    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x
    
    def _preproce_category_name(self, category_name):
        category_name = category_name.replace('_', ' ').strip()
        if '(' in category_name:
            category_name = category_name.split('(')[0].strip()
        return category_name
    
    def _insert_seg_token(self, category):
        return "<p> {category} </p> {seg}".format(category=category.strip(), seg='[SEG]')

    def _process_answer_for_explanatory(self, answer, catgeories_name):
        if not self.explanatory:
            return answer
        
        # cannot handle the case where multiple {seg} are adjacent
        if '{seg} {seg}' in answer or '{seg}, {seg}' in answer or '{seg} and {seg}' in answer:
            return None
        
        if answer.count('{seg}') != len(catgeories_name):
            return None
        
        i = 0
        new_answer = ''
        while '{seg}' in answer:
            category = catgeories_name[i]
            split1, split2 = answer.split('{seg}', 1)
            pos = split1.rfind(category)
            # if the category name is not in the answer or the category name is too far from the {seg}
            if pos == -1 or len(split1[pos:]) - len(category) > 5:
                category = category.replace(' ', '_')
                pos = split1.rfind(category)
            # retry with the category name replaced with '_'
            if pos == -1 or len(split1[pos:]) - len(category) > 5:
                return None
            else:
                new_answer += split1[:pos] + self._insert_seg_token(split1[pos:])
                answer = split2
                i += 1
        new_answer += answer

        return new_answer
    
    def _process_phrase_for_explanatory(self, phrases, catgeories_name):
        if not self.explanatory:
            return phrases
        
        for i in range(len(phrases)):
            phrase, category = phrases[i], catgeories_name[i]
            # check if the only one category name is in the phrase
            if f' {category}' in phrase:
                phrase = phrase.replace(category, self._insert_seg_token(category), 1)
                phrases[i] = phrase
            else:
                phrases[i] = self._insert_seg_token(category)
        return phrases
    
    def _process_category_for_explanatory(self, categories):
        if not self.explanatory:
            return categories
        
        new_categories = []
        for category in categories:
            new_categories.append(self._insert_seg_token(category))
        return new_categories
    
    def format_answer(self, raw_answer):
        if isinstance(raw_answer, str):
            return raw_answer
        elif isinstance(raw_answer, list):
            rephrased_target_list, category_target_list = raw_answer
            target_list = [rephrased_target_list[i] for i in range(len(rephrased_target_list))]

            target_answer = ''
            for i, sampled_class in enumerate(target_list):
                answer = sampled_class.strip().strip('.').lower()
                answer = answer.replace('[seg]', '[SEG]')
                if i == 0:
                    target_answer += answer[0].upper() + answer[1:]
                elif i == len(target_list) - 1:
                    target_answer += ' and ' + answer
                else:
                    target_answer += ', ' + answer
            target_answer += '.'
            return target_answer
        else:
            raise ValueError('Invalid answer type')

    def create_conversations(self, questions, answers):
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            question_template = random.choice(self.long_question_list)
            question = question_template.format(sent=question)
            if i == 0:
                question = self.begin_str + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], self.format_answer(answer))
        conversations.append(conv.get_prompt())
        return conversations