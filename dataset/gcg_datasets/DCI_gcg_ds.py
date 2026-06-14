import io
import os
import json
import random
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import GCG_LONG_QUESTIONS, GCG_SUB_QUESTIONS, GCG_SUB_ANSWERS
from dataset.utils.oss_dataset import OSSDataset


class Node(object):
    def __init__(self, data):
        self.id = data['idx']
        self.label = data['label']
        self.caption = data['caption'].strip()
        self.mask = data['outer_mask']
        self.parent = int(data['parent'])
        self.children = []
        self.mask_quality = data['mask_quality']

        self.height = data.get('height')
        self.width = data.get('width')

        if not self.caption.endswith('.'):
            self.caption += '.'
    
    def add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return f'Node {self.id}: {self.label} ({self.caption})'
    

class AnnotationsTree(object):
    LOW_MASK_QUALITY = [1, 2]

    def __init__(self, data, enable_filter=False):
        self.nodes = {}
        self.root = None
        
        for node_data in data:
            node = Node(node_data)
            self.nodes[node.id] = node
        
        for node in self.nodes.values():
            if node.id == -1:
                self.root = node.id
            else:
                self.nodes[node.parent].add_child(node.id)

        # remove nodes with low mask quality
        if enable_filter:
            self._filter_low_quality_nodes()
    
    def _filter_low_quality_nodes(self):
        for node in list(self.nodes.values()):
            if node.mask_quality in self.LOW_MASK_QUALITY:
                self._remove_node(node.id)

    def _remove_node(self, node_id):
        if self.root == node_id or node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        parent_id = node.parent
        if parent_id in self.nodes:
            parent = self.nodes[parent_id]
            parent.children.remove(node_id)
        
        # connect children to parent
        for child_id in node.children:
            child = self.nodes[child_id]
            child.parent = parent_id
            parent.add_child(child_id)
        
        self.nodes.pop(node_id)
    
    def get_global_caption(self):
        """
        Get the caption of the root node.
        """
        return self.nodes[self.root].caption
    
    def get_children_nodes(self, node_id):
        return [self.nodes[child_id] for child_id in self.nodes[node_id].children]

    def get_nodes_with_children(self, num_children=2):
        """
        Get nodes whose children are more than num_children.
        """
        return [node for node in self.nodes.values() if len(node.children) >= num_children and node.id != self.root]


class DCIGCGDataset(OSSDataset):
    """
    Dataset Class for Grounded Conversation Generation (GCG) proposed in GLaMM.
    """
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=True):
        super(DCIGCGDataset, self).__init__()
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

        self.question_templates = GCG_LONG_QUESTIONS
        self.sub_question_templates = GCG_SUB_QUESTIONS
        self.sub_answer_templates = GCG_SUB_ANSWERS
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.validation = validation

        # Defining paths
        self.base_dir = os.path.join(dataset_dir, "DCI")
        self.image_folder = os.path.join(dataset_dir, "SegmentAnything/imgs")
        annotations_file_dir = "val" if validation else "train"
        annotations_file_path = os.path.join(self.base_dir, annotations_file_dir)
        self.data_infos = self._load_annotations(annotations_file_path)
        print('\033[92m' + "----GCG-: DCI GCG dataset initialized----" + '\033[0m')
        print('\033[92m' + "----GCG-: Number of samples: {}----".format(len(self.data_infos)) + '\033[0m')

    def _load_annotations(self, ann_file):
        data_infos = self.oss_list_dir(ann_file, suffix=".json")
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def _parse_annotations(self, ann_info):

        root_mask = {
            'idx': -1,
            'label': 'image',
            'caption': ann_info['extra_caption'],
            'parent': -2,
            'outer_mask': None,
            'mask_quality': 0,
            'width': ann_info['width'],
            'height': ann_info['height'],
        }

        mask_data = [root_mask]
        for mask in ann_info['mask_data'].values():
            mask_data.append({
                'idx': mask['idx'],
                'label': mask['label'],
                'caption': mask['caption'],
                'parent': mask['parent'],
                'outer_mask': mask['outer_mask'],
                'mask_quality': mask['mask_quality']
            })

        annotations = {
            'image_path': os.path.join(self.image_folder, ann_info['image']),
            'mask_data': mask_data
            }
        return annotations
    
    def __getitem__(self, idx):
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, self.__len__() - 1)
        data_info = self.data_infos[idx]
        json_content = json.load(self.oss_get_file(data_info))
        data_item = self._parse_annotations(json_content)

        return self.process_data(data_item)

    def __len__(self):
        return len(self.data_infos)

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
        image_path = data_item['image_path']

        image = self.oss_load_img_cv2(image_path)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        questions, answers, masks = [], [], []

        anno_tree = AnnotationsTree(data_item['mask_data'], enable_filter=True)
        q, a, m = self.create_node_qa(anno_tree, anno_tree.root)
        questions.extend(q), answers.extend(a), masks.extend(m)

        nodes = anno_tree.get_nodes_with_children()
        random_nodes = random.sample(nodes, min(len(nodes), self.num_classes_per_sample - 1))
        for node in random_nodes:
            q, a, m = self.create_node_qa(anno_tree, node.id, is_sub_question=True)
            questions.extend(q), answers.extend(a), masks.extend(m)

        questions, conversations = self.create_conversations(questions, answers)
        if len(masks) == 0:
            print(f"Empty masks for {image_path}")
            return self.__getitem__(0)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)

        bboxes, label, selected_labels = None, None, None

        return (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, image_resize, questions,
        selected_labels)
    
    def create_node_qa(self, anno_tree, node_id, is_sub_question=False):
        questions, answers, masks = [], [], []

        cur_node = anno_tree.nodes[node_id]
        answer = cur_node.caption
        child_nodes = anno_tree.get_children_nodes(node_id)
        for i, node in enumerate(child_nodes):
            if i == 0:
                answer += " " +  random.choice(self.sub_answer_templates)
            answer += f" <p> {node.label.capitalize()} </p> [SEG]: " + node.caption
            decoded_mask = base64.b64decode(node.mask)
            mask_arr = np.array(Image.open(io.BytesIO(decoded_mask)))
            mask_arr = np.where(mask_arr > 0, 1, 0)
            masks.append(mask_arr)
        
        if is_sub_question:
            template = random.choice(self.sub_question_templates)
            template = template.format(label=cur_node.label.lower())
        else:
            template = random.choice(self.question_templates)

        questions.append(template.strip())        
        answers.append(answer)
        
        return questions, answers, masks
