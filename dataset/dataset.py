import torch
import numpy as np
from model.llava.mm_utils import tokenizer_image_token
from model.llava import conversation as conversation_lib

from dataset.caption_datasets.COCO_Caption_ds import CocoCapDataset
from dataset.caption_datasets.Flickr_Caption_ds import FlickrCapDataset
from dataset.caption_datasets.LLavaInstruct_vqa_ds import LLaVAInstructDataset
from dataset.caption_datasets.GranD_ShortCaption_ds import GrandShortCaptionDataset

from dataset.segm_datasets.Semantic_Segm_ds import SemanticSegmDataset
from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset
from dataset.segm_datasets.GranD_ReferringSegm_ds import GrandReferSegmDataset
from dataset.segm_datasets.Panoptic_Segm_ds import PanopticSegmDataset
from dataset.reason_datasets.reason_seg import ReasonSegDataset
from dataset.reason_datasets.multi_reason_seg import MultiReasonSegDataset

from dataset.gcg_datasets.GranDf_gcg_ds import GranDfDataset, OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset
from dataset.gcg_datasets.DCI_gcg_ds import DCIGCGDataset
from dataset.gcg_datasets.MGLMM_gcg_ds import MGLMMDataset

from dataset.region_datasets.Flickr_Region_ds import Flickr30kRegDataset
from dataset.region_datasets.GranD_ReferringRegion_ds import GrandReferRegDataset
from dataset.region_datasets.RefCOCO_VG_Region_ds import (RefCocoRegDataset, RefCocoGRegDataset, RefCocoPRegDataset,
                                                          VisualGenomeRegDataset)
from tools.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_PROMPT_START_TOKEN, DEFAULT_PROMPT_END_TOKEN


val_dataset_dict = {'CocoCapVal': CocoCapDataset,
                    'FlickrCapVal': FlickrCapDataset,
                    'RefCOCOgSegmVal': ReferSegmDataset,
                    'PsgGCGVal': OpenPsgGCGDataset,
                    'RefCocoGCGVal': RefCOCOgGCGDataset,
                    'FlickrGCGVal': Flickr30kGCGDataset,
                    'DCIGCGVal': DCIGCGDataset,
                    'MGLMMGCGVal': MGLMMDataset,
                    'ReasonSegmVal': ReasonSegDataset,
                    'PanopticSegmval': PanopticSegmDataset,}


class HybridDatasetBase(torch.utils.data.Dataset):
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config,
                 epoch_samples=500 * 8 * 2 * 10, batch_size=2, precision="fp32", image_size=224,
                 num_classes_per_sample=3, sample_rate=None):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.global_image_encoder = global_image_encoder
        self.dataset = dataset
        self.datasets_config = datasets_config
        self.epoch_samples = epoch_samples
        self.batch_size = batch_size
        self.precision = precision
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_list = dataset.split("||")
        self.sample_rate = np.array(sample_rate or [1] * len(self.dataset_list))
        self.sample_rate /= self.sample_rate.sum()
        self.all_datasets = self.create_datasets()

    def create_datasets(self):
        datasets = []
        for ds in self.dataset_list:
            dataset_cls = self.datasets_config.get(ds)
            if dataset_cls:
                if ds == 'Semantic_Segm':
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir, self.tokenizer, self.global_image_encoder, self.epoch_samples,
                            self.precision, self.image_size, self.num_classes_per_sample, self.semantic_segm_data, )
                        )
                elif ds == 'Refer_Segm':
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir, self.tokenizer, self.global_image_encoder, self.epoch_samples,
                            self.precision, self.image_size, self.num_classes_per_sample, self.refer_segm_data, )
                        )
                else:
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir, self.tokenizer, self.global_image_encoder, self.epoch_samples,
                            self.precision, self.image_size, self.num_classes_per_sample, )
                        )
        return datasets

    def __len__(self):
        return self.epoch_samples

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        data = selected_dataset[0]
        return (*data,)


class HybridCapDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10, batch_size=2,
                 precision="fp32", image_size=224, num_classes_per_sample=3,
                 dataset="CocoCap||LLaVaInstruct", sample_rate=[1, 1]):
        datasets_config = {"CocoCap": CocoCapDataset,
                           "LLaVaInstruct": LLaVAInstructDataset,
                           "GrandCaptionDataset": GrandShortCaptionDataset,
                           # Add other dataset mappings here
                           }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples, batch_size,
            precision, image_size, num_classes_per_sample, sample_rate
        )


class HybridRegDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10, batch_size=2,
                 precision="fp32", image_size=224, num_classes_per_sample=3,
                 dataset="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg||Flickr_Reg", sample_rate=[1, 1, 1, 1, 1]):
        datasets_config = {"RefCoco_Reg": RefCocoRegDataset,
                           "RefCocoG_Reg": RefCocoGRegDataset,
                           "RefCocoP_Reg": RefCocoPRegDataset,
                           "VisGen_Reg": VisualGenomeRegDataset,
                           "Flickr_Reg": Flickr30kRegDataset,
                           "GrandRefer_Reg": GrandReferRegDataset,
                           # Add other dataset mappings here
                           }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples, batch_size,
            precision, image_size, num_classes_per_sample, sample_rate
        )


class HybridSegDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10, batch_size=2,
                 precision="fp32", image_size=224, num_classes_per_sample=3,
                 dataset="Semantic_Segm||Refer_Segm||PSG_GCG||RefCoco_GCG||GranDf_GCG||Flickr_GCG",
                 sample_rate=[5,4,1,1,1,1],
                 semantic_segm_data="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
                 refer_segm_data="refcoco||refcocog||refcoco+||refclef"):
        self.semantic_segm_data = semantic_segm_data
        self.refer_segm_data = refer_segm_data
        datasets_config = {"Semantic_Segm": SemanticSegmDataset,
                           "Refer_Segm": ReferSegmDataset,
                           "PSG_GCG": OpenPsgGCGDataset,
                           "RefCoco_GCG": RefCOCOgGCGDataset,
                           "GranDf_GCG": GranDfDataset,
                           "MGLMM_GCG": MGLMMDataset,
                           "Flickr_GCG": Flickr30kGCGDataset,
                           "GrandRefer_Segm": GrandReferSegmDataset,
                           # Add other dataset mappings here
                           }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples, batch_size,
            precision, image_size, num_classes_per_sample, sample_rate
        )


def custom_collate_fn(batch, tokenizer=None, mm_use_im_start_end=True, mm_use_prompt_start_end=True, inference=False, added_token_num=575):
    # Initializing lists and counters
    image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
    bboxes_list, conversation_list, masks_list = [], [], []
    label_list, resize_list, questions_list = [], [], []
    selected_labels_list, offset_list, inferences = [], [0], []
    image_desc_list = []
    cnt = 0

    # Iterating through the batch
    for batch_item in batch:
        if len(batch_item) < 11:
            batch_item = batch_item + (None, ) * (11 - len(batch_item))

        (image_path, global_enc_image, grounding_enc_image, bboxes, conversations, masks, label, resize, questions,
         sampled_classes, image_desc) = batch_item

        image_path_list.append(image_path)
        global_enc_image_list.append(global_enc_image)
        grounding_enc_image_list.append(grounding_enc_image)
        bboxes_list.append(bboxes)
        conversation_list.extend(conversations)
        masks_list.append([] if masks is None else masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        selected_labels_list.append(sampled_classes)
        image_desc_list.append(image_desc)
        offset_list.append(cnt := cnt + len(conversations))
        inferences.append(inference)

    # Handling the conversation list
    if mm_use_im_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if mm_use_prompt_start_end:
            replace_token += (DEFAULT_PROMPT_START_TOKEN + DEFAULT_PROMPT_END_TOKEN)

        conversation_list = [conv.replace(DEFAULT_IMAGE_TOKEN, replace_token) for conv in conversation_list]

    # Tokenizing and padding input ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list],
        batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Preparing targets and handling conversation types
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    # conv_type == "llava_v1"
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    if not inferences[0]:
        # In case of training, truncating the input ids if they exceed the model's max length
        truncate_len = tokenizer.model_max_length - added_token_num
        if input_ids.shape[1] > truncate_len:
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
                )

    return {
        "image_paths": image_path_list,
        "global_enc_images": torch.stack(global_enc_image_list, dim=0),
        "grounding_enc_images": None if grounding_enc_image_list[0] is None else torch.stack(grounding_enc_image_list, dim=0),
        "bboxes": None if bboxes_list[0] is None else bboxes_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": None if masks_list[0] is None else masks_list,
        "label_list": None if label_list[0] is None else label_list,
        "resize_list": None if resize_list[0] is None else resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": selected_labels_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "images_desc": image_desc_list if image_desc_list[0] is not None else None
    }


def _process_conversation(conversation, target, tokenizer, sep, sep2):
    total_len = target.ne(tokenizer.pad_token_id).sum().item()
    rounds = conversation.split(sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX

    for rou in rounds:
        if not rou:
            break

        parts = rou.split(sep)
        assert len(parts) == 2, (len(parts), conversation)
        parts[0] += sep

        if DEFAULT_IMAGE_TOKEN in conversation:
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
        else:
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len

    target[cur_len:] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        assert cur_len == total_len
