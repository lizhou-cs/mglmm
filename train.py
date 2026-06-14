"""
train.py - MGLMM Model Training on Mixed Datasets
"""
import os
import sys
import time
import yaml
import json
import tqdm
import random
import warnings
import torch
import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model

from model.mglmm import MGLMMForCausalLM
from model.llava import conversation as conversation_lib
from dataset.dataset import val_dataset_dict, custom_collate_fn, HybridSegDataset, HybridCapDataset
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, Summary, 
                         dict_to_cuda, set_random_seed, intersectionAndUnionGPU)


def parse_args(args):
    parser = argparse.ArgumentParser(description="MGLMM Model Training")

    # Environment settings
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank")
    parser.add_argument("--rank", default=0, type=int, help="process rank")

    # Model-specific settings
    parser.add_argument("--version", default="./checkpoints/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default=None, type=str)
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--disable_data_format", action="store_true", default=False, help="Disable data format")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_segm_data", action="store_true", help="Use segmentation data")
    parser.add_argument("--use_gcg_data", action="store_true", help="Use GCG data")
    parser.add_argument("--weight_cap", default=0.5, type=float, help="Sampling weight for caption data")
    parser.add_argument("--weight_segm", default=0.5, type=float, help="Sampling weight for segmentation data")
    parser.add_argument("--weight_gcg", default=0.5, type=float, help="Sampling weight for GCG data")
    parser.add_argument("--cap_dataset", default="Caption||GranD_Caption", type=str, help="Choose from: Caption, GranD_Caption")
    parser.add_argument("--segm_dataset", default="Semantic_Segment||Referring_Segment", type=str, help="Choose from: Semantic_Segment, Referring_Segment, GranD_Segment")
    parser.add_argument("--gcg_dataset", default="GranD_GCG", type=str, help="Choose from: GranD_GCG, DCI_GCG")
    parser.add_argument("--cap_sample_rates", default="1,1", type=str)
    parser.add_argument("--segm_sample_rates", default="1,1", type=str)
    parser.add_argument("--gcg_sample_rates", default="1", type=str)
    parser.add_argument("--max_gt_per_sample", default=3, type=int)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)

    # Evaluation settings
    parser.add_argument("--val_dataset", default="CocoCapVal|RefCOCOgRegVal|RefCOCOgSegmVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal, DCIGCGDataset")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--ckpt_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="mglmm", type=str)

    return parser.parse_args(args)


def preprocess_args(args):
    # initialize log_dir when 'SUMMARY_DIR' is set
    args.log_dir = os.getenv('SUMMARY_DIR')
    args.rank = int(os.getenv('RANK', args.rank))
    args.world_size = int(os.getenv('WORLD_SIZE', torch.cuda.device_count()))
    args.distributed = args.world_size > 1

    return args


def initialize_environment(args):
    """ Set up logging and model directories. """
    if args.log_dir is None:
        args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

    if args.rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir, max_queue=30, flush_secs=120)
    return None


def save_arguments(args):
    args.ckpt_dir = os.path.join(args.ckpt_base_dir, args.exp_name)

    if args.rank == 0:
        # save args to yaml file
        args_dict = vars(args)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        with open(os.path.join(args.ckpt_dir, 'config.yaml'), 'w') as f:
            yaml.dump(args_dict, f, default_flow_style=False)


def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best):
    """ Saves the model checkpoint. """
    # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
    save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    save_dir = os.path.join(args.ckpt_dir, save_dir_name)
    # Ensure the directory exists
    if args.rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
        torch.save({"epoch": epoch, f"val_{metric_name}": metric_value}, os.path.join(save_dir, ckpt_filename))
    torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        special_tokens = []
        if args.mm_use_im_start_end:
            special_tokens.extend([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens.extend(reg_tokens + segmentation_tokens + phrase_tokens)
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the MGLMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", 
                   "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "vision_pretrained", "vision_tower", "mm_vision_select_layer", 
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", 
                   "mm_use_im_start_end", "bbox_token_idx", "seg_token_idx", "bop_token_idx", "eop_token_idx"]}
    
    model_args['mm_vision_tower'] = model_args['vision_tower']

    model = MGLMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def initialize_modules(model, tokenizer, args):
    device = args.local_rank if args.local_rank != -1 else 'cpu'

    # Initialize vision tower
    model.get_model().initialize_vision_modules(model.get_model().config, init_mm_projector = not args.pretrained)
    model.get_model().get_vision_tower().to(dtype=torch.bfloat16, device=device)
    print('\033[92m' + "---- Initialized Global Image Encoder from: {} ----".format(args.vision_tower) + '\033[0m')
    
    # Initialize MGLMM model
    if not args.pretrained:
        model.get_model().initialize_segment_model(args.vision_pretrained)
        print('\033[92m' + "---- Initialized Segment Encoder from: {} ----".format(args.vision_pretrained) + '\033[0m')
        
        modules_name = ['mm_projector'] + ['segment_encoder', 'text_hidden_fcs']
        for module_name in modules_name:
            module = getattr(model.get_model(), module_name)
            module.to(dtype=torch.bfloat16, device=device)


def prepare_model_for_training(model, tokenizer, args):
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Initialize other modules in the model
    initialize_modules(model, tokenizer, args)

    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    elif lora_r > 0:
        # Configure LoRA if applicable
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)
    else:
        assert False, "lora_r must be >= 0"

    # Make certain modules trainable
    set_trainable_modules(model, args)

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]


def prepare_model_for_evaluation(model, args):
    # Initialize other modules in the model
    initialize_modules(model, args)

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["vision_tower", "segment_encoder", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model, args):
    """ Make specified modules in the model trainable. """

    # Set requires_grad for vision tower
    for p in model.get_model().vision_tower.parameters():
        p.requires_grad = False

    # Set requires_grad for segment encoder
    for param in model.get_model().segment_encoder.parameters():
        param.requires_grad = False
    
    projection_modules = ["mm_projector", "text_hidden_fcs"]
    embed_modules = ["lm_head", "embed_tokens"]

    trainable_modules = projection_modules + embed_modules

    if args.train_mask_decoder:
        model.get_model().segment_encoder.mask_decoder.train()
        trainable_modules.append("mask_decoder")
    
    for module_name in projection_modules:
        module = getattr(model.get_model(), module_name)
        module.train()

    # Set requires_grad for trainable modules
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            param.requires_grad = True

    def count_parameters(model):
        total_params, trainable_params = 0, 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Name: {name}, Shape: {param.shape}, Type: {param.dtype}", file=sys.stderr)

        total_params_mb = total_params / (1024 * 1024)  # Convert to MB
        trainable_params_mb = trainable_params / (1024 * 1024)  # Convert to MB
        print('\033[92m' + "---- Total parameters: ----{} MB".format(total_params_mb) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{} MB".format(trainable_params_mb) + '\033[0m')

    count_parameters(model)


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0,
                                                           "betas": (args.beta1, args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0,
                                          "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    # TODO: remove sys.gettrace() when not needed
    if sys.gettrace() is not None:
        model_engine = model
        model_engine.to(device=args.local_rank)
        optimizer = torch.optim.AdamW(model_engine.parameters(), lr=args.lr, weight_decay=0.0, betas=(args.beta1, args.beta2))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steps_per_epoch)
        torch.distributed.init_process_group('nccl', world_size=1, rank=0)
    else:
        model_engine, optimizer, _, scheduler = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

    return model_engine, optimizer, scheduler


def initialize_train_datasets(args, tokenizer):
    # Dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer, "global_image_encoder": args.vision_tower,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * args.world_size,
                      "precision": args.precision, "image_size": args.image_size, "num_classes_per_sample": args.max_gt_per_sample}

    # Training datasets
    cap_train_dataset = HybridCapDataset(
        **common_ds_args, dataset=args.cap_dataset, sample_rate=[float(x) for x in args.cap_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_cap_data else None
    seg_train_dataset = HybridSegDataset(
        **common_ds_args, dataset=args.segm_dataset, sample_rate=[float(x) for x in args.segm_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_segm_data else None
    gcg_train_dataset = HybridSegDataset(
        **common_ds_args, dataset=args.gcg_dataset, sample_rate=[float(x) for x in args.gcg_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_gcg_data else None
    
    # Dataloader arguments
    collate_fn_args_train = partial(custom_collate_fn, tokenizer=tokenizer, 
                                    mm_use_im_start_end=args.mm_use_im_start_end, mm_use_prompt_start_end=args.mm_use_prompt_start_end, 
                                    inference=False, added_token_num=args.added_token_num)
    
    sampler_args = {"shuffle": False, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False, "collate_fn": collate_fn_args_train}

    # Training loaders
    cap_train_loader = torch.utils.data.DataLoader(
        cap_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(cap_train_dataset, **sampler_args), 
        **train_loader_args) if cap_train_dataset is not None else None
    seg_train_loader = torch.utils.data.DataLoader(
        seg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(seg_train_dataset, **sampler_args), 
        **train_loader_args) if seg_train_dataset is not None else None
    gcg_train_loader = torch.utils.data.DataLoader(
        gcg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(gcg_train_dataset, **sampler_args), 
        **train_loader_args) if gcg_train_dataset is not None else None

    return cap_train_loader, seg_train_loader, gcg_train_loader

def initialize_val_datasets(args, tokenizer):
    if args.no_eval:
        return None
    
    val_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer, "global_image_encoder": args.vision_tower, 
                   "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * args.world_size,
                    "precision": args.precision, "image_size": args.image_size, "num_classes_per_sample": args.max_gt_per_sample}
    
    val_datasets = []
    for val_dataset_name in args.val_dataset.split('|'):
        val_dataset_class = val_dataset_dict.get(val_dataset_name, None)
        if val_dataset_class is None:
            continue

        if val_dataset_name == 'RefCOCOgSegmVal':
            refer_segm_data = 'refcocog'
            all_datasets = refer_segm_data.split("||")
            for d in all_datasets:
                val_dataset = val_dataset_class(**val_ds_args, validation=True, refer_segm_data=d, split='val', 
                                                explanatory=not args.disable_data_format)
                val_dataset._set_len(len(val_dataset.refer_segm_data[d]['images']))
                val_datasets.append(val_dataset)
        else:
            val_datasets.append(val_dataset_class(**val_ds_args, validation=True))
    
    if val_datasets:
        inference_mode = args.mask_validation
        collate_fn_args_val = partial(custom_collate_fn, tokenizer=tokenizer, 
                                      mm_use_im_start_end=args.mm_use_im_start_end, mm_use_prompt_start_end=args.mm_use_prompt_start_end, 
                                      inference=inference_mode, added_token_num=args.added_token_num)

        sampler_args = {"shuffle": False, "drop_last": False}
        val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers, 
                           "pin_memory": False, "collate_fn": collate_fn_args_val}
        
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(combined_val_datasets,
                                                 sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args),
                                                 **val_loader_args)
    else:
        val_loader = None

    return val_loader

def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.ckpt_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")


def resume_config_from_checkpoint(args):
    if args.resume:
        if "ckpt_model_last_epoch" in args.resume:
            config_path = args.resume.replace("ckpt_model_last_epoch", "config.yaml")
        elif "ckpt_model_best" in args.resume:
            config_path = args.resume.replace("ckpt_model_best", "config.yaml")
        else:
            config_path = os.path.join(args.resume, "config.yaml")

        if os.path.exists(config_path):
            config = yaml.safe_load(open(config_path, "r"))
        else:
            raise FileNotFoundError(f"Config file not found for checkpoint: {args.resume}")
        
        print(f"Resuming training with config from {config_path}")
        ignore_keys = ['resume', 'log_dir', 'ckpt_dir', 'exp_name', 'local_rank', 'rank', 'batch_size', 'val_batch_size']
        for key, value in config.items():
            if key not in ignore_keys:
                print(f"Setting {key} to {value}")
                setattr(args, key, value)
    return args

def train_entrance(args):
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)

    model_engine, _, scheduler = initialize_deepspeed(model, tokenizer, args)
    resume_training_from_checkpoint(model_engine, args)
    
    cap_train_loader, seg_train_loader, gcg_train_loader = initialize_train_datasets(args, tokenizer)
    val_loader = initialize_val_datasets(args, tokenizer)
    
    active_dataloaders = []
    weights = []
    if args.use_cap_data:
        active_dataloaders.append(('cap', cap_train_loader))
        weights.append(args.weight_cap)
    if args.use_segm_data:
        active_dataloaders.append(('seg', seg_train_loader))
        weights.append(args.weight_segm)
    if args.use_gcg_data:
        active_dataloaders.append(('gcg', gcg_train_loader))
        weights.append(args.weight_gcg)

    # Assert that at least one dataset is active
    assert active_dataloaders, "Error: At least one dataset (segm or cap) must be active."

    dataset_iters = {'cap': iter(cap_train_loader) if args.use_cap_data else None,
                     'seg': iter(seg_train_loader) if args.use_segm_data else None, 
                     'gcg': iter(gcg_train_loader) if args.use_gcg_data else None}

    writer = initialize_environment(args)
    save_arguments(args)

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]
    dataset_choices = [idx for idx, _ in enumerate(active_dataloaders)]

    best_giou, best_ciou, best_val_loss = 0.0, 0.0, np.inf
    for epoch in range(args.start_epoch, args.epochs):
        random.seed(epoch_seeds[epoch])

        step_choices = random.choices(dataset_choices, weights=weights, k=args.steps_per_epoch)

        dataset_iters = train(active_dataloaders, model_engine, epoch, scheduler, writer, 
                              dataset_iters, args, step_choices, tokenizer=tokenizer)

        if args.mask_validation:
            giou, ciou = validate_model_performance(val_loader, model_engine, epoch, writer, args, tokenizer=tokenizer)
            is_best = giou > best_giou
            best_giou = max(giou, best_giou)
            best_ciou = ciou if is_best else best_ciou
            if args.rank == 0:  # Log the progress
                print(f"Epoch: {epoch}, giou: {giou}, ciou: {ciou}, best_giou: {best_giou}, best_ciou: {best_ciou}")
            save_checkpoint(model_engine, args, epoch, 'giou-ciou', f"{giou:.4f}-{ciou:.4f}", is_best)
        else:
            cur_val_loss = validate_model_performance(val_loader, model_engine, epoch, writer, args, tokenizer=tokenizer)
            is_best = cur_val_loss < best_val_loss
            best_val_loss = min(cur_val_loss, best_val_loss)
            if args.rank == 0:  # Log the progress
                print(f"Epoch: {epoch}, Current Validation Loss: {cur_val_loss:.4f}, Best Validation Loss: {best_val_loss:}")
            save_checkpoint(model_engine, args, epoch, 'loss', f"{cur_val_loss:.4f}", is_best)
    
    if args.rank == 0:
        writer.close()


def train(active_datasets, model, epoch, scheduler, writer, dataset_iters, args, step_choices, tokenizer=None):
    """Main training loop."""

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress."""
        log_step = global_step + epoch * args.steps_per_epoch
        if log_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.rank == 0:
                progress.display(global_step + 1)
                for key, tracker in trackers.items():
                    if tracker.avg == 0:
                        continue
                    writer.add_scalar(f"train/{key}", tracker.avg, log_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, log_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, log_step)

            for tracker in trackers.values():
                tracker.reset()
        
        if log_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], log_step)

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                "mask_loss": AverageMeter("MaskLoss", ":.4f")}
    
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for grad_step in range(args.grad_accumulation_steps):
            # Select data loader based on step choice
            dataset_type, data_loader = active_datasets[step_choices[global_step]]
            data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
            dataset_iters[dataset_type] = new_iter

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()

            output_dict = model(**data_batch)

            if global_step % 10 == 0 and grad_step == 0 and args.rank == 0:
                print(f"[{epoch}/{args.epochs}][{global_step}/{args.steps_per_epoch}]\n", file=sys.stderr)
                print_input_output(data_batch, output_dict, tokenizer)

            # Update training metrics
            for key, tracker in trackers.items():
                if key in output_dict and output_dict[key].item() != 0:
                    tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))

            model.backward(output_dict["loss"])
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()

    return dataset_iters


def eval_entrance(args):
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_evaluation(model, args)

    # load checkpoint
    if args.resume:
        model.load_state_dict(args.resume, strict=True)

    if sys.gettrace() is not None:
        warnings.warn("Running in debug mode, not using deepspeed")
        model_engine = model
        model_engine.to(device=args.local_rank)
        torch.distributed.init_process_group('nccl', world_size=1, rank=0)
    else:
        ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                     "gradient_accumulation_steps": args.grad_accumulation_steps,
                     "fp16": {"enabled": args.precision == "fp16"}, 
                     "bf16": {"enabled": args.precision == "bf16"}}
        model_engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

    # disable all training dataset
    val_loader = initialize_val_datasets(args, tokenizer)

    writer = initialize_environment(args)
    val_loss = validate_model_performance(val_loader, model_engine, 0, writer, args, tokenizer=tokenizer)
    print(f"Validation Loss: {val_loss:.4f}")


def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, args, tokenizer=None):
    if args.mask_validation:
        # For use with only segmentation/GCG type datasets
        trackers = {"intersection": AverageMeter("Intersec", ":.4f", Summary.SUM),
                    "union": AverageMeter("Union", ":.4f", Summary.SUM),
                    "gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM)}

        training_model.eval()
        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                results = training_model(**data_batch)

            pred_masks, gt_masks = results["pred_masks"], results["gt_masks"][0].int()
            assert len(pred_masks) == 1

            pred_masks = (pred_masks[0] > 0).int()
            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            for target, prediction in zip(gt_masks, pred_masks):
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                # handles no-object targets
                accuracy_iou[union_ == 0] += 1.0

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            accuracy_iou = accuracy_iou.cpu().numpy() / gt_masks.shape[0]
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])

        for meter in trackers.values():
            meter.all_reduce()

        iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
        class_iou = iou_per_class[1]
        global_iou = trackers["gIoU"].avg[1]

        if args.rank == 0:
            tensorboard_writer.add_scalar("val/giou", global_iou, current_epoch)
            tensorboard_writer.add_scalar("val/ciou", class_iou, current_epoch)
            print("giou: {:.4f}, ciou: {:.4f}".format(global_iou, class_iou))

        return global_iou, class_iou
    else:
        # Initializing performance trackers
        trackers = {"loss": AverageMeter("Loss", ":.4f"), "ce_loss": AverageMeter("CeLoss", ":.4f"), 
                    "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"), "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                    "mask_loss": AverageMeter("MaskLoss", ":.4f")}

        # Prepare model for validation phase
        # Hack to get the loss
        training_model.train()

        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                predictions = training_model(**data_batch)
            # Update performance metrics)
            for key, tracker in trackers.items():
                if key in predictions:
                    tracker.update(predictions[key].item(), data_batch["global_enc_images"].size(0))
            
            # print_input_output(data_batch, predictions, tokenizer)

        # Synchronize metrics across processes
        for tracker in trackers.values():
            tracker.all_reduce()
        # Calculate average validation loss
        avg_val_loss = trackers["ce_loss"].avg
        # Tensorboard logging for primary process
        if args.rank == 0:
            tensorboard_writer.add_scalar("val/loss", avg_val_loss, current_epoch)

        return avg_val_loss


def print_input_output(input_batch, output_dict, tokenizer, sample_num=2):
    labels = input_batch['labels']
    logits = output_dict['logits']
    sample_num = min(sample_num, len(labels))

    with torch.no_grad():
        pred_ans = []
        for i in range(sample_num):
            label = labels[i]
            # get logits of answer tokens
            pred_logits = logits[i][-label.size(0):]
            # shift predicted logits and labels
            pred_logits, label = pred_logits[:-1], label[1:]
            # remove padding tokens
            pred_logits = pred_logits[label != -100]
            pred_indices = torch.argmax(pred_logits, dim=-1)
            pred_ans.append(tokenizer.decode(pred_indices, skip_special_tokens=False))

    for i in range(sample_num):
        conversation = input_batch['conversation_list'][i]
        if (pos := conversation.find('USER: ')) != -1:
            conversation = conversation[pos:]

        print(f"image: {input_batch['image_paths'][i]}\n"
            f"conv: {[conversation]}\n"
            f"answer: {[pred_ans[i]]}\n", file=sys.stderr)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = resume_config_from_checkpoint(args)
    args = preprocess_args(args)

    set_random_seed(args.seed)

    if args.eval_only:
        print('\033[92m' + '---- Evaluation only mode. ----' + '\033[0m')        
        eval_entrance(args)
    else:
        print('\033[92m' + '---- Training mode. ----' + '\033[0m')
        train_entrance(args)
