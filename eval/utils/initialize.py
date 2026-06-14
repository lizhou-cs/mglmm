import torch
import transformers
from model.mglmm import MGLMMForCausalLM
from model.llava import conversation as conversation_lib


def process_args(args):
    if args.precision == "bf16":
        args.torch_dtype = torch.bfloat16
    elif args.precision == "fp32":
        args.torch_dtype = torch.float32
    else:
        assert False, "Invalid precision"
    return args


def setup_tokenizer(args):
    # Load tokenizer and add special tokens.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the MGLMM model. """
    model_args = {}
    model = MGLMMForCausalLM.from_pretrained(args.version, torch_dtype=args.torch_dtype, low_cpu_mem_usage=True, **model_args)
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_for_inference(model, tokenizer, args):
    # initialize modules that are not included in the pretrined model
    model.get_model().initialize_vision_modules(model.get_model().config, init_mm_projector=False)
    
    # Transfer the model to GPU and set the dtype
    modules_name = ['vision_tower', 'mm_projector', 'segment_encoder', 'text_hidden_fcs']
    for module_name in modules_name:
        module = getattr(model.get_model(), module_name)
        module.to(device='cuda', dtype=args.torch_dtype)

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    return model
