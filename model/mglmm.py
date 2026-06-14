import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from model.SAM.build_sam import sam_model_registry
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel


def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class MGLMMBaseModel:
    def __init__(self, config, **kwargs):
        super(MGLMMBaseModel, self).__init__(config)
        self.config = config

        # Set config attributes if they don't exist
        self.vision_pretrained = getattr(
            self.config, "vision_pretrained", kwargs.get("vision_pretrained", None)
        )
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_segment_model(checkpoint=self.vision_pretrained, delay_load=True)

    def initialize_segment_model(self, checkpoint, delay_load=False):
        # Initialize the segment encoder
        if "vit_h" in checkpoint:
            model_type = "vit_h"
        elif "vit_l" in checkpoint:
            model_type = "vit_l"
        elif "vit_b" in checkpoint:
            model_type = "vit_b"
        else:
            raise ValueError(f"Invalid model type in checkpoint: {checkpoint}")

        checkpoint = None if delay_load else checkpoint
        self.segment_encoder = sam_model_registry[model_type](checkpoint)

        # Configure segment encoder
        self._configure_segment_encoder()

        # Initialize the projection layer
        self._initialize_projection_layer()
    
    def _configure_segment_encoder(self):
        # Freezing segment model parameters
        for param in self.segment_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if self.config.train_mask_decoder:
            self.segment_encoder.mask_decoder.train()
            for param in self.segment_encoder.mask_decoder.parameters():
                param.requires_grad = True

    def _initialize_projection_layer(self):
        # initialize text projection layer
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), 
                                  nn.ReLU(inplace=True), 
                                  nn.Linear(in_dim, out_dim), 
                                  nn.Dropout(0.0)]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])


class MGLMMModel(MGLMMBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(MGLMMModel, self).__init__(config, **kwargs)


class MGLMMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = MGLMMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        configurations = {
            'mm_use_im_start_end': True, 
            'with_region': False,
            'seg_token_idx': None,
            'bbox_token_idx': None,
            'vision_tower': 'openai/clip-vit-large-patch14',
            'vision_pretrained': 'sam_vit_h_4b8939.pth',
            'pretrain_mm_mlp_adapter': None,
            'num_level_reg_features': 4,
            'use_cache': False
        }

        for cfg, val in configurations.items():
            setattr(config, cfg, getattr(config, cfg, kwargs.pop(cfg, val)))

        self._initialize_loss_weights(kwargs)
        self.added_token_num = kwargs.pop("added_token_num", None)
        
        if self.added_token_num is None:
            vision_tower_name = config.vision_tower.split('/')[-1]
            vision_tower_config = vision_tower_name.split('-')
            patch_size, image_size = int(vision_tower_config[-2].replace('patch', '')), int(vision_tower_config[-1])
            
            # added_token_num = image patches - one image placeholder
            self.added_token_num = (image_size // patch_size)**2 - 1

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

    def _encode_single_image(self, image):
        torch.cuda.empty_cache()
        return self.model.segment_encoder.image_encoder(image.unsqueeze(0))
    
    def get_segment_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)

    def model_forward(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      bboxes: torch.FloatTensor, input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, masks_list: List[torch.FloatTensor],
                      label_list: List[torch.Tensor], resize_list: List[tuple], inference: bool = False, images_desc=None, **kwargs, ):
        """_summary_

        Args:
            global_enc_images (torch.FloatTensor): input for clip vision encoder
            grounding_enc_images (torch.FloatTensor): input for segment encoder
            bboxes (torch.FloatTensor): _description_
            input_ids (torch.LongTensor): _description_
            labels (torch.LongTensor): _description_
            attention_masks (torch.LongTensor): _description_
            offset (torch.LongTensor): _description_
            masks_list (List[torch.FloatTensor]): _description_
            label_list (List[torch.Tensor]): _description_
            resize_list (List[tuple]): _description_
            inference (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # Handle inference or training paths
        if inference:
            output_hidden_states = self._inference_path(input_ids, global_enc_images, attention_masks)
        else:
            output, output_hidden_states = self._training_path(global_enc_images, images_desc, input_ids, labels, attention_masks, offset)

        if grounding_enc_images is not None:
            # Extract grounding encoder image embeddings
            image_embeddings = self.get_segment_encoder_embs(grounding_enc_images)
            assert image_embeddings.shape[0] == len(offset) - 1

            seg_token_mask = self._create_seg_token_mask(input_ids, from_output=False)
            _, prompt_embeddings = self._extract_seg_hidden_states(output_hidden_states, seg_token_mask, offset, infer=inference)

            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                prompt_embeddings, image_embeddings, resize_list, label_list
            )
        else:
            pred_masks = None

        if inference:
            output_dict = {"pred_masks": pred_masks, "gt_masks": masks_list, }
        else:
            # Calculate losses
            loss_dict = self._calculate_losses(pred_masks, masks_list, output)
            output_dict = {'logits': output['logits'], **loss_dict}
        
        return output_dict

    def _create_seg_token_mask(self, token_ids, from_output=False):
        seg_mask = token_ids[:, 1:] == self.config.seg_token_idx
        if from_output:
            # for ids from output, the output_ids and hidden_states are not shifted
            return torch.cat([torch.zeros((seg_mask.shape[0], self.added_token_num)).bool().cuda(), seg_mask], dim=1)
        else:
            # for ids from input , the input_ids and hidden_states are shifted
            return torch.cat([torch.zeros((seg_mask.shape[0], self.added_token_num)).bool().cuda(), 
                              seg_mask, torch.zeros((seg_mask.shape[0], 1)).bool().cuda()], dim=1)

    def _inference_path(self, input_ids, global_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i:i + 1], attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1], output_hidden_states=True, )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(self, global_enc_images, images_desc, input_ids, labels, attention_masks, offset):
        global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)

        output = super().forward(
            images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True, images_desc=images_desc, )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_global_enc_image(self, global_enc_image, offset):
        global_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0)

    def _extract_seg_hidden_states(self, output_hidden_states, seg_token_mask, offset, infer=False):
        # output_hidden_states: (batch_size, seq_len, hidden_size)
        # TODO: determine if we need to use the first layer of text_hidden_fcs or all layers
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        if not infer:
            seg_token_offset = seg_token_offset[offset]

        seg_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            seg_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, seg_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model.segment_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.segment_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.segment_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            
            # During inference, we have original size list in place of label list
            orig_size = label_list[i].shape if not infer else label_list[i]

            pred_mask = self.model.segment_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _calculate_losses(self, pred_masks, masks_list, output):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output):
        # Initialize loss components
        device = output.loss.device
        ce_loss = output.loss * self.ce_loss_weight

        mask_bce_loss = torch.tensor(0.0, device=device)
        mask_dice_loss = torch.tensor(0.0, device=device)
        num_masks = 0

        if pred_masks:
            # Iterate over batch and compute mask-related losses
            for pred_mask, gt_mask in zip(pred_masks, masks_list):
                if pred_mask.numel() > 0 and gt_mask != []:  # Ensure pred_mask is not empty
                    # Resize gt_mask to match pred_mask if needed
                    if gt_mask.shape[0] != pred_mask.shape[0]:
                        gt_mask = gt_mask[:pred_mask.shape[0]]

                    assert gt_mask.shape[0] == pred_mask.shape[
                        0], f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"

                    # Compute Binary Cross-Entropy Loss
                    mask_bce_loss += (compute_sigmoid_cross_entropy(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) *
                                      gt_mask.shape[0])
                    # Compute Dice Loss
                    mask_dice_loss += (
                            calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])
                    num_masks += gt_mask.shape[0]

        # Normalize the losses
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss
        return {"loss": total_loss, "ce_loss": ce_loss, 
                "mask_bce_loss": mask_bce_loss, "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss}

    @torch.no_grad()
    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes, max_tokens_new=32, **kwargs):
        generation_outputs = self.generate(images=global_enc_images, input_ids=input_ids, max_new_tokens=max_tokens_new,
            num_beams=1, output_hidden_states=True, return_dict_in_generate=True, )

        output_hidden_states = generation_outputs.hidden_states
        generated_output_ids = generation_outputs.sequences

        if grounding_enc_images is not None:
            image_embeddings = self.get_segment_encoder_embs(grounding_enc_images)
            seg_token_mask = self._create_seg_token_mask(generated_output_ids, from_output=True)
            _, prompt_embeddings = self._extract_seg_hidden_states(output_hidden_states, seg_token_mask, offset=None, infer=True)

            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                prompt_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )
        else:
            pred_masks = None
        return generated_output_ids, pred_masks
