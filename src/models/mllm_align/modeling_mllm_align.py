import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel
from src.models.mllm.modeling_mllm import MLLMPreTrainedModel
from .configuration_mllm_align import MLLMAlignConfig
import src.custom_utils as utils
from src.models.model_utils import (
    build_vision_encoder,
    build_projector,
    deepspeed_zero3_ctx,
    differentiable_all_gather_varlen
)
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
import torch.nn.functional as F
import torch.distributed as dist

logger = utils.get_logger()


def get_hera_topk_targets(
    tgt_norm_flat: torch.Tensor,
    local_bsz_flat: int,
    rank_offset: int,
    topk: Optional[int]
) -> torch.Tensor:
    all_tgt = differentiable_all_gather_varlen(tgt_norm_flat)
    tgt_kernel = tgt_norm_flat @ all_tgt.T # (local_N, global_N)
    tgt_kernel[:, rank_offset:rank_offset + local_bsz_flat].fill_diagonal_(float('-inf'))

    _, topk_idxs = tgt_kernel.topk(topk, dim=-1)
    targets = torch.zeros_like(tgt_kernel)
    targets.scatter_(1, topk_idxs, 1.0 / topk)

    return targets


def distributed_hera_loss(
    preds_flat: torch.Tensor,
    targets: torch.Tensor,
    local_bsz_flat: int,
    rank_offset: int,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Contrastive loss over all ranks for a single representation type.

    preds_flat : (local_N, H, D) — will be L2-normalised inside
    targets    : (local_N, global_N) — soft labels, rows sum to 1
    Returns scalar loss averaged over heads.
    """
    preds_flat = F.normalize(preds_flat, dim=-1)
    all_preds = differentiable_all_gather_varlen(preds_flat)
    all_preds = all_preds.transpose(
        0, 2)                    # (D, H, global_N)
    nheads = preds_flat.size(1)
    loss = 0.0
    for i in range(nheads):
        logits = preds_flat[:, i] @ all_preds[:, i] \
            * logit_scale[min(i, len(logit_scale) - 1)]
        logits[:, rank_offset:rank_offset + local_bsz_flat].fill_diagonal_(-1e6)
        loss += F.cross_entropy(logits, targets, reduction='mean')
    return loss / nheads
    

@dataclass
class CausalLMAlignOutputWithPast(CausalLMOutputWithPast):
    loss_lm: Optional[torch.FloatTensor | float] = None
    loss_ra: Optional[torch.FloatTensor | float] = None


class CustomOProj(nn.Linear):
    def __init__(
        self,
        n_heads: int,
        head_idxs: List[int],
        in_features: int,
        out_features: int,
        n_kv_heads: int = 0,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        assert not bias
        super().__init__(in_features, out_features, bias, device, dtype)
        self.n_heads = n_heads
        self.head_idxs = head_idxs
        self.n_kv_heads = n_kv_heads
        self.reset()

    def reset(self):
        self._saved_input = None

    def forward(self, input):
        # Store input for later recomputation outside checkpointed region.
        # With use_reentrant=False, this tensor maintains its gradient connection.
        self._saved_input = input
        # Standard o_proj: combine all heads
        return F.linear(input, self.weight, self.bias)

    def compute_head_predictions(self) -> torch.Tensor:
        """
        Recompute per-head predictions from saved input.
        Must be called AFTER the LLM forward pass completes, outside the
        checkpointed region, so gradients flow properly.
        """
        input = self._saved_input
        bsz, seq_len, input_dim = input.shape
        headwise_input = input.reshape(
            bsz, seq_len, self.n_heads, -1).contiguous()

        head_dim = headwise_input.size(-1)
        W_O_sliced = self.weight.view(
            self.weight.size(0), self.n_heads, head_dim)
        headwise_output = torch.einsum(
            "bshd, mhd -> bshm", headwise_input, W_O_sliced)
        
        if self.n_kv_heads:
            out_dim = W_O_sliced.size(0)
            n_q_heads_per_grp = self.n_heads // self.n_kv_heads
            headwise_output = headwise_output.reshape(
                bsz, seq_len, self.n_kv_heads, n_q_heads_per_grp, out_dim).sum(3)
        
        headwise_output = headwise_output[:, :, self.head_idxs] # N, L, n_heads, D
        
        self._saved_input = None
        return headwise_output


class MLLMAlignModel(MLLMPreTrainedModel):
    config_class = MLLMAlignConfig

    def __init__(
        self,
        config: MLLMAlignConfig,
        tgt_vision_model: Optional[PreTrainedModel] = None,
        **kwargs
    ):
        if config.ra_llm_layers_heads_gqa:
            n_kv_heads = config.text_config.num_key_value_heads
            all_head_idxs = [
                head_idx for layer in config.ra_llm_layers_heads.values() for head_idx in layer]
            assert all(i < n_kv_heads for i in all_head_idxs), "When `ra_llm_layers_heads_gqa` is True, the head indices in `ra_llm_layers_heads` should correspond to GQA group indices, which should be less than the number of key-value heads."

        super().__init__(config, **kwargs)

        if tgt_vision_model is None:
            tgt_vision_model = build_vision_encoder(
                config.tgt_vision_model_name)
        else:
            assert tgt_vision_model.config.name_or_path == config.tgt_vision_model_name, f"The provided `tgt_vision_model` has a different `name_or_path` ({tgt_vision_model.config.name_or_path}) from the specified `tgt_vision_model_name` ({config.tgt_vision_model_name}). Please make sure they match."
        tgt_vision_model.requires_grad_(False)
        tgt_vision_model.eval()
        self.tgt_vision_model = tgt_vision_model

        n_layers = None
        if self.config.ra_llm_layers:
            n_layers = len(self.config.ra_llm_layers)
        elif self.config.ra_llm_layers_heads:
            n_layers = sum(len(x)
                           for x in self.config.ra_llm_layers_heads.values())
        tgt_proj_kwargs = dict(
            projector_type=config.tgt_projector_type,
            input_size=config.text_config.hidden_size,
            output_size=config.tgt_vision_config.hidden_size,
            bias=False,
            n_layers=n_layers
        )
        if config.tgt_proj_intermediate_size:
            tgt_proj_kwargs['intermediate_size'] = config.tgt_proj_intermediate_size
        self.tgt_projector = build_projector(**tgt_proj_kwargs).to(self.language_model.dtype)

        self.o_proj_register = {}

        n_logit_scales = 1
        if config.hera_untie_logit_scale:
            n_logit_scales = n_layers
        logit_scale = torch.tensor(
            [config.hera_logit_scale_init_val] * n_logit_scales, dtype=torch.float32)
        self.register_buffer("logit_scale", logit_scale)
        if config.hera_learn_logit_scale:
            self.logit_scale = nn.Parameter(logit_scale)

    def register_modules_for_alignment(self):
        if self.config.ra_llm_layers_heads:
            layer_modulelist = []

            def find_lm_layer_modulelist(root, n_layers):
                for name, child in root.named_children():
                    if isinstance(child, nn.ModuleList) and 'layer' in name.lower() and len(child) == n_layers:
                        layer_modulelist.append(child)
                    find_lm_layer_modulelist(child, n_layers)

            find_lm_layer_modulelist(
                self.language_model, self.language_model.config.num_hidden_layers)
            assert len(layer_modulelist) == 1
            layer_modulelist = layer_modulelist[0]

            for layer_idx, head_idxs in self.config.ra_llm_layers_heads.items():
                found = []

                def find_o_proj(root):
                    for name, child in root.named_children():
                        if name == 'o_proj':
                            found.append(root)
                            found.append(child)
                            break
                        find_o_proj(child)
                find_o_proj(layer_modulelist[layer_idx])
                o_proj_parent, original_o_proj = found
                n_kv_heads = 0
                if self.config.ra_llm_layers_heads_gqa:
                    n_kv_heads = self.language_model.config.num_key_value_heads
                with deepspeed_zero3_ctx(original_o_proj.parameters()):
                    o_proj_parent.o_proj = CustomOProj(
                        self.language_model.config.num_attention_heads,
                        head_idxs,
                        in_features=original_o_proj.weight.size(1),
                        out_features=original_o_proj.weight.size(0),
                        n_kv_heads=n_kv_heads,
                        bias=original_o_proj.bias is not None,
                        device=original_o_proj.weight.device,
                        dtype=original_o_proj.weight.dtype
                    )
                    o_proj_parent.o_proj.load_state_dict(
                        original_o_proj.state_dict())
                self.o_proj_register[layer_idx] = o_proj_parent.o_proj

            logger.info(f"Total alignment layers: {len(self.o_proj_register)}")

    def get_heads_predictions_and_reset(self, module_list: nn.ModuleList = None):
        preds = []
        i = 0
        for proj in self.o_proj_register.values():
            # Recompute predictions outside checkpointed region for proper gradients
            head_pred = proj.compute_head_predictions()
            if module_list is None:
                preds.append(head_pred)
            else:
                for j in range(head_pred.size(2)):
                    preds.append(module_list[i](head_pred[:, :, j:j + 1]))
                    i += 1
        return torch.cat(preds, dim=2)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        tgt_pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache,
                                        List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMAlignOutputWithPast]:
        assert inputs_embeds is None

        with torch.no_grad():
            visual_embeds, visual_embeds_pool = self.vision_model_forward(
                pixel_values)
            if self.tgt_vision_model is self.vision_model and \
               tgt_pixel_values is None and \
                    self.config.vision_layer_idx == self.config.tgt_vision_layer_idx:
                tgt_visual_embeds = visual_embeds
                tgt_visual_embeds_pool = visual_embeds_pool
            else:
                tgt_pixel_values = pixel_values if tgt_pixel_values is None else tgt_pixel_values
                tgt_visual_embeds, tgt_visual_embeds_pool = self.vision_model_forward(
                    tgt_pixel_values,
                    vision_model=self.tgt_vision_model,
                    vision_layer_idx=self.config.tgt_vision_layer_idx,
                    skip_left_visual_tokens=self.config.tgt_skip_left_visual_tokens
                )

        visual_embeds = self.projector(visual_embeds)

        inputs_embeds, attention_mask, labels, position_ids, visual_position_ids = self.embed_tokens(
            input_ids, attention_mask, labels, visual_embeds, position_ids)
        input_ids = None

        # assuming the LLM uses RoPE, we do not need, `position_ids` for most cases.
        # If we do not set it to None, then there will be problem in `prepare_inputs_for_generation` if using full attention mask on image tokens
        position_ids = None

        align_llm_layer_post_mlp = bool(self.config.ra_llm_layers)
        if align_llm_layer_post_mlp:
            output_hidden_states = True

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )

        if return_dict:
            outputs['visual_position_ids'] = visual_position_ids

        image_mask &= (labels != -100).any(dim=1)
        # `force_loss_ra` is required to avoid deadlocks in distributed training with model sharding, i.e., DeepSpeed ZeRO-3
        force_loss_ra = not image_mask.any() and not (labels != -100).any(1).all()
        if self.config.hera:
            n_imgs = image_mask.sum()
            all_n_imgs = [torch.empty_like(n_imgs)
                          for _ in range(utils.get_world_size())]
            dist.all_gather(all_n_imgs, n_imgs)
            if force_loss_ra or any(len(labels) != local_n_imgs.item() for local_n_imgs in all_n_imgs):
                force_loss_ra = True
                
        if force_loss_ra:
            n_dummy_vembeds = int(inputs_embeds.size(1) ** 0.5) ** 2
            if self.config.hera:
                # Assuming that the seq length is larger than 1,
                # this guarantees that no empty sequences will be produced while gathering the `preds`.
                # Conversely, if we stick with `n_dummy_vembeds = int(inputs_embeds.size(1) ** 0.5) ** 2`,
                # that is required to preserve the 2D spatial structure in case of spatial mismatch with the tgt vision encoder,
                # then `preds` will contain empty sequences for the samples without images, which will cause NaNs in the loss when gathering across the batch for the contrastive loss.
                n_dummy_vembeds = 1
            visual_position_ids[0] = [[0, n_dummy_vembeds]]
            image_mask[0] = True

        if align_llm_layer_post_mlp:
            layers_hidden_state = [outputs['hidden_states'][i + 1]
                                   for i in self.config.ra_llm_layers]
            if self.config.tgt_projector_type.startswith('multi_'):
                preds = torch.stack([self.tgt_projector[i](
                    lhs) for i, lhs in enumerate(layers_hidden_state)], dim=2)
            else:
                preds = self.tgt_projector(
                    torch.stack(layers_hidden_state, dim=2))
        else:
            if self.config.tgt_projector_type.startswith('multi_'):
                preds = self.get_heads_predictions_and_reset(
                    self.tgt_projector)
            else:
                preds = self.tgt_projector(
                    self.get_heads_predictions_and_reset())

        if self.config.hera:
            def _pool_multiturn(i, vis_pos, blacklist_tokens=None):
                idxs = labels[i] != -100
                if blacklist_tokens is not None:
                    for tok_id in blacklist_tokens:
                        idxs &= (labels[i] != tok_id)
                idxs = idxs.nonzero().squeeze(-1)
                return preds[i, idxs].mean(dim=0, keepdim=True)
            
            preds = torch.stack([_pool_multiturn(i, vp)
                                    for i, vp in enumerate(visual_position_ids) if image_mask[i]])

        else:
            preds = torch.stack([preds[i, vis_pos[0][0]:vis_pos[0][1]]
                                 for i, vis_pos in enumerate(visual_position_ids) if image_mask[i]])

        tgt_visual_embeds = tgt_visual_embeds[image_mask]
        tgt_visual_embeds_pool = tgt_visual_embeds_pool[image_mask].unsqueeze(1)

        if self.config.hera:
            del tgt_visual_embeds
            tgt_visual_embeds = tgt_visual_embeds_pool.flatten(0, 1)
            
            local_bsz_flat = len(tgt_visual_embeds)
            rank, local_rank = utils.get_rank(), utils.get_local_rank()
            all_bsz_flat = torch.empty(
                utils.get_world_size(), dtype=torch.long, device=local_rank)
            dist.all_gather_into_tensor(all_bsz_flat, torch.tensor(
                local_bsz_flat, device=local_rank))
            rank_offset = all_bsz_flat.cumsum(0)[rank].item() - local_bsz_flat

            tgt_visual_embeds = F.normalize(tgt_visual_embeds, dim=-1)
            targets = get_hera_topk_targets(
                tgt_visual_embeds, 
                local_bsz_flat, 
                rank_offset,
                self.config.hera_topk
            )
            del tgt_visual_embeds

            logit_scale = self.logit_scale.exp().clamp(max=100.0)
            loss_ra = distributed_hera_loss(
                preds.flatten(0, 1), 
                targets, 
                local_bsz_flat, 
                rank_offset, 
                logit_scale
            )

        else:
            del tgt_visual_embeds_pool

            if preds.size(1) != tgt_visual_embeds.size(1):
                bsz, patches, n_heads, hs = preds.shape
                patches_per_side = int(patches ** 0.5)
                tgt_patches_per_side = int(tgt_visual_embeds.size(1) ** 0.5)
                tgt_hs = tgt_visual_embeds.size(2)
                tgt_visual_embeds = F.interpolate(
                    tgt_visual_embeds.permute(0, 2, 1).reshape(
                        bsz, tgt_hs, tgt_patches_per_side, tgt_patches_per_side
                    ),
                    (patches_per_side, patches_per_side),
                    mode='bilinear',
                    align_corners=False
                ).flatten(-2).reshape(bsz, tgt_hs, -1).permute(0, 2, 1)            
            
            if tgt_visual_embeds.dim() == 4:
                loss_ra = - F.cosine_similarity(
                    preds.unsqueeze(3), tgt_visual_embeds.unsqueeze(2), dim=-1).mean()
            else:
                loss_ra = - F.cosine_similarity(
                    preds, tgt_visual_embeds.unsqueeze(2), dim=-1).mean()

        if force_loss_ra:
            loss_ra *= 0.0
            logger.info(
                f"[Rank {utils.get_rank()}] `loss_ra` set to 0.0. Reason can be: no images in the batch, or mismatch in the number of images across GPUs (if training with HeRA).")

        outputs['loss'] += (loss_ra * self.config.ra_loss_w)
        return CausalLMAlignOutputWithPast(
            **outputs,
            loss_lm=outputs['loss'].item(),
            loss_ra=loss_ra.item()
        )


__all__ = [
    'MLLMAlignModel'
]
