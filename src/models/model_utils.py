from transformers import (
    AutoModel,
    AutoConfig,
    CLIPVisionModel,
    CLIPVisionConfig,
    SiglipVisionModel,
    SiglipVisionConfig
)
from enum import StrEnum
import torch.nn as nn
from transformers.integrations import is_deepspeed_zero3_enabled
import deepspeed
from contextlib import nullcontext
import torch
from pathlib import Path
import src.custom_utils as utils
import torch.distributed as dist

logger = utils.get_logger()


class ProjectorType(StrEnum):
    IDENTITY = 'identity'
    LINEAR = 'linear'
    MULTI_LINEAR = 'multi_linear'
    MLP = 'mlp'
    MULTI_MLP = 'multi_mlp'
    MLP_VIRAL = 'mlp_viral'


def build_projector(projector_type, **kwargs):
    if projector_type == ProjectorType.IDENTITY:
        return nn.Identity()
    elif projector_type == ProjectorType.LINEAR:
        return nn.Linear(
            in_features=kwargs['input_size'],
            out_features=kwargs['output_size'],
            bias=kwargs['bias']
        )
    elif projector_type == ProjectorType.MULTI_LINEAR:
        n_layers = kwargs['n_layers']
        return nn.ModuleList(
            nn.Linear(kwargs['input_size'],
                      kwargs['output_size'], bias=kwargs['bias']) for _ in range(n_layers)
        )
    elif projector_type == ProjectorType.MLP:
        intermediate_size = kwargs.get('intermediate_size', kwargs['output_size'])
        return nn.Sequential(
            nn.Linear(kwargs['input_size'],
                      intermediate_size, bias=kwargs['bias']),
            nn.GELU(),
            nn.Linear(intermediate_size,
                      kwargs['output_size'], bias=kwargs['bias'])
        )
    elif projector_type == ProjectorType.MULTI_MLP:
        n_layers = kwargs['n_layers']
        intermediate_size = kwargs.get('intermediate_size', kwargs['output_size'])
        return nn.ModuleList(
            nn.Sequential(
                nn.Linear(kwargs['input_size'],
                        intermediate_size, bias=kwargs['bias']),
                nn.GELU(),
                nn.Linear(intermediate_size,
                        kwargs['output_size'], bias=kwargs['bias'])
            )  for _ in range(n_layers)
        )   
    elif projector_type == ProjectorType.MLP_VIRAL:
        intermediate_size = kwargs.get('intermediate_size', kwargs['output_size'])
        return nn.Sequential(
                    nn.Linear(kwargs['input_size'], intermediate_size),
                    nn.SiLU(),
                    nn.Linear(intermediate_size, intermediate_size),
                    nn.SiLU(),
                    nn.Linear(intermediate_size, kwargs['output_size'])
                )
    else:
        raise ValueError(f'Projector type {projector_type} not supported')


def build_vision_encoder(name_or_path: str):
    vision_cls = AutoModel
    config_cls = AutoConfig
    if 'openai/clip' in name_or_path:
        vision_cls = CLIPVisionModel
        config_cls = CLIPVisionConfig
    elif 'siglip' in name_or_path.lower():
        vision_cls = SiglipVisionModel
        config_cls = SiglipVisionConfig
    config = None
    if utils.DEBUG_LOAD_FIRST_VIT_LAYERS:
        config = config_cls.from_pretrained(name_or_path)
        config.num_hidden_layers = utils.DEBUG_LOAD_FIRST_VIT_LAYERS
        logger.info(f"Using a {utils.DEBUG_LOAD_FIRST_VIT_LAYERS}-layer vision model for debugging")
    return vision_cls.from_pretrained(name_or_path, config=config)


def deepspeed_zero3_ctx(params_iter):
    ctx = nullcontext()
    if is_deepspeed_zero3_enabled():
        ctx = deepspeed.zero.GatheredParameters(params_iter, modifier_rank=0)
    return ctx


def load_torch_weights(module: nn.Module, path: str | Path):
    weights = torch.load(path, map_location='cpu')
    if is_deepspeed_zero3_enabled():
        with deepspeed.zero.GatheredParameters(module.parameters(), modifier_rank=0):
            module.load_state_dict(weights)
    else:
        module.load_state_dict(weights)    


class DiffAllGatherVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        if not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Exchange batch sizes
        local_bs = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
        all_bs = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
        dist.all_gather(all_bs, local_bs)
        all_bs = [s.item() for s in all_bs]

        # Allocate correctly-sized output tensors (no padding!)
        gathered = [
            torch.zeros(bs, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
            for bs in all_bs
        ]
        dist.all_gather(gathered, tensor.contiguous())

        ctx.rank = rank
        ctx.all_batch_sizes = all_bs

        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if not dist.is_initialized():
            return grad_output
        
        # Sum gradient contributions across all ranks
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output)
        
        # Return this rank's slice
        splits = grad_output.split(ctx.all_batch_sizes, dim=0)
        return splits[ctx.rank].contiguous()


def differentiable_all_gather_varlen(tensor):
    return DiffAllGatherVarlen.apply(tensor)
