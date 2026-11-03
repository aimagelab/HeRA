from dataclasses import dataclass
from typing import Optional
from src.models.conversations import CONV_TEMPLATES
from transformers import AutoConfig, TrainingArguments
from src.models import ProjectorType
import src.custom_utils as utils
from transformers.trainer_utils import get_last_checkpoint
import random

logger = utils.get_logger()

@dataclass
class CustomTrainingArguments(TrainingArguments):
    resume_from_last_checkpoint: bool = False
    proj_learning_rate: Optional[float] = None
    tgt_proj_learning_rate: Optional[float] = None
    logit_scale_lr: Optional[float] = None
    train_proj_only: bool = False
    freeze_proj: bool = False
    freeze_tgt_proj: bool = False
    group_by_mod_lens: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.resume_from_last_checkpoint:
            try:
                check = get_last_checkpoint(self.output_dir)
            except FileNotFoundError:
                check = False
            if not check:
                self.resume_from_last_checkpoint = False
                logger.warning(f"No valid checkpoint found in {self.output_dir}. Setting `resume_from_last_checkpoint` to False")
        if self.resume_from_last_checkpoint and self.resume_from_checkpoint is None:
            self.resume_from_checkpoint = True
            logger.info(f"Setting `resume_from_checkpoint` to True because x`resume_from_last_checkpoint` is set to True and no checkpoint was specified in `resume_from_checkpoint`.")

        if self.gradient_checkpointing:
            self.gradient_checkpointing_kwargs = {'use_reentrant': False}


@dataclass
class DataArguments:
    train_data_path: Optional[str] = None
    train_image_folder: Optional[str] = None
    eval_data_path: Optional[str] = None
    eval_image_folder: Optional[str] = None
    conv_template: str = CONV_TEMPLATES.PLAIN
    prompt_max_length: Optional[int] = None

    lmdb_root: Optional[str] = None


@dataclass
class ModelArguments:
    attn_implementation: str = 'sdpa'

    checkpoint_path: Optional[str] = None
    override_bidir_attn_visual_tokens: bool = False

    # from scratch
    language_model_name: Optional[str] = None
    disable_rope_scaling: bool = False
    vision_model_name: Optional[str] = None
    vision_layer_idx: int = -2
    skip_left_visual_tokens: int = 1
    projector_type: Optional[ProjectorType] = ProjectorType.LINEAR
    projector_bias: bool = True
    bidir_attn_visual_tokens: Optional[bool] = None
    image_size: Optional[int] = None
    image_aspect_ratio: Optional[str] = None

    load_proj_weights: Optional[str] = None
    load_model_weights: Optional[str] = None

    # HeRA
    tgt_vision_model_name: Optional[str] = None
    tgt_image_processor: bool = False
    tgt_image_size: Optional[int] = None
    tgt_vision_layer_idx: int = -1
    tgt_skip_left_visual_tokens: int = 1
    tgt_projector_type: Optional[ProjectorType] = ProjectorType.LINEAR
    tgt_proj_intermediate_size: Optional[int] = None
    ra_llm_layers: Optional[str] = None
    ra_llm_layers_heads: Optional[str] = None
    ra_llm_layers_heads_gqa: bool = False
    ra_loss_w: float = 1.0
    hera: bool = False
    hera_topk: int = 10
    hera_learn_logit_scale: bool = True
    hera_logit_scale_init_val: float = 2.659260036932778
    hera_untie_logit_scale: bool = False

    def __post_init__(self):
        assert not self.ra_llm_layers or not self.ra_llm_layers_heads
        if self.ra_llm_layers_heads:
            # "0: 0 3: 4 5   7 -1 : 9"
            # layer 0 -> heads 0 and 3
            # layer 3 -> heads 4, 5, and 7
            # last layer -> head 9
            splitstr = self.ra_llm_layers_heads.split(':')
            layersheads = {}
            for i in range(1, len(splitstr)):
                layer = int(splitstr[i - 1].split()[-1])
                heads = [int(x) for x in splitstr[i].split()]
                if len(heads) > 1 and i < len(splitstr) - 1:
                    heads = heads[:-1]
                layersheads[layer] = heads
            self.ra_llm_layers_heads =  layersheads
        
        if self.ra_llm_layers:
            self.ra_llm_layers = [int(x) for x in self.ra_llm_layers.split()]


def postprocess_args(training_args: TrainingArguments, model_args: ModelArguments, data_args: DataArguments):
    model_args.seed = training_args.seed
    model_args.gradient_checkpointing = training_args.gradient_checkpointing
    model_args.train_proj_only = training_args.train_proj_only
    model_args.freeze_proj = training_args.freeze_proj
    model_args.freeze_tgt_proj = training_args.freeze_tgt_proj

    data_args.image_aspect_ratio = model_args.image_aspect_ratio
    data_args.tgt_image_size = model_args.tgt_image_size

    return training_args, model_args, data_args