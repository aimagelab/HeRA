from transformers import AutoConfig
from src.models.model_utils import ProjectorType
from src.models.mllm.configuration_mllm import MLLMConfig
from typing import Dict, List, Optional


class MLLMAlignConfig(MLLMConfig):
    model_type = "mllm-align"

    def __init__(
        self,
        tgt_vision_model_name: Optional[str] = None,
        tgt_vision_layer_idx: Optional[int] = None,
        tgt_skip_left_visual_tokens: Optional[int] = None, 
        tgt_projector_type: Optional[ProjectorType] = ProjectorType.LINEAR,
        tgt_proj_intermediate_size: Optional[int] = None,
        ra_llm_layers: Optional[List[int]] = None,
        ra_llm_layers_heads: Optional[Dict[int, List[int]]] = None,
        ra_llm_layers_heads_gqa: bool = False,
        ra_loss_w: float = 1.0,
        hera: bool = False,
        hera_topk: Optional[int] = 10,
        hera_learn_logit_scale: bool = True,
        hera_logit_scale_init_val: float = 2.659260036932778, # np.log(1/0.07)
        hera_untie_logit_scale: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.tgt_vision_model_name = tgt_vision_model_name
        self.tgt_vision_config = None
        if self.tgt_vision_model_name is None or self.tgt_vision_model_name == self.vision_config.name_or_path:
                self.tgt_vision_config = self.vision_config
        else:
            self.tgt_vision_config = AutoConfig.from_pretrained(self.tgt_vision_model_name)
            if hasattr(self.tgt_vision_config, 'vision_config'):
                self.tgt_vision_config = self.tgt_vision_config.vision_config
        
        assert tgt_vision_layer_idx is None or tgt_vision_layer_idx < 0, "`tgt_vision_layer_idx` is used as a negative index, i.e. -1 --> last layer, -2 --> second-last layer, etc."
        self.tgt_vision_layer_idx = tgt_vision_layer_idx
        self.tgt_skip_left_visual_tokens = tgt_skip_left_visual_tokens
        self.tgt_projector_type = tgt_projector_type
        self.tgt_proj_intermediate_size = tgt_proj_intermediate_size
        
        self.ra_llm_layers = ra_llm_layers
        self.ra_llm_layers_heads = ra_llm_layers_heads
        self.ra_llm_layers_heads_gqa = ra_llm_layers_heads_gqa
        self.ra_loss_w = ra_loss_w

        self.hera = hera
        self.hera_topk = hera_topk
        self.hera_learn_logit_scale = hera_learn_logit_scale
        self.hera_logit_scale_init_val = hera_logit_scale_init_val
        self.hera_untie_logit_scale = hera_untie_logit_scale

    def to_dict(self):
        ret = super().to_dict()
        ret.pop('tgt_vision_config')
        return ret
    

__all__ = [
    'MLLMAlignConfig'
]