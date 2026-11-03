from typing import Dict, Optional, Union
from transformers import PretrainedConfig, AutoConfig
from ..conversations import DEFAULT_IMAGE_TOKEN
from src.models.model_utils import ProjectorType


class MLLMConfig(PretrainedConfig):
    model_type = 'mllm'
    is_composition = True
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        text_config: Optional[Union[PretrainedConfig, Dict]] = None,
        vision_config: Optional[Union[PretrainedConfig, Dict]] = None,
        projector_type: Optional[ProjectorType] = ProjectorType.LINEAR,
        projector_bias: Optional[bool] = False,
        image_token: str = DEFAULT_IMAGE_TOKEN,
        image_token_id: Optional[int] = None,
        bidir_attn_visual_tokens: bool = False,
        vision_layer_idx: int = -2,
        skip_left_visual_tokens: int = 1,
        image_size: Optional[int] = None,
        image_aspect_ratio: Optional[str] = None,
        **kwargs
    ):
        assert bidir_attn_visual_tokens is not None
        super().__init__(**kwargs)

        if isinstance(text_config, PretrainedConfig) or text_config is None:
            self.text_config = text_config
        else:
            self.text_config = AutoConfig.for_model(text_config.pop('model_type'), **text_config)

        if isinstance(vision_config, PretrainedConfig) or vision_config is None:
            self.vision_config = vision_config
        else:
            vision_model_type = vision_config.pop('model_type')
            if vision_model_type == 'clip':
                vision_model_type = 'clip_vision_model'
            elif vision_model_type == 'siglip':
                vision_model_type = 'siglip_vision_model'
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config)
        
        self.projector_type = projector_type
        self.projector_bias = projector_bias
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.bidir_attn_visual_tokens = bidir_attn_visual_tokens
        assert vision_layer_idx < 0, "Change `MLLM.vision_model_forward` to use positive indexing to select the visual layer."
        self.vision_layer_idx = vision_layer_idx
        self.skip_left_visual_tokens = skip_left_visual_tokens
        self.image_size = image_size
        self.image_aspect_ratio = image_aspect_ratio

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

__all__ = [
    "MLLMConfig"
]