from .configuration_mllm_align import *
from .modeling_mllm_align import *
from transformers import AutoConfig, AutoModel

AutoConfig.register(MLLMAlignConfig.model_type, MLLMAlignConfig)
AutoModel.register(MLLMAlignConfig, MLLMAlignModel)
