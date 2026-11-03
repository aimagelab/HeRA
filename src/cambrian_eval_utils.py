import torch
from src.models.mllm import MLLMForCausalLM
from transformers import AutoTokenizer, AutoImageProcessor, AutoConfig
import src.custom_utils as utils

logger = utils.get_logger()


def process_images(images, image_processor, image_aspect_ratio='pad'):
    if image_aspect_ratio == 'pad':
        if isinstance(images, list):
            images = [utils.expand2square(image, tuple(int(x*255) for x in image_processor.image_mean)) for image in images]
        else:
            images = utils.expand2square(images, tuple(int(x*255) for x in image_processor.image_mean))
    
    if not isinstance(images, list):
        images = [images]

    return image_processor(images, return_tensors='pt').pixel_values


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


DTYPE_PREFIX_BF16 = "bf16__"


def apply_dtype_prefix_from_answers_file(model_name: str, answers_file: str) -> str:
    """Mirror the `bf16__` filename prefix onto model_name when the answers
    file was produced by a bf16 run.

    The Cambrian eval pipeline names answers files as
    `{dtype_prefix}answers_<run_name>.jsonl`, where `dtype_prefix` is `bf16__`
    for bf16 runs (see scripts/eval/cambrian/eval_cambrian_single.sh).
    Tabulation keys off the `model` column of experiments.csv, so we propagate
    the prefix here to keep bf16 and fp16 runs as distinct rows. Idempotent.
    """
    import os as _os
    base = _os.path.splitext(_os.path.basename(answers_file))[0]
    if base.startswith(DTYPE_PREFIX_BF16) and not model_name.startswith(DTYPE_PREFIX_BF16):
        return f"{DTYPE_PREFIX_BF16}{model_name}"
    return model_name


def resolve_torch_dtype(torch_dtype):
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    dtype_mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    normalized_torch_dtype = torch_dtype.lower()
    if normalized_torch_dtype not in dtype_mapping:
        raise ValueError(
            f"Unsupported torch_dtype '{torch_dtype}'. Expected one of: {', '.join(dtype_mapping)}."
        )

    return dtype_mapping[normalized_torch_dtype]


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, torch_dtype="float16", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    lm_torch_dtype = resolve_torch_dtype(torch_dtype)
    logger.info(f"Loading model with torch dtype {lm_torch_dtype}.")

    kwargs['lm_kwargs'] = dict(
        torch_dtype=lm_torch_dtype,
        attn_implementation='flash_attention_2' if use_flash_attn else 'sdpa'
    )

    config = None
    if utils.DEBUG_LOAD_FIRST_LLM_LAYERS:
        config = AutoConfig.from_pretrained(model_path)
        config.text_config.num_hidden_layers = utils.DEBUG_LOAD_FIRST_LLM_LAYERS
    model = MLLMForCausalLM.from_pretrained(model_path, config=config, **kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_config.name_or_path)
    model.init_vision_tokenizer(tokenizer)
    image_processor = AutoImageProcessor.from_pretrained(model.config.vision_config.name_or_path)
    model.init_image_processor(image_processor)
    
    # workaround: we hard-code `mm_use_im_start_end = False` in MLLMConfig to be compatible with llava and cambrian code.
    model.config.mm_use_im_start_end = False

    context_len = 4096

    return tokenizer, model, image_processor, context_len
