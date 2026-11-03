"""
Answer generation for AMBER. Note this is intended to be run separately for generative and discriminative modes,
in order to parse correctly the yes/no answers in the latter.  
"""

import argparse
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, set_seed

from src.models.conversations import (
    CONV_MAPPING,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from src.models.mllm import MLLMForCausalLM


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def process_images(images, image_processor, image_aspect_ratio='pad'):
    if not isinstance(images, list):
        images = [images]
    if image_aspect_ratio == 'pad':
        bg = tuple(int(x * 255) for x in image_processor.image_mean)
        images = [expand2square(im, bg) for im in images]
    return image_processor(images, return_tensors='pt').pixel_values


def get_model_name_from_path(model_path):
    parts = model_path.strip("/").split("/")
    if parts[-1].startswith('checkpoint-'):
        return parts[-2] + "_" + parts[-1]
    return parts[-1]


_DTYPE_MAP = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
              "fp16": torch.float16, "float16": torch.float16}


def load_model(model_path, device_map="auto", attn_implementation="sdpa", dtype="bf16"):
    lm_kwargs = dict(
        torch_dtype=_DTYPE_MAP[dtype],
        attn_implementation=attn_implementation,
    )
    model = MLLMForCausalLM.from_pretrained(model_path, device_map=device_map, lm_kwargs=lm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_config.name_or_path)
    model.init_vision_tokenizer(tokenizer)
    image_processor = AutoImageProcessor.from_pretrained(model.config.vision_config.name_or_path)
    model.init_image_processor(image_processor)
    model.config.mm_use_im_start_end = False
    return tokenizer, model, image_processor


def eval_model(args):
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if "generative" in args.question_file:
        mode = "generative"
    elif "discriminative" in args.question_file:
        mode = "discriminative"
    else:
        raise ValueError("This script is intended to be run separately for generative and discriminative modes.")
    

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor = load_model(
        model_path, attn_implementation=args.attn_implementation, dtype=args.dtype)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))

    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".json"):
        raise ValueError("Answers file must be a .json file")

    answers_dir = os.path.dirname(answers_file)
    os.makedirs(answers_dir, exist_ok=True)

    results = []
    for line in tqdm(questions, total=len(questions)):
        qs = line["query"]
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs += args.test_prompt

        conv = CONV_MAPPING[args.conv_mode].new_empty()
        conv.add_sys_prompt()
        conv.add_message(dict(role='user', content=qs))
        prompt = conv.get_prompt(add_generation_prompt=True)

        image = Image.open(os.path.join(args.image_folder, line["image"])).convert('RGB')
        pixel_values = process_images(
            [image], image_processor,
            image_aspect_ratio=model.config.image_aspect_ratio,
        )

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device='cuda', non_blocking=True)
        pixel_values = pixel_values.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if mode == "discriminative": 
            raw_response = outputs
            if not args.test_prompt:
                # regex matching is active only if the user does not provide 
                # specific answering instructions via prompt
                match = re.search(r'^\s*(yes|no)\b', outputs, re.IGNORECASE)
                if match:
                    outputs = match.group(1)
                else:
                    outputs = ""
                    print(f"Warning: Could not parse yes/no answer from model output: '{raw_response}' for question id {line['id']}")

        entry = {
            "id": line["id"],
            "prompt": line["query"] + args.test_prompt,
            "response": outputs.capitalize(),
            "model_id": model_name,
            "metadata": {},
        }
        if mode == "discriminative":
            entry["raw_response"] = raw_response
        results.append(entry)

    with open(answers_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--conv_mode", type=str, required=True, choices=list(CONV_MAPPING.keys()))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--test_prompt", type=str, default="")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "bfloat16", "fp16", "float16"])
    args = parser.parse_args()
    print(args)

    eval_model(args)
