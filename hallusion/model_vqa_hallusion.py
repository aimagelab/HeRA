import argparse
import json
import os
import random

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


def process_images(images, image_processor, image_aspect_ratio="pad"):
    if not isinstance(images, list):
        images = [images]
    if image_aspect_ratio == "pad":
        bg = tuple(int(x * 255) for x in image_processor.image_mean)
        images = [expand2square(im, bg) for im in images]
    return image_processor(images, return_tensors="pt").pixel_values


def get_model_name_from_path(model_path):
    parts = model_path.strip("/").split("/")
    if parts[-1].startswith("checkpoint-"):
        return parts[-2] + "_" + parts[-1]
    return parts[-1]


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


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
    model.eval()
    return tokenizer, model, image_processor


def load_questions(question_file):
    question_file = os.path.expanduser(question_file)
    if question_file.endswith(".jsonl"):
        with open(question_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(question_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("questions", "data", "samples"):
            if isinstance(payload.get(key), list):
                return payload[key]

        # Accept a single-sample JSON object as input for quick testing.
        if "question" in payload and ("filename" in payload or "image" in payload):
            return [payload]

    raise ValueError(
        "Question file must be a JSON list, JSONL, or a dict containing a questions list"
    )


def resolve_image_path(image_folder, image_path):
    if image_path is None:
        return None

    image_path = os.path.expanduser(image_path)
    candidates = []

    if os.path.isabs(image_path):
        candidates.append(image_path)
    else:
        candidates.append(image_path)

    if image_folder:
        image_folder = os.path.expanduser(image_folder)
        candidates.append(os.path.join(image_folder, image_path))
        candidates.append(os.path.join(image_folder, image_path.lstrip("./")))

    for cand in candidates:
        norm = os.path.normpath(cand)
        if os.path.exists(norm):
            return norm

    return os.path.normpath(candidates[-1])


_YES_NO_SUFFIX = "\nAnswer the question using a single word or phrase: Yes or No."


def build_prompt(question, conv, use_image, mm_use_im_start_end):
    qs = question + _YES_NO_SUFFIX

    if use_image:
        if mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv.add_sys_prompt()
    conv.add_message(dict(role="user", content=qs))
    return conv.get_prompt(add_generation_prompt=True)


def eval_model(args):
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor = load_model(
        model_path,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
    )
    model_device = next(model.parameters()).device

    questions = load_questions(args.question_file)

    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a .jsonl file")

    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for sample_idx, line in enumerate(tqdm(questions, total=len(questions))):
            idx        = line.get("question_id", sample_idx)
            question   = line.get("question", line.get("text", ""))
            image_file = line.get("filename")   # may be None for no-visual samples

            if not question:
                raise ValueError(
                    f"Missing question text for sample index {sample_idx}"
                )

            has_image = (image_file is not None) and (
                str(line.get("visual_input", "1")) != "0"
            )

            conv   = CONV_MAPPING[args.conv_mode].new_empty()
            prompt = build_prompt(
                question,
                conv,
                use_image=has_image,
                mm_use_im_start_end=getattr(model.config, "mm_use_im_start_end", False),
            )

            input_ids = (
                tokenizer(prompt, return_tensors="pt")
                .input_ids
                .to(device=model_device, non_blocking=True)
            )

            pixel_values = None
            if has_image:
                image_path = resolve_image_path(args.image_folder, image_file)
                try:
                    image = Image.open(image_path).convert("RGB")
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Image not found: {image_path!r}  "
                        f"(sample index {sample_idx}, question_id={idx})"
                    )
                pixel_values = process_images(
                    [image],
                    image_processor,
                    image_aspect_ratio=model.config.image_aspect_ratio,
                ).to(device=model_device, non_blocking=True)

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

            output_text = (
                tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            )


            result = dict(line)
            result.update(
                {

                    "question_id": idx,
                    "image_id":    image_file,
                    "prompt":      question,
                    "text":        output_text,
                    "answer":      output_text,
                    "model_id":    model_name,
                    "metadata":    {},
                }
            )
            ans_file.write(json.dumps(result, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file",type=str, required=True)
    parser.add_argument(
        "--answers_file", type=str, default="./answers/hallusion_answers.jsonl"
    )
    parser.add_argument(
        "--conv_mode", type=str, required=True, choices=list(CONV_MAPPING.keys())
    )
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--top_p",          type=float, default=None)
    parser.add_argument("--num_beams",      type=int,   default=1)
    parser.add_argument("--max_new_tokens", type=int,   default=512)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "bfloat16", "fp16", "float16"],
    )
    args = parser.parse_args()
    print(args)
    eval_model(args)