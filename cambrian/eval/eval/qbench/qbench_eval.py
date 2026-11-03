import argparse
import csv
import json
import math
import os
import random
import tarfile

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from src.cambrian_eval_utils import get_model_name_from_path, load_pretrained_model, process_images
from src.models.conversations import CONV_MAPPING, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def process(line, args, tokenizer, image_processor, model_config, images):
    qs = line["question"] + " Options:"
    options = line["candidates"]
    for i in range(len(options)):
        option = chr(ord('A')+i)
        text = options[i]
        qs += f"\n{option}. {text}"

    qs += f"\n{args.question_extension}"

    image_path = images[line["img_path"]]
    input_image = Image.open(image_path).convert('RGB')

    if input_image is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = CONV_MAPPING[args.conv_mode].new_empty()
    conv.add_sys_prompt()
    conv.add_message(dict(role='user', content=qs))
    prompt = conv.get_prompt(add_generation_prompt=True)
    if input_image is None:
        image = None
        image_size = None
        pixel_values = None
    else:
        image = input_image
        image_size = [image.size]
        pixel_values = process_images([image], image_processor, image_aspect_ratio=model_config.image_aspect_ratio)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

    return input_ids, pixel_values, image_size, prompt

def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result

def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, torch_dtype=args.torch_dtype)

    images = {}
    file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="images_llvisionqa.tar", repo_type="dataset")
    extract_dir = os.path.dirname(file_path)
    if not os.path.exists(extract_dir+"/images"):
        with tarfile.open(file_path, "r:") as tar:
            tar.extractall(path=extract_dir)

    files_in_dir = os.listdir(extract_dir+"/images/")
    for file in files_in_dir:
        images[file] = extract_dir+"/images/"+file

    questions = []
    dev_file_path = hf_hub_download(repo_id="teowu/LLVisionQA-QBench", filename="llvisionqa_dev.json", repo_type="dataset")

    with open(dev_file_path, "r") as json_file:
        questions.extend(json.load(json_file))
    
    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")

    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    ans_file = open(chunk_file, "w")

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)
    for line in tqdm(questions, total=len(questions)):
        idx = idx+1
        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue
        
        input_ids, pixel_values, image_sizes, prompt = process(line, args, tokenizer, image_processor, model.config, images)
        answer = line["correct_ans"] 
        options = line["candidates"]
        reverse_options = {}
        for ind, option in enumerate(options):
            reverse_options[option] = chr(ord('A')+ind)
        #gt_answer
        gt_answer = reverse_options[answer]
        qn_type = line["type"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        pixel_values = pixel_values.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": prompt,
            "answer": outputs,
            "gt_answer": gt_answer,
            "type": qn_type,
            "model_id": model_name
        }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Please answer directly with only the letter of the correct option and nothing else.")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", type=str, default="float16")
    args = parser.parse_args()

    eval_model(args)

