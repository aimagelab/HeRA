import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path
import json
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

from src.custom_utils import DEBUG
from src.models import MLLMForCausalLM
from src.models.conversations import DEFAULT_IMAGE_TOKEN, CONV_MAPPING, CONV_TEMPLATES


def apply_pooling(embeddings, pooling_mode, attention_mask=None, labels=None):
    if labels is not None and pooling_mode != 'avg':
        raise NotImplementedError()
    
    if pooling_mode == 'eos':
        eos_idx = int(attention_mask.sum().item()) - 1 if attention_mask is not None else -1
        return embeddings[eos_idx].clone()
    elif pooling_mode == 'avg':
        if labels is not None:
            assert len(embeddings) == len(labels)
            valid = embeddings[labels != -100]
        else:
            valid = embeddings[attention_mask.bool()] if attention_mask is not None else embeddings
        return valid.mean(dim=0)
    elif pooling_mode == 'wavg':
        valid = embeddings[attention_mask.bool()] if attention_mask is not None else embeddings
        weights = (torch.arange(1, valid.size(0) + 1) ** 2).to(dtype=valid.dtype, device=valid.device)
        weights /= weights.sum()
        return (valid * weights[:, None]).sum(0)
    raise ValueError(f"Unknown pooling mode: {pooling_mode}")


def save_layer_embds(embeddings_dict, output_path, model_name, n_layers, prefix=""):
    output_dir = Path(output_path) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx in embeddings_dict:
        save_file(embeddings_dict[layer_idx], output_dir / f"{model_name}__L{layer_idx}.safetensors")
    prefix_str = f"{prefix} " if prefix else ""
    num_samples = len(embeddings_dict[0]) if embeddings_dict else 0
    print(f"Saved {prefix_str}embeddings for {num_samples} samples across {n_layers} layers to {output_dir}/")


def save_head_embds(module, input, output, nheads, n_kv_heads=None):
    bsz, seq_len, input_dim = input[0].shape
    head_dim = input_dim // nheads
    headwise_input = input[0].reshape(bsz, seq_len, nheads, -1).contiguous()
    W_O_sliced = module.weight.view(module.weight.size(0), nheads, head_dim)
    headwise_output = torch.einsum("bshd, mhd -> bshm", headwise_input, W_O_sliced)
    if n_kv_heads is not None:
        n_query_groups = nheads // n_kv_heads
        headwise_output = headwise_output.view(bsz, seq_len, n_kv_heads, n_query_groups, input_dim).sum(3)
    h_embds.append(headwise_output.cpu())
    return output


class MultimodalDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.ann_paths = sorted(Path(args.annotation_root).glob('*.json'))
        print(f"Found {len(self.ann_paths)} JSON files in {args.annotation_root}")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = Path(args.image_root)
        self.args = args
        self.image_token = args.image_token

    def __len__(self):
        return len(self.ann_paths)

    def __getitem__(self, idx):
        with open(self.ann_paths[idx], 'r') as f:
            anns = json.load(f)

        img_filename = next(iter(anns))
        sample_id = Path(img_filename).stem
        caption = next(iter(anns.values()))['dense_caption']['caption']

        img_path = self.image_root / img_filename
        if not img_path.exists():
            print(f"Warning: image not found for sample {sample_id}, skip.")
            return None

        img = Image.open(img_path).convert('RGB')
        pixel_values = self.image_processor(images=img, return_tensors="pt")['pixel_values']

        if self.args.conv_template == CONV_TEMPLATES.PLAIN:
            text = f"{self.image_token}\n{caption}"
        else:
            conv = CONV_MAPPING[self.args.conv_template].new_empty()
            conv.add_sys_prompt()
            conv.add_message(dict(role='user', content=f"{self.image_token}\nDescribe the image."))
            conv.add_message(dict(role='assistant', content=caption))
            text = conv.get_prompt()

        input_ids = self.tokenizer(text, return_tensors="pt")['input_ids']

        labels = None
        if self.args.conv_template != CONV_TEMPLATES.PLAIN:
            labels = conv.get_labels(input_ids[0], self.tokenizer.convert_ids_to_tokens(input_ids[0]))

        return {
            'id': sample_id,
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'labels': labels
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--annotation_root', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--pooling_mode', type=str, default='avg', choices=['eos', 'avg', 'wavg'],
                        help='Pooling strategy for global embedding')
    parser.add_argument('--conv_template', type=str, default=CONV_TEMPLATES.PLAIN)
    parser.add_argument('--headwise', action='store_true', help='Extract per-head embeddings')
    parser.add_argument('--headwise_gqa', action='store_true')
    parser.add_argument('--image_token', type=str, default=DEFAULT_IMAGE_TOKEN)
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    args = parser.parse_args()

    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]

    model = MLLMForCausalLM.from_pretrained(args.checkpoint_path, device_map='cuda')
    model.get_language_model().lm_head = nn.Identity()  # skip lm_head for faster feature extraction
    model.to('cuda', dtype=dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

    model.init_vision_tokenizer(tokenizer, resize_embeds=False)
    args.image_token = model.config.image_token

    image_processor = AutoImageProcessor.from_pretrained(model.config.vision_config.name_or_path)
    model.init_image_processor(image_processor)

    n_layers = model.get_language_model().config.num_hidden_layers
    n_heads = model.get_language_model().config.num_attention_heads
    print(f"MLLM has {n_layers} layers, {n_heads} heads")

    dataset = MultimodalDataset(args, tokenizer, image_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x[0],
        num_workers=0 if DEBUG else 2,
    )

    if args.headwise:
        def get_o_proj(root, projs):
            for name, child in root.named_children():
                if name == 'o_proj':
                    projs.append(child)
                get_o_proj(child, projs)

        all_o_proj = []
        get_o_proj(model.language_model, all_o_proj)
        if not len(all_o_proj):
            print(f"Cannot extract head-wise features for class `{model.language_model.__class__.__name__}`")
            args.headwise = False
        else:
            assert len(all_o_proj) == model.language_model.config.num_hidden_layers
            if args.headwise_gqa:
                if not hasattr(model.language_model.config, 'num_key_value_heads'):
                    print(f"Model config does not have `num_key_value_heads`, cannot do headwise GQA pooling")
                    n_kv_heads = None
                else:
                    n_kv_heads = model.language_model.config.num_key_value_heads
                    n_heads = n_kv_heads
            else:
                n_kv_heads = None
            for proj in all_o_proj:
                proj.register_forward_hook(partial(save_head_embds, nheads=model.language_model.config.num_attention_heads, n_kv_heads=n_kv_heads))

    global h_embds
    h_embds = []

    all_l_embds = defaultdict(dict)
    all_h_embds = defaultdict(dict)

    for sample in tqdm(dataloader, desc="Extracting features"):
        input_ids = sample['input_ids'].to(device=model.device)
        pixel_values = sample['pixel_values'].to(dtype=dtype, device=model.device)
        sample_id = sample['id']
        labels = sample['labels']

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            vis_start, vis_end = outputs['visual_position_ids'][0][0]
            # (n_layers, seq_len, hidden_size) — skip embedding layer (index 0)
            all_hs = torch.stack(outputs.hidden_states[1:]).squeeze(1).cpu()
            del outputs

        if args.headwise:
            h_all = torch.stack(h_embds).squeeze(1).permute(0, 2, 1, 3).cpu()

        if args.conv_template != CONV_TEMPLATES.PLAIN:
            labels = labels[vis_start + 1:]

        hs = all_hs[:, vis_end:]    # (n_layers, seq_len, D)
        h_hs = h_all[:, :, vis_end:] if args.headwise else None

        for layer_idx in range(n_layers):
            all_l_embds[layer_idx][sample_id] = apply_pooling(hs[layer_idx], args.pooling_mode, labels=labels)
            if args.headwise:
                lh = h_hs[layer_idx]                             # (n_heads, seq_len, D)
                all_h_embds[layer_idx][sample_id] = torch.stack(
                    [apply_pooling(lh[h], args.pooling_mode, labels=labels) for h in range(n_heads)]
                )

        h_embds = []

        model_name = Path(args.checkpoint_path).stem

    save_layer_embds(all_l_embds, args.output_path, model_name, n_layers)
    if args.headwise:
        save_layer_embds(all_h_embds, str(Path(args.output_path) / ('gqa_heads' if args.headwise_gqa else 'heads')), model_name, n_layers,
                              prefix="heads")

    print("Done!")
