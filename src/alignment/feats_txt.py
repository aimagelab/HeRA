import argparse
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import save_file
from pathlib import Path
import json
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from functools import partial
import src.custom_utils as utils


class TextDataset(Dataset):
    def __init__(self, args, tokenizer):
        self.ann_paths = sorted(Path(args.annotation_root).glob('*.json'))
        print(f"Found {len(self.ann_paths)} JSON files in {args.annotation_root}")
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.ann_paths)

    def __getitem__(self, idx):
        with open(self.ann_paths[idx], 'r') as f:
            anns = json.load(f)

        sample_id = Path(next(iter(anns))).stem
        text = next(iter(anns.values()))['dense_caption']['caption']

        return {'id': sample_id, 'text': text}


def collate_fn(batch, tokenizer=None):
    texts = [x['text'] for x in batch]
    ret = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    ret['id'] = [x['id'] for x in batch]
    return ret


def apply_pooling(embeddings, pooling_mode, attention_mask=None):
    if pooling_mode == 'eos':
        eos_idx = int(attention_mask.sum().item()) - 1 if attention_mask is not None else -1
        return embeddings[eos_idx].clone()
    elif pooling_mode == 'avg':
        valid = embeddings[attention_mask.bool()] if attention_mask is not None else embeddings
        return valid.mean(dim=0)
    elif pooling_mode == 'wavg':
        valid = embeddings[attention_mask.bool()] if attention_mask is not None else embeddings
        weights = (torch.arange(1, valid.size(0) + 1) ** 2).to(dtype=valid.dtype, device=valid.device)
        weights /= weights.sum()
        return (valid * weights[:, None]).sum(0)
    raise ValueError(f"Unknown pooling mode: {pooling_mode}")


def save_layer_embds(embeddings_dict, output_path, model_name_or_path, num_layers, prefix=""):
    output_dir = Path(output_path) / model_name_or_path
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_name_or_path).name
    for layer_idx in embeddings_dict:
        save_file(embeddings_dict[layer_idx], output_dir / f"{model_name}__L{layer_idx}.safetensors")
    num_samples = len(embeddings_dict[0]) if embeddings_dict else 0
    prefix_str = f"{prefix} " if prefix else ""
    print(f"Saved {prefix_str}embeddings for {num_samples} samples across {num_layers} layers to {output_dir}/")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--annotation_root', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pooling_mode', type=str, default='avg')
    parser.add_argument('--chat_template', action='store_true')
    parser.add_argument('--headwise', action='store_true')
    parser.add_argument('--headwise_gqa', action='store_true')
    parser.add_argument('--layerwise_after_sa', action='store_true')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = None
    if utils.DEBUG_LOAD_FIRST_LLM_LAYERS:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.num_hidden_layers = utils.DEBUG_LOAD_FIRST_LLM_LAYERS
        print(f"DEBUG: Loading only first {utils.DEBUG_LOAD_FIRST_LLM_LAYERS} layers of the model")
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
        device_map='cuda',
    )

    dataset = TextDataset(args, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=0 if utils.DEBUG else 2,
    )

    if args.headwise:
        def get_o_proj(root, projs):
            for name, child in root.named_children():
                if name == 'o_proj':
                    projs.append(child)
                get_o_proj(child, projs)

        all_o_proj = []
        get_o_proj(model, all_o_proj)
        if not len(all_o_proj):
            print(f"Cannot extract head-wise features for class `{model.__class__.__name__}`")
            args.headwise = False
        else:
            assert len(all_o_proj) == model.config.num_hidden_layers
            if args.headwise_gqa:
                if not hasattr(model.config, 'num_key_value_heads'):
                    print(f"Model config does not have `num_key_value_heads`, cannot do headwise GQA pooling")
                    n_kv_heads = None
                else:
                    n_kv_heads = model.config.num_key_value_heads
            else:
                n_kv_heads = None
            for proj in all_o_proj:
                proj.register_forward_hook(partial(save_head_embds, nheads=model.config.num_attention_heads, n_kv_heads=n_kv_heads))

    global h_embds
    h_embds = []

    all_l_embds = defaultdict(dict)
    all_h_embds = defaultdict(dict)

    for batch in tqdm(dataloader, desc="Extracting features"):
        if batch is None:
            continue
        input_ids = batch['input_ids'].to(device=model.device)
        attention_mask = batch['attention_mask'].to(device=model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = torch.stack(outputs.hidden_states[1:]).cpu()  # (num_layers, bsz, seq_len, D)
            attention_mask = attention_mask.cpu()
            del outputs
            if args.headwise:
                h_embds = torch.stack(h_embds).permute(0, 3, 1, 2, 4)  # (num_layers, n_heads, bsz, seq_len, D)
                if args.layerwise_after_sa:
                    hidden_states = h_embds.sum(1)

        for i, sample_id in enumerate(batch['id']):
            layers_embds = hidden_states[:, i]   # (num_layers, seq_len, D)
            for layer_idx, curr_layer_embds in enumerate(layers_embds):
                all_l_embds[layer_idx][sample_id] = apply_pooling(curr_layer_embds, args.pooling_mode, attention_mask[i])
            if args.headwise:
                for layer_idx in range(len(layers_embds)):
                    lh = h_embds[layer_idx, :, i]  # (n_heads, seq_len, D)
                    all_h_embds[layer_idx][sample_id] = torch.stack(
                        [apply_pooling(lh[h], args.pooling_mode, attention_mask[i]) for h in range(lh.size(0))]
                    )

        h_embds = []

    save_layer_embds(all_l_embds, args.output_path, args.model_name_or_path,
                          model.config.num_hidden_layers)
    if args.headwise:
        save_layer_embds(
            all_h_embds,
            str(Path(args.output_path) / ('gqa_heads' if args.headwise_gqa else 'heads')),
            args.model_name_or_path,
            model.config.num_hidden_layers,
            prefix="heads",
        )

    print("Done!")
