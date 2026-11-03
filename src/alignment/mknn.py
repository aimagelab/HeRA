from safetensors import safe_open
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from safetensors.torch import load_file


def load_visual_embeddings(feats_img_path):
    embeddings = {}
    with safe_open(feats_img_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            embeddings[key] = f.get_tensor(key)
    return embeddings


def get_text_layer_files(feats_txt_path):
    feats_txt_path = Path(feats_txt_path)

    if feats_txt_path.is_file():
        parent = feats_txt_path.parent
        name_parts = feats_txt_path.stem.split('__L')
        if len(name_parts) == 2:
            base_pattern = f"{name_parts[0]}__L*.safetensors"
            layer_files = sorted(parent.glob(base_pattern))
        else:
            raise ValueError(f"Could not parse layer pattern from {feats_txt_path}")
    elif feats_txt_path.is_dir():
        layer_files = sorted(feats_txt_path.glob("*__L*.safetensors"))
    else:
        raise ValueError(f"Path {feats_txt_path} does not exist")

    if not layer_files:
        raise ValueError(f"No layer files found for pattern")

    layer_files_with_idx = []
    for layer_file in layer_files:
        stem = layer_file.stem
        layer_idx = int(stem.split('__L')[-1])
        layer_files_with_idx.append((layer_idx, layer_file))

    layer_files_with_idx.sort(key=lambda x: x[0])
    return layer_files_with_idx


def load_single_text_layer(layer_file_path, device='cpu'):
    return load_file(layer_file_path, device=device)


def dict_to_matrix(embeddings_dict, sample_ids=None):
    if sample_ids is None:
        sample_ids = sorted(embeddings_dict.keys())
    embeddings_list = [embeddings_dict[sid] for sid in sample_ids]
    embeddings_matrix = torch.stack(embeddings_list)
    is_headwise = embeddings_matrix.ndim == 3  # N x H x D
    return embeddings_matrix, sample_ids, is_headwise


def remove_outliers(feats, q, exact=False, max_threshold=None):
    if q == 1:
        return feats

    if exact:
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()

    if max_threshold is not None:
        max_threshold = max(max_threshold, q_val)

    return feats.clamp(-q_val, q_val)


def compute_cosine_similarity_matrix(embeddings_matrix):
    if embeddings_matrix.ndim == 3:
        N, H, D = embeddings_matrix.shape
        embeddings_matrix = embeddings_matrix.permute(1, 0, 2)  # H x N x D

        similarity_matrices = []
        for h in range(H):
            head_embeds = embeddings_matrix[h]
            head_embeds = remove_outliers(head_embeds.to(torch.float32), q=0.95)
            head_embeds_norm = F.normalize(head_embeds.to(torch.float32), p=2, dim=-1)
            sim_matrix = torch.mm(head_embeds_norm, head_embeds_norm.t())
            sim_matrix.fill_diagonal_(float('-1e6'))  # exclude self-similarity
            similarity_matrices.append(sim_matrix)

        return torch.stack(similarity_matrices)  # H x N x N

    embeddings_matrix = remove_outliers(embeddings_matrix.to(torch.float32), q=0.95)
    embeddings_normalized = F.normalize(embeddings_matrix.to(torch.float32), p=2, dim=-1)
    similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
    similarity_matrix.fill_diagonal_(float('-1e6'))  # exclude self-similarity
    return similarity_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats_img_path', type=str, required=True)
    parser.add_argument('--feats_txt_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--topk_heads', type=int, default=10)
    args = parser.parse_args()

    visual_embeddings = load_visual_embeddings(args.feats_img_path)
    print(f"Loaded {len(visual_embeddings)} visual embeddings")

    text_layer_files = get_text_layer_files(args.feats_txt_path)
    num_layers = len(text_layer_files)
    print(f"Found {num_layers} text layer files")

    first_layer_embeddings = load_single_text_layer(text_layer_files[0][1])

    visual_ids = set(visual_embeddings.keys())
    text_ids = set(first_layer_embeddings.keys())
    common_ids = visual_ids & text_ids

    if len(common_ids) != len(visual_ids) or len(common_ids) != len(text_ids):
        print(f"Warning: Sample ID mismatch! Visual: {len(visual_ids)}, Text: {len(text_ids)}, Common: {len(common_ids)}")
        visual_embeddings = {k: v for k, v in visual_embeddings.items() if k in common_ids}

    del first_layer_embeddings

    visual_emb_matrix, sample_ids, _ = dict_to_matrix(visual_embeddings)
    print(f"Visual embeddings matrix shape: {visual_emb_matrix.shape}")

    Kimg = compute_cosine_similarity_matrix(visual_emb_matrix)
    knn_img = Kimg.topk(args.k, dim=1).indices  # (N, k)
    del visual_embeddings, visual_emb_matrix, Kimg

    mknn_scores = []
    text_is_headwise = None
    num_heads = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('Using CUDA acceleration')

    for layer_idx, layer_file in tqdm(text_layer_files, desc="Processing layers"):
        layer_embeddings = load_single_text_layer(layer_file, device=device)

        if len(common_ids) != len(text_ids):
            layer_embeddings = {k: v for k, v in layer_embeddings.items() if k in common_ids}

        layer_emb_matrix, _, is_hw = dict_to_matrix(layer_embeddings, sample_ids=sample_ids)
        del layer_embeddings

        if text_is_headwise is None:
            text_is_headwise = is_hw
            if text_is_headwise:
                num_heads = layer_emb_matrix.shape[1]
                print(f"Text embeddings are head-wise with {num_heads} heads")

        Ktxt = compute_cosine_similarity_matrix(layer_emb_matrix)
        if knn_img.device != Ktxt.device:
            knn_img = knn_img.to(device=Ktxt.device)

        if text_is_headwise:
            knn_txt = Ktxt.topk(args.k, dim=2).indices  # (H, N, k)
            # img: (1, N, k, 1), txt: (H, N, 1, k)
            layer_mnn = ((knn_img[None, :, :, None] == knn_txt[:, :, None, :]).sum((2, 3)) / args.k).mean(1)
            mknn_scores.append(layer_mnn.cpu())
        else:
            knn_txt = Ktxt.topk(args.k, dim=1).indices  # (N, k)
            # equation 10 from "The Platonic Representation Hypothesis"
            layer_mnn = ((knn_img[:, :, None] == knn_txt[:, None, :]).sum((1, 2)) / args.k).mean()
            mknn_scores.append(layer_mnn.cpu())

        del layer_emb_matrix, Ktxt, knn_txt

    mknn_scores = torch.stack(mknn_scores).numpy()  # (L,) or (L, H)
    del knn_img

    visual_encoder_name = Path(args.feats_img_path).stem
    llm_name = Path(args.feats_txt_path).name

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_filename_base = f"mknn_k{args.k}__{visual_encoder_name}__{llm_name}"

    if text_is_headwise:
        plot_filename_base = 'headwise_' + plot_filename_base
        num_layers, num_heads = mknn_scores.shape

        import numpy as np
        flat_scores = mknn_scores.flatten()
        top_k = min(args.topk_heads, len(flat_scores))
        top_k_indices = np.argsort(flat_scores)[-top_k:][::-1]
        top_k_coords = [(idx // num_heads, idx % num_heads) for idx in top_k_indices]
        top_k_scores = [mknn_scores[layer_idx, head_idx] for layer_idx, head_idx in top_k_coords]
        vmax = top_k_scores[0]

        fig, ax = plt.subplots(figsize=(max(10, num_heads * 0.5), max(8, num_layers * 0.4)))
        im = ax.imshow(mknn_scores, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0.0, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MKNN Alignment Score', fontsize=12)
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels(range(num_heads))
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels(range(num_layers))
        ax.set_xlabel('Head Index', fontsize=12)
        ax.set_ylabel('Layer Index', fontsize=12)
        ax.set_title(f'Head-wise Cross-Modal Alignment: {visual_encoder_name} vs {llm_name}\n(k={args.k})', fontsize=14)
        plt.tight_layout()
        plot_path_jpg = output_path / f"{plot_filename_base}.jpg"
        plt.savefig(plot_path_jpg, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()

        all_indices = [(i, j) for i in range(num_layers) for j in range(num_heads)]
        all_scores = [mknn_scores[i, j] for i, j in all_indices]
        sorted_pairs = sorted(zip(all_indices, all_scores), key=lambda x: x[1], reverse=True)
        mknn_scores_list = [[f"L{layer_idx}H{head_idx}", float(score)]
                            for (layer_idx, head_idx), score in sorted_pairs]
        metadata = {"mknn_scores": mknn_scores_list}
        metadata_path = output_path / f"{plot_filename_base}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"JPG: {plot_path_jpg}")
        print(f"JSON: {metadata_path}")

    else:
        layer_indices = list(range(len(mknn_scores)))
        layer_labels = [str(idx) if idx != -1 else str(num_layers - 1) for idx in layer_indices]
        max_score_idx = mknn_scores.argmax()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(layer_indices)), mknn_scores, color='steelblue', edgecolor='black', alpha=0.7)
        plt.text(max_score_idx, mknn_scores[max_score_idx],
                 f'{mknn_scores[max_score_idx]:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.xlabel(f'Layer Index\n({llm_name})', fontsize=12)
        plt.ylabel('MKNN Alignment Score', fontsize=12)
        plt.title(f'Cross-Modal Alignment: {visual_encoder_name} vs {llm_name}\n(k={args.k})', fontsize=14)
        plt.xticks(range(len(layer_indices)), layer_labels, rotation=45 if len(layer_indices) > 20 else 0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plot_path_jpg = output_path / f"{plot_filename_base}.jpg"
        plt.savefig(plot_path_jpg, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()

        layer_score_pairs = [(i, mknn_scores[i]) for i in range(len(mknn_scores))]
        sorted_layers = sorted(layer_score_pairs, key=lambda x: x[1], reverse=True)
        mknn_scores_list = [[f"L{layer_idx}", float(score)] for layer_idx, score in sorted_layers]
        metadata = {"mknn_scores": mknn_scores_list}
        metadata_path = output_path / f"{plot_filename_base}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"JPG: {plot_path_jpg}")
        print(f"JSON: {metadata_path}")
        print(f"MKNN scores per layer: {dict(zip(layer_labels, mknn_scores))}")
