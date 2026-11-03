# HallusionBench Evaluation
This folder contains the answer-generation and metric computation scripts for HallusionBench.

## Requirements
- Install project dependencies from the repo root: `pip install -r requirements.txt`
- You need the HallusionBench question JSON and the image folder from the official dataset.

## Generate Answers (single node)
From the repo root:
```bash
python hallusion/model_vqa_hallusion.py \
	--model_path /path/to/checkpoint \
	--conv_mode <conv_mode> \
	--question_file /path/to/HallusionBench.json \
	--image_folder /path/to/images \
	--answers_file /path/to/answers/hallusion_answers.jsonl \
	--attn_implementation sdpa \
	--dtype bf16 \
	--temperature 0.0 \
	--seed 42 \
	--max_new_tokens 512
```
`--model_path` accepts a 🤗 Hub id or a local checkpoint directory, and `--conv_mode` must match the checkpoint (see the [checkpoints table](../README.md#checkpoints)).

## Compute Metrics
```bash
python hallusion/hallusion.py \
	--answers_file /path/to/answers/hallusion_answers.jsonl \
	--question_file /path/to/HallusionBench.json \
	--output_json /path/to/answers/hallusion_metrics.json
```

## Slurm Evaluation
Use the provided Slurm script and edit its placeholder paths (marked with `TODO`) for your cluster.
The script generates the answers and then computes the metrics.
```bash
sbatch scripts/eval/hallusion_bench/eval_hallusion.sh <model_path> <conv_mode> [attn_implementation] [temperature] [seed] [max_new_tokens]
```
For example:
```bash
sbatch scripts/eval/hallusion_bench/eval_hallusion.sh aimagelab/HeRA-Qwen3-8B-SigLIP2 qwen3
```

## Benchmark Setup
Benchmark setup followed the instructions in the
[HallusionBench repository](https://github.com/tianyi-lab/hallusionbench).