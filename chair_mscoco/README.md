# CHAIR-MSCOCO Evaluation
This folder contains the answer-generation and metric computation scripts for CHAIR on MSCOCO.
The model describes 500 images sampled from the MSCOCO validation set, and CHAIR<sub>s</sub> / CHAIR<sub>i</sub> measure sentence- and object-level hallucination rates.

## Requirements
- Install project dependencies from the repo root: `pip install -r requirements.txt`
- Download the NLTK resources used to compute the metrics:
```bash
python -c "import nltk; [nltk.download(x, download_dir='/path/to/nltk_data') for x in ('punkt_tab', 'averaged_perceptron_tagger_eng', 'wordnet')]"
export NLTK_DATA=/path/to/nltk_data
```

## Benchmark Setup
Benchmark setup follows the instructions in the [less-is-more repository](https://github.com/yuezih/less-is-more/tree/main#chair-evaluation):
- the test set `chair-500.jsonl`;
- the 500 test images (collected from MSCOCO with their script, or downloaded from their link);
- the MSCOCO 2014 annotations (`annotations_trainval2014.zip`), used to compute the metrics.

The scripts expect the following layout:
```
chair_mscoco/
├── data/
│   └── chair-500.jsonl
└── MSCOCO/
    ├── images/
    │   ├── 000000002239.jpg
    │   └── ...
    └── annotations/
        ├── captions_train2014.json
        ├── captions_val2014.json
        ├── instances_train2014.json
        └── instances_val2014.json
```

## Generate Answers (single node)
From the repo root:
```bash
python chair_mscoco/model_vqa_chairmscoco.py \
	--model_path aimagelab/HeRA-Qwen2.5-3B-SigLIP2 \
	--conv_mode qwen2_5 \
	--question_file /path/to/chair_mscoco/data/chair-500.jsonl \
	--image_folder /path/to/chair_mscoco/MSCOCO/images \
	--answers_file /path/to/answers/chairmscoco_answers.jsonl \
	--attn_implementation sdpa \
	--dtype bf16 \
	--temperature 0.0 \
	--seed 42 \
	--max_new_tokens 512
```
`--model_path` accepts a 🤗 Hub id or a local checkpoint directory, and `--conv_mode` must match the checkpoint (see the [checkpoints table](../README.md#checkpoints)).

## Compute Metrics
```bash
python chair_mscoco/chair.py \
	--cap_file /path/to/answers/chairmscoco_answers.jsonl \
	--scores_file /path/to/answers/chairmscoco_scores.json \
	--save_path /path/to/answers/chairmscoco_answers_details.json \
	--image_id_key image_id \
	--caption_key text \
	--coco_path /path/to/chair_mscoco/MSCOCO/annotations \
	--cache /path/to/chair_mscoco/MSCOCO/annotations/chair_cache.pkl
```
The CHAIR evaluator is built from the MSCOCO annotations on the first run and cached in `--cache`, which makes later runs much faster.
We report CHAIR<sub>s</sub> and CHAIR<sub>i</sub>.

## Slurm Evaluation
Use the provided scripts and edit their placeholder paths (marked with `TODO`) for your cluster.
```bash
# generate answers
sbatch scripts/eval/chair_mscoco/eval_chairmscoco.sh <model_path> <conv_mode> \
	[attn_implementation] [temperature] [seed] [max_new_tokens]

# compute metrics, once the generation job has finished
bash scripts/eval/chair_mscoco/compute_chairmscoco_scores.sh <model_path> [temperature] [seed] [max_new_tokens]
```
For example:
```bash
sbatch scripts/eval/chair_mscoco/eval_chairmscoco.sh aimagelab/HeRA-Qwen3-8B-SigLIP2 qwen3
bash scripts/eval/chair_mscoco/compute_chairmscoco_scores.sh aimagelab/HeRA-Qwen3-8B-SigLIP2
```
