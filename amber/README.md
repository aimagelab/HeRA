# AMBER Evaluation
This folder contains the answer-generation and metric computation scripts for [AMBER](https://github.com/junyangwang0410/AMBER), adapted from the official repository.
AMBER is evaluated in two separate runs: the generative task (image descriptions) and the discriminative task (Yes/No questions).

## Requirements
- Install project dependencies from the repo root: `pip install -r requirements.txt`
- Download the spaCy model and the NLTK resources used to compute the metrics:
```bash
python -m spacy download en_core_web_lg
python -c "import nltk; [nltk.download(x, download_dir='/path/to/nltk_data') for x in ('punkt_tab', 'averaged_perceptron_tagger_eng', 'wordnet')]"
export NLTK_DATA=/path/to/nltk_data
```

## Benchmark Setup
Benchmark setup follows the instructions in the [AMBER repository](https://github.com/junyangwang0410/AMBER#getting-started):
- the `data` folder of the AMBER repository (queries, annotations, and metric files);
- the 1,004 images, downloaded from the link in the AMBER repository.

The scripts expect the following layout:
```
amber/
├── data/
│   ├── annotations.json
│   ├── metrics.txt
│   ├── relation.json
│   ├── safe_words.txt
│   └── query/
│       ├── query_generative.json
│       └── query_discriminative.json
└── image/
    ├── AMBER_1.jpg
    └── ...
```

## Generate Answers (single node)
From the repo root:
```bash
# generative task
python amber/model_vqa_amber.py \
	--model_path aimagelab/HeRA-Qwen2.5-3B-SigLIP2 \
	--conv_mode qwen2_5 \
	--question_file /path/to/amber/data/query/query_generative.json \
	--image_folder /path/to/amber/image \
	--answers_file /path/to/answers/amber_generative_answers.json \
	--attn_implementation sdpa \
	--dtype bf16 \
	--temperature 0.0 \
	--seed 42 \
	--max_new_tokens 512

# discriminative task
python amber/model_vqa_amber.py \
	--model_path aimagelab/HeRA-Qwen2.5-3B-SigLIP2 \
	--conv_mode qwen2_5 \
	--question_file /path/to/amber/data/query/query_discriminative.json \
	--image_folder /path/to/amber/image \
	--answers_file /path/to/answers/amber_discriminative_answers.json \
	--attn_implementation sdpa \
	--dtype bf16 \
	--temperature 0.0 \
	--seed 42 \
	--max_new_tokens 3 \
	--test_prompt " Answer only with Yes or No. Use exactly one word. Do not use commas dots or symbols."
```
`--model_path` accepts a 🤗 Hub id or a local checkpoint directory, and `--conv_mode` must match the checkpoint (see the [checkpoints table](../README.md#checkpoints)).
The task is inferred from the name of the question file, so generative and discriminative answers must be generated in separate runs.
In the discriminative task, `--test_prompt` is appended to every query to enforce Yes/No answers.

## Compute Metrics
```bash
# generative task
python amber/inference.py \
	--inference_data /path/to/answers/amber_generative_answers.json \
	--scores_file /path/to/answers/amber_generative_scores.json \
	--word_association /path/to/amber/data/relation.json \
	--safe_words /path/to/amber/data/safe_words.txt \
	--annotation /path/to/amber/data/annotations.json \
	--metrics /path/to/amber/data/metrics.txt \
	--evaluation_type g

# discriminative task
python amber/inference.py \
	--inference_data /path/to/answers/amber_discriminative_answers.json \
	--scores_file /path/to/answers/amber_discriminative_scores.json \
	--word_association /path/to/amber/data/relation.json \
	--safe_words /path/to/amber/data/safe_words.txt \
	--annotation /path/to/amber/data/annotations.json \
	--metrics /path/to/amber/data/metrics.txt \
	--evaluation_type d
```
We report CHAIR, Cover, Hal, and Cog for the generative task, and Accuracy, Precision, Recall, and F1 for the discriminative task.

## Slurm Evaluation
Use the provided scripts and edit their placeholder paths (marked with `TODO`) for your cluster.
The scripts use the settings above, including the discriminative `--test_prompt`.
```bash
# generate answers
sbatch scripts/eval/amber/eval_amber.sh <model_path> <conv_mode> <generative|discriminative> \
	[attn_implementation] [temperature] [seed]

# compute metrics, once the generation job has finished
bash scripts/eval/amber/compute_amber_scores.sh <model_path> <generative|discriminative> [temperature] [seed]
```
For example:
```bash
sbatch scripts/eval/amber/eval_amber.sh aimagelab/HeRA-Qwen3-8B-SigLIP2 qwen3 generative
sbatch scripts/eval/amber/eval_amber.sh aimagelab/HeRA-Qwen3-8B-SigLIP2 qwen3 discriminative

bash scripts/eval/amber/compute_amber_scores.sh aimagelab/HeRA-Qwen3-8B-SigLIP2 generative
bash scripts/eval/amber/compute_amber_scores.sh aimagelab/HeRA-Qwen3-8B-SigLIP2 discriminative
```
