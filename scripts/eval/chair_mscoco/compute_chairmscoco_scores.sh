#!/bin/bash
# Computes CHAIR metrics from the answers produced by scripts/eval/chair_mscoco/eval_chairmscoco.sh (no GPU needed).
#
# Usage:
#   bash scripts/eval/chair_mscoco/compute_chairmscoco_scores.sh <model_path> [temperature] [seed] [max_new_tokens]
# e.g.
#   bash scripts/eval/chair_mscoco/compute_chairmscoco_scores.sh aimagelab/HeRA-Qwen2.5-3B-SigLIP2

# TODO: set up python env and load the needed modules / container image

# TODO: adjust this
export CODE_ROOT=/path/to/HeRA
# TODO: adjust this (directory where the nltk resources were downloaded)
export NLTK_DATA=/datasets/nltk_data

# --- args ---
model_path="$1"                        # same value passed to eval_chairmscoco.sh
temperature="${2:-0.0}"
seed="${3:-42}"
max_new_tokens="${4:-512}"
model_name="${model_path%/}"
model_name="${model_name##*/}"

dtype="bf16"

# TODO: adjust these paths
out_dir="/eval/chair_mscoco/${model_name}"
answers_file="${out_dir}/answers_chairmscoco_seed${seed}_temp${temperature}_tok${max_new_tokens}_${dtype}__${model_name}.jsonl"
scores_file="${out_dir}/scores_chairmscoco_seed${seed}_temp${temperature}_tok${max_new_tokens}_${dtype}__${model_name}.json"
# MSCOCO 2014 annotations (instances_* and captions_* for train and val).
chair_coco_path="/datasets/chair_mscoco/MSCOCO/annotations"
# The CHAIR evaluator is built from the annotations on the first run and cached here.
chair_cache="/datasets/chair_mscoco/MSCOCO/annotations/chair_cache.pkl"
save_path="${answers_file%.jsonl}_details.json"

echo "Inference data: $answers_file"
echo "Scores out:     $scores_file"
echo "Details out:    $save_path"

CONTAINER_CMD="if [[ -e $scores_file ]]; then \
    echo \"Refusing to overwrite existing scores file: $scores_file\"; \
    exit 1; \
fi && \
if [[ -e $save_path ]]; then \
    echo \"Refusing to overwrite existing details file: $save_path\"; \
    exit 1; \
fi && \
cd $CODE_ROOT && export PYTHONPATH=$CODE_ROOT && \
python chair_mscoco/chair.py \
    --cap_file $answers_file \
    --scores_file $scores_file \
    --image_id_key image_id \
    --caption_key text \
    --cache $chair_cache \
    --coco_path $chair_coco_path \
    --save_path $save_path"

# TODO: adjust path bindings
singularity exec \
    -B ...:/datasets \
    -B ...:/eval \
    -B /tmp:/tmp \
    $SIF \
    bash -c "$CONTAINER_CMD"
