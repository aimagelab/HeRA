#!/bin/bash
# Computes AMBER metrics from the answers produced by scripts/eval/amber/eval_amber.sh (no GPU needed).
#
# Usage:
#   bash scripts/eval/amber/compute_amber_scores.sh <model_path> <generative|discriminative> [temperature] [seed]
# e.g.
#   bash scripts/eval/amber/compute_amber_scores.sh aimagelab/HeRA-Qwen2.5-3B-SigLIP2 generative

# TODO: set up python env and load the needed modules / container image

# TODO: adjust this
export CODE_ROOT=/path/to/HeRA
# TODO: adjust this (directory where the nltk resources were downloaded)
export NLTK_DATA=/datasets/nltk_data

# --- args ---
model_path="$1"                        # same value passed to eval_amber.sh
mode="${2:-generative}"                # generative | discriminative
temperature="${3:-0.0}"
seed="${4:-42}"
model_name="${model_path%/}"
model_name="${model_name##*/}"

dtype="bf16"

case "$mode" in
  generative)     max_new_tokens=512; evaluation_type="g" ;;
  discriminative) max_new_tokens=3;   evaluation_type="d" ;;
  *) echo "Unknown mode: $mode (expected generative|discriminative)"; exit 1 ;;
esac

# TODO: adjust these
amber_data="/datasets/amber/data"
out_dir="/eval/amber/${model_name}"
answers_file="${out_dir}/answers_amber_${mode}_seed${seed}_temp${temperature}_tok${max_new_tokens}_${dtype}__${model_name}.json"
scores_file="${out_dir}/scores_amber_${mode}_seed${seed}_temp${temperature}_tok${max_new_tokens}_${dtype}__${model_name}.json"

echo "Inference data: $answers_file"
echo "Scores out:     $scores_file"
echo "Mode:           $mode ($evaluation_type)"

CONTAINER_CMD="if [[ -e $scores_file ]]; then \
    echo \"Refusing to overwrite existing scores file: $scores_file\"; \
    exit 1; \
fi && \
cd $CODE_ROOT && export PYTHONPATH=$CODE_ROOT && \
python amber/inference.py \
    --inference_data $answers_file \
    --scores_file $scores_file \
    --word_association $amber_data/relation.json \
    --safe_words $amber_data/safe_words.txt \
    --annotation $amber_data/annotations.json \
    --metrics $amber_data/metrics.txt \
    --evaluation_type $evaluation_type"

# TODO: adjust path bindings
singularity exec \
    -B ...:/datasets \
    -B ...:/eval \
    -B /tmp:/tmp \
    $SIF \
    bash -c "$CONTAINER_CMD"
