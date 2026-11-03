#!/bin/bash
#SBATCH --job-name=mknn
#SBATCH --output=/logs/alignment/%x-%j
#SBATCH --error=/logs/alignment/%x-%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=
#SBATCH --cpus-per-task=
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=02:00:00
#SBATCH --array=0-5

set -x

# TODO: set up python env and load the needed modules / container image

export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false

# TODO: adjust this
export CODE_ROOT=~/git/HeRA

# TODO: adjust this
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-8B"
    "lmsys/vicuna-7b-v1.5"
    "meta-llama/Meta-Llama-3-8B-Instruct"
)
model_name="${MODELS[$SLURM_ARRAY_TASK_ID]}"

vision_model="dinov2-large"
pool_mode="avg"

feats_img_path="/data/alignment/features/visual/${vision_model}.safetensors"
feats_txt_path="/data/alignment/features/text/${pool_mode}/${model_name}"
feats_txt_heads_path="/data/alignment/features/text/${pool_mode}/heads/${model_name}"
output_path="/data/alignment/results/${pool_mode}"

CONTAINER_CMD="cd $CODE_ROOT && export PYTHONPATH=$PYTHONPATH:.:/user-software/venv/pytorch/lib/python3.12/site-packages && \
python -u src/alignment/mknn.py \
    --feats_img_path $feats_img_path \
    --feats_txt_path $feats_txt_path \
    --output_path $output_path \
    --k 10 && \
\
python -u src/alignment/mknn.py \
    --feats_img_path $feats_img_path \
    --feats_txt_path $feats_txt_heads_path \
    --output_path ${output_path}/heads \
    --k 10 \
    --topk_heads 10 \
"

# TODO: adjust path bindings
srun --kill-on-bad-exit=1 --wait=0 \
singularity exec \
    -B ...:/hf_models \
    -B ...:/data \
    -B /tmp:/tmp \
    $SIF \
    bash -c "$CONTAINER_CMD"
