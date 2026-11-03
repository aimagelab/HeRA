#!/bin/bash
#SBATCH --job-name=feats_txt
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

# LUMI / AMD GPUs only: per-node MIOpen kernel cache
# export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_NODEID}
# export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# TODO: adjust this
export CODE_ROOT=~/git/HeRA

export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false

# TODO: adjust this
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-8B"
    "lmsys/vicuna-7b-v1.5"
    "meta-llama/Meta-Llama-3-8B-Instruct"
)
model_name_or_path="${MODELS[$SLURM_ARRAY_TASK_ID]}"

pool_mode="avg"
output_path="/data/alignment/features/text/${pool_mode}"

CONTAINER_CMD="cd $CODE_ROOT && export PYTHONPATH=$PYTHONPATH:.:/user-software/venv/pytorch/lib/python3.12/site-packages && \
python -u src/alignment/feats_txt.py \
    --model_name_or_path $model_name_or_path \
    --annotation_root /datasets/GranD/annotations \
    --output_path $output_path \
    --batch_size 1 \
    --pooling_mode $pool_mode \
    --headwise \
"

# TODO: adjust path bindings
srun --kill-on-bad-exit=1 --wait=0 \
singularity exec \
    -B ...:/hf_models \
    -B ...:/datasets \
    -B ...:/data \
    -B /tmp:/tmp \
    $SIF \
    bash -c "$CONTAINER_CMD"
