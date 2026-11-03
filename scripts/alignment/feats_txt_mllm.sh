#!/bin/bash
#SBATCH --job-name=feats_txt_mllm
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
checkpoint_path="/checkpoints/hera/st1_hera_qwen2_5_7B_dinov2l"
conv_template="plain"

pool_mode="avg"
output_path="/data/alignment/features/mllm/${pool_mode}"

CONTAINER_CMD="cd $CODE_ROOT && export PYTHONPATH=$PYTHONPATH:.:/user-software/venv/pytorch/lib/python3.12/site-packages && \
python -u src/alignment/feats_txt_mllm.py \
    --checkpoint_path $checkpoint_path \
    --image_root /datasets/GranD/images \
    --annotation_root /datasets/GranD/annotations \
    --output_path $output_path \
    --pooling_mode $pool_mode \
    --conv_template $conv_template \
    --headwise \
"

# TODO: adjust path bindings
srun --kill-on-bad-exit=1 --wait=0 \
singularity exec \
    -B ...:/hf_models \
    -B ...:/datasets \
    -B ...:/checkpoints \
    -B ...:/data \
    -B /tmp:/tmp \
    $SIF \
    bash -c "$CONTAINER_CMD"
