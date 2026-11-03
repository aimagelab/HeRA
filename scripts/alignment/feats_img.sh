#!/bin/bash
#SBATCH --job-name=feats_img
#SBATCH --output=/logs/alignment/%x-%j
#SBATCH --error=/logs/alignment/%x-%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=
#SBATCH --cpus-per-task=
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=01:00:00

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
vision_model_name="facebook/dinov2-large"
output_path="/data/alignment/features/visual"

CONTAINER_CMD="cd $CODE_ROOT && export PYTHONPATH=$PYTHONPATH:.:/user-software/venv/pytorch/lib/python3.12/site-packages && \
python -u src/alignment/feats_img.py \
    --model_name_or_path $vision_model_name \
    --image_root /datasets/GranD/images \
    --annotation_root /datasets/GranD/annotations \
    --output_path $output_path \
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
