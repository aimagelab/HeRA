#!/bin/bash
#SBATCH --job-name=eval_cambrian
# TODO: adjust log paths
#SBATCH --output=/logs/cambrian/%x-%j
#SBATCH --error=/logs/cambrian/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --account=
#SBATCH --time=02:00:00
#SBATCH --array=0-22

# Usage (one benchmark per array task):
#   sbatch scripts/eval/cambrian/eval_cambrian.sh <model_path> <conv_mode> [output_dir]
# e.g.
#   sbatch scripts/eval/cambrian/eval_cambrian.sh aimagelab/HeRA-Qwen2.5-3B-SigLIP2 qwen2_5

# TODO: set up python env and load the needed modules / container image

# TODO: adjust this
export CODE_ROOT=/path/to/HeRA

export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
# TODO: adjust these
export HF_HUB_CACHE="..."
export HF_DATASETS_CACHE="..."
# Uncomment once the checkpoint, the base LLM tokenizer, the vision processor and the benchmarks are in the cache.
# export HF_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# LUMI / AMD GPUs only: per-job MIOpen kernel cache, to avoid contention/corruption on a shared /tmp
# ("Cannot open database file" -> RuntimeError: miopenStatusInternalError during the SigLIP conv2d).
# export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
# export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
# mkdir -p "$MIOPEN_USER_DB_PATH"
# export SINGULARITYENV_MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH
# export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR

# --- args ---
ckpt="$1"                              # HF Hub id (e.g. aimagelab/HeRA-Qwen2.5-3B-SigLIP2) or local checkpoint dir
conv_mode="$2"                         # e.g. qwen2_5 (see the checkpoints table in the README)
# TODO: adjust this
_eval_output_dir="/eval/cambrian"
eval_output_dir="${3:-$_eval_output_dir}"

gpu_devices="0"

echo "Model checkpoint: $ckpt"
echo "Conversation mode: $conv_mode"
echo "Evaluation output directory: $eval_output_dir"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

export CUDA_VISIBLE_DEVICES="$gpu_devices"

benchmarks=(
    gqa
    vizwiz
    scienceqa
    textvqa
    pope
    mme
    mmbench_en
    mmbench_cn
    seed
    mmmu
    mathvista
    ai2d
    chartqa
    ocrbench
    mmstar
    realworldqa
    qbench
    blink
    mmvp
    vstar
    ade
    omni
    coco
)

benchmark=${benchmarks[$SLURM_ARRAY_TASK_ID]}
benchmark_output_dir="${eval_output_dir}/${benchmark}"

echo "Running benchmark: $benchmark (array index: $SLURM_ARRAY_TASK_ID)"
start_time=$(date +%s)
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Starting benchmark $benchmark at $timestamp"


CONTAINER_CMD="mkdir -p $benchmark_output_dir && echo \"Created evaluation directory: $benchmark_output_dir\" && \
cd $CODE_ROOT && export PYTHONPATH=$CODE_ROOT && echo \"$0 PYTHONPATH: $CODE_ROOT\" && \
if scripts/eval/cambrian/eval_cambrian_single.sh \
    --benchmark $benchmark \
    --ckpt $ckpt \
    --conv_mode $conv_mode \
    --output_dir $benchmark_output_dir; then
    echo 'Successfully completed benchmark: $benchmark'

else
    echo \"Error: Failed to complete benchmark: $benchmark\"
    exit 1
fi
"

# TODO: adjust path bindings (bind your checkpoint dir too when evaluating a local checkpoint)
srun --kill-on-bad-exit=1 --wait=0 \
singularity exec \
    -B ...:/hf_models \
    -B ...:/datasets \
    -B ...:/eval \
    -B /tmp:/tmp \
    $SIF \
    bash -c "$CONTAINER_CMD"

end_time=$(date +%s)
duration=$(( (end_time - start_time) / 60 ))
cur_timestamp=$(date +"%Y-%m-%d %H:%M:%S")

echo "Benchmark $benchmark completed in $duration minutes"
echo "Finished at: $cur_timestamp"

echo "Benchmark $benchmark evaluation completed successfully!"
echo "Results saved in: $benchmark_output_dir"
