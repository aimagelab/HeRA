#!/bin/bash
#SBATCH --job-name=eval_chairmscoco
# TODO: adjust log paths
#SBATCH --output=/logs/chair_mscoco/%x-%j
#SBATCH --error=/logs/chair_mscoco/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --account=
#SBATCH --time=02:00:00

# Usage:
#   sbatch scripts/eval/chair_mscoco/eval_chairmscoco.sh <model_path> <conv_mode> \
#       [attn_implementation] [temperature] [seed] [max_new_tokens]
# e.g.
#   sbatch scripts/eval/chair_mscoco/eval_chairmscoco.sh aimagelab/HeRA-Qwen2.5-3B-SigLIP2 qwen2_5

# TODO: set up python env and load the needed modules / container image

# TODO: adjust this
export CODE_ROOT=/path/to/HeRA

export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# TODO: adjust this
export HF_HUB_CACHE="..."
# Uncomment once the checkpoint, the base LLM tokenizer and the vision processor are in the cache.
# export HF_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# --- args ---
model_path="$1"                        # HF Hub id (e.g. aimagelab/HeRA-Qwen2.5-3B-SigLIP2) or local checkpoint dir
conv_mode="$2"                         # e.g. qwen2_5 (see the checkpoints table in the README)
attn_implementation="${3:-sdpa}"       # sdpa | flash_attention_2 | eager
temperature="${4:-0.0}"                # sampling temperature
seed="${5:-42}"                        # random seed
max_new_tokens="${6:-512}"
model_name="${model_path%/}"
model_name="${model_name##*/}"

dtype="bf16"

# TODO: adjust these
question_file="/datasets/chair_mscoco/data/chair-500.jsonl"
image_folder="/datasets/chair_mscoco/MSCOCO/images"
out_dir="/eval/chair_mscoco/${model_name}"
answers_file="${out_dir}/answers_chairmscoco_seed${seed}_temp${temperature}_tok${max_new_tokens}_${dtype}__${model_name}.jsonl"

start_time=$(date +%s)
echo "Checkpoint:    $model_path"
echo "Conv mode:     $conv_mode"
echo "Attn impl.:    $attn_implementation"
echo "Dtype:         $dtype"
echo "Temperature:   $temperature"
echo "Seed:          $seed"
echo "Max new tok.:  $max_new_tokens"
echo "Question file: $question_file"
echo "Image folder:  $image_folder"
echo "Answers file:  $answers_file"

CONTAINER_CMD="mkdir -p $out_dir && \
if [[ -e $answers_file ]]; then \
    echo \"Refusing to overwrite existing answers file: $answers_file\"; \
    exit 1; \
fi && \
cd $CODE_ROOT && export PYTHONPATH=$CODE_ROOT && \
python chair_mscoco/model_vqa_chairmscoco.py \
    --model_path $model_path \
    --image_folder $image_folder \
    --question_file $question_file \
    --answers_file $answers_file \
    --conv_mode $conv_mode \
    --temperature $temperature \
    --seed $seed \
    --max_new_tokens $max_new_tokens \
    --attn_implementation $attn_implementation \
    --dtype $dtype"

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
echo "=== eval_chairmscoco generation finished: $model_name in ${duration} minutes ==="
echo "Results saved in: $answers_file"
