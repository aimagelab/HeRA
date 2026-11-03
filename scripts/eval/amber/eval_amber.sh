#!/bin/bash
#SBATCH --job-name=eval_amber
# TODO: adjust log paths
#SBATCH --output=/logs/amber/%x-%j
#SBATCH --error=/logs/amber/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=
#SBATCH --account=
#SBATCH --time=03:30:00

# Usage:
#   sbatch scripts/eval/amber/eval_amber.sh <model_path> <conv_mode> <generative|discriminative> \
#       [attn_implementation] [temperature] [seed]
# e.g.
#   sbatch scripts/eval/amber/eval_amber.sh aimagelab/HeRA-Qwen2.5-3B-SigLIP2 qwen2_5 generative

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
mode="$3"                              # generative | discriminative
attn_implementation="${4:-sdpa}"       # sdpa | flash_attention_2 | eager
temperature="${5:-0.0}"                # sampling temperature
seed="${6:-42}"                        # random seed
model_name="${model_path%/}"
model_name="${model_name##*/}"

dtype="bf16"

# In discriminative mode, the instruction below is appended to every query to enforce Yes/No answers.
case "$mode" in
  generative)     max_new_tokens=512; test_prompt="" ;;
  discriminative) max_new_tokens=3;   test_prompt=" Answer only with Yes or No. Use exactly one word. Do not use commas dots or symbols." ;;
  *) echo "Unknown mode: $mode (expected generative|discriminative)"; exit 1 ;;
esac

# TODO: adjust these
question_file="/datasets/amber/data/query/query_${mode}.json"
image_folder="/datasets/amber/image"
out_dir="/eval/amber/${model_name}"
answers_file="${out_dir}/answers_amber_${mode}_seed${seed}_temp${temperature}_tok${max_new_tokens}_${dtype}__${model_name}.json"

start_time=$(date +%s)
echo "Checkpoint:     $model_path"
echo "Conv mode:      $conv_mode"
echo "Mode:           $mode"
echo "Attn impl.:     $attn_implementation"
echo "Dtype:          $dtype"
echo "Temperature:    $temperature"
echo "Seed:           $seed"
echo "Test prompt:    $test_prompt"
echo "Question file:  $question_file"
echo "Image folder:   $image_folder"
echo "Answers file:   $answers_file"

CONTAINER_CMD="mkdir -p $out_dir && \
if [[ -e $answers_file ]]; then \
    echo \"Refusing to overwrite existing answers file: $answers_file\"; \
    exit 1; \
fi && \
cd $CODE_ROOT && export PYTHONPATH=$CODE_ROOT && \
python amber/model_vqa_amber.py \
    --model_path $model_path \
    --image_folder $image_folder \
    --question_file $question_file \
    --answers_file $answers_file \
    --conv_mode $conv_mode \
    --temperature $temperature \
    --seed $seed \
    --max_new_tokens $max_new_tokens \
    --attn_implementation $attn_implementation \
    --dtype $dtype \
    --test_prompt \"$test_prompt\""

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
echo "=== eval_amber generation finished: $model_name ($mode) in ${duration} minutes ==="
echo "Results saved in: $answers_file"
