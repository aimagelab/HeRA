#!/bin/bash
#SBATCH --job-name=st1_llava
#SBATCH --output=/logs/hera/%x-%j
#SBATCH --error=/logs/hera/%x-%j
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=
#SBATCH --cpus-per-task=
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=06:00:00

set -x

# TODO: set up python env and load the needed modules / container image

# LUMI / AMD GPUs only: Slingshot network interfaces for NCCL (RCCL) and a per-node MIOpen kernel cache
# export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# export NCCL_NET_GDR_LEVEL=PHB
# export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_NODEID}
# export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# TODO: adjust this
export CODE_ROOT=~/git/HeRA

export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=
export WANDB_PROJECT=
export OMP_NUM_THREADS=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))

# TODO: adjust this
IDX=${SLURM_ARRAY_TASK_ID:-0}

LLMS=(
    "Qwen/Qwen2.5-3B-Instruct" "qwen2_5_3B"
    "Qwen/Qwen2.5-7B-Instruct" "qwen2_5_7B"
    "Qwen/Qwen3-4B-Instruct-2507" "qwen3_4B"
    "Qwen/Qwen3-8B" "qwen3_8B"
    "lmsys/vicuna-7b-v1.5" "vicuna_7B"
    "meta-llama/Meta-Llama-3-8B-Instruct" "llama3_8B"
)

stride=2
language_model_name="${LLMS[(($IDX * $stride))]}"
_llm_suffix="${LLMS[(($IDX * $stride + 1))]}"

vision_model_name="google/siglip2-so400m-patch14-384"
skip_left_visual_tokens=0 # siglip2 has no CLS token
vision_layer_idx=-1 # If using CLIP, LLaVA advises to use the second-last, e.g., -2

run_name="${SLURM_JOB_NAME}_${_llm_suffix}"
output_dir="/checkpoints/hera/${run_name}" # TODO: adjust this

# TODO: adjust to obtain global batch size 256 according to LLaVA
per_device_train_batch_size=16
gradient_accumulation_steps=1

# TODO: adjust this
train_data_path="/datasets/llava_lmdb/blip_laion_cc_sbu_558k.json"
lmdb_root="/datasets/llava_lmdb"

echo "Nodes: ${SLURM_NNODES}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "GPUs: ${SLURM_GPUS_PER_NODE}"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"
echo "WORLD SIZE: ${WORLD_SIZE}"

CONTAINER_CMD="cd $CODE_ROOT && export PYTHONPATH=$PYTHONPATH:. && \
export MASTER_ADDR=${MASTER_ADDR} && \
export MASTER_PORT=${MASTER_PORT} && \
export WORLD_SIZE=${WORLD_SIZE} && \
export RANK=\$SLURM_PROCID && \
export LOCAL_RANK=\$SLURM_LOCALID && \
python -u src/train/train.py \
    --seed 42 \
    --deepspeed deepspeed/zero2.json \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --output_dir $output_dir \
    --report_to wandb \
    --run_name $run_name \
    --logging_steps 5 \
    --bf16 True \
    --num_train_epochs 1 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --eval_strategy no \
    --train_data_path $train_data_path \
    --lmdb_root $lmdb_root \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --local_rank \$LOCAL_RANK \
    --language_model_name $language_model_name \
    --attn_implementation sdpa \
    --conv_template plain \
    --vision_model_name $vision_model_name \
    --vision_layer_idx $vision_layer_idx \
    --skip_left_visual_tokens $skip_left_visual_tokens \
    --bidir_attn_visual_tokens False \
    --projector_type mlp \
    --projector_bias True \
    --train_proj_only \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1e-3 \
    --lr_scheduler_type cosine \
    --weight_decay 0. \
    --warmup_ratio 0.03
"
# TODO: adjust path bindings
srun --kill-on-bad-exit=1 --wait=0 \
singularity exec \
    -B ...:/hf_models \
    -B ...:/datasets \
    -B ...:/checkpoints \
    -B /tmp:/tmp \
    $SIF \
    bash -c "$CONTAINER_CMD"
