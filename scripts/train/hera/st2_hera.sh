#!/bin/bash
#SBATCH --job-name=st2_hera
#SBATCH --output=/logs/hera/%x-%j
#SBATCH --error=/logs/hera/%x-%j
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=
#SBATCH --cpus-per-task=
#SBATCH --account=
#SBATCH --partition=
#SBATCH --time=12:00:00

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
    "Qwen/Qwen2.5-3B-Instruct" "qwen2_5_3B" "qwen2_5"
    "Qwen/Qwen2.5-7B-Instruct" "qwen2_5_7B" "qwen2_5"
    "Qwen/Qwen3-4B-Instruct-2507" "qwen3_4B" "qwen3_inst"
    "Qwen/Qwen3-8B" "qwen3_8B" "qwen3"
    "lmsys/vicuna-7b-v1.5" "vicuna_7B" "vicuna_v1"
    "meta-llama/Meta-Llama-3-8B-Instruct" "llama3_8B" "llama3"
)

stride=3
language_model_name="${LLMS[(($IDX * $stride))]}"
_llm_suffix="${LLMS[(($IDX * $stride + 1))]}"
conv_template="${LLMS[(($IDX * $stride + 2))]}"

# TODO: adjest this: path to the projector weights obtained from stage 1
load_proj_weights="/checkpoints/hera/st1_hera_${_llm_suffix}_dinov2l/projector.pt"

vision_model_name="google/siglip2-so400m-patch14-384"
skip_left_visual_tokens=0 # siglip2 has no CLS token
vision_layer_idx=-1 # If using CLIP, LLaVA advises to use the second-last, e.g., -2

# We notice that, allowing bi-directional attention among visual tokens helps lowering the alignment loss,
# achieving higher score of MKNN after stage 1. Despite that, performance after stage 2 are very similar no matter 
# we use causal or bi-directional attention among visual tokens. 
# However, during stage 1, we detected instabilites when training some LLMs without HeRA if using the bi-directional mask on visual tokens,
# namely that the training loss has some spurious huge spikes. Because of this, we recommend sticking with the usual
# causal mask, i.e. `bidir_attn_visual_tokens=False`, when training the LLaVA baselines, 
# and (optionally) set `bidir_attn_visual_tokens=True` when HeRA is active.
# Because bi-directional attention mask is not supported by FlashAttention 2, we use the `sdpa` attention implementation.
bidir_attn_visual_tokens=True

run_name="${SLURM_JOB_NAME}_${_llm_suffix}_dinov2l"
output_dir="/checkpoints/hera/${run_name}" # TODO: adjust this

# TODO: adjust to obtain global batch size 128 according to LLaVA
per_device_train_batch_size=4
gradient_accumulation_steps=1

# % --- HeRA --- %
tgt_vision_model_name="facebook/dinov2-large"
tgt_vision_layer_idx=-1
tgt_skip_left_visual_tokens=1
# Select the heads on which to apply the HeRA loss.
# These are specified as a string following the notation `{layer_idx1}:{head_idx1, head_idx2,...}`.
# Refer to Table 5 in the paper for the head details of each LLM.
# In the example below, that means:
# - 4th layer; 4th head
# - 5th layer; 9th, 10th, 14th, and 15th heads
ra_llm_layers_heads="3:4 5:9 10 14 15"
# We recommend setting `hera_topk` to 10 if the effective batch size (not taking grad_acc into account) is >= 128
hera_topk=10
# % ------------ %

# TODO: adjust this
train_data_path="/datasets/llava_lmdb/llava_v1_5_mix665k.json"
lmdb_root="/datasets/llava_lmdb"

echo "Nodes: ${SLURM_NNODES}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "GPUs: ${SLURM_GPUS_PER_NODE}"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"
echo "WORLD SIZE: ${WORLD_SIZE}"

CONTAINER_CMD="cd $CODE_ROOT && export PYTHONPATH=$PYTHONPATH:.:/user-software/venv/pytorch/lib/python3.12/site-packages && \
export MASTER_ADDR=${MASTER_ADDR} && \
export MASTER_PORT=${MASTER_PORT} && \
export WORLD_SIZE=${WORLD_SIZE} && \
export RANK=\$SLURM_PROCID && \
export LOCAL_RANK=\$SLURM_LOCALID && \
python -u src/train/train.py \
    --seed 42 \
    --deepspeed deepspeed/zero3.json \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --output_dir $output_dir \
    --report_to wandb \
    --run_name $run_name \
    --logging_steps 5 \
    --bf16 True \
    --num_train_epochs 1 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --eval_strategy no \
    --train_data_path $train_data_path \
    --lmdb_root $lmdb_root \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --local_rank \$LOCAL_RANK \
    --language_model_name $language_model_name \
    --attn_implementation sdpa \
    --conv_template $conv_template \
    --vision_model_name $vision_model_name \
    --vision_layer_idx $vision_layer_idx \
    --skip_left_visual_tokens $skip_left_visual_tokens \
    --bidir_attn_visual_tokens $bidir_attn_visual_tokens \
    --projector_type mlp \
    --projector_bias True \
    --load_proj_weights $load_proj_weights \
    --tgt_vision_model_name $tgt_vision_model_name \
    --tgt_image_processor \
    --tgt_vision_layer_idx $tgt_vision_layer_idx \
    --tgt_skip_left_visual_tokens $tgt_skip_left_visual_tokens \
    --tgt_projector_type identity \
    --ra_llm_layers_heads '$ra_llm_layers_heads' \
    --ra_loss_w 0.01 \
    --hera True \
    --hera_topk $hera_topk \
    --hera_learn_logit_scale \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --image_aspect_ratio pad \
    --group_by_mod_lens
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
