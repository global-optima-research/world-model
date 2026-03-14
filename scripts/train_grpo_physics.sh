#!/bin/bash
# Phase 2: 物理 Reward GRPO 训练
#
# 前置步骤:
#   1. python scripts/preprocess_physics_embeddings.py (预计算 embeddings)
#   2. pip install -e DanceGRPO --no-deps
#
# 用法: bash scripts/train_grpo_physics.sh

set -e

cd "$(dirname "$0")/.."

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT="physics-grpo"

# 预计算 prompt embeddings (如果不存在)
if [ ! -f "data/physics_rl_embeddings/videos2caption.json" ]; then
    echo ">>> 预计算物理 prompt embeddings..."
    CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_physics_embeddings.py \
        --model_path LongLive/wan_models/Wan2.1-T2V-1.3B \
        --prompt_file scripts/physics_prompts.txt \
        --output_dir data/physics_rl_embeddings
fi

echo ">>> 开始 GRPO 训练..."
mkdir -p videos outputs/grpo_physics

# GPU 选择
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,7}
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l | tr -d ' ')
echo ">>> 使用 ${NGPU} 张 GPU: ${CUDA_VISIBLE_DEVICES}"

# 关键改动 (对齐 DanceGRPO 原始配置):
#   --t 5:               最短可计算光流的视频 (5帧, 4帧间隔)
#   --h 512 --w 512:     正方形，与 DanceGRPO 原始一致
#   --num_generations 8: 增加候选数，advantage 估计更稳定
#   --gradient_accumulation_steps 8: 与 num_generations 对齐，累积完整 group 再 step
#   --clip_range 0.2:    放宽 clipping，视频场景 ratio 变化大于单帧
#   --sampling_steps 20: 对齐 DanceGRPO 原始
#   --max_train_steps 100: 10 epochs × 10 steps/epoch

torchrun --nproc_per_node=${NGPU} --master_port 29502 \
    fastvideo/train_grpo_physics.py \
    --seed 42 \
    --pretrained_model_name_or_path /data/xuhao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b \
    --vae_model_path /data/xuhao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b \
    --cache_dir data/.cache \
    --data_json_path data/physics_rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 100 \
    --learning_rate 5e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir outputs/grpo_physics_v7 \
    --h 512 \
    --w 512 \
    --t 5 \
    --sampling_steps 20 \
    --eta 0.3 \
    --lr_warmup_steps 5 \
    --sampler_seed 42 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --use_physics_reward \
    --num_generations 8 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 0.2 \
    --adv_clip_max 5.0 \
    --cfg_infer 5.0
