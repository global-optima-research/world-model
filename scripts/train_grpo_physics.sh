#!/bin/bash
# Phase 2: 物理 Reward GRPO 训练 — 流体单域 PoC
#
# 前置步骤:
#   1. python scripts/preprocess_physics_embeddings.py (预计算 embeddings)
#   2. 确保 DanceGRPO 已安装
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

# 关键参数说明:
#   --t 33:              生成 33 帧视频 (而非单帧图像)
#   --num_generations 4: 每个 prompt 生成 4 个候选 (显存限制)
#   --sampling_steps 20: denoising 步数
#   --eta 0.3:           SDE 噪声系数 (用于计算 log prob)
#   --use_physics_reward: 使用物理 reward 替代 HPSv2

# GPU 选择: 根据当前空闲 GPU 调整
# 默认使用 GPU 5,7 (2卡), 若 8 卡空闲可改为 0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,7}
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l | tr -d ' ')
echo ">>> 使用 ${NGPU} 张 GPU: ${CUDA_VISIBLE_DEVICES}"

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
    --gradient_accumulation_steps 4 \
    --max_train_steps 50 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 25 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir outputs/grpo_physics \
    --h 480 \
    --w 832 \
    --t 33 \
    --sampling_steps 10 \
    --eta 0.3 \
    --lr_warmup_steps 5 \
    --sampler_seed 42 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --use_physics_reward \
    --num_generations 4 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-2 \
    --adv_clip_max 5.0 \
    --cfg_infer 5.0
