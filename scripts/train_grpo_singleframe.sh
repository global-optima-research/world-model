#!/bin/bash
# 单帧 GRPO 训练 (对齐 DanceGRPO 原始配置)
#
# 目的: 验证 GRPO pipeline 梯度流通，确认 reward 能提升
# Reward: CLIP score (text-image alignment)
#
# 用法: bash scripts/train_grpo_singleframe.sh

set -e

cd "$(dirname "$0")/.."

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT="grpo-singleframe"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 预计算 prompt embeddings (如果不存在)
if [ ! -f "data/physics_rl_embeddings/videos2caption.json" ]; then
    echo ">>> 预计算 prompt embeddings..."
    CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_physics_embeddings.py \
        --model_path LongLive/wan_models/Wan2.1-T2V-1.3B \
        --prompt_file scripts/physics_prompts.txt \
        --output_dir data/physics_rl_embeddings
fi

echo ">>> 开始单帧 GRPO 训练..."
mkdir -p videos outputs/grpo_singleframe

# GPU 选择
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,7}
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l | tr -d ' ')
echo ">>> 使用 ${NGPU} 张 GPU: ${CUDA_VISIBLE_DEVICES}"

# 对齐 DanceGRPO 原始 Wan 配置:
#   --t 1:               单帧图像 (与 DanceGRPO 原始一致)
#   --sampling_steps 20: 20步去噪 (与 DanceGRPO 原始一致)
#   --h 512 --w 512:     正方形
#   --num_generations 8: 每 prompt 8 候选 (原始 12，我们 GPU 少)
#   --gradient_accumulation_steps 8: = batch_size * num_generations
#   --clip_range 1e-4:   DanceGRPO 原始值
#   --learning_rate 1e-5: DanceGRPO 原始值
#   --use_clip_reward:   CLIP score 替代 HPSv2
#   --ignore_last:       与 DanceGRPO 一致
#   --timestep_fraction 0.6: 与 DanceGRPO 一致

torchrun --nproc_per_node=${NGPU} --master_port 29503 \
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
    --max_train_steps 200 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 100 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir outputs/grpo_singleframe \
    --h 512 \
    --w 512 \
    --t 1 \
    --sampling_steps 20 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 42 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --use_clip_reward \
    --num_generations 8 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --clip_range 0.2 \
    --adv_clip_max 5.0 \
    --cfg_infer 5.0 \
    --num_ppo_epochs 1
