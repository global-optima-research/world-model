#!/bin/bash
# Week 1 环境搭建脚本 — 在 8x RTX 5090 上运行
# 用法: bash scripts/setup.sh

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Step 1: 下载 Wan 2.1 基础模型 + LongLive 权重"
echo "=========================================="

cd LongLive

# 下载 Wan 2.1 1.3B 基础模型（T5 encoder, VAE, DiT weights）
echo ">>> 下载 Wan2.1-T2V-1.3B ..."
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B

# 下载 LongLive fine-tuned weights（base checkpoint + LoRA）
echo ">>> 下载 LongLive 权重 ..."
huggingface-cli download Efficient-Large-Model/LongLive --local-dir longlive_models

echo ""
echo "=========================================="
echo "Step 2: 安装 LongLive 依赖"
echo "=========================================="

pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Step 3: 安装 NVIDIA Warp（可微物理模拟器）"
echo "=========================================="

pip install warp-lang

echo ""
echo "=========================================="
echo "Step 4: 安装运动估计工具"
echo "=========================================="

# RAFT / UniMatch 光流估计（后续用于奖励计算）
pip install torchvision  # RAFT 内置于 torchvision

echo ""
echo "=========================================="
echo "Step 5: Clone DanceGRPO（GRPO 训练框架）"
echo "=========================================="

cd "$PROJECT_ROOT"
if [ ! -d "DanceGRPO" ]; then
    git clone --depth 1 https://github.com/XueZeyue/DanceGRPO.git
    echo ">>> DanceGRPO cloned"
else
    echo ">>> DanceGRPO already exists, skipping"
fi

echo ""
echo "=========================================="
echo "完成！下一步："
echo "  1. 跑 LongLive 推理: bash scripts/run_inference.sh"
echo "  2. 跑 Warp demo: python scripts/test_warp.py"
echo "=========================================="
