#!/bin/bash
# LongLive 推理脚本 — 单卡 RTX 5090 即可
# 用法: bash scripts/run_inference.sh

set -e

cd "$(dirname "$0")/../LongLive"

# 生成物理场景视频，用于建立 baseline 视觉印象
torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  inference.py \
  --config_path configs/longlive_inference.yaml
