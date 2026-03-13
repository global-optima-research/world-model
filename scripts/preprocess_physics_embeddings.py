"""
预计算物理 prompt 的 T5 embeddings

使用 Wan 原生的 T5EncoderModel (非 HuggingFace 格式)，
从 models_t5_umt5-xxl-enc-bf16.pth 加载权重。

DanceGRPO 训练时直接加载预计算的 embedding，避免每次都跑 T5。

用法:
CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_physics_embeddings.py \
    --model_path LongLive/wan_models/Wan2.1-T2V-1.3B \
    --prompt_file scripts/physics_prompts.txt \
    --output_dir data/physics_rl_embeddings
"""

import argparse
import json
import os
import sys
import torch

# 添加 LongLive 到 sys.path 以使用 Wan 原生 T5
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LongLive"))

from wan.modules.t5 import T5EncoderModel


NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

SEQ_LEN = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Wan2.1-T2V-1.3B (original format)")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "negative_prompt_embeds"), exist_ok=True)

    # 加载 prompts
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"加载 {len(prompts)} 条 prompt")

    # 加载 Wan 原生 T5 text encoder
    checkpoint_path = os.path.join(args.model_path, "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = os.path.join(args.model_path, "google", "umt5-xxl")
    print(f"加载 T5 text encoder: {checkpoint_path}")
    print(f"Tokenizer: {tokenizer_path}")

    device = torch.cuda.current_device()
    text_encoder = T5EncoderModel(
        text_len=SEQ_LEN,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
    )
    print("T5 encoder 加载完成")

    # 预计算 embeddings
    annotations = []

    # 先计算 negative prompt embedding (只需一次)
    print("编码 negative prompt...")
    with torch.no_grad():
        neg_context = text_encoder([NEGATIVE_PROMPT], device)
    # neg_context 是 list of tensors, 每个 [seq_len, 4096]
    neg_embed = neg_context[0].cpu()  # [variable_len, 4096]
    # 填充到固定长度 SEQ_LEN
    if neg_embed.shape[0] < SEQ_LEN:
        pad = torch.zeros(SEQ_LEN - neg_embed.shape[0], neg_embed.shape[1])
        neg_embed = torch.cat([neg_embed, pad], dim=0)
    else:
        neg_embed = neg_embed[:SEQ_LEN]
    print(f"Negative embed shape: {neg_embed.shape}")

    for idx, prompt in enumerate(prompts):
        print(f"[{idx}/{len(prompts)}] {prompt[:60]}...")

        with torch.no_grad():
            pos_context = text_encoder([prompt], device)
        pos_embed = pos_context[0].cpu()  # [variable_len, 4096]
        # 填充到固定长度 SEQ_LEN
        if pos_embed.shape[0] < SEQ_LEN:
            pad = torch.zeros(SEQ_LEN - pos_embed.shape[0], pos_embed.shape[1])
            pos_embed = torch.cat([pos_embed, pad], dim=0)
        else:
            pos_embed = pos_embed[:SEQ_LEN]

        prompt_embed_file = f"prompt_{idx:04d}.pt"
        negative_file = f"negative_{idx:04d}.pt"

        torch.save(pos_embed, os.path.join(args.output_dir, "prompt_embed", prompt_embed_file))
        torch.save(neg_embed, os.path.join(args.output_dir, "negative_prompt_embeds", negative_file))

        annotations.append({
            "prompt_embed_path": prompt_embed_file,
            "negative_prompt_embeds_path": negative_file,
            "caption": prompt,
            "length": 1,
        })

    # 保存索引
    json_path = os.path.join(args.output_dir, "videos2caption.json")
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"\n完成! 保存到 {args.output_dir}")
    print(f"  - {len(annotations)} 条 prompt embeddings")
    print(f"  - 每个 embed shape: [{SEQ_LEN}, 4096]")
    print(f"  - 索引文件: {json_path}")

    del text_encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
