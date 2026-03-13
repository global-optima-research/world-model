"""
预计算物理 prompt 的 T5 embeddings

DanceGRPO 训练时直接加载预计算的 embedding，避免每次都跑 T5。
复用 DanceGRPO 的预处理脚本逻辑。

用法:
CUDA_VISIBLE_DEVICES=0 python scripts/preprocess_physics_embeddings.py \
    --model_path LongLive/wan_models/Wan2.1-T2V-1.3B \
    --prompt_file scripts/physics_prompts.txt \
    --output_dir data/physics_rl_embeddings
"""

import argparse
import json
import os
import torch
from diffusers import WanPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "negative_prompt_embeds"), exist_ok=True)

    # 加载 prompts
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"加载 {len(prompts)} 条 prompt")

    # 加载 pipeline (只需要 text encoder)
    print(f"加载模型: {args.model_path}")
    pipe = WanPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # 预计算 embeddings
    annotations = []
    for idx, prompt in enumerate(prompts):
        print(f"[{idx}/{len(prompts)}] {prompt[:60]}...")

        # Encode prompt
        prompt_embeds, negative_prompt_embeds, _, _ = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            do_classifier_free_guidance=True,
            device="cuda",
        )

        # 保存
        prompt_embed_file = f"prompt_{idx:04d}.pt"
        negative_file = f"negative_{idx:04d}.pt"

        torch.save(
            prompt_embeds.squeeze(0).cpu(),
            os.path.join(args.output_dir, "prompt_embed", prompt_embed_file),
        )
        torch.save(
            negative_prompt_embeds.squeeze(0).cpu(),
            os.path.join(args.output_dir, "negative_prompt_embeds", negative_file),
        )

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
    print(f"  - 索引文件: {json_path}")

    # 清理 GPU 显存
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
