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
from transformers import AutoTokenizer, UMT5EncoderModel


NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def encode_prompt(tokenizer, text_encoder, prompt, device, max_length=256):
    """Encode a single prompt using UMT5"""
    tokens = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            prompt_embeds = outputs.last_hidden_state

    return prompt_embeds.squeeze(0).cpu(), attention_mask.squeeze(0).cpu()


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

    # 加载 T5 text encoder (UMT5-XXL)
    t5_path = os.path.join(args.model_path, "google", "umt5-xxl")
    print(f"加载 T5 text encoder: {t5_path}")
    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    text_encoder = UMT5EncoderModel.from_pretrained(
        t5_path, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    print("T5 encoder 加载完成")

    # 预计算 embeddings
    annotations = []
    for idx, prompt in enumerate(prompts):
        print(f"[{idx}/{len(prompts)}] {prompt[:60]}...")

        prompt_embeds, _ = encode_prompt(tokenizer, text_encoder, prompt, "cuda")
        negative_embeds, _ = encode_prompt(tokenizer, text_encoder, NEGATIVE_PROMPT, "cuda")

        prompt_embed_file = f"prompt_{idx:04d}.pt"
        negative_file = f"negative_{idx:04d}.pt"

        torch.save(prompt_embeds, os.path.join(args.output_dir, "prompt_embed", prompt_embed_file))
        torch.save(negative_embeds, os.path.join(args.output_dir, "negative_prompt_embeds", negative_file))

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

    del text_encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
