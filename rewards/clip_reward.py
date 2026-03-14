"""
CLIP Score Reward Model

使用 openai/clip-vit-large-patch14 计算 text-image alignment score。
用于单帧 GRPO 训练的 PoC 验证。
"""

import os
import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPRewardModel:
    # 5090 本地缓存路径
    LOCAL_CACHE = "/data/xuhao/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"

    def __init__(self, device="cuda", model_name="openai/clip-vit-large-patch14"):
        self.device = device
        # 优先使用本地缓存（5090 无法访问 HuggingFace）
        load_path = self.LOCAL_CACHE if os.path.exists(self.LOCAL_CACHE) else model_name
        self.model = CLIPModel.from_pretrained(load_path, local_files_only=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(load_path, local_files_only=True)
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def score(self, image_tensor, text):
        """
        计算 CLIP score。

        Args:
            image_tensor: (C, H, W) in [0, 1], float tensor
            text: str, prompt text

        Returns:
            reward: scalar tensor on self.device
        """
        # 转换为 PIL 用于 processor
        from torchvision.transforms.functional import to_pil_image
        pil_image = to_pil_image(image_tensor.clamp(0, 1).cpu())

        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        # cosine similarity, 已经归一化到 [0, 1] 范围
        logits = outputs.logits_per_image[0, 0]
        # 归一化到 [0, 1]（CLIP score 一般在 15-35 范围，/100 映射到合理区间）
        reward = (logits / 100.0).clamp(0, 1)
        return torch.tensor([reward.item()], device=self.device, dtype=torch.float32)
