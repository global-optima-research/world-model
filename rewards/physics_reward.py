"""
PhysicsRewardModel — 统一物理 Reward 接口

整合光流基础 reward + 流体专项 reward，提供 DanceGRPO 接入接口。
后续扩展: 软体 reward, 布料 reward, Warp 模拟对比 reward。
"""

import torch
from .flow_reward import OpticalFlowReward
from .fluid_reward import FluidPhysicsReward


class PhysicsRewardModel:
    """
    统一物理 Reward 模型

    DanceGRPO 接入方式:
        reward_model = PhysicsRewardModel(device="cuda:0")
        score = reward_model.score_video(video_tensor, prompt)
    """

    def __init__(self, device="cuda:0", mode="auto"):
        """
        Args:
            device: CUDA 设备
            mode: reward 模式
                - "auto": 根据 prompt 自动选择 (TODO: 需要 prompt 分类器)
                - "flow_only": 仅用光流基础 reward
                - "fluid": 光流 + 流体物理
                - "softbody": 光流 + 软体物理 (TODO)
                - "cloth": 光流 + 布料物理 (TODO)
        """
        self.device = device
        self.mode = mode
        self.flow_reward = OpticalFlowReward(device=device)
        self.fluid_reward = FluidPhysicsReward(device=device)

    @torch.no_grad()
    def score_video(
        self,
        video: torch.Tensor,
        prompt: str = "",
        return_details: bool = False,
    ) -> float:
        """
        对一个视频计算物理 reward score

        Args:
            video: (T, C, H, W) in [0, 1], float32
                   或 (T, H, W, C) in [0, 255], uint8 (会自动转换)
            prompt: 文本 prompt (用于场景分类)
            return_details: 是否返回各分项分数

        Returns:
            score: float, 越高越好
            (可选) details: dict, 各分项分数
        """
        video = self._normalize_video(video)

        # 提取光流
        flows = self.flow_reward.extract_flows(video)

        # 基础光流 reward
        flow_score, flow_details = self.flow_reward.compute_reward(video)

        # 根据模式选择额外 reward
        mode = self._detect_mode(prompt) if self.mode == "auto" else self.mode

        if mode == "fluid":
            fluid_score, fluid_details = self.fluid_reward.compute_reward(flows)
            # 综合: 0.4 基础光流 + 0.6 流体专项
            total_score = 0.4 * flow_score + 0.6 * fluid_score
            details = {"flow": flow_details, "fluid": fluid_details, "mode": "fluid"}
        elif mode == "softbody":
            # TODO: 实现软体 reward
            total_score = flow_score
            details = {"flow": flow_details, "mode": "softbody (fallback to flow)"}
        elif mode == "cloth":
            # TODO: 实现布料 reward
            total_score = flow_score
            details = {"flow": flow_details, "mode": "cloth (fallback to flow)"}
        else:
            total_score = flow_score
            details = {"flow": flow_details, "mode": "flow_only"}

        if return_details:
            return total_score, details
        return total_score

    def score_videos_batch(
        self,
        videos: list,
        prompts: list = None,
    ) -> torch.Tensor:
        """
        批量打分，用于 GRPO 的 group 候选评估

        Args:
            videos: list of (T, C, H, W) tensors
            prompts: list of str

        Returns:
            scores: (N,) tensor
        """
        if prompts is None:
            prompts = [""] * len(videos)

        scores = []
        for video, prompt in zip(videos, prompts):
            score = self.score_video(video, prompt)
            scores.append(score)

        return torch.tensor(scores, device=self.device)

    def _normalize_video(self, video: torch.Tensor) -> torch.Tensor:
        """统一视频格式为 (T, C, H, W) in [0, 1]"""
        if video.dtype == torch.uint8:
            video = video.float() / 255.0

        # (T, H, W, C) → (T, C, H, W)
        if video.ndim == 4 and video.shape[-1] in (1, 3):
            video = video.permute(0, 3, 1, 2)

        # (B, T, C, H, W) → (T, C, H, W) 去掉 batch dim
        if video.ndim == 5:
            video = video[0]

        return video.to(self.device)

    def _detect_mode(self, prompt: str) -> str:
        """
        简单关键词匹配检测物理场景类型

        TODO: 后续用 LLM 做更精确的场景分类
        """
        prompt_lower = prompt.lower()

        fluid_keywords = [
            "water", "liquid", "pour", "flow", "splash", "ripple",
            "wave", "rain", "waterfall", "coffee", "smoke", "mist",
            "steam", "fog", "river", "ocean", "flood", "drip", "puddle",
        ]
        softbody_keywords = [
            "jelly", "rubber", "bounce", "elastic", "stretch", "squeeze",
            "deform", "clay", "dough", "balloon", "sponge", "wobble",
        ]
        cloth_keywords = [
            "cloth", "fabric", "flag", "curtain", "sheet", "tablecloth",
            "silk", "paper", "wave", "fold", "drape", "flutter",
        ]

        fluid_count = sum(1 for kw in fluid_keywords if kw in prompt_lower)
        soft_count = sum(1 for kw in softbody_keywords if kw in prompt_lower)
        cloth_count = sum(1 for kw in cloth_keywords if kw in prompt_lower)

        max_count = max(fluid_count, soft_count, cloth_count)
        if max_count == 0:
            return "flow_only"

        if fluid_count == max_count:
            return "fluid"
        elif soft_count == max_count:
            return "softbody"
        else:
            return "cloth"
