"""
光流基础物理 Reward

从生成视频中提取光流，基于物理先验计算 reward:
1. 时间一致性: 相邻帧的光流应平滑变化（非跳变）
2. 空间平滑性: 同一物体区域的光流应连续
3. 守恒性: 运动物体的面积应大致守恒（不凭空出现/消失）

这是最基础的物理 reward，不依赖模拟器，用于快速验证 GRPO pipeline。
"""

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class OpticalFlowReward:
    """基于光流的物理 reward（不依赖模拟器）"""

    def __init__(self, device="cuda:0"):
        self.device = device
        # RAFT-Small: 轻量级光流模型，推理快
        weights = Raft_Small_Weights.DEFAULT
        self.flow_model = raft_small(weights=weights).to(device).eval()
        self.transforms = weights.transforms()

    @torch.no_grad()
    def extract_flows(self, video: torch.Tensor) -> torch.Tensor:
        """
        从视频提取光流序列

        Args:
            video: (T, C, H, W) in [0, 1], float32

        Returns:
            flows: (T-1, 2, H, W) 光流场
        """
        T = video.shape[0]
        flows = []
        for t in range(T - 1):
            frame1 = video[t].unsqueeze(0)
            frame2 = video[t + 1].unsqueeze(0)
            # RAFT 需要 (B, C, H, W) in [0, 255] 范围
            f1, f2 = self.transforms(frame1 * 255, frame2 * 255)
            flow = self.flow_model(f1.to(self.device), f2.to(self.device))[-1]  # 取最后一次迭代
            flows.append(flow.squeeze(0))
        return torch.stack(flows)  # (T-1, 2, H, W)

    def temporal_consistency_reward(self, flows: torch.Tensor) -> float:
        """
        时间一致性: 相邻帧光流的变化应平滑

        物理直觉: 真实物体的加速度有限，速度不会在帧间剧烈跳变
        reward = -mean(||flow_t - flow_{t-1}||²)

        高 reward = 光流平滑变化（物理合理）
        低 reward = 光流剧烈跳变（非物理）
        """
        if flows.shape[0] < 2:
            return 0.0
        flow_diff = flows[1:] - flows[:-1]  # (T-2, 2, H, W)
        acceleration = (flow_diff ** 2).mean()
        # 实测 acceleration 量级 ~1000-10000, sigma 调大
        reward = torch.exp(-acceleration / 10000.0)
        return reward.item()

    def spatial_smoothness_reward(self, flows: torch.Tensor) -> float:
        """
        空间平滑性: 光流场应局部连续（非噪声）

        物理直觉: 同一刚体/流体区域的速度场应空间连续
        reward = -mean(||∇flow||²)
        """
        # Sobel-like 梯度
        dx = flows[:, :, :, 1:] - flows[:, :, :, :-1]  # 水平梯度
        dy = flows[:, :, 1:, :] - flows[:, :, :-1, :]  # 垂直梯度
        smoothness = (dx ** 2).mean() + (dy ** 2).mean()
        # 实测 smoothness 量级 ~50-200
        reward = torch.exp(-smoothness / 200.0)
        return reward.item()

    def motion_magnitude_reward(self, flows: torch.Tensor) -> float:
        """
        运动幅度: 视频应有适当的运动量（不能太静也不能太乱）

        物理直觉: 物理场景应有可观测的运动
        reward 在运动幅度适中时最高
        """
        magnitude = torch.sqrt((flows ** 2).sum(dim=1))  # (T-1, H, W)
        mean_mag = magnitude.mean()

        # 实测 magnitude 量级 ~30-100 pixels/frame
        # 鼓励适度运动，高斯型 reward 在 target 附近最高
        target = 50.0
        sigma = 40.0
        reward = torch.exp(-((mean_mag - target) ** 2) / (2 * sigma ** 2))
        return reward.item()

    def gravity_consistency_reward(self, flows: torch.Tensor) -> float:
        """
        重力一致性: 自由运动物体应有向下的加速度分量

        物理直觉: 没有支撑的物体应受重力加速
        检测: 光流的垂直分量 (v_y) 是否随时间增大
        """
        if flows.shape[0] < 3:
            return 0.0

        # 提取垂直方向光流的平均值 (正值 = 向下)
        vy = flows[:, 1, :, :].mean(dim=(1, 2))  # (T-1,)

        # 检查是否有向下加速趋势
        # 简单方法: 后半段 vy 应大于前半段
        mid = len(vy) // 2
        if mid == 0:
            return 0.5
        vy_first = vy[:mid].mean()
        vy_second = vy[mid:].mean()

        # 有加速 → reward 高
        accel = vy_second - vy_first
        reward = torch.sigmoid(accel)  # 加速向下 → >0.5
        return reward.item()

    def compute_reward(
        self,
        video: torch.Tensor,
        weights: dict = None,
    ) -> float:
        """
        计算综合光流物理 reward

        Args:
            video: (T, C, H, W) in [0, 1]
            weights: 各分项权重

        Returns:
            reward: float, 越高越好
        """
        if weights is None:
            weights = {
                "temporal": 0.3,
                "spatial": 0.3,
                "motion": 0.2,
                "gravity": 0.2,
            }

        flows = self.extract_flows(video)

        scores = {
            "temporal": self.temporal_consistency_reward(flows),
            "spatial": self.spatial_smoothness_reward(flows),
            "motion": self.motion_magnitude_reward(flows),
            "gravity": self.gravity_consistency_reward(flows),
        }

        reward = sum(weights[k] * scores[k] for k in weights)
        return reward, scores
