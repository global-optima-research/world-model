"""
流体物理 Reward — 基于 Warp SPH 模拟器

核心思路:
1. 从生成视频提取光流 → 速度场 (velocity field)
2. 用 Warp SPH 模拟同场景的流体运动 → 参考速度场
3. 对比两个速度场的一致性 → reward

Level 1 (本文件): 不依赖场景解析，直接用物理先验约束:
- Navier-Stokes 不可压缩性: div(v) ≈ 0
- 质量守恒: 流体区域面积不应剧变
- 重力加速度方向一致性

Level 2 (后续): 接入 Warp SPH 做 prompt-conditioned 模拟对比
"""

import torch
import torch.nn.functional as F
import numpy as np


class FluidPhysicsReward:
    """流体物理 reward，基于 Navier-Stokes 先验"""

    def __init__(self, device="cuda:0"):
        self.device = device

    def divergence_free_reward(self, flows: torch.Tensor) -> float:
        """
        不可压缩性约束: div(v) = ∂vx/∂x + ∂vy/∂y ≈ 0

        对于不可压缩流体（水、大部分液体），速度场的散度应接近零。
        这是 Navier-Stokes 方程最基本的约束。

        Args:
            flows: (T-1, 2, H, W) 光流场

        Returns:
            reward: float, div 越小 reward 越高
        """
        vx = flows[:, 0, :, :]  # (T-1, H, W)
        vy = flows[:, 1, :, :]

        # 中心差分计算偏导
        dvx_dx = (vx[:, :, 2:] - vx[:, :, :-2]) / 2.0  # ∂vx/∂x
        dvy_dy = (vy[:, 2:, :] - vy[:, :-2, :]) / 2.0  # ∂vy/∂y

        # 对齐尺寸
        min_h = min(dvx_dx.shape[1], dvy_dy.shape[1])
        min_w = min(dvx_dx.shape[2], dvy_dy.shape[2])
        dvx_dx = dvx_dx[:, :min_h, :min_w]
        dvy_dy = dvy_dy[:, :min_h, :min_w]

        divergence = dvx_dx + dvy_dy  # 应接近 0
        div_magnitude = (divergence ** 2).mean()

        # 实测 div_magnitude 量级 ~50-200
        reward = torch.exp(-div_magnitude / 200.0)
        return reward.item()

    def mass_conservation_reward(self, flows: torch.Tensor, threshold: float = 1.0) -> float:
        """
        质量守恒: 运动区域（流体区域）的面积不应剧变

        物理直觉: 流体不会凭空产生或消失
        通过光流幅度阈值检测"运动区域"，检查其面积随时间的变化

        Args:
            flows: (T-1, 2, H, W)
            threshold: 光流幅度阈值，大于此值认为是运动区域
        """
        magnitude = torch.sqrt((flows ** 2).sum(dim=1))  # (T-1, H, W)
        motion_mask = (magnitude > threshold).float()  # 二值化运动区域

        # 每帧的运动区域面积
        areas = motion_mask.sum(dim=(1, 2))  # (T-1,)

        if areas.shape[0] < 2:
            return 1.0

        # 面积变化率
        area_change = torch.abs(areas[1:] - areas[:-1]) / (areas[:-1] + 1e-6)
        mean_change = area_change.mean()

        # 变化率小 → 守恒 → reward 高
        reward = torch.exp(-mean_change)
        return reward.item()

    def vorticity_reward(self, flows: torch.Tensor) -> float:
        """
        涡度合理性: curl(v) = ∂vy/∂x - ∂vx/∂y

        流体的涡度应:
        1. 空间上连续（不出现孤立的涡度点）
        2. 时间上缓慢衰减（粘性耗散）

        高涡度区域应有空间连续性，而非随机噪声。
        """
        vx = flows[:, 0, :, :]
        vy = flows[:, 1, :, :]

        # 涡度 = ∂vy/∂x - ∂vx/∂y
        dvy_dx = (vy[:, :, 2:] - vy[:, :, :-2]) / 2.0
        dvx_dy = (vx[:, 2:, :] - vx[:, :-2, :]) / 2.0

        min_h = min(dvy_dx.shape[1], dvx_dy.shape[1])
        min_w = min(dvy_dx.shape[2], dvx_dy.shape[2])
        dvy_dx = dvy_dx[:, :min_h, :min_w]
        dvx_dy = dvx_dy[:, :min_h, :min_w]

        vorticity = dvy_dx - dvx_dy  # (T-1, H', W')

        # 涡度的空间梯度应小（空间连续性）
        vort_dx = vorticity[:, :, 1:] - vorticity[:, :, :-1]
        vort_dy = vorticity[:, 1:, :] - vorticity[:, :-1, :]
        vort_smoothness = (vort_dx ** 2).mean() + (vort_dy ** 2).mean()

        # 涡度的空间平滑性，实测量级 ~10-100
        if vorticity.shape[0] >= 2:
            vort_magnitude = (vorticity ** 2).mean(dim=(1, 2))
            # 检查后半段是否比前半段小
            mid = len(vort_magnitude) // 2
            if mid > 0:
                decay = vort_magnitude[:mid].mean() - vort_magnitude[mid:].mean()
                decay_reward = torch.sigmoid(decay)  # 有衰减 → > 0.5
            else:
                decay_reward = torch.tensor(0.5)
        else:
            decay_reward = torch.tensor(0.5)

        smoothness_reward = torch.exp(-vort_smoothness / 100.0)
        reward = 0.6 * smoothness_reward.item() + 0.4 * decay_reward.item()
        return reward

    def surface_tension_reward(self, flows: torch.Tensor) -> float:
        """
        表面张力近似: 流体边界处速度应指向内部（表面张力使液滴趋于圆形）

        简化实现: 检测运动区域边界，边界处的法向光流分量应指向内部
        """
        magnitude = torch.sqrt((flows ** 2).sum(dim=1))  # (T-1, H, W)

        # 检测运动区域边界（Sobel 梯度）
        mag_padded = F.pad(magnitude.unsqueeze(1), (1, 1, 1, 1), mode='replicate')
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device)
        sobel_y = sobel_x.T
        edge_x = F.conv2d(mag_padded, sobel_x.reshape(1, 1, 3, 3))
        edge_y = F.conv2d(mag_padded, sobel_y.reshape(1, 1, 3, 3))
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze(1)

        # 边界区域的光流应更平滑
        # 边界处的光流梯度不应过大
        border_mask = (edge_magnitude > edge_magnitude.mean()).float()
        if border_mask.sum() < 10:
            return 0.5  # 没有明显边界，中性 reward

        # 边界处光流变化率
        flow_grad_x = torch.abs(flows[:, :, :, 1:] - flows[:, :, :, :-1]).mean(dim=1)
        flow_grad_y = torch.abs(flows[:, :, 1:, :] - flows[:, :, :-1, :]).mean(dim=1)

        min_h = min(border_mask.shape[1], flow_grad_y.shape[1])
        min_w = min(border_mask.shape[2], flow_grad_x.shape[2])

        border_flow_grad = (
            flow_grad_x[:, :min_h, :min_w] * border_mask[:, :min_h, :min_w]
            + flow_grad_y[:, :min_h, :min_w] * border_mask[:, :min_h, :min_w]
        )
        mean_border_grad = border_flow_grad.sum() / (border_mask[:, :min_h, :min_w].sum() + 1e-6)

        # 实测量级 ~5-50
        reward = torch.exp(-mean_border_grad / 50.0)
        return reward.item()

    def compute_reward(
        self,
        flows: torch.Tensor,
        weights: dict = None,
    ) -> tuple:
        """
        计算流体物理综合 reward

        Args:
            flows: (T-1, 2, H, W) 预先计算好的光流
            weights: 各分项权重

        Returns:
            (reward, scores_dict)
        """
        if weights is None:
            weights = {
                "divergence_free": 0.35,
                "mass_conservation": 0.25,
                "vorticity": 0.25,
                "surface_tension": 0.15,
            }

        scores = {
            "divergence_free": self.divergence_free_reward(flows),
            "mass_conservation": self.mass_conservation_reward(flows),
            "vorticity": self.vorticity_reward(flows),
            "surface_tension": self.surface_tension_reward(flows),
        }

        reward = sum(weights[k] * scores[k] for k in weights)
        return reward, scores
