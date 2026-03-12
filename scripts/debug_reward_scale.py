"""快速检查光流值的量级，用于校准 reward 归一化参数"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchvision.io as io
from rewards.flow_reward import OpticalFlowReward

device = "cuda:0"
reward = OpticalFlowReward(device=device)

# 加载第一个视频 (倒水)
video, _, _ = io.read_video("outputs/physics_baseline/rank0-0-0_lora.mp4", pts_unit="sec")
indices = torch.linspace(0, video.shape[0]-1, 32).long()
video = video[indices].permute(0,3,1,2).float() / 255.0

flows = reward.extract_flows(video)
print(f"flows shape: {flows.shape}")
print(f"flows range: [{flows.min():.2f}, {flows.max():.2f}]")
print(f"flows abs mean: {flows.abs().mean():.4f}")
print(f"flows magnitude mean: {torch.sqrt((flows**2).sum(dim=1)).mean():.4f}")

# 各分项的原始值 (未归一化)
diff = flows[1:] - flows[:-1]
acceleration = (diff**2).mean()
print(f"\nacceleration (temporal): {acceleration:.4f}")

dx = flows[:,:,:,1:] - flows[:,:,:,:-1]
dy = flows[:,:,1:,:] - flows[:,:,:-1,:]
smoothness = (dx**2).mean() + (dy**2).mean()
print(f"smoothness (spatial): {smoothness:.4f}")

mag = torch.sqrt((flows**2).sum(dim=1)).mean()
print(f"mean magnitude (motion): {mag:.4f}")

# 流体分项
vx, vy = flows[:,0], flows[:,1]
dvx_dx = (vx[:,:,2:] - vx[:,:,:-2]) / 2.0
dvy_dy = (vy[:,2:,:] - vy[:,:-2,:]) / 2.0
min_h = min(dvx_dx.shape[1], dvy_dy.shape[1])
min_w = min(dvx_dx.shape[2], dvy_dy.shape[2])
div = dvx_dx[:,:min_h,:min_w] + dvy_dy[:,:min_h,:min_w]
print(f"divergence mean sq: {(div**2).mean():.4f}")
