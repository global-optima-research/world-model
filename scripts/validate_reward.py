"""
验证物理 Reward 信号质量

用 baseline 生成的 20 个物理视频测试:
1. Reward 信号是否能区分不同物理场景
2. 流体专项 reward 是否对流体场景给出更高分
3. 各分项 reward 的分布和区分度

这是 Phase 1 的关键决策点: reward 信号必须有意义才值得继续 GRPO 训练。

用法: python scripts/validate_reward.py --video_dir outputs/physics_baseline
"""

import argparse
import os
import sys
import torch
import torchvision.io as io

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.physics_reward import PhysicsRewardModel


PROMPT_MAP = {
    0: ("fluid", "A glass of water being slowly poured from a pitcher"),
    1: ("fluid", "Thick smoke rising from a candle flame, swirling and dispersing"),
    2: ("fluid", "A waterfall cascading down rocks into a pool"),
    3: ("fluid", "A cup of coffee being stirred with a spoon"),
    4: ("fluid", "Rain drops falling into a puddle, creating expanding circular ripples"),
    5: ("softbody", "A soft jelly cube wobbling on a white plate after being gently poked"),
    6: ("softbody", "A rubber ball dropped onto a hard floor, bouncing and deforming"),
    7: ("softbody", "A balloon filled with water being squeezed by a hand"),
    8: ("softbody", "A piece of soft clay being pressed flat by a hand"),
    9: ("softbody", "An elastic rubber band being stretched and released"),
    10: ("cloth", "A red flag waving in strong wind on a flagpole"),
    11: ("cloth", "A white tablecloth being pulled off a table in slow motion"),
    12: ("cloth", "A silk curtain blowing gently in a breeze"),
    13: ("cloth", "A bedsheet being shaken out and floating down onto a bed"),
    14: ("cloth", "A piece of paper falling through the air, fluttering"),
    15: ("rigid", "A bowling ball rolling down a lane and striking the pins"),
    16: ("rigid", "A pendulum swinging back and forth in a grandfather clock"),
    17: ("rigid", "Two billiard balls colliding on a green pool table"),
    18: ("rigid", "A basketball bouncing on a wooden court floor"),
    19: ("rigid", "A stack of wooden blocks being knocked over by a marble"),
}


def load_video(path: str, max_frames: int = 32) -> torch.Tensor:
    """加载视频，均匀采样 max_frames 帧"""
    video, _, info = io.read_video(path, pts_unit="sec")
    # video: (T, H, W, C) uint8
    T = video.shape[0]
    if T > max_frames:
        indices = torch.linspace(0, T - 1, max_frames).long()
        video = video[indices]
    # → (T, C, H, W) float [0, 1]
    video = video.permute(0, 3, 1, 2).float() / 255.0
    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="outputs/physics_baseline")
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("物理 Reward 信号验证")
    print("=" * 60)

    # 初始化 reward model
    print(f"\n初始化 PhysicsRewardModel (device={args.device})...")
    reward_model = PhysicsRewardModel(device=args.device, mode="auto")
    print("完成\n")

    results = {}
    category_scores = {"fluid": [], "softbody": [], "cloth": [], "rigid": []}

    for idx in range(20):
        video_path = os.path.join(args.video_dir, f"rank0-{idx}-0_lora.mp4")
        if not os.path.exists(video_path):
            print(f"  跳过 {video_path} (不存在)")
            continue

        category, prompt = PROMPT_MAP[idx]

        print(f"[{idx:2d}] {category:8s} | {prompt[:50]}...")
        video = load_video(video_path, max_frames=args.max_frames)

        score, details = reward_model.score_video(video, prompt, return_details=True)
        results[idx] = {"score": score, "category": category, "details": details}
        category_scores[category].append(score)

        # 打印分项
        mode = details.get("mode", "unknown")
        flow_d = details.get("flow", {})
        print(f"         总分: {score:.4f} (mode={mode})")
        print(f"         光流: temporal={flow_d.get('temporal', 0):.3f} "
              f"spatial={flow_d.get('spatial', 0):.3f} "
              f"motion={flow_d.get('motion', 0):.3f} "
              f"gravity={flow_d.get('gravity', 0):.3f}")
        if "fluid" in details:
            fd = details["fluid"]
            print(f"         流体: div_free={fd.get('divergence_free', 0):.3f} "
                  f"mass={fd.get('mass_conservation', 0):.3f} "
                  f"vorticity={fd.get('vorticity', 0):.3f} "
                  f"surface={fd.get('surface_tension', 0):.3f}")
        print()

    # 汇总
    print("=" * 60)
    print("各类别平均分")
    print("=" * 60)
    for cat in ["fluid", "softbody", "cloth", "rigid"]:
        scores = category_scores[cat]
        if scores:
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
            print(f"  {cat:10s}: {mean:.4f} ± {std:.4f}  (n={len(scores)})")
        else:
            print(f"  {cat:10s}: 无数据")

    # 区分度分析
    print("\n" + "=" * 60)
    print("区分度分析")
    print("=" * 60)
    all_scores = [results[i]["score"] for i in results]
    if all_scores:
        print(f"  全局: mean={sum(all_scores)/len(all_scores):.4f}, "
              f"min={min(all_scores):.4f}, max={max(all_scores):.4f}, "
              f"range={max(all_scores)-min(all_scores):.4f}")

    fluid_mean = sum(category_scores["fluid"]) / max(len(category_scores["fluid"]), 1)
    rigid_mean = sum(category_scores["rigid"]) / max(len(category_scores["rigid"]), 1)
    print(f"  流体 vs 刚体 差异: {fluid_mean - rigid_mean:+.4f}")
    print(f"  (正值说明流体 reward 对流体场景有更高评分)")

    print("\n结论:")
    score_range = max(all_scores) - min(all_scores) if all_scores else 0
    if score_range > 0.1:
        print("  ✅ Reward 信号有足够区分度，可以用于 GRPO 训练")
    elif score_range > 0.05:
        print("  ⚠️ Reward 信号区分度一般，可能需要调整权重或增加分项")
    else:
        print("  ❌ Reward 信号区分度不足，需要重新设计 reward 函数")


if __name__ == "__main__":
    main()
