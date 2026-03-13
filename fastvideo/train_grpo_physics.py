"""
物理 Reward GRPO 训练脚本

Fork 自 DanceGRPO/fastvideo/train_grpo_wan_2_1.py
核心修改:
  1. 替换 HPSv2 reward → PhysicsRewardModel
  2. 支持多帧视频 (--t 33) 而非单帧
  3. 视频解码后直接传给物理 reward，不保存中间 mp4

原始版权: FastVideo Team + ByteDance (Apache 2.0)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from einops import rearrange

from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from diffusers.optimization import get_scheduler
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
from safetensors.torch import save_file
import json

# DanceGRPO utilities — 从 DanceGRPO 目录导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DanceGRPO'))
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.logging_ import main_print

# 物理 Reward
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rewards.physics_reward import PhysicsRewardModel

# Dataset — 复用 DanceGRPO 的 Wan 2.1 RL dataset
from fastvideo.dataset.latent_wan_2_1_rl_datasets import LatentDataset, latent_collate_function


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def flux_step(
    model_output, latents, eta, sigmas, index, prev_sample, grpo, sde_solver,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output
    pred_original_sample = latents - sigma * model_output
    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        log_prob = (
            (-((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2)))
            - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample


def run_sample_step(args, z, progress_bar, sigma_schedule, transformer, encoder_hidden_states, negative_prompt_embeds, grpo_sample):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            transformer.eval()
            if args.cfg_infer > 1:
                with torch.autocast("cuda", torch.bfloat16):
                    pred = transformer(
                        hidden_states=torch.cat([z, z], dim=0),
                        timestep=torch.cat([timesteps, timesteps], dim=0),
                        encoder_hidden_states=torch.cat([encoder_hidden_states, negative_prompt_embeds], dim=0),
                        return_dict=False,
                    )[0]
                    model_pred, uncond_pred = pred.chunk(2)
                    pred = uncond_pred.to(torch.float32) + args.cfg_infer * (model_pred.to(torch.float32) - uncond_pred.to(torch.float32))
            else:
                with torch.autocast("cuda", torch.bfloat16):
                    pred = transformer(
                        hidden_states=z, timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )[0]
            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        return z, latents, all_latents, all_log_probs


def grpo_one_step(args, latents, pre_latents, encoder_hidden_states, negative_prompt_embeds, transformer, timesteps, i, sigma_schedule):
    transformer.train()
    if args.cfg_infer > 1:
        with torch.autocast("cuda", torch.bfloat16):
            pred = transformer(
                hidden_states=torch.cat([latents, latents], dim=0),
                timestep=torch.cat([timesteps, timesteps], dim=0),
                encoder_hidden_states=torch.cat([encoder_hidden_states, negative_prompt_embeds], dim=0),
                return_dict=False,
            )[0]
            model_pred, uncond_pred = pred.chunk(2)
            pred = uncond_pred.to(torch.float32) + args.cfg_infer * (model_pred.to(torch.float32) - uncond_pred.to(torch.float32))
    else:
        with torch.autocast("cuda", torch.bfloat16):
            pred = transformer(
                hidden_states=latents, timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob


def decode_video_tensor(vae, latents):
    """VAE 解码 latents → video tensor (T, C, H, W) in [0, 1]"""
    vae.enable_tiling()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = vae.decode(latents, return_dict=False)[0]  # (B, C, T, H, W)
    # → (T, C, H, W) in [0, 1]
    video = video[0].permute(1, 0, 2, 3).clamp(0, 1).float()  # (T, C, H, W)
    return video


def sample_reference_model(
    args, device, transformer, vae, encoder_hidden_states, negative_prompt_embeds,
    physics_reward_model, caption,
):
    """采样 + 物理 reward 打分"""
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, sample_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_indices = torch.chunk(torch.arange(B), B)

    all_latents = []
    all_log_probs = []
    all_rewards = []

    if args.init_same_noise:
        input_latents = torch.randn(
            (1, IN_CHANNELS, latent_t, latent_h, latent_w),
            device=device, dtype=torch.bfloat16,
        )

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_negative_prompt_embeds = negative_prompt_embeds[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]

        if not args.init_same_noise:
            input_latents = torch.randn(
                (1, IN_CHANNELS, latent_t, latent_h, latent_w),
                device=device, dtype=torch.bfloat16,
            )

        progress_bar = tqdm(range(0, sample_steps), desc=f"Sampling {index}/{B}", disable=int(os.environ.get("LOCAL_RANK", 0)) > 0)
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args, input_latents.clone(), progress_bar, sigma_schedule,
                transformer, batch_encoder_hidden_states, batch_negative_prompt_embeds, True,
            )
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)

        # ===== 物理 Reward (替换 HPSv2) =====
        video_tensor = decode_video_tensor(vae, latents)

        rank = int(os.environ.get("RANK", 0))
        # 可选: 保存视频用于可视化
        if rank == 0 and index < 4:
            video_processor = VideoProcessor(vae_scale_factor=8)
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
                    ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
                    dec_lat = latents / ls + lm
                    vid = vae.decode(dec_lat, return_dict=False)[0]
                    decoded = video_processor.postprocess_video(vid)
            export_to_video(decoded[0], f"./videos/physics_{rank}_{index}.mp4", fps=16)

        # 计算物理 reward
        with torch.no_grad():
            physics_score = physics_reward_model.score_video(video_tensor, batch_caption[0])
            reward = torch.tensor([physics_score], device=device, dtype=torch.float32)
        all_rewards.append(reward)

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)

    return all_rewards, all_latents, all_log_probs, sigma_schedule


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def train_one_step(
    args, device, transformer, vae, physics_reward_model,
    optimizer, lr_scheduler, encoder_hidden_states, negative_prompt_embeds,
    caption, max_grad_norm,
):
    total_loss = 0.0
    optimizer.zero_grad()

    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        negative_prompt_embeds = repeat_tensor(negative_prompt_embeds)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, (list, tuple)):
            caption = [item for item in caption for _ in range(args.num_generations)]

    reward, all_latents, all_log_probs, sigma_schedule = sample_reference_model(
        args, device, transformer, vae, encoder_hidden_states, negative_prompt_embeds,
        physics_reward_model, caption,
    )

    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps = torch.tensor(timestep_values, device=device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[:, :-1][:, :-1],
        "next_latents": all_latents[:, 1:][:, :-1],
        "log_probs": all_log_probs[:, :-1],
        "rewards": reward.to(torch.float32),
        "encoder_hidden_states": encoder_hidden_states,
        "negative_prompt_embeds": negative_prompt_embeds,
    }

    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank() == 0:
        print(f"rewards: {gathered_reward.tolist()}")
        with open('./reward_physics.txt', 'a') as f:
            f.write(f"{gathered_reward.mean().item()}\n")

    # Advantage 计算
    if args.use_group:
        n = len(samples["rewards"]) // args.num_generations
        advantages = torch.zeros_like(samples["rewards"])
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
        samples["advantages"] = advantages

    perms = torch.stack([
        torch.randperm(len(samples["timesteps"][0])) for _ in range(batch_size)
    ]).to(device)

    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][torch.arange(batch_size).to(device)[:, None], perms]

    samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
    samples_batched_list = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

    train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
    for i, sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            new_log_probs = grpo_one_step(
                args, sample["latents"][:, _], sample["next_latents"][:, _],
                sample["encoder_hidden_states"], sample["negative_prompt_embeds"],
                transformer, sample["timesteps"][:, _], perms[i][_], sigma_schedule,
            )

            adv = torch.clamp(sample["advantages"], -args.adv_clip_max, args.adv_clip_max)
            ratio = torch.exp(new_log_probs - sample["log_probs"][:, _])
            unclipped_loss = -adv * ratio
            clipped_loss = -adv * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()

        if (i + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if dist.get_rank() % 8 == 0:
            print(f"reward={sample['rewards'].item():.4f} adv={sample['advantages'].item():.4f} loss={loss.item():.6f}")
        dist.barrier()

    return total_loss, grad_norm.item()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if args.seed is not None:
        set_seed(args.seed + rank)

    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs("videos", exist_ok=True)

    # ===== 物理 Reward Model =====
    main_print("--> Loading PhysicsRewardModel...")
    physics_reward_model = PhysicsRewardModel(device=f"cuda:{local_rank}", mode="auto")
    main_print("--> PhysicsRewardModel loaded")

    # ===== Transformer =====
    main_print(f"--> Loading model from {args.pretrained_model_name_or_path}")
    transformer = WanTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to(device)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, (WanTransformerBlock,), args.selective_checkpointing)

    # ===== VAE =====
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)

    main_print("--> Model loaded")
    transformer.train()

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed)
    train_dataloader = DataLoader(
        train_dataset, sampler=sampler, collate_fn=latent_collate_function,
        pin_memory=True, batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers, drop_last=True,
    )

    if rank <= 0:
        wandb.init(project="physics-grpo", config=vars(args))

    main_print("***** Running Physics GRPO Training *****")
    main_print(f"  Num prompts = {len(train_dataset)}")
    main_print(f"  Video size = {args.h}x{args.w}x{args.t}")
    main_print(f"  Num generations per prompt = {args.num_generations}")
    main_print(f"  Max train steps = {args.max_train_steps}")

    progress_bar = tqdm(range(0, args.max_train_steps), desc="Steps", disable=local_rank > 0)
    step_times = deque(maxlen=100)

    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for step, (prompt_embeds, negative_prompt_embeds, caption) in enumerate(train_dataloader):
            prompt_embeds = prompt_embeds.to(device)
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            start_time = time.time()

            if (step - 1) % args.checkpointing_steps == 0 and step != 1:
                cpu_state = transformer.state_dict()
                if rank <= 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{step}-{epoch}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_file(cpu_state, os.path.join(save_dir, "diffusion_pytorch_model.safetensors"))
                    config_dict = dict(transformer.config)
                    if "dtype" in config_dict:
                        del config_dict["dtype"]
                    with open(os.path.join(save_dir, "config.json"), "w") as f:
                        json.dump(config_dict, f, indent=4)
                main_print(f"--> Checkpoint saved at step {step}")
                dist.barrier()

            if step > args.max_train_steps:
                break

            loss, grad_norm = train_one_step(
                args, device, transformer, vae, physics_reward_model,
                optimizer, lr_scheduler, prompt_embeds, negative_prompt_embeds,
                caption, args.max_grad_norm,
            )

            step_time = time.time() - start_time
            step_times.append(step_time)
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}", "step_time": f"{step_time:.1f}s", "grad_norm": f"{grad_norm:.3f}",
            })
            progress_bar.update(1)

            if rank <= 0:
                wandb.log({
                    "train_loss": loss, "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time, "grad_norm": grad_norm,
                }, step=step)

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_latent_t", type=int, default=1)
    # model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    # diffusion
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--precondition_outputs", action="store_true")
    # training
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=50)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_steps", type=int, default=5)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--use_cpu_offload", action="store_true")
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--train_sp_batch_size", type=int, default=1)
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    # lr scheduler
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--master_weight_type", type=str, default="fp32")
    # GRPO
    parser.add_argument("--h", type=int, default=480)
    parser.add_argument("--w", type=int, default=832)
    parser.add_argument("--t", type=int, default=33)
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--loss_coef", type=float, default=1.0)
    parser.add_argument("--use_group", action="store_true", default=False)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--use_physics_reward", action="store_true", default=False)
    parser.add_argument("--use_hpsv2", action="store_true", default=False)
    parser.add_argument("--ignore_last", action="store_true", default=False)
    parser.add_argument("--init_same_noise", action="store_true", default=False)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--timestep_fraction", type=float, default=0.6)
    parser.add_argument("--clip_range", type=float, default=1e-4)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--cfg_infer", type=float, default=5.0)

    args = parser.parse_args()
    main(args)
