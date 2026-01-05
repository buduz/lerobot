#!/bin/bash

# 设置 CUDA_VISIBLE_DEVICES 环境变量
export CUDA_VISIBLE_DEVICES=3

# 设置采取的策略网络
policy_type="diffusion"

# 设置训练步数
batch_size=64
steps=200_000
save_freq=100_000

# panda_wristcam数据集
repo_id="AllTasks/3-shot"
root_dir="/home/zhiheng/data/lerobot/Few-shot/3-shot-summary"
job_name="${repo_id}_${policy_type}_${steps}_steps_b${batch_size}"
output_dir="/home/zhiheng/project/LAR_baseline/lerobot/outputs/DP/3-shot-modified"

# 运行训练脚本
# Args:
#     wandb.disable_artifact: 是否禁用 WandB 远程存储 checkpoints 功能
#     job_name: 可作为 wandb 的 name 配置(run记录名称)
python -m lerobot.scripts.train \
    --dataset.repo_id="$repo_id" \
    --dataset.root="$root_dir" \
    --policy.type="$policy_type" \
    --policy.push_to_hub=False \
    --num_workers=16 \
    --batch_size=$batch_size \
    --steps=$steps \
    --save_freq=$save_freq \
    --wandb.enable=False \
    --wandb.project="baseline_DP" \
    --wandb.mode="offline" \
    --wandb.disable_artifact=True \
    --job_name="$job_name" \
    --output_dir="$output_dir"
