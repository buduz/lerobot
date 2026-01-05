import dataclasses
import logging
import os
import json
import torch
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import imageio
import numpy as np
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from service import ExternalRobotInferenceClient
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# import matplotlib.pyplot as plt

# def plot_prediction_vs_groundtruth(gt: np.ndarray, pred: np.ndarray, save_path: str):
#     """
#     将真实值和推理值随时间变化的关系画图并保存。
    
#     参数:
#         gt (np.ndarray): shape 为 (n, d) 的真实值
#         pred (np.ndarray): shape 为 (n, d) 的推理值
#         save_path (str): 图像保存路径，如 'output/pred_vs_gt.png'
#     """
#     assert gt.shape == pred.shape, "gt 和 pred 的 shape 必须相同"
#     n, d = gt.shape

#     # 创建输出目录（如有必要）
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     # 创建图像
#     fig, axes = plt.subplots(d, 1, figsize=(10, 2 * d), sharex=True)

#     for i in range(d):
#         axes[i].plot(range(n), gt[:, i], label='Ground Truth', color='blue', linestyle='-')
#         axes[i].plot(range(n), pred[:, i], label='Prediction', color='red', linestyle='--')
#         axes[i].set_ylabel(f'Dim {i+1}')
#         axes[i].legend(loc='upper right')
#         axes[i].grid(True)

#     axes[-1].set_xlabel('Time Step')

#     plt.suptitle('Ground Truth vs Prediction Over Time (All Dimensions)')
#     plt.tight_layout(rect=[0, 0, 1, 0.96])

#     # 保存图像
#     plt.savefig(save_path, dpi=300)
#     plt.close()

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 5555
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # Dataset parameters
    #################################################################################################################
    dataset_path: str = "/home/zhiheng/data/lerobot/Few-shot/50-shot-summary"

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)


def eval_dp(args: Args) -> None:
    device = "cuda"
    # Set random seed
    np.random.seed(args.seed)
    # ------------ step 1: load dataset ------------
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.images.image": [-0.1, 0.0],
        "observation.images.wrist_image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(args.dataset_path, delta_timestamps=delta_timestamps)
    item_idx = random.randint(0, len(dataset) - 1)
    data = dataset[item_idx]
    policy_client = ExternalRobotInferenceClient(host=args.host, port=args.port)

    state = data["observation.state"][1]       # tensor, /255
    image = data['observation.images.image'][1] 
    wrist_image = data['observation.images.wrist_image'][1] 
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)
    wrist_image = wrist_image.to(device, non_blocking=True)
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)
    wrist_image = wrist_image.unsqueeze(0)

    element = {
            "observation.images.image": image.detach().cpu().numpy(),
            "observation.images.wrist_image": wrist_image.detach().cpu().numpy(),
            "observation.state": state.detach().cpu().numpy(),
    }
    element = {k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v) for k, v in element.items()}
    action_chunk = policy_client.get_action(element)
    # plot_prediction_vs_groundtruth(data['action.actions'], action_chunk, save_path="/home/zhiheng/project/Isaac-GR00T/compare.png")
    print(action_chunk)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_dp)