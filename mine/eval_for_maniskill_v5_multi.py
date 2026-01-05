"""Validation of maniskill for different tasks

"""
import sys
# sys.path.append("/home/zetyun/ManiSkill-all") # 选择环境环境
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional
import gymnasium as gym
import numpy as np
import torch
import tyro
import math
import json
import os
from tqdm import tqdm

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import mani_skill.examples.benchmarking.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper # import benchmark env code
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from service import ExternalRobotInferenceClient
# from web_socket import WebSocketInferenceClient

EMBODIMENT_TAGS = {
    "panda_wristcam": "panda",
    "xarm6_robotiq_wristcam": "xarm6",
    "xarm7_robotiq_wristcam": "xarm7",
    "widowxai_wristcam": "widowxai"
}
# TASKS = {
#     "panda_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PullCubeTool-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     "xarm6_robotiq_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PullCubeTool-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     "xarm7_robotiq_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PullCubeTool-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     "widowxai_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PullCubeTool-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
# }
# TASKS = {
#     "panda_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     "xarm6_robotiq_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     # "xarm7_robotiq_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     # "widowxai_wristcam": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
# }

# TASKS = {
#     "panda_wristcam_rel": ["PickCube-v1", "PushCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     "xarm6_robotiq_wristcam_rel": ["PickCube-v1", "PushCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     # "xarm7_robotiq_wristcam_rel": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
#     # "widowxai_wristcam_rel": ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PlaceSphere-v1", "LiftPegUpright-v1"],
# }
TASKS = {
    "panda_wristcam": ["PushCube-v1", "PlaceSphere-v1", "PullCube-v1", "LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"],
    "xarm6_robotiq_wristcam": ["PushCube-v1", "PlaceSphere-v1", "PullCube-v1", "LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"],
    "widowxai_wristcam": ["PushCube-v1", "PlaceSphere-v1", "PullCube-v1", "LiftPegUpright-v1", "PickCube-v1", "StackCube-v1"],
}

TASK_INSTRUCTIONS = {
    "PickCube-v1": "Pick up the cube.",
    "PullCube-v1": "Pull the cube to the target position.",
    "PushCube-v1": "Push the cube to the target position.",
    "StackCube-v1": "Stack the cube on top of the other cube.",
    "PullCubeTool-v1": "Pick up the cube tool and use it to bring the cube closer.",
    "PlaceSphere-v1": "Pick up the ball and place it in the target position.",
    "LiftPegUpright-v1": "Pick up the peg and place it upright.",
}
STEP_LENGTHS = {
    "PickCube-v1": 500,
    "PullCube-v1": 500,
    "PushCube-v1": 500,
    "StackCube-v1": 800,
    "PullCubeTool-v1": 800,
    "PlaceSphere-v1": 500,
    "LiftPegUpright-v1": 700,
}

generalized_tasks = {
      "panda_wristcam": ["PushCube-v1", "PlaceSphere-v1", ],
      "xarm6_robotiq_wristcam": ["PullCube-v1", "LiftPegUpright-v1"],
      "widowxai_wristcam": ["PickCube-v1", "StackCube-v1"],
}
device = "cuda"
# BENCHMARK_ENVS = ["PickCube-v1"]
@dataclass
class EvalConfig:
    """Configuration for evaluation

    Args:
   """
    host: str = "0.0.0.0"
    port: int = 5550
    url: Optional[str] = None
    resize_size: int = 224
    replan_steps: int = 1 # 取默认的1，lerobot的dp在policy端维护action队列
    # env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = BENCHMARK_ENVS[INDEX]
    """Environment ID"""
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_ee_delta_pose"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    cpu_sim: bool = True
    """Whether to use the CPU or GPU simulation"""
    seed: int = 0
    save_example_image: bool = False
    control_freq: Optional[int] = 60
    sim_freq: Optional[int] = 120
    num_cams: Optional[int] = None
    """Number of cameras. Only used by benchmark environments"""
    cam_width: Optional[int] = None
    """Width of cameras. Only used by benchmark environments"""
    cam_height: Optional[int] = None
    """Height of cameras. Only used by benchmark environments"""
    render_mode: str = "rgb_array"
    """Which set of cameras/sensors to render for video saving. 'cameras' value will save a video showing all sensor/camera data in the observation, e.g. rgb and depth. 'rgb_array' value will show a higher quality render of the environment running."""
    save_video: bool = False
    """Whether to save videos"""
    save_results: Optional[str] = None
    """Path to save results to. Should be path/to/results.csv"""
    save_path: str = '/home/zhiheng/project/LAR_baseline/eval_result/DP/10-shot/10w'        # /home/pc/maniskill/videos/ms3_benchmark_qwen_dit
    shader: str = "default"
    num_per_task: int = 50

def main(args: EvalConfig):
    os.makedirs(args.save_path, exist_ok=True)
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq
    
    if args.url:
        policy_client = WebSocketInferenceClient(url=args.url)
    else:
        policy_client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    
    kwargs = dict()

    for robot_uids, tasks in TASKS.items():
        total_successes = 0.0
        success_dict = {}
        for env_id in tasks:
            if not args.cpu_sim:
                env = gym.make(
                    env_id,
                    num_envs=num_envs,
                    obs_mode=args.obs_mode,
                    robot_uids=robot_uids,     
                    sensor_configs=dict(shader_pack=args.shader),
                    human_render_camera_configs=dict(shader_pack=args.shader),
                    viewer_camera_configs=dict(shader_pack=args.shader),
                    render_mode=args.render_mode,
                    control_mode=args.control_mode,
                    sim_config=sim_config,
                    **kwargs
                )
                if isinstance(env.action_space, gym.spaces.Dict):
                    env = FlattenActionSpaceWrapper(env)
                base_env: BaseEnv = env.unwrapped
            else:
                def make_env():
                    def _init():
                        env = gym.make(env_id,
                                    obs_mode=args.obs_mode,
                                    sim_config=sim_config,
                                    robot_uids=robot_uids,
                                    sensor_configs=dict(shader_pack=args.shader),
                                    human_render_camera_configs=dict(shader_pack=args.shader),
                                    viewer_camera_configs=dict(shader_pack=args.shader),
                                    render_mode=args.render_mode,
                                    control_mode=args.control_mode,
                                    **kwargs)
                        env = CPUGymWrapper(env, )
                        return env
                    return _init
                # mac os system does not work with forkserver when using visual observations
                env = AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver" if sys.platform == "darwin" else None) if args.num_envs > 1 else make_env()()
                base_env = make_env()().unwrapped

            base_env.print_sim_details()
            
            task_successes = 0.0
            for seed in tqdm(range(args.num_per_task)):
                images = []
                video_nrows = int(np.sqrt(num_envs))
                with torch.inference_mode():
                    env.reset(seed=seed+2025)
                    env.step(env.action_space.sample())  # warmup step
                    obs, info = env.reset(seed=seed+2025)
                    if args.save_video:
                        images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())
                        # images.append(obs["sensor_data"]["third_view_camera"]["rgb"].cpu().numpy())
                    task_description = TASK_INSTRUCTIONS[env_id]
                    step_length = STEP_LENGTHS[env_id]
                    N = step_length // args.replan_steps
                    # N = 100
                    with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
                        for i in range(N):         
                            # print(i)   
                            state = np.concatenate(
                                        (
                                            obs["extra"]["tcp_pose"],
                                            obs["agent"]["qpos"][-1:],
                                        )
                                    )
                            img = torch.from_numpy(obs["sensor_data"]["third_view_camera"]["rgb"]).type(torch.float32) / 255
                            img = img.permute(2, 0, 1).contiguous()
                            wrist_img = torch.from_numpy(obs["sensor_data"]["hand_camera"]["rgb"]).type(torch.float32) / 255
                            wrist_img = wrist_img.permute(2, 0, 1).contiguous()
                            state = torch.from_numpy(state).type(torch.float32)

                            element = {
                                    "observation.images.image": img.unsqueeze(0),
                                    "observation.images.wrist_image": wrist_img.unsqueeze(0),
                                    "observation.state": state.unsqueeze(0),
                            }
                            element = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in element.items()}

                            action = policy_client.get_action(element).cpu().numpy().squeeze(0)
                            if 'widowxai' in robot_uids:
                                pred_action = np.concatenate([action, action[-1:]])
                            else:
                                pred_action = action
                            # pred_action[pred_action[:, -1] == 0, -1] = -1
                            obs, rew, terminated, truncated, info = env.step(pred_action)
                            if args.save_video:
                                images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())
                                # images.append(obs["sensor_data"]["third_view_camera"]["rgb"].cpu().numpy())
                            terminated = terminated if args.cpu_sim else terminated.item()
                            if terminated:
                                task_successes += 1
                                total_successes += 1
                                break
                            if terminated:
                                break
                    profiler.log_stats("env.step")

                    if args.save_video:
                        images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
                        images_to_video(
                            images,
                            output_dir=args.save_path,
                            video_name=f"{robot_uids}-{env_id}-{seed}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}--success={terminated}",
                            fps=30,
                        )
                        del images
            env.close()
            print(f"Task Success Rate: {task_successes / args.num_per_task}")
            success_dict[env_id] = task_successes / args.num_per_task
        print(f"Total Success Rate: {total_successes / (args.num_per_task * len(tasks))}")
        success_dict['total_success'] = total_successes / (args.num_per_task * len(tasks))
        with open(f"{args.save_path}/{robot_uids}_success_dict.json", "w") as f:
            json.dump(success_dict, f)
    
    def load_json(file_path):
        """读取 JSON 文件并返回 Python 对象"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(data, file_path):
        """将 Python 字典保存为 JSON 文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # 计算跨本体平均成功率
    json_files = [
        fn for fn in os.listdir(args.save_path)
        if fn.endswith(".json") and fn != "aver.json"
    ]

    total_success_list = []
    for fn in json_files:
        fp = os.path.join(args.save_path, fn)
        data = load_json(fp)
        total_success_list.append(data["total_success"])

    aver_success = sum(total_success_list) / len(total_success_list)

    out_path = os.path.join(args.save_path, "aver.json")
    save_json({"aver_success": aver_success}, out_path)

    # 计算泛化任务的平均成功率
    success_dict = {}
    success_list = []
    for robot_uids, tasks in generalized_tasks.items():
        for fn in json_files:
            if not fn.startswith(robot_uids):
                continue
            fp = os.path.join(args.save_path, fn)
            data = load_json(fp)
            task_successes = []
            for task in tasks:
                task_successes.append(data[task])
            success_dict[robot_uids] = sum(task_successes) / len(task_successes)
            success_list.append(sum(task_successes) / len(task_successes))
        
    aver_success = sum(success_list) / len(success_list)
    success_dict['generalized_aver_success'] = aver_success
    out_path = os.path.join(args.save_path, f"generalized_aver.json")
    save_json(success_dict, out_path)

if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
