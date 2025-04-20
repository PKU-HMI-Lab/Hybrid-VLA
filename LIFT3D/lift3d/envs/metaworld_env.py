from typing import Optional

import gymnasium
import numpy as np
import torch
import tqdm
from gymnasium import Wrapper
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import *
from scipy.spatial.transform import Rotation
from termcolor import colored

from lift3d.envs.evaluator import Evaluator
from lift3d.helpers.common import Logger, set_seed
from lift3d.helpers.graphics import PointCloud
from lift3d.helpers.gymnasium import VideoWrapper
from lift3d.helpers.mujoco import camera_name_to_id, generate_point_cloud


def load_mw_policy(task_name):
    if task_name == "peg-insert-side":
        agent = SawyerPegInsertionSideV2Policy()
    else:
        task_name = task_name.split("-")
        task_name = [s.capitalize() for s in task_name]
        task_name = "Sawyer" + "".join(task_name) + "V2Policy"
        agent = eval(task_name)()
    return agent


# see: http://arxiv.org/abs/1910.10897 Appendix A
METAWORLD_LANGUAGE_DESCRIPTION = {
    "assembly": "Pick up a nut and place it onto a peg.",
    "basketball": "Dunk the basketball into the basket.",
    "bin-picking": "Grasp the puck from one bin and place it into another bin.",
    "box-close": "Grasp the cover and close the box with it.",
    "button-press-topdown": "Press a button from the top.",
    "button-press-topdown-wall": "Bypass a wall and press a button from the top.",
    "button-press": "Press a button.",
    "button-press-wall": "Bypass a wall and press a button.",
    "coffee-button": "Push a button on the coffee machine.",
    "coffee-pull": "Pull a mug from a coffee machine.",
    "coffee-push": "Push a mug into a coffee machine.",
    "dial-turn": "Rotate a dial 180 degrees.",
    "disassemble": "pick a nut out of the a peg.",
    "door-close": "Close a door with a revolving joint.",
    "door-lock": "Lock the door by rotating the lock clockwise.",
    "door-open": "Open a door with a revolving joint.",
    "door-unlock": "Unlock the door by rotating the lock counter-clockwise.",
    "drawer-close": "Push and close a drawer.",
    "drawer-open": "Open a drawer.",
    "faucet-close": "Rotate the faucet clockwise.",
    "faucet-open": "Rotate the faucet counter-clockwise.",
    "hammer": "Hammer a screw on the wall.",
    "hand-insert": "Insert the gripper into a hole.",
    "handle-press-side": "Press a handle down sideways.",
    "handle-press": "Press a handle down.",
    "handle-pull-side": "Pull a handle up sideways.",
    "handle-pull": "Pull a handle up.",
    "lever-pull": "Pull a lever down 90 degrees.",
    "peg-insert-side": "Insert a peg sideways.",
    "peg-unplug-side": "Unplug a peg sideways.",
    "pick-out-of-hole": "Pick up a puck from a hole.",
    "pick-place": "Pick and place a puck to a goal.",
    "pick-place-wall": "Pick a puck, bypass a wall and place the puck.",
    "plate-slide-back-side": "Get a plate from the cabinet sideways.",
    "plate-slide-back": "Get a plate from the cabinet.",
    "plate-slide-side": "Slide a plate into a cabinet sideways.",
    "plate-slide": "Slide a plate into a cabinet.",
    "push-back": "Pull a puck to a goal.",
    "push": "Push a puck to a goal.",
    "push-wall": "Bypass a wall and push a puck to a goal.",
    "reach": "reach a goal position.",
    "reach-wall": "Bypass a wall and reach a goal.",
    "shelf-place": "pick and place a puck onto a shelf.",
    "soccer": "Kick a soccer into the goal.",
    "stick-pull": "Grasp a stick and pull a box with the stick.",
    "stick-push": "Grasp a stick and push a box with the stick.",
    "sweep-into": "Sweep a puck into a hole.",
    "sweep": "Sweep a puck off the table.",
    "window-close": "Close a door with a revolving joint.",
    "window-open": "Open a window with a revolving joint.",
}


class MetaWorldEnv(gymnasium.Env):
    TASK_POINT_CLOUD_BOUDNS = {
        "default": [-10, -10, -10, 10, 10, 10],
        # 'corner': [-1.78, 0.62, -20, -0.53, 1.37, 0.60],
        "corner": [-1.66, 0.8, -0.6, -0.48, 1.38, 10],
        "corner2": [-1.97, 0.45, -1.1, -0.73, 1.08, 10],
        "behindGripper": [-10, -0.69, -0.259, 10, 10, 10],
        "basketball": [-2, -2, -2, 2, 2, 2],
    }
    CAMERAS = ("topview", "corner", "corner2", "corner3", "behindGripper", "gripperPOV")

    def __init__(
        self,
        task_name,
        max_episode_length=200,
        image_size=224,
        camera_name="corner",
        use_point_crop=True,
        num_points=1024,
        point_cloud_camera_names=["corner"],
    ):
        super(MetaWorldEnv, self).__init__()

        # task_name
        self.task_name = task_name
        self.env: MujocoEnv = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
            f"{task_name}-v2-goal-observable"
        ]()

        # Language
        self.text = METAWORLD_LANGUAGE_DESCRIPTION[task_name]

        # max_episode_length
        self.max_episode_length = max_episode_length

        # image_size
        self.image_size = image_size
        self.env.mujoco_renderer.width = image_size
        self.env.mujoco_renderer.height = image_size
        self.env.mujoco_renderer.model.vis.global_.offwidth = image_size
        self.env.mujoco_renderer.model.vis.global_.offheight = image_size

        # camera_name
        self.camera_name = camera_name
        self.point_cloud_camera_names = point_cloud_camera_names

        # point cloud support
        x_angle = 61.4
        y_angle = -7
        z_angle = 45  # 绕z轴的旋转角度
        rotation_x = Rotation.from_euler("x", x_angle, degrees=True).as_matrix()
        rotation_y = Rotation.from_euler("y", y_angle, degrees=True).as_matrix()
        rotation_z = Rotation.from_euler("z", z_angle, degrees=True).as_matrix()

        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        self.use_point_crop = use_point_crop
        self.crop_table = False if "top" not in self.camera_name else True
        self.num_points = num_points

        if task_name in MetaWorldEnv.TASK_POINT_CLOUD_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = (
                MetaWorldEnv.TASK_POINT_CLOUD_BOUDNS[task_name]
            )
        elif self.camera_name in MetaWorldEnv.TASK_POINT_CLOUD_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = (
                MetaWorldEnv.TASK_POINT_CLOUD_BOUDNS[self.camera_name]
            )
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = (
                MetaWorldEnv.TASK_POINT_CLOUD_BOUDNS["default"]
            )
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]

        self.pc_transform = {
            "corner": np.array(
                [
                    [-0.66173422, -0.48809537, 0.56909642],
                    [-0.31361979, 0.86966611, 0.38121317],
                    [0.68099225, -0.0737819, 0.7285642],
                ]
            ),
            "corner2": np.array(
                [
                    [0.56914086, -0.56424844, 0.59808225],
                    [0.23069754, 0.80774597, 0.54251738],
                    [-0.78921311, -0.17079271, 0.58989196],
                ]
            ),
        }
        # others
        self.env._freeze_rand_vec = (
            False  # ! Crucial for diversity during data collection
        )

    def get_robot_state(self):
        pos_hand = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env.data.body("rightclaw"),
            self.env.data.body("leftclaw"),
        )
        gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
        return np.append(pos_hand, gripper_distance_apart)

    def get_rgb(self):
        renderer: MujocoRenderer = self.env.mujoco_renderer
        camera_id = camera_name_to_id(renderer.model, self.camera_name)
        viewer = renderer._get_viewer(render_mode="rgb_array")
        image = viewer.render(render_mode="rgb_array", camera_id=camera_id)
        image = np.flip(image, axis=0)
        return image

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = generate_point_cloud(
            self.env.mujoco_renderer, self.point_cloud_camera_names
        )

        # whether to use rgb
        if not use_rgb:
            point_cloud = point_cloud[..., :3]

        # transform point cloud
        point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform[self.camera_name].T

        # crop point cloud
        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]
        # print(f"point_cloud.shape: {point_cloud.shape}")
        point_cloud = PointCloud.point_cloud_sampling(
            point_cloud, self.num_points, "fps"
        )
        # depth = depth[::-1]
        return point_cloud, depth

    def get_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        raw_state = self.env._get_obs()
        point_cloud, depth = self.get_point_cloud()
        obs_dict = {
            "image": obs_pixels,
            "robot_state": robot_state,
            "raw_state": raw_state,
            "point_cloud": point_cloud,
            "depth": depth,
        }
        return obs_dict

    def step(self, action: np.array):
        raw_state, reward, terminated, truncated, env_info = self.env.step(action)
        self.cur_step += 1
        obs_dict = self.get_obs()
        truncated = truncated or self.cur_step >= self.max_episode_length
        env_info["gripper_proprio"] = obs_dict["raw_state"][:4]
        return obs_dict, reward, terminated, truncated, env_info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        raw_obs = self.env.reset(seed=seed, options=options)
        self.cur_step = 0
        obs_dict = self.get_obs()
        return obs_dict


class MetaWorldEvaluator(Evaluator):
    def __init__(
        self,
        task_name,
        max_episode_length=200,
        image_size=224,
        camera_name="corner",
        use_point_crop=True,
        num_points=1024,
        point_cloud_camera_names=["corner"],
    ):
        self.env = MetaWorldEnv(
            task_name=task_name,
            max_episode_length=max_episode_length,
            image_size=image_size,
            camera_name=camera_name,
            use_point_crop=use_point_crop,
            num_points=num_points,
            point_cloud_camera_names=point_cloud_camera_names,
        )
        self.env = VideoWrapper(self.env)

    def evaluate(self, num_episodes, policy, verbose: bool = False):
        task_name = Wrapper.get_wrapper_attr(self.env, "task_name")

        if verbose:
            success_list, rewards_list = [], []
            video_steps_list = []
        else:
            total_success, total_rewards = 0, 0

        for i in tqdm.tqdm(
            range(num_episodes),
            desc=f'Evaluating in MetaWorld <{colored(task_name, "red")}>',
        ):
            raw_state = self.env.reset()["raw_state"]
            obs_dict = Wrapper.get_wrapper_attr(self.env, "get_obs")()
            truncated = terminated = False
            rewards = 0
            success = False
            # ! Metaworld environment will never terminates, it only truncates.
            while not truncated and not terminated:
                obs_img = obs_dict["image"]
                obs_point_cloud = obs_dict["point_cloud"]
                obs_robot_state = obs_dict["robot_state"]
                device = next(policy.parameters()).device
                obs_img_tensor = (
                    torch.from_numpy(obs_img).float().unsqueeze(0).to(device)
                )
                obs_point_cloud_tensor = (
                    torch.from_numpy(obs_point_cloud).float().unsqueeze(0).to(device)
                )
                obs_robot_state_tensor = (
                    torch.from_numpy(obs_robot_state).float().unsqueeze(0).to(device)
                )
                obs_img_tensor = obs_img_tensor.permute(0, 3, 1, 2)
                batch_size = obs_img_tensor.shape[0]
                input_data = {
                    "images": obs_img_tensor,
                    "point_clouds": obs_point_cloud_tensor,
                    "robot_states": obs_robot_state_tensor,
                    "texts": [self.env.text] * batch_size,
                }
                with torch.no_grad():
                    action = policy(**input_data)
                action = action.to("cpu").detach().numpy().squeeze()
                obs_dict, reward, terminated, truncated, info = self.env.step(action)
                rewards += reward
                success = success or info["success"]

            if verbose:
                video_steps_list.append(self.env.get_frames().transpose(0, 3, 1, 2))
                success_list.append(success)
                rewards_list.append(rewards)
            else:
                total_success += success
                total_rewards += rewards

        if verbose:
            return_value = (
                sum(success_list) / num_episodes,
                sum(rewards_list) / num_episodes,
            )
            self.success_list = success_list
            self.rewards_list = rewards_list
            self.video_steps_list = video_steps_list
        else:
            avg_success = total_success / num_episodes
            avg_rewards = total_rewards / num_episodes
            return_value = avg_success, avg_rewards

        return return_value

    def callback_verbose(self, wandb_logger):
        import plotly.express as px
        import plotly.graph_objects as go
        import wandb

        fig1 = go.Figure(
            data=[
                go.Bar(
                    x=["Success", "Failure"],
                    y=[
                        sum(self.success_list),
                        len(self.success_list) - sum(self.success_list),
                    ],
                )
            ]
        )
        fig2 = px.box(self.rewards_list, title="Rewards distribution")
        wandb_logger.log({"Charts/success_failure": fig1})
        wandb_logger.log({"Charts/rewards_distribution": fig2})

        for i, (success, rewards, video_steps) in enumerate(
            zip(self.success_list, self.rewards_list, self.video_steps_list)
        ):
            if success:
                wandb_logger.log(
                    {
                        f"validation/video_steps_success_{i}": wandb.Video(
                            video_steps, fps=30
                        ),
                    }
                )
            else:
                wandb_logger.log(
                    {
                        f"validation/video_steps_failure_{i}": wandb.Video(
                            video_steps, fps=30
                        ),
                    }
                )


if __name__ == "__main__":
    set_seed(0)
    env = MetaWorldEnv(
        task_name="assembly",
        max_episode_length=500,
        image_size=480,
        camera_name="corner",
        use_point_crop=True,
        num_points=1024,
    )
    obs_dict = env.reset()
    first_obs = obs_dict["raw_state"][-3:]
    Logger.log_info(f"First observation: {first_obs}")

    obs_dict = env.reset()
    second_obs = obs_dict["raw_state"][-3:]
    Logger.log_info(f"Second observation: {second_obs}")
