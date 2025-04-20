import os
import pathlib
import sys
import traceback
from typing import List, Optional, Tuple, Union

import gymnasium
import numpy as np
import rlbench  # Not explicitly used but for registering the environment
import torch
import tqdm
import wandb
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench import Environment
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation
from rlbench.gym import RLBenchEnv as RLBenchEnv_Original
from rlbench.observation_config import ObservationConfig
from rlbench.task_environment import TaskEnvironment
from rlbench.utils import name_to_task_class
from scipy.spatial.transform import Rotation
from termcolor import colored

from lift3d.envs.evaluator import Evaluator
from lift3d.helpers.common import Logger
from lift3d.helpers.graphics import PointCloud, Quaternion
from lift3d.helpers.gymnasium import VideoWrapper


class RLBenchActionMode(object):
    @staticmethod
    def eepose_then_gripper_action_mode(absolute):
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=absolute),
            gripper_action_mode=Discrete(),
        )
        return action_mode

    def __new__(cls):
        if cls is ActionMode:
            raise TypeError("ActionMode cannot be instantiated directly")


class RLBenchObservationConfig(object):
    @staticmethod
    def single_view_config(
        camera_name: str, image_size: Tuple[int, int] | int
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)
        if camera_name == "wrist":
            obs_config.wrist_camera.set_all(True)
            obs_config.wrist_camera.image_size = image_size
        elif camera_name == "front":
            obs_config.front_camera.set_all(True)
            obs_config.front_camera.image_size = image_size
        elif camera_name == "left_shoulder":
            obs_config.left_shoulder_camera.set_all(True)
            obs_config.left_shoulder_camera.image_size = image_size
        elif camera_name == "right_shoulder":
            obs_config.right_shoulder_camera.set_all(True)
            obs_config.right_shoulder_camera.image_size = image_size
        elif camera_name == "overhead":
            obs_config.overhead_camera.set_all(True)
            obs_config.overhead_camera.image_size = image_size
        else:
            raise ValueError(f"Invalid view name: {camera_name}")
        return obs_config

    @staticmethod
    def multi_view_config(camera_names: List[str], image_size: Tuple[int, int] | int):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)
        for camera_name in camera_names:
            if camera_name == "wrist":
                obs_config.wrist_camera.set_all(True)
                obs_config.wrist_camera.image_size = image_size
            elif camera_name == "front":
                obs_config.front_camera.set_all(True)
                obs_config.front_camera.image_size = image_size
            elif camera_name == "left_shoulder":
                obs_config.left_shoulder_camera.set_all(True)
                obs_config.left_shoulder_camera.image_size = image_size
            elif camera_name == "right_shoulder":
                obs_config.right_shoulder_camera.set_all(True)
                obs_config.right_shoulder_camera.image_size = image_size
            elif camera_name == "overhead":
                obs_config.overhead_camera.set_all(True)
                obs_config.overhead_camera.image_size = image_size
            else:
                raise ValueError(f"Invalid view name: {camera_name}")
        return obs_config

    def __new__(cls):
        if cls is ObservationConfig:
            raise TypeError("ObservationConfig cannot be instantiated directly")


class RLBenchEnv(gymnasium.Env):

    POINT_CLOUD_BOUNDS = {"default": [-1.5, -1.5, 0.76, 1.5, 1.5, 3.0]}

    def __init__(
        self,
        task_name,
        action_mode: Optional[ActionMode] = None,
        obs_config: ObservationConfig = ObservationConfig(),
        camera_name: str = "front",
        point_cloud_camera_names: List[int] = ["front"],
        use_point_crop=True,
        num_points: int = 1024,
        max_episode_length: int = 200,
        headless: bool = True,
        verbose_warnings: bool = False,
        cinematic_record_enabled: bool = False,
        cinematic_cam_resolution: Tuple[int, int] = (1280, 720),
        cinematic_rotate_speed: float = 0.005,
        cinematic_cam_fps: int = 30,
    ) -> None:
        super(RLBenchEnv, self).__init__()

        self.task_name = task_name
        self.env = RLBenchEnv_Original(
            task_class=name_to_task_class(task_name),
            action_mode=action_mode,
            obs_config=obs_config,
            headless=headless,
        )
        self.camera_name = camera_name
        self.max_episode_length = max_episode_length

        # Language
        self.text = max(self.env.rlbench_task_env._task.init_episode(index=0), key=len)

        # Point cloud
        self.point_cloud_camera_names = point_cloud_camera_names
        self.use_point_crop = use_point_crop
        self.num_points = num_points
        x_min, y_min, z_min, x_max, y_max, z_max = self.POINT_CLOUD_BOUNDS["default"]
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]

        # Error handling
        self.error_type_counts = {
            "IKError": 0,
            "ConfigurationPathError": 0,
            "InvalidActionError": 0,
        }
        self.verbose_warnings = verbose_warnings

        # Cinematic recording
        self.cinematic_record_enabled = cinematic_record_enabled
        self.cinematic_flag = False
        self.cinematic_cam_resolution = cinematic_cam_resolution
        self.cinematic_rotate_speed = cinematic_rotate_speed
        self.cinematic_cam_fps = cinematic_cam_fps
        cam_placeholder = Dummy("cam_cinematic_placeholder")
        cam = VisionSensor.create(self.cinematic_cam_resolution)
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)
        cam_motion = CircleCameraMotion(
            cam, Dummy("cam_cinematic_base"), self.cinematic_rotate_speed
        )
        self.tr = TaskRecorder(self.env, cam_motion, fps=self.cinematic_cam_fps)
        if cinematic_record_enabled:
            self.set_cinematic_record(True)

    def set_cinematic_record(self, value: bool):
        if self.cinematic_flag == value:
            return

        action_mode = self.env.action_mode

        if isinstance(action_mode, MoveArmThenGripper):
            arm_action_mode: Union[EndEffectorPoseViaIK, EndEffectorPoseViaPlanning] = (
                self.env.action_mode.arm_action_mode
            )
            if not isinstance(
                arm_action_mode, EndEffectorPoseViaPlanning
            ) and not isinstance(arm_action_mode, EndEffectorPoseViaIK):
                raise ValueError(
                    "Only [EndEffectorPoseViaPlanning/EndEffectorPoseViaIK] are supported for cinematic recording"
                )
            if value:
                arm_action_mode.register_callback(self.tr.take_snap)
            else:
                arm_action_mode.deregister_callback(self.tr.take_snap)
        else:
            raise ValueError(
                "Only BaseArmThenGripper is supported for cinematic recording"
            )

        self.cinematic_flag = value

    def get_rlbench_env_obs(self):
        obs: Observation = self.env.rlbench_task_env.get_observation()
        return obs

    def get_robot_state(self):
        obs: Observation = self.get_rlbench_env_obs()
        arm_joint_state = obs.joint_positions
        arm_pose_state = obs.gripper_pose
        gripper_state = obs.gripper_open
        robot_state = np.concatenate(
            (arm_joint_state, arm_pose_state, np.array([gripper_state]))
        )

        return robot_state

    def get_rgb(self):
        if self.camera_name == "default":
            frame = self.env.render()
        else:
            obs: Observation = self.get_rlbench_env_obs()
            if self.camera_name == "wrist":
                frame = obs.wrist_rgb
            elif self.camera_name == "front":
                frame = obs.front_rgb
            elif self.camera_name == "left_shoulder":
                frame = obs.left_shoulder_rgb
            elif self.camera_name == "right_shoulder":
                frame = obs.right_shoulder_rgb
            elif self.camera_name == "overhead":
                frame = obs.overhead_rgb
            else:
                raise ValueError(f"Invalid camera name: {self.camera_name}")
        return frame

    def get_depth(self):
        if self.camera_name == "default":
            depth = self.env.render(mode="depth")
        else:
            obs: Observation = self.get_rlbench_env_obs()
            if self.camera_name == "wrist":
                depth = obs.wrist_depth
            elif self.camera_name == "front":
                depth = obs.front_depth
            elif self.camera_name == "left_shoulder":
                depth = obs.left_shoulder_depth
            elif self.camera_name == "right_shoulder":
                depth = obs.right_shoulder_depth
            elif self.camera_name == "overhead":
                depth = obs.overhead_depth
            else:
                raise ValueError(f"Invalid camera name: {self.camera_name}")
        return depth

    def get_point_cloud_single_view(self, camera_name: str):
        obs: Observation = self.get_rlbench_env_obs()
        if camera_name == "wrist":
            point_cloud = obs.wrist_point_cloud
            image = obs.wrist_rgb
        elif camera_name == "front":
            point_cloud = obs.front_point_cloud
            image = obs.front_rgb
        elif camera_name == "left_shoulder":
            point_cloud = obs.left_shoulder_point_cloud
            image = obs.left_shoulder_rgb
        elif camera_name == "right_shoulder":
            point_cloud = obs.right_shoulder_point_cloud
            image = obs.right_shoulder_rgb
        elif camera_name == "overhead":
            point_cloud = obs.overhead_point_cloud
            image = obs.overhead_rgb
        else:
            raise ValueError(f"Invalid camera name: {camera_name}")
        point_cloud = np.concatenate((point_cloud, image), axis=-1)
        point_cloud = point_cloud.reshape(-1, 6)
        return point_cloud

    def get_point_cloud(self):
        point_clouds = []
        for camera_name in self.point_cloud_camera_names:
            point_cloud = self.get_point_cloud_single_view(camera_name)
            point_clouds.append(point_cloud)
        point_cloud = np.concatenate(point_clouds, axis=0)

        # crop point cloud
        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = PointCloud.point_cloud_sampling(
            point_cloud, self.num_points, "fps"
        )

        return point_cloud

    def get_obs(self):
        obs_image = self.get_rgb()
        depth = self.get_depth()
        # point_cloud = self.get_point_cloud()
        robot_state = self.get_robot_state()
        obs_dict = {
            "image": obs_image,
            "depth": depth,
            "robot_state": robot_state,
            # "point_cloud": point_cloud,
        }
        return obs_dict

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminated = True
            reward = 0.0
            info = {"error": str(e)}
            if isinstance(e, IKError):
                self.error_type_counts["IKError"] += 1
            elif isinstance(e, ConfigurationPathError):
                self.error_type_counts["ConfigurationPathError"] += 1
            elif isinstance(e, InvalidActionError):
                self.error_type_counts["InvalidActionError"] += 1
            if self.verbose_warnings:
                Logger.log_warning(f"Error (type: {type(e)}) in RLBenchEnv.step(): {e}")
                tb_str = traceback.format_exc()
                Logger.log_warning("Traceback information:")
                print(tb_str)

        self.cur_step += 1
        obs_dict = self.get_obs()
        terminated = terminated or self.cur_step >= self.max_episode_length
        truncated = False
        return obs_dict, reward, terminated, truncated, info

    def reset(self):
        self.cur_step = 0
        obs = self.env.reset()
        obs_dict = self.get_obs()
        if self.cinematic_flag:
            self.tr.take_snap(obs=None)
        return obs_dict


class CameraMotion(object):
    """Base class for camera motion"""

    def __init__(self, cam: VisionSensor):
        # Hook the camera
        self.cam = cam

    def step(self):
        # Move the camera
        raise NotImplementedError()

    def save_pose(self):
        # Save the current camera pose
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        # Restore the camera pose
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):
    """Rotate the camera around a point"""

    def __init__(
        self,
        cam: VisionSensor,
        origin: Dummy,
        speed: float,
        init_rotation: float = np.deg2rad(180),
    ):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])

    def step(self):
        self.origin.rotate([0, 0, self.speed])


class TaskRecorder(object):
    def __init__(self, env: Environment, cam_motion: CameraMotion, fps=30):
        self._env = env
        self._cam_motion = cam_motion
        self._fps = fps
        self._snaps = []
        self._current_snaps = []

    def record_end(self, scene, steps=60, step_scene=True):
        for _ in range(steps):
            if step_scene:
                scene.step()
            self.take_snap(scene.get_observation())

    def take_snap(self, obs: Observation):
        self._cam_motion.step()
        self._current_snaps.append(
            (self._cam_motion.cam.capture_rgb() * 255.0).astype(np.uint8)
        )

    def save(self, path, lang_goal):
        print(colored("[INFO]", "blue"), "Converting to cinematic video ...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # OpenCV QT version can conflict with PyRep, so import here
        import cv2

        image_size = self._cam_motion.cam.get_resolution()
        video = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            self._fps,
            tuple(image_size),
        )
        for image in self._current_snaps:
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = (0.45 * image_size[0]) / 640
            font_thickness = 2

            lang_textsize = cv2.getTextSize(
                lang_goal, font, font_scale, font_thickness
            )[0]
            lang_textX = (image_size[0] - lang_textsize[0]) // 2

            frame = cv2.putText(
                frame,
                lang_goal,
                org=(lang_textX, image_size[1] - 35),
                fontScale=font_scale,
                fontFace=font,
                color=(0, 0, 0),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

            video.write(frame)
        video.release()
        print(colored("[INFO]", "blue"), f"Cinematic video saved to {path}")
        self._current_snaps = []

    def get_frames(self):
        frames = np.array(self._current_snaps)
        self._current_snaps = []
        return frames


class RLBenchEvaluator(Evaluator):
    def __init__(
        self,
        task_name,
        image_size: int,
        action_mode: Optional[ActionMode] = None,
        camera_name: str = "front",
        point_cloud_camera_names: List[str] = ["front"],
        use_point_crop=True,
        num_points=1024,
        max_episode_length: int = 200,
        rotation_representation: str = "quaternion",  # 'quaternion' or 'euler'
        headless: bool = True,
        verbose_warnings: bool = False,
        cinematic_record_enabled: bool = False,
        cinematic_cam_resolution: Tuple[int, int] = (1280, 720),
        cinematic_rotate_speed: float = 0.005,
        cinematic_cam_fps: int = 30,
        require_video_wrapper: bool = True,
    ) -> None:
        if rotation_representation not in ["quaternion", "euler"]:
            raise ValueError(
                f"Invalid rotation representation: {rotation_representation}"
            )
        self.rotation_representation = rotation_representation
        camera_names = list(set([camera_name]) | set(point_cloud_camera_names))
        obs_config = RLBenchObservationConfig.multi_view_config(
            camera_names=camera_names, image_size=(image_size, image_size)
        )
        self.env = RLBenchEnv(
            task_name=task_name,
            action_mode=action_mode,
            obs_config=obs_config,
            camera_name=camera_name,
            point_cloud_camera_names=point_cloud_camera_names,
            use_point_crop=use_point_crop,
            num_points=num_points,
            max_episode_length=max_episode_length,
            headless=headless,
            verbose_warnings=verbose_warnings,
            cinematic_record_enabled=cinematic_record_enabled,
            cinematic_cam_resolution=cinematic_cam_resolution,
            cinematic_rotate_speed=cinematic_rotate_speed,
            cinematic_cam_fps=cinematic_cam_fps,
        )
        if require_video_wrapper:
            self.env = VideoWrapper(self.env)

    def evaluate(
        self,
        num_episodes,
        policy,
        verbose: bool = False,
        verbose_with_cinematic: bool = False,
    ):
        task_name = self.env.task_name

        if verbose:
            video_steps_list, video_cinematic_list = [], []
            success_list, rewards_list = [], []
            if verbose_with_cinematic:
                self.env.set_cinematic_record(True)
        else:
            total_success, total_rewards = 0, 0

        for i in tqdm.tqdm(
            range(num_episodes),
            desc=f'Evaluating in RLbench env <{colored(task_name, "red")}> with text <{colored(self.env.text, "red")}>',
        ):

            if (
                not verbose
                and i == num_episodes - 1
                and self.env.cinematic_record_enabled
            ):
                self.env.set_cinematic_record(True)

            obs = self.env.reset()
            obs_dict = self.env.get_obs()
            terminated = False
            rewards = 0
            success = False
            while not terminated:
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

                if self.rotation_representation == "quaternion":
                    x, y, z, qw, qx, qy, qz, gripper = action
                    unit_quat = Quaternion.normalize_quaternion([qw, qx, qy, qz])
                    action = np.array(
                        [
                            x,
                            y,
                            z,
                            unit_quat[0],
                            unit_quat[1],
                            unit_quat[2],
                            unit_quat[3],
                            gripper,
                        ]
                    )
                elif self.rotation_representation == "euler":
                    x, y, z, roll, pitch, yaw, gripper = action
                    unit_quat = Rotation.from_euler(
                        "xyz", [roll, pitch, yaw], degrees=False
                    ).as_quat(scalar_first=False)
                    action = np.array(
                        [
                            x,
                            y,
                            z,
                            unit_quat[0],
                            unit_quat[1],
                            unit_quat[2],
                            unit_quat[3],
                            gripper,
                        ]
                    )
                else:
                    raise ValueError(
                        f"Invalid rotation representation: {self.rotation_representation}"
                    )
                
                obs_dict, reward, terminated, truncated, info = self.env.step(action)
                rewards += reward
                success = success or bool(reward)
                if terminated:
                    break

            if verbose:
                video_steps_list.append(self.env.get_frames().transpose(0, 3, 1, 2))
                if verbose_with_cinematic:
                    video_cinematic_list.append(
                        self.env.tr.get_frames().transpose(0, 3, 1, 2)
                    )
                else:
                    video_cinematic_list.append(None)
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
            self.video_cinematic_list = video_cinematic_list
        else:
            avg_success = total_success / num_episodes
            avg_rewards = total_rewards / num_episodes
            return_value = (avg_success, avg_rewards)

        error_type_counts = self.env.error_type_counts
        if any(error_type_counts.values()):
            Logger.log_warning("Error type counts:")
            for k, v in error_type_counts.items():
                Logger.log_warning(f"{k}: {v}")
        self.env.set_cinematic_record(False)

        return return_value

    def callback(self, logging_info: dict):
        if self.env.cinematic_record_enabled:
            task_recorder = self.env.tr
            frames = task_recorder.get_frames()
            frames = frames.transpose(0, 3, 1, 2)
            logging_info.update(
                {"validation/video_cinematic": wandb.Video(frames, fps=30)}
            )

    def callback_verbose(self, wandb_logger):
        import plotly.express as px
        import plotly.graph_objects as go

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

        for i, (success, rewards, video_steps, video_cinematic) in enumerate(
            zip(
                self.success_list,
                self.rewards_list,
                self.video_steps_list,
                self.video_cinematic_list,
            )
        ):
            wandb_info = {}
            if success:
                wandb_info.update(
                    {
                        f"validation/video_steps_success_{i}": wandb.Video(
                            video_steps, fps=30
                        )
                    }
                )
                if video_cinematic is not None:
                    wandb_info.update(
                        {
                            f"validation/video_cinematic_success_{i}": wandb.Video(
                                video_cinematic, fps=30
                            )
                        }
                    )
            else:
                wandb_info.update(
                    {
                        f"validation/video_steps_failure_{i}": wandb.Video(
                            video_steps, fps=30
                        )
                    }
                )
                if video_cinematic is not None:
                    wandb_info.update(
                        {
                            f"validation/video_cinematic_failure_{i}": wandb.Video(
                                video_cinematic, fps=30
                            )
                        }
                    )
            wandb_logger.log(wandb_info)


if __name__ == "__main__":

    class RandomPolicy(torch.nn.Module):
        def __init__(self, output_size):
            super(RandomPolicy, self).__init__()
            self.output_size = output_size
            self.dummy_params = torch.nn.Parameter(torch.rand(1))

        def forward(self, images, point_clouds, robot_states, texts):
            return torch.rand(self.output_size)

    task_name = "reach_target"
    evaluator = RLBenchEvaluator(
        task_name="open_box",
        image_size=224,
        action_mode=RLBenchActionMode.eepose_then_gripper_action_mode(absolute=False),
        camera_name="front",
        point_cloud_camera_names=["front"],
        use_point_crop=True,
        num_points=1024,
        max_episode_length=200,
        headless=True,
        verbose_warnings=False,
        cinematic_record_enabled=False,
        cinematic_cam_resolution=(1280, 720),
        cinematic_rotate_speed=0.005,
        cinematic_cam_fps=30,
    )
    num_episodes = 10
    policy = RandomPolicy(output_size=evaluator.env.unwrapped.env.action_space.shape[0])
    avg_success, avg_rewards = evaluator.evaluate(num_episodes, policy)
    print(f"Average success rate: {avg_success}, average rewards: {avg_rewards}")
