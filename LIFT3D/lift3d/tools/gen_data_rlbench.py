import argparse
import copy
import os
import pathlib
import pickle
import sys

import numpy as np
import tqdm
import zarr
from numcodecs import MsgPack
from rlbench.backend.observation import Observation
from rlbench.demo import keypoint_discovery
from rlbench.utils import get_stored_demos
from termcolor import colored

from lift3d.dataset import RLBenchDataset
from lift3d.envs import RLBenchEnv, RLBenchObservationConfig
from lift3d.helpers.common import (
    Logger,
    save_depth_image,
    save_rgb_image,
    save_video_imageio,
)
from lift3d.helpers.graphics import EEpose, PointCloud, Quaternion

import shutil

def action_extracter(data_curr, data_next, rotation_representation):
    arm_action = EEpose.pose_delta_7DoF(
        pose1_7d=data_curr.gripper_pose,
        pose2_7d=data_next.gripper_pose,
    )

    if rotation_representation == "quaternion":
        x, y, z, qx, qy, qz, qw = arm_action
        arm_action = np.concatenate(
            [
                np.array([x, y, z]),
                Quaternion.ensure_positive_real_part(
                    np.array([qx, qy, qz, qw]), scalar_first=False
                ),
            ],
            axis=0,
        )
    elif rotation_representation == "euler":
        arm_action = EEpose.pose_7DoF_to_6DoF(
            arm_action, scalar_first=False, degrees=False
        )
    else:
        raise ValueError("Rotation representation should be either quaternion or euler")

    gripper_action = data_next.gripper_open
    action = np.append(arm_action, gripper_action)
    return action

def action_extracter_ch(data_curr, data_next, rotation_representation):
    position1 = data_curr.gripper_pose[:3]
    position2 = data_next.gripper_pose[:3]
    delta_postion = position2 - position1
    arm_action = np.concatenate([delta_postion, data_next.gripper_pose[3:]])

    if rotation_representation == "quaternion":
        x, y, z, qx, qy, qz, qw = arm_action
        arm_action = np.concatenate(
            [
                np.array([x, y, z]),
                Quaternion.ensure_positive_real_part(
                    np.array([qx, qy, qz, qw]), scalar_first=False
                ),
            ],
            axis=0,
        )
    elif rotation_representation == "euler":
        arm_action = EEpose.pose_7DoF_to_6DoF(
            arm_action, scalar_first=False, degrees=False
        )
    else:
        raise ValueError("Rotation representation should be either quaternion or euler")

    gripper_action = data_next.gripper_open
    action = np.append(arm_action, gripper_action)
    return action


def robot_state_extracter(data: Observation):
    arm_joint_state = data.joint_positions
    arm_pose_state = data.gripper_pose
    x, y, z, qx, qy, qz, qw = arm_pose_state
    arm_pose_state = np.concatenate(
        [
            np.array([x, y, z]),
            Quaternion.ensure_positive_real_part(
                np.array([qx, qy, qz, qw]), scalar_first=False
            ),
        ],
        axis=0,
    )
    gripper_state = data.gripper_open
    robot_state = np.concatenate(
        (arm_joint_state, arm_pose_state, np.array([gripper_state]))
    )
    return robot_state


def image_extracter(data: Observation):
    image = data.front_rgb
    return image

def recreate_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)

def main(args):
    # Report the arguments
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with arguments:'
    )
    Logger.log_info(f"RLBench data root: {args.rlbench_data_root}")
    Logger.log_info(f"Task name: {args.task_name}")
    Logger.log_info(f"Camera name: {args.camera_name}")
    Logger.log_info(f"Rotation representation: {args.rotation_representation}")
    Logger.log_info(f"Image size: {args.image_size}")
    Logger.log_info(f"Number of episodes: {args.num_episodes}")
    Logger.log_info(f"Only keypoints: {args.only_keypoints}")
    Logger.log_info(f"Interval: {args.interval}")
    Logger.log_info(f"Number of points: {args.num_points}")
    Logger.log_info(f"Save directory: {args.save_dir}")
    task_name = args.task_name
    if args.rotation_representation not in ["quaternion", "euler"]:
        raise ValueError("Rotation representation should be either quaternion or euler")
    if args.camera_name != "front":
        raise ValueError("Only front camera is used for fixed manipulation")

    camera_names = list(set(args.point_cloud_camera_names) | set([args.camera_name]))
    Logger.log_info(f"Active camera names: {camera_names}")
    Logger.print_seperator()

    obs_config = RLBenchObservationConfig.multi_view_config(
        camera_names=camera_names,
        image_size=(args.image_size, args.image_size),
    )

    # Make directories
    video_dir = os.path.join(
        args.save_dir, "visualized_data", args.task_name, "videos", args.camera_name
    )
    image_dir = os.path.join(
        args.save_dir, "visualized_data", args.task_name, "images", args.camera_name
    )
    depth_dir = os.path.join(
        args.save_dir, "visualized_data", args.task_name, "depths", args.camera_name
    )
    text_dir = os.path.join(
        args.save_dir, "visualized_data", args.task_name, "texts", args.camera_name
    )
    recreate_directory(video_dir)
    recreate_directory(image_dir)
    recreate_directory(depth_dir)
    recreate_directory(text_dir)

    for_rlds_dir = os.path.join(
        args.save_dir, "for_rlds", args.task_name
    )
    recreate_directory(for_rlds_dir)

    # Convert source data to dataset
    total_count = 0
    img_arrays = []
    robot_state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    texts = []

    dataset_root = pathlib.Path(args.rlbench_data_root).expanduser()
    task_data_dir = dataset_root / task_name
    num_variations = len(
        list(filter(lambda x: x.startswith("variation"), os.listdir(task_data_dir)))
    )
    Logger.log_info(
        f'Found {colored(num_variations, "red")} variations for task {colored(task_name, "red")}'
    )
    iterable = (
        tqdm.tqdm(range(args.num_episodes)) if args.quiet else range(args.num_episodes)
    )
    for i in iterable:
        demo = get_stored_demos(
            amount=1,
            image_paths=None,
            dataset_root=dataset_root,
            variation_number=i % num_variations,
            task_name=task_name,
            obs_config=obs_config,
            random_selection=False,
            from_episode_number=i // num_variations,
        )[0]

        description_path = (
            task_data_dir
            / f"variation{i % num_variations}"
            / "variation_descriptions.pkl"
        )
        with open(description_path, "rb") as f:
            descriptions = pickle.load(f)
        description = max(descriptions, key=len)

        total_count_sub = 0
        img_arrays_sub = []
        robot_state_arrays_sub = []
        action_arrays_sub = []
        texts_sub = []

        if not args.quiet:
            Logger.log_info(f'episode {colored(i, "red")}')

        key_points = keypoint_discovery(demo)
        action_points = (
            [0] + key_points
            if args.only_keypoints
            else sorted(list(set(range(0, len(demo), args.interval)) | set(key_points)))
        )

        if not args.quiet:
            Logger.log_info(
                f'extracted {colored(len(action_points), "red")} action points with '
                f'{colored(len(key_points), "red")} key points from '
                f'{colored(len(demo), "red")} steps'
            )

        episode_np_list =[]
        demo = [demo[i] for i in action_points]
        for j in range(1, len(demo)):
            total_count_sub += 1
            # action
            action = action_extracter_ch(
                demo[j - 1], demo[j], args.rotation_representation
            )
            # robot state
            robot_state = robot_state_extracter(demo[j - 1])
            # observation
            img = image_extracter(demo[j - 1])

            # record data
            img_arrays_sub.append(img)
            robot_state_arrays_sub.append(robot_state)
            action_arrays_sub.append(action)
            texts_sub.append(description)

            # if j == 1:
            #     print(img.shape)
            #     print(robot_state)
            #     print(action)
            #     print(description)

            episode_np_list.append(
                {
                'keypoint': action_points[j-1],
                'image': img,
                'state': robot_state,
                'action': action,
                'language_instruction': description,
                }
            )

        print(f'episode{i}: ', len(demo), '  ', len(episode_np_list))
        # save .npy data, one .npy for one episode
        np.save(os.path.join(for_rlds_dir, f'episode{i}.npy'), episode_np_list)

        # save visualized data
        sample_video_array = np.stack(img_arrays_sub, axis=0)
        sample_video_array = np.concatenate([sample_video_array, image_extracter(demo[j])[np.newaxis, ...]], axis=0)
        save_video_imageio(
            sample_video_array,
            os.path.join(video_dir, f"episode_{i}.mp4"),
            quiet=args.quiet,
        )

        for stepp in range(len(img_arrays_sub)):
            save_rgb_image(
                img_arrays_sub[stepp],
                os.path.join(image_dir, f"episode_{i}_{stepp}.png"),
                quiet=args.quiet,
            )

        save_depth_image(
            demo[0].front_depth,
            os.path.join(depth_dir, f"episode_{i}.png"),
            quiet=args.quiet,
        )

        with open(os.path.join(text_dir, f"{task_name}_episode_{i}.txt"), "w") as f:
            f.write(description)
            f.write('\n')
            f.write(" ".join(map(str, action_points)))


        # release memory
        del (
            demo,
            key_points,
            action_points,
            img_arrays_sub,
            robot_state_arrays_sub,
            action_arrays_sub,
            texts_sub,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlbench-data-root", type=str, default="~/Data/RLBench_224")
    parser.add_argument("--task-name", type=str, default="open_box")
    parser.add_argument("--camera-name", type=str, default="front")
    parser.add_argument(
        "--point-cloud-camera-names",
        type=str,
        nargs="+",
        default=["front", "overhead", "wrist", "left_shoulder", "right_shoulder"],
    )
    parser.add_argument(
        "--rotation-representation",
        type=str,
        default="quaternion",
        help="quaternion or euler",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=120)
    parser.add_argument("--only-keypoints", action="store_true", default=True)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "data"
            / "rlbench_i1_quaternion"
        ),
    )
    parser.add_argument("--quiet", action="store_true", default=True)
    main(parser.parse_args())
