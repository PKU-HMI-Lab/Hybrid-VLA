import argparse
import copy
import os
import pathlib
import sys

import numpy as np
import tqdm
import zarr
from gymnasium import Wrapper
from numcodecs import MsgPack
from termcolor import colored, cprint

from lift3d.dataset import MetaWorldDataset
from lift3d.envs import METAWORLD_LANGUAGE_DESCRIPTION, MetaWorldEnv, load_mw_policy
from lift3d.helpers.common import (
    Logger,
    save_point_cloud_ply,
    save_rgb_image,
    save_video_imageio,
)
from lift3d.helpers.gymnasium import VideoWrapper


def main(args):
    # Report the arguments
    Logger.log_info(
        f'Running {colored(pathlib.Path(__file__).absolute(), "red")} with arguments:'
    )
    Logger.log_info(f"Task name: {args.task_name}")
    Logger.log_info(f"Camera name: {args.camera_name}")
    Logger.log_info(f"Image size: {args.image_size}")
    Logger.log_info(f"Number of episodes: {args.num_episodes}")
    Logger.log_info(f"Episode length: {args.episode_length}")
    Logger.log_info(f"Save directory: {args.save_dir}")
    Logger.print_seperator()

    task_name = args.task_name

    # Create the save directory
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    video_dir = (
        pathlib.Path(args.save_dir)
        / "visualized_data"
        / "videos"
        / task_name
        / args.camera_name
    )
    image_dir = (
        pathlib.Path(args.save_dir)
        / "visualized_data"
        / "images"
        / task_name
        / args.camera_name
    )
    point_cloud_dir = (
        pathlib.Path(args.save_dir)
        / "visualized_data"
        / "point_clouds"
        / task_name
        / args.camera_name
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    point_cloud_dir.mkdir(parents=True, exist_ok=True)

    env = MetaWorldEnv(
        task_name=task_name,
        image_size=args.image_size,
        camera_name=args.camera_name,
        point_cloud_camera_names=[
            args.camera_name,
        ],
    )
    original_episode_length = env.max_episode_length
    env.max_episode_length = args.episode_length
    Logger.log_info(
        f"Original episode length: {original_episode_length}, New episode length: {env.max_episode_length}"
    )

    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    robot_state_arrays = []
    raw_state_arrays = []
    action_arrays = []
    reward_arrays = []
    episode_ends_arrays = []
    env_info_arrays = []
    texts = []
    failure_count = 0

    mw_policy = load_mw_policy(task_name)

    # loop over episodes
    description = METAWORLD_LANGUAGE_DESCRIPTION[task_name]
    episode_idx = 0
    if args.quiet:
        process_bar = tqdm.tqdm(range(args.num_episodes))
    while episode_idx < args.num_episodes:
        raw_state = env.reset()["raw_state"]
        obs_dict = Wrapper.get_wrapper_attr(env, "get_obs")()
        truncated = False
        ep_reward = 0.0
        ep_success = False
        ep_success_times = 0
        img_arrays_sub = []
        point_cloud_arrays_sub = []
        robot_state_arrays_sub = []
        raw_state_arrays_sub = []
        action_arrays_sub = []
        reward_arrays_sub = []
        env_info_arrays_sub = []
        texts_sub = []
        total_count_sub = 0

        while not truncated:
            total_count_sub += 1

            obs_img = obs_dict["image"]
            obs_robot_state = obs_dict["robot_state"]
            obs_point_cloud = obs_dict["point_cloud"]

            action = mw_policy.get_action(raw_state)
            obs_dict, reward, terminated, truncated, env_info = env.step(action)
            raw_state = obs_dict["raw_state"]
            ep_reward += reward

            img_arrays_sub.append(obs_img)
            point_cloud_arrays_sub.append(obs_point_cloud)
            robot_state_arrays_sub.append(obs_robot_state)
            raw_state_arrays_sub.append(raw_state)
            action_arrays_sub.append(action)
            reward_arrays_sub.append(reward)
            env_info_arrays_sub.append(env_info)
            texts_sub.append(description)

            ep_success = ep_success or env_info["success"]
            ep_success_times += env_info["success"]

            if truncated:
                break

        if not ep_success or ep_success_times < 5:
            cprint(
                f"Task: {args.task_name} Episode: {episode_idx} failed with reward {ep_reward} and success times {ep_success_times}",
                "red",
            )
            failure_count += 1
            if failure_count > 5:
                cprint(
                    f"Failed to generate enough successful episodes. Exiting...", "red"
                )
                exit(1)
            continue
        else:
            failure_count = 0
            total_count += total_count_sub
            if args.quiet:
                process_bar.update(1)

            # save visualized dadta
            sample_video_array = np.stack(img_arrays_sub, axis=0)
            save_video_imageio(
                sample_video_array,
                video_dir / f"episode_{episode_idx}.mp4",
                quiet=args.quiet,
            )
            save_rgb_image(
                img_arrays_sub[0],
                image_dir / f"episode_{episode_idx}_rgb.png",
                quiet=args.quiet,
            )
            save_point_cloud_ply(
                point_cloud_arrays_sub[0],
                point_cloud_dir / f"episode_{episode_idx}_point_cloud.ply",
                quiet=args.quiet,
            )

            # merge episode data into dataset
            episode_ends_arrays.append(
                copy.deepcopy(total_count)
            )  # the index of the last step of the episode
            img_arrays.extend(copy.deepcopy(img_arrays_sub))
            point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
            robot_state_arrays.extend(copy.deepcopy(robot_state_arrays_sub))
            action_arrays.extend(copy.deepcopy(action_arrays_sub))
            reward_arrays.extend(copy.deepcopy(reward_arrays_sub))
            raw_state_arrays.extend(copy.deepcopy(raw_state_arrays_sub))
            env_info_arrays.extend(copy.deepcopy(env_info_arrays_sub))
            texts.extend(copy.deepcopy(texts_sub))

            # release memory
            del (
                img_arrays_sub,
                point_cloud_arrays_sub,
                robot_state_arrays_sub,
                action_arrays_sub,
                reward_arrays_sub,
                raw_state_arrays_sub,
                env_info_arrays_sub,
                texts_sub,
            )

            # print episode info
            if not args.quiet:
                cprint(
                    "Episode Index: {}, Episode End: {}, Reward: {}, Success Times: {}".format(
                        episode_idx, total_count, ep_reward, ep_success_times
                    ),
                    "green",
                )
            
            episode_idx += 1

    # Merge data
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3:  # make channel last
        img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    robot_state_arrays = np.stack(robot_state_arrays, axis=0)
    raw_state_arrays = np.stack(raw_state_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    reward_arrays = np.stack(reward_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)
    texts = np.array(texts, dtype=object)

    # Save data
    Logger.log_info("Saving data to zarr file...", end="", flush=True)
    zarr_dir = pathlib.Path(args.save_dir) / f"{task_name}_{args.camera_name}.zarr"
    zarr_root = zarr.group(zarr_dir)
    zarr_data = zarr_root.create_group("data", overwrite=True)
    zarr_meta = zarr_root.create_group("meta", overwrite=True)
    img_chunk_size = (
        100,
        img_arrays.shape[1],
        img_arrays.shape[2],
        img_arrays.shape[3],
    )
    point_cloud_chunk_size = (
        100,
        point_cloud_arrays.shape[1],
        point_cloud_arrays.shape[2],
    )
    robot_state_chunk_size = (100, robot_state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    zarr_data.create_dataset(
        "images",
        data=img_arrays,
        chunks=img_chunk_size,
        dtype="uint8",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_clouds",
        data=point_cloud_arrays,
        chunks=point_cloud_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "robot_states",
        data=robot_state_arrays,
        chunks=robot_state_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "actions",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends", data=episode_ends_arrays, dtype="int64", compressor=compressor
    )
    zarr_data.create_dataset(
        "texts", data=texts, dtype=object, compressor=compressor, object_codec=MsgPack()
    )
    print("Done")
    Logger.log_info(f"Dataset Info:\n{zarr_root.tree()}")
    Logger.print_seperator()
    del (
        img_arrays,
        point_cloud_arrays,
        robot_state_arrays,
        action_arrays,
        episode_ends_arrays,
        texts,
    )
    del zarr_root, zarr_data, zarr_meta
    Logger.log_info("Delete the data in memory")

    # validate the saved data
    dataset = MetaWorldDataset(
        data_dir=zarr_dir,
        split="custom",
        custom_split_size=max(10, args.num_episodes // 10),
    )
    dataset.print_info()

    Logger.log_ok("All data saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="assembly")
    parser.add_argument("--camera-name", type=str, default="corner")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(
            pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "metaworld"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    main(parser.parse_args())
