import argparse
import os
import pathlib
import sys

import numpy as np

from lift3d.envs import RLBenchActionMode, RLBenchEnv, RLBenchObservationConfig
from lift3d.helpers.common import (
    Logger,
    save_depth_image,
    save_point_cloud_plotly,
    save_point_cloud_ply,
    save_rgb_image,
)
from lift3d.helpers.gymnasium import VideoWrapper


def main(args):
    result_dir = pathlib.Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=False)
    cameras = list(set(args.point_cloud_cameras) | set([args.camera_name]))
    obs_config = RLBenchObservationConfig.multi_view_config(cameras, args.image_size)
    env = RLBenchEnv(
        task_name=args.task_name,
        camera_name=args.camera_name,
        action_mode=action_mode,
        cinematic_record_enabled=True,
        obs_config=obs_config,
        point_cloud_camera_names=cameras,
        num_points=args.num_points,
        use_point_crop=True,
    )
    env = VideoWrapper(env)
    obs = env.reset()

    # Save RGB image
    image = obs["image"]
    Logger.log_info(f"Image shape: {image.shape}, type: {type(image)}")
    save_rgb_image(image, result_dir / "image.png")

    # Save depth image
    depth = obs["depth"]
    Logger.log_info(f"Depth shape: {depth.shape}, type: {type(depth)}")
    if depth.ndim == 2:
        depth = depth[np.newaxis, :]
    for i, d in enumerate(depth):
        save_depth_image(d, result_dir / f"depth_{i}.png")

        # Save point cloud
    point_cloud = obs["point_cloud"]
    Logger.log_info(
        f"Point cloud shape: {point_cloud.shape}, type: {type(point_cloud)}"
    )
    save_point_cloud_ply(point_cloud, result_dir / "point_cloud.ply")
    for camera_name in args.point_cloud_cameras:
        point_cloud_sub = env.get_point_cloud_single_view(camera_name)
        save_point_cloud_ply(
            point_cloud_sub, result_dir / f"point_cloud_{camera_name}.ply"
        )
    save_point_cloud_plotly(
        point_cloud, result_dir / "point_cloud.html", visualize=True
    )

    Logger.log_ok("All observations saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="open_box")
    parser.add_argument("--camera-name", type=str, default="front")
    parser.add_argument(
        "--point-cloud-cameras",
        type=str,
        nargs="+",
        default=["front", "wrist", "overhead", "left_shoulder", "right_shoulder"],
    )
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--result-dir",
        type=str,
        default=str(
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "results"
            / "sample_obs_rlbench"
        ),
    )
    main(parser.parse_args())
