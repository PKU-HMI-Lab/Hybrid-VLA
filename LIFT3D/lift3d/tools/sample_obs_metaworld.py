import argparse
import os
import pathlib
import sys

import numpy as np

from lift3d.envs import MetaWorldEnv
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
    env = MetaWorldEnv(
        task_name=args.task_name,
        image_size=args.image_size,
        camera_name=args.camera_name,
        num_points=args.num_points,
        use_point_crop=True,
        point_cloud_camera_names=args.point_cloud_cameras,
    )
    env = VideoWrapper(env)
    obs = env.reset()

    # Save RGB image
    image = obs["image"]
    Logger.log_info(
        f"Image shape: {image.shape}, type: {type(image)}, dtype: {image.dtype}, mean: {np.mean(image)}"
    )
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
    save_point_cloud_plotly(
        array=point_cloud,
        file_path=result_dir / "point_cloud.html",
        visualize=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="assembly")
    parser.add_argument("--camera-name", type=str, default="corner")
    parser.add_argument(
        "--point-cloud-cameras", type=str, nargs="+", default=["corner"]
    )
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--result-dir",
        type=str,
        default=str(
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "results"
            / "sample_obs_metaworld"
        ),
    )
    main(parser.parse_args())
