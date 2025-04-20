import argparse
import os
import pathlib
import sys

from gymnasium import Wrapper

from lift3d.envs import MetaWorldEnv, load_mw_policy
from lift3d.helpers.common import Logger
from lift3d.helpers.gymnasium import VideoWrapper


def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    env = MetaWorldEnv(
        task_name=args.task_name,
        image_size=args.image_size,
        camera_name=args.camera_name,
    )
    env = VideoWrapper(env)
    mw_policy = load_mw_policy(args.task_name)
    raw_state = env.reset()["raw_state"]
    obs_dict = Wrapper.get_wrapper_attr(env, "get_obs")()
    terminated = False
    while not terminated:
        action = mw_policy.get_action(raw_state)
        obs_dict, reward, terminated, truncated, env_info = env.step(action)
        raw_state = obs_dict["raw_state"]
        if terminated or truncated:
            break
    env.save_video(
        pathlib.Path(args.result_dir, f"{args.task_name}_{args.camera_name}.mp4")
    )
    Logger.log_ok(
        f"Saved video to {args.result_dir}/{args.task_name}_{args.camera_name}.mp4"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="assembly")
    parser.add_argument("--camera-name", type=str, default="corner")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--result-dir",
        type=str,
        default=str(
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "results"
            / "export_videos_metaworld"
        ),
    )
    main(parser.parse_args())
