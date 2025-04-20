import os
import pathlib
import random
import sys
import uuid

import cv2
import hydra
import imageio
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch
import wandb
from PIL import Image
from termcolor import colored, cprint


def set_seed(seed):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def clear_directory(directory_path):
    """Clear the contents of a directory.

    Args:
        directory_path (str): path to the directory
    """
    if not os.path.exists(directory_path):
        print(f"The directory '{directory_path}' does not exist.")
        return
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            clear_directory(dir_path)


def save_video_cv2(array, save_path, fps=30.0, quiet=False):
    """
    Save video from numpy array using cv2.
    Args:
        array (numpy.ndarray): (n_frames, height, width, 3)
        save_path (str): path to save video
        fps (float): frame per second
    """
    n_frames, height, width, _ = array.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (height, width))
    for frame in array:
        video_writer.write(frame)
    video_writer.release()
    if not quiet:
        Logger.log_info(f"Video (cv2) saved to {save_path}")


def save_video_imageio(array, save_path, fps=30.0, quiet=False):
    """
    Save video from numpy array using imageio.
    Args:
        array (numpy.ndarray): (n_frames, height, width, 3)
        save_path (str): path to save video
        fps (float): frame per second
    """
    writer = imageio.get_writer(save_path, fps=fps)  # fps是帧率
    for frame in array:
        writer.append_data(frame)
    writer.close()
    if not quiet:
        Logger.log_info(f"Video (imageio) saved to {save_path}")


class Logger:
    """Logger class for printing messages with different colors."""

    @staticmethod
    def log(hint, *args, **kwargs):
        color = kwargs.pop("color", "blue")
        print(colored(f"[{hint}]", color), *args, **kwargs)

    @staticmethod
    def log_info(*args, **kwargs):
        Logger.log("INFO", *args, **kwargs, color="blue")

    @staticmethod
    def log_warning(*args, **kwargs):
        Logger.log("WARNING", *args, **kwargs, color="yellow")

    @staticmethod
    def log_error(*args, **kwargs):
        Logger.log("ERROR", *args, **kwargs, color="red")

    @staticmethod
    def log_ok(*args, **kwargs):
        Logger.log("OK", *args, **kwargs, color="green")

    @staticmethod
    def log_notice(*args, **kwargs):
        Logger.log("NOTICE", *args, **kwargs, color="magenta")

    @staticmethod
    def print_seperator(char="-", color="cyan"):
        cprint(char * os.get_terminal_size().columns, color)


def get_env_variable(var_name):
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{var_name}' does not exist.")
    return value


class WandBLogger:
    def __init__(self, config, hyperparameters=None):
        try:
            get_env_variable("WANDB_API_KEY")
            get_env_variable("WANDB_USER_EMAIL")
            get_env_variable("WANDB_USERNAME")
        except:
            Logger.log_error("Please set up the environment variables for wandb.")
            Logger.log_error(
                "Required: WANDB_API_KEY, WANDB_USER_EMAIL, WANDB_USERNAME"
            )
            exit(1)

        self.run = wandb.init(
            project=config["project"],
            config=hyperparameters,
            group=config["group"],
            name=config["name"],
            notes=config["notes"],
            reinit=config["reinit"],
            mode=config["mode"],
            id=uuid.uuid4().hex,
            dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)


def save_rgb_image(image_array, save_path, quiet=False):
    """save image to file

    Args:
        image_array (ndarray): shape (H, W, 3)
        save_path (str): path to save image
    """
    image = Image.fromarray(image_array)
    save_path = os.path.abspath(save_path)
    image.save(save_path)
    if not quiet:
        Logger.log_info(f"RGB image saved to {save_path}")


def save_depth_image(image_array, save_path, quiet=False):
    """save image to file

    Args:
        image_array (ndarray): shape (H, W)
        save_path (str): path to save image
    """
    depth_map_normalized = (
        (image_array - image_array.min())
        / (image_array.max() - image_array.min())
        * 255
    ).astype(np.uint8)
    image = Image.fromarray(depth_map_normalized)
    save_path = os.path.abspath(save_path)
    image.save(save_path)
    if not quiet:
        Logger.log_info(f"Depth image saved to {save_path}")


def save_point_cloud_ply(array, file_path, quiet=False):
    """Save point clouds (N x 6 array) as a PLY file.

    Args:
        array (numpy.ndarray): (N, 6) array with x, y, z, R, G, B values
        file_path (str): path to save the point cloud
        quiet (bool, optional): suppress log messages. Defaults to False
    """
    if isinstance(file_path, pathlib.Path):
        file_path = str(file_path)

    # Extract points and colors from the array
    points = array[:, 0:3]  # x, y, z coordinates
    colors = array[:, 3:6] / 255.0  # Normalize RGB colors to [0, 1]

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Write the point cloud to a PLY file
    o3d.io.write_point_cloud(file_path, point_cloud)

    if not quiet:
        Logger.log_info(f"Point cloud saved to {file_path}")


def save_point_cloud_plotly(array, file_path, visualize=False):
    """Save point clouds (N x 6 array) as an interactive HTML file.

    Args:
        array (numpy.ndarray): (N, 6) array with x, y, z, R, G, B values
        file_path (str): path to save the point cloud
        quiet (bool, optional): suppress log messages. Defaults to False
    """
    save_dir = pathlib.Path(file_path).parent
    if save_dir.exists() is False:
        save_dir.mkdir(parents=True, exist_ok=True)
        Logger.log_warning(f"Directory '{save_dir}' does not exist. Created.")

    fig = go.Figure()
    rgb = array[:, 3:6] / 255.0
    colors = ["rgb({},{},{})".format(r * 255, g * 255, b * 255) for r, g, b in rgb]

    trace = go.Scatter3d(
        x=array[:, 0],
        y=array[:, 1],
        z=array[:, 2],
        mode="markers",
        marker=dict(size=2, color=colors, colorscale="Viridis", opacity=0.8),
    )

    layout = go.Layout(
        title="Interactive Point Clouds",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )
    fig = go.Figure(data=[trace], layout=layout)

    fig.write_html(file_path)

    Logger.log_info(f"Point clouds visualization saved as {file_path}")

    if visualize:
        fig.show()


def float_list_formatter(float_list):
    return [f"{x:.5f}" for x in float_list]
