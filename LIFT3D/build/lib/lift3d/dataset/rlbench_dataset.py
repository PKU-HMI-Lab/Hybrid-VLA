import torch
import zarr
from termcolor import colored

from lift3d.helpers.common import Logger


class RLBenchDataset(torch.utils.data.Dataset):
    """
    Dataset for RLBench Benchmark.
    """

    SPLIT_SIZE = {"train": 100, "validation": 20, "custom": None}

    def __init__(self, data_dir, split: str = None, custom_split_size: int = None):
        zarr_root = zarr.open_group(data_dir, mode="r")
        self._episode_ends = zarr_root["meta"]["episode_ends"][:]

        if split not in self.SPLIT_SIZE:
            raise ValueError(f"Invalid split: {split}")

        if split == "custom" and custom_split_size is None:
            raise ValueError(f"custom_split_size must be provided for split: {split}")

        begin_index, end_index = (
            (0, self._episode_ends[self.SPLIT_SIZE["train"] - 1])
            if split == "train"
            else (
                (
                    self._episode_ends[self.SPLIT_SIZE["train"] - 1],
                    self._episode_ends[
                        self.SPLIT_SIZE["train"] + self.SPLIT_SIZE["validation"] - 1
                    ],
                )
                if split == "validation"
                else (0, self._episode_ends[custom_split_size - 1])
            )
        )

        # (T, H, W, C) -> (T, C, H, W)
        self._images = zarr_root["data"]["images"][begin_index:end_index].transpose(
            0, 3, 1, 2
        )
        # Logger.log_notice(f'images shape: {self._images.shape}')
        assert self._images.shape[1] == 3
        self._point_clouds = zarr_root["data"]["point_clouds"][begin_index:end_index]
        self._robot_states = zarr_root["data"]["robot_states"][begin_index:end_index]
        self._actions = zarr_root["data"]["actions"][begin_index:end_index]
        self._texts = zarr_root["data"]["texts"][begin_index:end_index]
        assert len(self._images) == len(self._robot_states) == len(self._actions)
        self._dataset_size = len(self._actions)

    def __getitem__(self, idx):
        image = torch.from_numpy(self._images[idx]).float()
        point_cloud = torch.from_numpy(self._point_clouds[idx]).float()
        robot_state = torch.from_numpy(self._robot_states[idx]).float()
        action = torch.from_numpy(self._actions[idx]).float()
        text = self._texts[idx]
        return image, point_cloud, robot_state, torch.zeros((0,)), action, text

    def __len__(self):
        return self._dataset_size

    def print_info(self):
        Logger.log_info(f"RLBench Dataset Info:")
        Logger.log_info(
            f'images ({colored(self._images.dtype, "red")}): {colored(self._images.shape, "red")}, range: [{colored(self._images.min(), "red")}, {colored(self._images.max(), "red")}]'
        )
        Logger.log_info(
            f'point_cloud ({colored(self._point_clouds.dtype, "red")}): {colored(self._point_clouds.shape, "red")}, range: [{colored(self._point_clouds.min(), "red")}, {colored(self._point_clouds.max(), "red")}]'
        )
        Logger.log_info(
            f'robot_state ({colored(self._robot_states.dtype, "red")}): {colored(self._robot_states.shape, "red")}, range: [{colored(self._robot_states.min(), "red")}, {colored(self._robot_states.max(), "red")}]'
        )
        Logger.log_info(
            f'action ({colored(self._actions.dtype, "red")}): {colored(self._actions.shape, "red")}, range: [{colored(self._actions.min(), "red")}, {colored(self._actions.max(), "red")}]'
        )
        Logger.log_info(
            f'text ({colored(type(self._texts[0]), "red")}): {colored(len(self._texts), "red")}'
        )
        Logger.log_info(
            f'episode_ends ({colored(self._episode_ends.dtype, "red")}): {colored(self._episode_ends.shape, "red")}, range: [{colored(self._episode_ends.min(), "red")}, {colored(self._episode_ends.max(), "red")}]'
        )
        Logger.print_seperator()


if __name__ == "__main__":
    data_dir = "data/rlbench/close_box.zarr"
    dataset = RLBenchDataset(data_dir, split="custom", custom_split_size=120)
    actions = dataset._actions
    Logger.log_notice(f"episode_ends: {dataset._episode_ends}")
    Logger.log_notice(f"texts: {dataset._texts}")
