import os
import pathlib
import subprocess
import sys

from lift3d.helpers.common import Logger

RLBENCH_DATA_ROOT = "~/Data/RLBench_224"
DATASET_ROOT = "data/rlbench"


TASKS = [
    "close_box",
    "put_rubbish_in_bin",
    "close_laptop_lid",
    "water_plants",
    "unplug_charger",
    "toilet_seat_down",
]


def main():
    code_root = pathlib.Path(__file__).resolve().parent.parent
    tool_path = code_root / "tools" / "gen_data_rlbench.py"

    for task in TASKS:
        cmd = [
            "python",
            str(tool_path),
            "--rlbench-data-root",
            RLBENCH_DATA_ROOT,
            "--task-name",
            task,
            "--camera-name",
            "front",
            "--point-cloud-camera-names",
            "front",
            "--num-points",
            "1024",
            "--rotation-representation",
            "quaternion",
            "--image-size",
            "224",
            "--num-episodes",
            "120",
            "--only-keypoints",
            "--save-dir",
            DATASET_ROOT,
            "--quiet",
        ]
        Logger.log_info(" ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
