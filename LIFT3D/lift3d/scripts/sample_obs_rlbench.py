import os
import pathlib
import subprocess
import sys

from lift3d.helpers.common import Logger

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
    tool_path = code_root / "tools" / "sample_obs_rlbench.py"
    for task in TASKS:
        cmd = [
            "python",
            str(tool_path),
            "--task-name",
            task,
            "--camera-name",
            "front",
            "--image-size",
            "224",
            "--result-dir",
            f"results/sample_obs_rlbench/{task}",
        ]
        Logger.log_info(" ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
