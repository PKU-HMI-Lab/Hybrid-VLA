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
    tool_path = "third_party/RLBench/tools/cinematic_recorder.py"
    for task in TASKS:
        cmd = [
            "python",
            tool_path,
            "--save_dir",
            "results/rlbench_cinematic",
            "--tasks",
            str(task),
        ]
        Logger.log_info(" ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
