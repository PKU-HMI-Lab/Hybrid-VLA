import os
import pathlib
import subprocess
import sys

from lift3d.helpers.common import Logger

CAMERAS = [
    "corner",
    "corner2",
]


TASKS = [
    "assembly",
    "button-press",
    "bin-picking",
    "hammer",
    "drawer-open",
]


def main():
    code_root = pathlib.Path(__file__).resolve().parent.parent
    tool_path = code_root / "tools" / "gen_data_metaworld.py"
    for task in TASKS:
        for camera in CAMERAS:
            cmd = [
                "python",
                str(tool_path),
                "--task-name",
                task,
                "--camera-name",
                camera,
                "--image-size",
                str(224),
                "--num-episodes",
                str(30),
                "--save-dir",
                "data/metaworld",
                "--episode-length",
                str(200),
                "--quiet",
            ]
            Logger.log_info(" ".join(cmd))
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
