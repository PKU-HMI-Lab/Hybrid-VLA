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
    tool_path = code_root / "tools" / "sample_obs_metaworld.py"
    for task in TASKS:
        for camera in CAMERAS:
            cmd = [
                "python",
                str(tool_path),
                "--task-name",
                task,
                "--camera-name",
                camera,
                "--point-cloud-cameras",
                camera,
                "--image-size",
                "224",
                "--result-dir",
                f"results/sample_obs_metaworld/{task}_{camera}",
            ]
            Logger.log_info(" ".join(cmd))
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
