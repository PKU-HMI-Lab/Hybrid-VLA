import os

import gymnasium
import numpy as np
from termcolor import colored, cprint

from lift3d.helpers.common import save_rgb_image, save_video_imageio


class VideoWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super(VideoWrapper, self).__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.frames = []
        frame = self.env.get_rgb()
        self.frames.append(frame)
        self.step_count = 1
        return obs

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        frame = self.env.get_rgb()
        self.frames.append(frame)
        return result

    def get_frames(self):
        frames = np.stack(self.frames, axis=0)  # (T, H, W, C)
        return frames

    def save_video(self, save_path, fps=30.0):
        frames = self.get_frames()
        save_video_imageio(frames, save_path, fps=fps)

    def save_images(self, save_dir, quiet=False):
        frames = self.get_frames()
        for i, frame in enumerate(frames):
            save_path = os.path.join(save_dir, f"{i:03d}.png")
            save_rgb_image(frame, save_path, quiet=quiet)
        if quiet:
            print(colored("[INFO]", "blue"), f"{0:03d}~{i:03d}.png saved to {save_dir}")

    def __getattr__(self, name: str):
        return getattr(self.env, name)
