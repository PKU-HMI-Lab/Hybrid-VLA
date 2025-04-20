import numpy as np
from typing import List
from rlbench.backend.observation import Observation


class Demo(object):

    def __init__(self, observations: List[Observation], random_seed=None, num_reset_attempts=None):
        self._observations = observations
        self.random_seed = random_seed
        self.num_reset_attempts = num_reset_attempts

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i) -> Observation:
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    arm_small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and arm_small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo: Demo,
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    STOP_BUFFER_SIZE = 4

    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
            if len(episode_keypoints) > 1 and (episode_keypoints[-1] - STOP_BUFFER_SIZE - 1) <= episode_keypoints[-2]:
                episode_keypoints.pop(-2)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError
    