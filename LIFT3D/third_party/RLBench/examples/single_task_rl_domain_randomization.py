import numpy as np

from rlbench import Environment
from rlbench import ObservationConfig
from rlbench import RandomizeEvery
from rlbench import VisualRandomizationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import ReachTarget


class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


obs_config = ObservationConfig()
obs_config.set_all(True)

# We will borrow some from the tests dir
rand_config = VisualRandomizationConfig(
    image_directory='../tests/unit/assets/textures')

action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(
    action_mode, obs_config=obs_config, headless=False,
    randomize_every=RandomizeEvery.EPISODE, frequency=1,
    visual_randomization_config=rand_config
)
env.launch()

task = env.get_task(ReachTarget)

agent = Agent(env.action_shape)

training_steps = 120
episode_length = 20
obs = None
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        descriptions, obs = task.reset()
        print(descriptions)
    action = agent.act(obs)
    obs, reward, terminate = task.step(action)

print('Done')
env.shutdown()
