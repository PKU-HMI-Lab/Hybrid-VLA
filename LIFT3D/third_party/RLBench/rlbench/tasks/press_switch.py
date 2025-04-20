from typing import List, Tuple
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition
import numpy as np

class PressSwitch(Task):

    def init_task(self) -> None:
        switch_joint = Joint('joint')
        self.register_success_conditions([JointCondition(switch_joint, 1.0)])

    def init_episode(self, index: int) -> List[str]:
        return ['press switch',
                'turn the switch on or off',
                'flick the switch']

    def variation_count(self) -> int:
        return 1
    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        return [0, 0, np.pi* 3/ 4.], [0, 0, 4*np.pi/4]
