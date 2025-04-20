from typing import List,Tuple
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.task import Task
import numpy as np


class ChangeClock(Task):

    def init_task(self) -> None:
        self.register_success_conditions([
            DetectedCondition(Shape('clock_needle_minute'),
                              ProximitySensor('detector_minute0'))
        ])

    def init_episode(self, index: int) -> List[str]:
        return [
            'change the clock to show time 12.15',
            'adjust the time to 12.15',
            'change the clock to 12.15',
            'set the clock to 12.15',
            'turn the knob on the back of the clock until the time shows 12.15',
            'rotate the wheel on the clock to make it show 12.15',
            'make the clock say 12.15',
            'turn the knob on the back of the clock 90 degrees'
        ]

    def variation_count(self) -> int:
        return 1
    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        return [0, 0, np.pi* 3/ 4.], [0, 0, 4*np.pi/4]