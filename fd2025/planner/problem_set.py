from dataclasses import dataclass

import numpy as np
from rpbench.articulated.world.utils import CylinderSkelton
from skrobot.coordinates import Coordinates

from fd2025.perception.perception_node import FridgeEnvDetection
from fd2025.planner.bridge import create_task_param


@dataclass
class TampProblem:
    detection: FridgeEnvDetection
    target_co: Coordinates

    def to_param(self) -> np.ndarray:
        return create_task_param(self.detection, self.target_co)


def problem_single_object_blocking() -> TampProblem:
    fridge_param = np.array([0.55971283, 0.30126796, 0.42723835, 2.35101264])
    cylinder1 = CylinderSkelton(0.0260, 0.1111)
    cylinder1.translate([0.78746959, 0.31389666, 1.04915313])
    cylidner2 = CylinderSkelton(0.0320, 0.116)
    cylidner2.translate([0.73337129, 0.45384931, 1.0517442])
    cylinders = [cylinder1, cylidner2]
    detection = FridgeEnvDetection(fridge_param, cylinders)
    co = Coordinates()
    co.translate([0.3, -0.1, 1.05])
    return TampProblem(detection, co)
