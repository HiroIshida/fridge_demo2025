from dataclasses import dataclass

import numpy as np
from rpbench.articulated.pr2.jskfridge import JskFridgeReachingTask
from rpbench.articulated.world.utils import CylinderSkelton
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.viewers import PyrenderViewer

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
    cylidner2 = CylinderSkelton(0.026, 0.116)
    cylidner2.translate([0.73337129, 0.45384931, 1.0517442])
    cylinders = [cylinder1, cylidner2]
    detection = FridgeEnvDetection(fridge_param, cylinders)
    co = Coordinates()
    co.translate([0.3, -0.1, 1.05])
    return TampProblem(detection, co)


def problem_single_object_blocking_hard() -> TampProblem:
    fridge_param = np.array([0.55971283, 0.30126796, 0.42723835, 2.35101264])
    cylinder1 = CylinderSkelton(0.0260, 0.1111)
    cylinder1.translate([0.78746959, 0.31389666, 1.04915313])
    cylidner2 = CylinderSkelton(0.026, 0.116)
    cylidner2.translate([0.73337129, 0.45384931, 1.0517442])
    cylinders = [cylinder1, cylidner2]
    detection = FridgeEnvDetection(fridge_param, cylinders)
    co = Coordinates()
    co.translate([0.3, -0.03, 1.05])
    return TampProblem(detection, co)


def problem_single_object_blocking_hard2() -> TampProblem:
    fridge_param = np.array([0.55971283, 0.30126796, 0.42723835, 2.35101264])
    cylinder1 = CylinderSkelton(0.0260, 0.1111)
    cylinder1.translate([0.78746959, 0.31389666, 1.04915313])
    cylidner2 = CylinderSkelton(0.026, 0.116)
    cylidner2.translate([0.73337129, 0.45384931, 1.0517442])
    cylinders = [cylinder1, cylidner2]
    detection = FridgeEnvDetection(fridge_param, cylinders)
    co = Coordinates()
    co.translate([0.3, -0.03, 1.05])
    return TampProblem(detection, co)


def problem_double_object2_blocking() -> TampProblem:
    fridge_param = np.array([0.55971283, 0.30126796, 0.42723835, 2.35101264])
    cylinder1 = CylinderSkelton(0.0260, 0.1111)
    cylinder1.translate([0.81, 0.34389666, 1.04915313])
    cylidner2 = CylinderSkelton(0.026, 0.116)
    cylidner2.translate([0.75, 0.41384931, 1.0517442])
    cylinders = [cylinder1, cylidner2]
    detection = FridgeEnvDetection(fridge_param, cylinders)
    co = Coordinates()
    co.translate([0.33, -0.03, 1.05])
    return TampProblem(detection, co)


if __name__ == "__main__":
    problem = problem_double_object2_blocking()
    task = JskFridgeReachingTask.from_task_param(problem.to_param())
    v = PyrenderViewer()
    task.world.visualize(v)
    ax = Axis()
    ax.newcoords(problem.target_co)
    v.add(ax)
    v.show()
    import time

    time.sleep(1000)
