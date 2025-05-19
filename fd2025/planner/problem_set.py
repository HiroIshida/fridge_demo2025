from dataclasses import dataclass

import numpy as np
from rpbench.articulated.pr2.jskfridge import AV_INIT, JskFridgeReachingTask
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


def problem_triple_object_blocking() -> TampProblem:
    fridge_param = np.array([0.55971283, 0.30126796, 0.42723835, 2.35101264])
    cylinder1 = CylinderSkelton(0.0260, 0.1111)
    cylinder1.translate([0.81, 0.34389666, 1.04915313])
    cylidner2 = CylinderSkelton(0.026, 0.116)
    cylidner2.translate([0.75, 0.41384931, 1.0517442])
    cylinder3 = CylinderSkelton(0.0260, 0.1111)
    cylinder3.translate([0.86, 0.38389666, 1.04915313])
    cylinders = [cylinder1, cylidner2, cylinder3]
    detection = FridgeEnvDetection(fridge_param, cylinders)
    co = Coordinates()
    co.translate([0.33, -0.03, 1.05])
    return TampProblem(detection, co)


def problem_triple_object_blocking2() -> TampProblem:
    fridge_param = np.array([0.43971283, 0.48126796, 0.42723835, 2.35101264])
    cylinder1 = CylinderSkelton(0.0260, 0.1111)
    cylinder1.translate([0.68, 0.52389666, 1.04915313])
    cylidner2 = CylinderSkelton(0.026, 0.116)
    cylidner2.translate([0.62, 0.59384931, 1.0517442])
    cylinder3 = CylinderSkelton(0.0260, 0.1111)
    cylinder3.translate([0.73, 0.56389666, 1.04915313])
    cylinders = [cylinder3, cylidner2, cylinder1]
    detection = FridgeEnvDetection(fridge_param, cylinders)
    co = Coordinates()
    co.translate([0.33, -0.03, 1.05])
    return TampProblem(detection, co)


if __name__ == "__main__":
    from skrobot.models import PR2

    pr2 = PR2(use_tight_joint_limit=False)
    pr2.angle_vector(AV_INIT)
    problem = problem_triple_object_blocking2()
    task = JskFridgeReachingTask.from_task_param(problem.to_param())
    base_pose = task.description[-3:]
    pr2.translate(np.hstack([base_pose[:2], 0.0]))
    pr2.rotate(base_pose[2], "z")
    v = PyrenderViewer()
    v.add(pr2)
    task.world.visualize(v)
    ax = Axis()
    ax.newcoords(problem.target_co)
    v.add(ax)
    v.show()
    import time

    time.sleep(1000)
