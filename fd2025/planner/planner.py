from dataclasses import dataclass

import numpy as np
from rpbench.articulated.pr2.jskfridge import JskFridgeVerticalReachingTask
from rpbench.articulated.world.jskfridge import get_fridge_model
from rpbench.articulated.world.utils import CylinderSkelton
from skrobot.coordinates import Coordinates, rpy_angle
from skrobot.model.primitives import Axis

from fd2025.perception.perception_node import FridgeEnvDetection, PerceptionDebugNode


@dataclass
class Transform2d:
    trans: np.ndarray
    rot: float

    def __mul__(self, other):
        trans12, rot12 = self.trans, self.rot
        trans23, rot23 = other.trans, other.rot
        rot23_mat = np.array([[np.cos(rot23), -np.sin(rot23)], [np.sin(rot23), np.cos(rot23)]])
        rot13 = rot12 + rot23
        trans13 = trans23 + rot23_mat @ trans12
        return Transform2d(trans13, rot13)

    def __rmul__(self, other):
        return other.__mul__(self)

    def inv(self):
        inv_rot = -self.rot
        c = np.cos(inv_rot)
        s = np.sin(inv_rot)
        rot_mat_inv = np.array([[c, -s], [s, c]])
        inv_trans = -rot_mat_inv @ self.trans
        return Transform2d(inv_trans, inv_rot)


def create_task_param(detection: FridgeEnvDetection, target_coords: Coordinates):
    tf_frdge_to_robot = Transform2d(detection.fridge_param[:2], detection.fridge_param[2])
    tf_robot_to_fridge = tf_frdge_to_robot.inv()

    target_region = get_fridge_model().regions[1]
    tf_fridge_to_region = Transform2d(-target_region.box.worldpos()[:2], 0)
    tf_robot_to_region = tf_robot_to_fridge * tf_fridge_to_region

    task_param = np.zeros(
        1 + JskFridgeVerticalReachingTask.get_world_type().N_MAX_OBSTACLES * 4 + 7
    )
    head = 0
    task_param[0] = len(detection.cylinders)
    head += 1
    for cylinder in detection.cylinders:
        assert isinstance(cylinder, CylinderSkelton)
        tf_cylinder_to_robot = Transform2d(cylinder.worldpos()[:2], 0.0)
        tf_cylinder_to_region = tf_cylinder_to_robot * tf_robot_to_region
        task_param[head : head + 2] = tf_cylinder_to_region.trans
        task_param[head + 2] = cylinder.height
        task_param[head + 3] = cylinder.radius
        head += 4
    head = JskFridgeVerticalReachingTask.get_world_type().N_MAX_OBSTACLES * 4 + 1

    # target position
    target_pos = target_coords.worldpos()
    task_param[head : head + 3] = target_pos
    task_param[head + 3] = rpy_angle(target_coords.worldrot())[0][0]  # yaw
    head += 4

    # pr2 position
    task_param[head : head + 2] = tf_robot_to_fridge.trans
    task_param[head + 2] = tf_robot_to_fridge.rot
    return task_param


if __name__ == "__main__":
    node = PerceptionDebugNode("20250326_082328")
    detection = node.percept()
    v = detection.visualize()
    ax = Axis()
    ax.translate([0.75, +0.4, 1.1])  # easy
    task_param = create_task_param(detection, ax.copy_worldcoords())
    task = JskFridgeVerticalReachingTask.from_task_param(task_param)

    from rpbench.articulated.pr2.jskfridge import AV_INIT
    from skrobot.models import PR2
    from skrobot.viewers import PyrenderViewer

    v = PyrenderViewer()
    task.world.visualize(v)
    pr2 = PR2(use_tight_joint_limit=False)
    pr2.angle_vector(AV_INIT)
    base_pose = task.description[-3:]
    pr2.translate(np.hstack([base_pose[:2], 0.0]))
    pr2.rotate(base_pose[2], "z")
    v.add(pr2)
    v.show()
    import time

    time.sleep(1000)
