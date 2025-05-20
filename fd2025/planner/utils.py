import numpy as np


def gripper_distance(pr2_model, dist, arm="arms"):
    if arm == "larm":
        joint = pr2_model.l_gripper_l_finger_joint
    elif arm == "rarm":
        joint = pr2_model.r_gripper_l_finger_joint
    else:
        raise ValueError("Invalid arm arm argument. You can specify " "'larm', 'rarm'.")

    def _dist(angle):
        return 0.0099 * (18.4586 * np.sin(angle) + np.cos(angle) - 1.0101)

    # calculate joint_angle from approximated equation
    max_dist = _dist(joint.max_angle)
    dist = max(min(dist, max_dist), 0)
    d = dist / 2.0
    angle = 2 * np.arctan(
        (9137 - np.sqrt(2) * np.sqrt(-5e9 * (d**2) - 5e7 * d + 41739897))
        / (5 * (20000 * d + 199))
    )
    return joint.name, angle


if __name__ == "__main__":
    from skrobot.models.pr2 import PR2
    from skrobot.viewers import PyrenderViewer

    pr2 = PR2()
    joint_name, angle = gripper_distance(pr2, 0.07, arm="larm")
    pr2.__dict__[joint_name].joint_angle(angle)
    v = PyrenderViewer()
    v.add(pr2)
    v.show()
    import time

    time.sleep(1000)
