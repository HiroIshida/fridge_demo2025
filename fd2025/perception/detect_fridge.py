from dataclasses import dataclass
from typing import Optional

import numpy as np
import rospy
from rpbench.articulated.world.jskfridge import FridgeModel
from scipy.optimize import minimize
from skrobot.sdf import UnionSDF

from fd2025.perception.magic import POINTCLOUD_OFFSET


@dataclass
class FridgeModelReconciler:
    _next_initial_guess: Optional[np.ndarray] = None

    @staticmethod
    def instantiate_fridge_model(param: np.ndarray) -> FridgeModel:
        x, y, yaw, angle = param
        model = FridgeModel(angle)
        model.translate([x, y, 0])
        model.rotate(yaw, "z")
        return model

    def reconcile(self, points: np.ndarray) -> Optional[np.ndarray]:
        # remove nan inf by np.isfinite
        points = points[np.all(np.isfinite(points), axis=1)]

        # tmp: delete points of fetch robot behind the door
        # bools_fetch = np.all(
        #     [points[:, 0] > 0.4, points[:, 1] < -0.3, points[:, 2] < 0.7], axis=0
        # )
        # points = points[~bools_fetch]

        # tiny calibration error easily make planning problem infeasible though its actually feasible
        points += POINTCLOUD_OFFSET
        data = points[points[:, 0] < 2.0]

        # determine yaw first
        data_lower = data[data[:, 2] < 0.7]
        if len(data_lower) < 100:
            rospy.logwarn("Too few points to determine yaw")
            return None  # to small to determine yaw
        x_tmp = -data_lower[:, 1]  # -90 degree rotated
        y_tmp = data_lower[:, 0]  # -90 degree rotatied
        A = np.vstack([x_tmp, np.ones(len(x_tmp))]).T
        a, b = np.linalg.lstsq(A, y_tmp, rcond=None)[0]
        yaw = np.arctan(a)

        # we can detremine t_min, t_max by using data_lower
        # because from the min and max value of the lower points
        # we can confine the range of the fridge in x direction
        """ Old
        W_fridge = FridgeParameter().W
        n_x_tmp = len(x_tmp)
        x_tmp_sorted = np.sort(x_tmp)
        x_tmp_min = x_tmp_sorted[int(n_x_tmp * 0.05)]  # robust to outliers
        x_tmp_max = x_tmp_sorted[int(n_x_tmp * 0.95)]  # robust to outliers
        width_tmp = x_tmp_max - x_tmp_min
        if width_tmp > W_fridge:
            x_tmp_min = x_tmp_sorted[int(n_x_tmp * 0.1)]  # robust to outliers
            x_tmp_max = x_tmp_sorted[int(n_x_tmp * 0.9)]  # robust to outliers
            width_tmp = x_tmp_max - x_tmp_min
            if width_tmp > W_fridge:
                return None
        t_min = x_tmp_max - 0.5 * W_fridge
        t_max = x_tmp_min + 0.5 * W_fridge
        """

        n_sample = 500
        data = data[np.random.choice(data.shape[0], n_sample, replace=False), :]

        def compute_loss(param: np.ndarray) -> float:
            t, angle = param
            x = b + (0.05 / np.cos(yaw)) + a * t
            y = -t
            model = self.instantiate_fridge_model(np.array([x, y, yaw, angle]))

            loss_constraint = 0.0
            if angle < np.pi * 0.75:
                loss_constraint += 10.0 * (np.pi * 0.75 - angle) ** 2
            if angle > 0.95 * np.pi:
                loss_constraint += 10.0 * (angle - np.pi) ** 2

            sdfs = []
            for link in model.links:
                if link != model.shelf:
                    sdfs.append(link.sdf)
            sdf = UnionSDF(sdfs)
            errors = sdf(data)
            loss = np.sum(errors**2)

            # as the bbo cannot take into accoutn the inequality constraint
            # we add penalty term to the loss
            # penalty_width_rate = (max(t_min - t, 0.0) + max(t - t_max, 0.0)) / W_fridge
            penalty_width_rate = 0.0  # NOTE: found that data_lower often contains the points of the shelf, so this penaly make estimation worse
            return (loss + loss_constraint) * (1.0 + 100 * penalty_width_rate)

        if self._next_initial_guess is None:
            x_guess = np.array([0.0, 2.6])
        else:
            x_guess = self._next_initial_guess
        ret_init_guess = minimize(lambda x: compute_loss(x), x_guess, method="Nelder-Mead")
        if not ret_init_guess.success:
            return None
        t, angle = ret_init_guess.x
        x = b + (0.05 / np.cos(yaw)) + a * t
        y = -t
        self._next_initial_guess = np.array([t, angle])
        return np.array([x, y, yaw, angle])
