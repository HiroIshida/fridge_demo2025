from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from hifuku.batch_network import BatchFCN
from hifuku.core import SolutionLibrary
from hifuku.domain import JSKFridge
from hifuku.script_utils import load_library
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from rpbench.articulated.pr2.jskfridge import JskFridgeVerticalReachingTask
from rpbench.articulated.vision import create_heightmap_z_slice
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


class ActionEdge:
    ...


class ReachAction(ActionEdge):
    ...


@dataclass
class MoveCanAction(ActionEdge):
    idx_can: int
    # detail
    start_pos: Optional[np.ndarray] = None
    target_pos: Optional[np.ndarray] = None


@dataclass
class MoveBaseAction(ActionEdge):
    # detail
    start_pose: np.ndarray
    target_pose: np.ndarray


class Node:
    state: np.ndarray
    parent: Optional["Node"]
    action: Optional[ActionEdge]


class FeasibilityCheckerBatchImageJit:
    ae_modeL_shared: torch.jit.ScriptModule

    def __init__(self, n_batch: int):
        lib: SolutionLibrary = load_library(JSKFridge, "cuda", postfix="0.2")
        self.dummy_encoded = torch.zeros(n_batch, 200).float().cuda()
        self.biases = torch.tensor(lib.biases).float().cuda()
        self.max_admissible_cost = lib.max_admissible_cost

        # ae part
        dummy_input = torch.zeros(n_batch, 1, 112, 112).float().cuda()
        traced = torch.jit.trace(lib.ae_model_shared.eval(), (dummy_input,))
        self.ae_model_shared = torch.jit.optimize_for_inference(traced)

        # vector part
        linear_list = []
        expander_list = []
        for pred in lib.predictors:
            linear_list.append(pred.linears)
            expander_list.append(pred.description_expand_linears)
        fcn_linears_batch = BatchFCN(linear_list).cuda()
        fcn_expanders_batch = BatchFCN(expander_list).cuda()

        class Tmp(nn.Module):
            def __init__(self, fcn_linears, fcn_expanders, n):
                super().__init__()
                self.fcn_linears = fcn_linears
                self.fcn_expanders = fcn_expanders
                self.n = n

            def forward(self, bottlenecks, descriptor):
                # bottlenecks: (n_batch, n_latent)
                n_batch = bottlenecks.shape[0]
                expanded = self.fcn_expanders(descriptor.unsqueeze(0))
                expanded_repeat = expanded.repeat(n_batch, 1, 1)
                bottlenecks_repeat = bottlenecks.unsqueeze(1).repeat(1, self.n, 1)
                concat = torch.cat((bottlenecks_repeat, expanded_repeat), dim=2)
                tmp = self.fcn_linears(concat)
                return tmp.squeeze(2)

            def cuda(self):
                self.fcn_linears.cuda()
                self.fcn_expanders.cuda()
                return super().cuda()

        tmp = Tmp(fcn_linears_batch, fcn_expanders_batch, len(lib.predictors)).cuda()
        dummy_input = (torch.zeros(n_batch, 200).float().cuda(), torch.zeros(7).float().cuda())
        traced = torch.jit.trace(tmp, dummy_input)
        self.batch_predictor = torch.jit.optimize_for_inference(traced)

        # warm up
        vector = np.random.randn(7).astype(np.float32)
        hmaps = [np.random.randn(112, 112).astype(np.float32) for _ in range(10)]
        for _ in range(10):
            self.infer(vector, hmaps)

    def infer(self, vector: np.ndarray, mat_lits: List[np.ndarray]):
        n_batch_actual = len(mat_lits)
        vector = torch.from_numpy(vector).float().cuda()

        mats = torch.stack([torch.from_numpy(mat).float() for mat in mat_lits]).unsqueeze(1).cuda()
        encoded = self.ae_model_shared.forward(mats)
        self.dummy_encoded[:n_batch_actual] = encoded

        costs = self.batch_predictor(self.dummy_encoded, vector)
        cost_calibrated = costs[:n_batch_actual] + self.biases
        min_costs, min_indices = torch.min(cost_calibrated, dim=1)
        return (
            min_costs.cpu().detach().numpy() < self.max_admissible_cost,
            min_indices.cpu().detach().numpy(),
        )


class TampSolver:
    def __init__(self):
        self._checker = FeasibilityCheckerBatchImageJit(20)

    def solve(self, task_param: np.ndarray):
        task_init = JskFridgeVerticalReachingTask.from_task_param(task_param)
        self._hypothetical_check(task_init)

    def _hypothetical_check(self, task: JskFridgeVerticalReachingTask) -> bool:
        region = get_fridge_model().regions[task.world.attention_region_index]
        obstacles = task.world.get_obstacle_list()
        n_obs = len(obstacles)
        hmap_list = []
        for n in range(n_obs, 0, -1):
            for comb in combinations(obstacles, n):
                lst = list(comb)
                hmap = create_heightmap_z_slice(region.box, lst, 112)
                hmap_list.append(hmap)
        feasibilities, _ = self._checker.infer(task.description, hmap_list)

    def is_feasible(self, task: JskFridgeVerticalReachingTask) -> bool:
        problem = task.export_problem()
        conf = OMPLSolverConfig(n_max_call=1000000, timeout=0.1, n_max_ik_trial=1000)
        solver = OMPLSolver(conf)
        ret = solver.solve(problem)
        return ret.traj is None


if __name__ == "__main__":
    node = PerceptionDebugNode("20250326_082328")
    detection = node.percept()
    ax = Axis()
    ax.translate([0.3, -0.1, 1.05])  # easy
    task_param = create_task_param(detection, ax.copy_worldcoords())
    solver = TampSolver()
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    solver.solve(task_param)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=False))

    debug = False
    if debug:
        v = detection.visualize()
        task = JskFridgeVerticalReachingTask.from_task_param(task_param)
        # ret = task.solve_default()

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
        v.add(ax)
        v.show()
        import time

        time.sleep(1000)
