import copy
from dataclasses import dataclass
from typing import ClassVar, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
from hifuku.batch_network import BatchFCN
from hifuku.core import SolutionLibrary
from hifuku.domain import JSKFridge
from hifuku.script_utils import load_library
from plainmp.ik import solve_ik
from plainmp.ompl_solver import OMPLSolver, set_random_seed
from plainmp.psdf import CylinderSDF, Pose
from plainmp.robot_spec import PR2LarmSpec
from plainmp.trajectory import Trajectory
from rpbench.articulated.pr2.jskfridge import (
    AV_INIT,
    JskFridgeReachingTask,
    larm_reach_clf,
)
from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.jskfridge import get_fridge_model, get_fridge_model_sdf
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

    task_param = np.zeros(1 + JskFridgeReachingTask.get_world_type().N_MAX_OBSTACLES * 4 + 7)
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
    head = JskFridgeReachingTask.get_world_type().N_MAX_OBSTACLES * 4 + 1

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
    ae_model_shared: torch.jit.ScriptModule

    def __init__(self, lib: SolutionLibrary, n_batch: int):
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
        hmaps = [np.random.randn(112, 112).astype(np.float32) for _ in range(n_batch)]
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

    def infer_single(self, vector: np.ndarray, mat: np.ndarray):
        feasibilities, libtraj_idx = self.infer(vector, [mat])
        return feasibilities[0], libtraj_idx[0]


class TampSolver:
    CYLINDER_PREGRASP_OFFSET: ClassVar[float] = 0.06

    def __init__(self):
        self._lib = load_library(JSKFridge, "cuda", postfix="0.2")
        conf = copy.deepcopy(JSKFridge.solver_config)
        conf.n_max_call *= 2  # ensure enough time
        self._mp_solver = OMPLSolver(conf)
        self._checker = FeasibilityCheckerBatchImageJit(self._lib, 30)
        self._checker_single = FeasibilityCheckerBatchImageJit(
            self._lib, 1
        )  # for non-batch inference
        # Initialize region box as a member variable
        self._region_box = get_fridge_model().regions[1].box

        # setup specs
        self._pr2_spec = PR2LarmSpec(use_fixed_uuid=False)
        pr2 = self._pr2_spec.get_robot_model(deepcopy=False)
        pr2.angle_vector(AV_INIT)
        self._pr2_spec.reflect_kin_to_skrobot_model(pr2)

    def solve(self, task_param: np.ndarray):
        task_init = JskFridgeReachingTask.from_task_param(task_param)

        target_pose, base_pose = task_init.description[:4], task_init.description[4:7]
        larm_reach_clf.set_base_pose(base_pose)
        target_pose_xyzrpy = np.hstack([target_pose[:3], 0.0, 0.0, target_pose[3]])
        if not larm_reach_clf.predict(target_pose_xyzrpy):
            return None

        # check if solvable without any replacement
        co = Coordinates(task_init.description[:3])
        co.rotate(task_init.description[3], "z")
        obstacle_list = task_init.world.get_obstacle_list()
        if self.is_valid_target_pose(co, obstacle_list, is_grasping=False):
            print("valid target pose without any replacement")
            hmap_now = create_heightmap_z_slice(self._region_box, obstacle_list, 112)
            is_est_feasible, _ = self._checker_single.infer_single(task_init.description, hmap_now)
            if is_est_feasible:
                print("solvable without any replacement")
                return task_init

        return self._hypothetical_obstacle_delete_check(task_init)

    def _hypothetical_obstacle_delete_check(self, task: JskFridgeReachingTask) -> bool:

        obstacles = task.world.get_obstacle_list()

        # consider removing single obstacle
        for i in range(len(obstacles)):
            indices_remain = list(set(range(len(obstacles))) - {i})
            lst_remain = [obstacles[i] for i in indices_remain]
            hmap = create_heightmap_z_slice(self._region_box, lst_remain, 112)
            is_est_feasible, _ = self._checker_single.infer_single(task.description, hmap)
            if is_est_feasible:
                return self._plan_obstacle_relocation(task.description, obstacles, indices_remain)
        return None

    def _plan_obstacle_relocation(
        self, description: np.ndarray, obstacles: List[CylinderSkelton], indices_remain: List[int]
    ):
        n_obs = len(obstacles)
        indices_move = list(set(range(n_obs)) - set(indices_remain))
        assert len(indices_move) == 1  # TODO: support multiple
        obstacles = copy.deepcopy(obstacles)
        remove_idx = indices_move[0]
        obstacle_remove = obstacles[remove_idx]

        pose_final_reaching_target = description[:4]
        co_final_reaching_target = Coordinates(pose_final_reaching_target[:3])
        co_final_reaching_target.rotate(pose_final_reaching_target[3], "z")

        # find reacable pregrasp pose
        pregrasp_pose = None
        create_heightmap_z_slice(self._region_box, obstacles, 112)
        for pregrasp_pose_cand in self._sample_possible_pre_grasp_pose(remove_idx, obstacles):
            description_tweak = description.copy()
            description_tweak[:4] = pregrasp_pose_cand
            solution = self.solve_motion_plan(obstacles, description_tweak)
            if solution is not None:
                pregrasp_pose = pregrasp_pose_cand
                q_final = solution._points[-1]
                q_grasp = self.solve_grasp_plan(q_final, remove_idx, obstacles)
                if q_grasp is not None:
                    self._traj_to_pregrasp = solution
                    self._q_grasp = q_grasp
                    break
                print("grasp pose is not reachable")
        if pregrasp_pose is None:
            return None

        # determine relocation target
        for relocation_target in self._sample_possible_relocation_target_pose(
            remove_idx, obstacles, co_final_reaching_target
        ):
            # 1. post-relocate feasibility check
            obstacle_remove.newcoords(Coordinates(relocation_target))
            solution = self.solve_motion_plan(obstacles, description)
            if solution is None:
                continue

            # 2. post-relocate reachability check
            for pregrasp_cand_pose in self._sample_possible_pre_grasp_pose(remove_idx, obstacles):
                description_tweak = description.copy()
                description_tweak[:4] = pregrasp_cand_pose
                solution = self.solve_motion_plan(obstacles, description_tweak)
                if solution is not None:
                    q_grasp = self.solve_grasp_plan(solution._points[-1], remove_idx, obstacles)
                    if q_grasp is not None:
                        # 3. check if the grasp pose is reachable
                        self._traj_final_reach = solution
                        self._q_grasp = q_grasp
                        break
                    print("grasp pose is not reachable")
            return solution
        return None

    def _sample_possible_pre_grasp_pose(
        self, i_pick: int, obstacles: List[CylinderSkelton]
    ) -> Iterator[np.ndarray]:
        # ============================================================
        # >> DEPEND ON rpbench JSKFridgeTaskBase.sample_pose() method!!
        region = get_fridge_model().regions[1]
        center = region.box.worldpos()[:2]
        D, W, H = region.box.extents
        horizontal_margin = 0.08
        depth_margin = 0.03
        width_effective = np.array([D - 2 * depth_margin, W - 2 * horizontal_margin])
        center = region.box.worldpos()[:2]
        lb = center - 0.5 * width_effective
        ub = center + 0.5 * width_effective
        z = 1.07  # the value is fixed for task (check by actually sample task!)

        obstacle = obstacles[i_pick]
        co_baseline = obstacle.copy_worldcoords()
        z_offset = z - obstacle.worldpos()[2]
        co_baseline.translate([0.0, 0.0, z_offset])
        n_max_iter = 10
        for _ in range(n_max_iter):
            co_cand = co_baseline.copy_worldcoords()
            pos = co_cand.worldpos()
            if np.any(pos[:2] < lb) or np.any(pos[:2] > ub):
                continue
            yaw = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
            co_cand.rotate(yaw, "z")
            co_cand.translate([-self.CYLINDER_PREGRASP_OFFSET, 0.0, 0.0])
            is_reachable = larm_reach_clf.predict(
                co_cand
            )  # assuming that base pose is already set in solve()
            if is_reachable:
                if self.is_valid_target_pose(co_cand, obstacles, is_grasping=False):
                    yield np.hstack([co_cand.worldpos(), yaw])
        # << DEPEND ON rpbench JSKFridgeTaskBase.sample_pose() method!!
        # ============================================================

    def _sample_possible_relocation_target_pose(
        self,
        i_pick: int,
        obstacles: List[CylinderSkelton],
        co_final_reaching_target,
        n_budget: int = 100,
    ) -> Iterator[np.ndarray]:
        center2d = self._region_box.worldpos()[:2]
        radius = obstacles[i_pick].radius
        lb = center2d - 0.5 * self._region_box.extents[:2] + radius
        ub = center2d + 0.5 * self._region_box.extents[:2] - radius
        indices_remain = list(set(range(len(obstacles))) - {i_pick})
        other_obstacles_pos = np.array([obstacles[i].worldpos()[:2] for i in indices_remain])
        other_obstacles_radius = np.array([obstacles[i].radius for i in indices_remain])
        obstacle_pick = obstacles[i_pick]
        pos2d_original = obstacle_pick.worldpos()[:2]
        pos2d_cands = np.random.uniform(lb, ub, (n_budget, 2))
        z = obstacle_pick.worldpos()[2]
        dists_from_original = np.linalg.norm(pos2d_cands - pos2d_original, axis=1)
        sorted_indices = np.argsort(dists_from_original)
        pos2d_cands = pos2d_cands[sorted_indices]

        for pos2d in pos2d_cands:
            if other_obstacles_pos.size > 0:
                distances = np.linalg.norm(other_obstacles_pos - pos2d, axis=1)
                min_distances = distances - other_obstacles_radius - radius
                is_any_collision = np.any(min_distances < 0)
                if is_any_collision:
                    continue
            new_obs_co = Coordinates(np.hstack([pos2d, z]))
            obstacles[i_pick].newcoords(new_obs_co)
            if not self.is_valid_target_pose(
                co_final_reaching_target, obstacles, is_grasping=False
            ):
                continue
            yield new_obs_co.worldpos()

    def solve_motion_plan(
        self, obstacles: List[CylinderSkelton], description: np.ndarray
    ) -> Optional[Trajectory]:

        hmap_current = create_heightmap_z_slice(self._region_box, obstacles, 112)
        feasible, traj_idx = self._checker_single.infer_single(description, hmap_current)
        if not feasible:
            return None

        obstacles_param = self.obstacles_to_obstacles_param(obstacles, self._region_box)
        world = JskFridgeReachingTask.get_world_type()(obstacles_param[: len(obstacles) * 4])
        task = JskFridgeReachingTask(world, description)
        problem = task.export_problem()
        ret = self._mp_solver.solve(problem, self._lib.init_solutions[traj_idx])
        return ret.traj

    def solve_grasp_plan(
        self, q_now: np.ndarray, i_pick: int, obstacles: List[CylinderSkelton]
    ) -> Optional[np.ndarray]:
        # remove taget cylinder from the collision obstacls and check if the reach toward the
        # grasp position is feasible
        obstacles[i_pick]
        model = self._pr2_spec.get_robot_model()
        self._pr2_spec.set_skrobot_model_state(model, q_now)
        self._pr2_spec.reflect_skrobot_model_to_kin(model)
        co = model.l_gripper_tool_frame.copy_worldcoords()
        co.translate([self.CYLINDER_PREGRASP_OFFSET, 0.0, 0.0])

        sdf = get_fridge_model_sdf()
        for i, obstacle in enumerate(obstacles):
            if i == i_pick:
                continue
            sdf.add(CylinderSDF(obstacle.radius, obstacle.height, Pose(obstacle.worldpos())))
        coll_cst = self._pr2_spec.create_collision_const()
        coll_cst.set_sdf(sdf)
        yaw = rpy_angle(co.worldrot())[0][0]
        pose_goal = np.hstack([co.worldpos(), 0.0, 0.0, yaw])
        pose_cst = self._pr2_spec.create_gripper_pose_const(pose_goal)
        lb, ub = self._pr2_spec.angle_bounds()
        ret = solve_ik(pose_cst, None, lb, ub, q_seed=q_now, max_trial=1)
        if ret.success:
            return ret.q
        return None

    @staticmethod
    def obstacles_to_obstacles_param(obstacles: List[CylinderSkelton], region_box) -> np.ndarray:
        world_type = JskFridgeReachingTask.get_world_type()
        obstacles_param = np.zeros(world_type.N_MAX_OBSTACLES * 4)

        for j, obs in enumerate(obstacles):
            pos = obs.worldpos()
            region_pos = region_box.worldpos()
            H_region = region_box.extents[2]

            pos_relative = pos - region_pos
            pos_relative[2] += 0.5 * H_region - 0.5 * obs.height

            idx = j * 4
            obstacles_param[idx : idx + 2] = pos_relative[:2]  # x, y
            obstacles_param[idx + 2] = obs.height  # height
            obstacles_param[idx + 3] = obs.radius  # radius

        return obstacles_param

    @staticmethod
    def is_valid_target_pose(
        co: Coordinates, obstacles: List[CylinderSkelton], *, is_grasping: bool
    ) -> bool:
        assert not is_grasping, "currently not supported"
        sdf = get_fridge_model_sdf()
        for obs in obstacles:
            cylinder_sdf = CylinderSDF(obs.radius, obs.height, Pose(obs.worldpos()))
            sdf.add(cylinder_sdf)

        # ============================================================
        # >> DEPEND ON rpbench JSKFridgeTaskBase.sample_pose() method!!
        if sdf.evaluate(co.worldpos()) < 0.03:
            return False
        co_dummy = co.copy_worldcoords()
        co_dummy.translate([-0.07, 0.0, 0.0])

        if sdf.evaluate(co_dummy.worldpos()) < 0.04:
            return False
        co_dummy.translate([-0.07, 0.0, 0.0])
        if sdf.evaluate(co_dummy.worldpos()) < 0.04:
            return False
        # << DEPEND ON rpbench JSKFridgeTaskBase.sample_pose() method!!
        # ============================================================
        return True


if __name__ == "__main__":
    np_seed = 0
    np.random.seed(np_seed)
    set_random_seed(0)
    node = PerceptionDebugNode("20250326_082328")
    detection = node.percept()
    detection.cylinders = [detection.cylinders[2]]
    # detection.cylinders = []

    ax = Axis()
    ax.translate([0.3, -0.1, 1.05])  # easy
    task_param = create_task_param(detection, ax.copy_worldcoords())
    task = JskFridgeReachingTask.from_task_param(task_param)
    solver = TampSolver()
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    solver.solve(task_param)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=False))

    import hashlib
    import pickle

    md5_hash_of_task = hashlib.md5(pickle.dumps(task)).hexdigest()
    print(f"md5 hash of task: {md5_hash_of_task}")
    # print(task.to_task_param())

    debug = True
    if debug:
        v = detection.visualize()
        task = JskFridgeReachingTask.from_task_param(task_param)
        # ret = task.solve_default()
        # print(ret)

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
        spec = PR2LarmSpec()
        v.add(pr2)
        v.add(ax)
        v.show()
        import time

        # for q in solver._traj_to_pregrasp.resample(100):
        for q in solver._traj_final_reach.resample(100):
            spec.set_skrobot_model_state(pr2, q)
            v.redraw()
            time.sleep(0.1)

        time.sleep(1000)
