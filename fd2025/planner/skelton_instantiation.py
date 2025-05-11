import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore")

import copy
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np
from hifuku.domain import JSKFridge
from hifuku.script_utils import load_library
from plainmp.ik import solve_ik
from plainmp.ompl_solver import OMPLSolver, set_random_seed
from plainmp.psdf import CylinderSDF, Pose
from plainmp.robot_spec import PR2LarmSpec
from plainmp.trajectory import Trajectory
from rpbench.articulated.pr2.jskfridge import (
    AV_INIT,
    Q_INIT,
    JskFridgeReachingTask,
    larm_reach_clf,
)
from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.jskfridge import get_fridge_model, get_fridge_model_sdf
from rpbench.articulated.world.utils import CylinderSkelton
from skrobot.coordinates import Coordinates, rpy_angle
from skrobot.model.robot_model import RobotModel

from fd2025.planner.inference import FeasibilityCheckerBatchImageJit
from fd2025.planner.problem_set import problem_single_object_blocking

# globals
CYLINDER_PREGRASP_OFFSET = 0.06


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


class MotionPlanerBase(ABC):
    @abstractmethod
    def is_feasible(self, obstacles: List[CylinderSkelton], description: np.ndarray) -> bool:
        ...

    @abstractmethod
    def solve_motion_plan(
        self, obstacles: List[CylinderSkelton], description: np.ndarray
    ) -> Optional[Trajectory]:
        ...

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


class CoverlibMotionPlanner(MotionPlanerBase):
    def __init__(self):
        self._lib = load_library(JSKFridge, "cuda", postfix="0.1")
        conf = copy.deepcopy(JSKFridge.solver_config)
        conf.n_max_call *= 2  # ensure enough time
        self._mp_solver = OMPLSolver(conf)
        self._checker_single = FeasibilityCheckerBatchImageJit(self._lib, 1, 7)
        self._region_box = get_fridge_model().regions[1].box

    def is_feasible(self, obstacles: List[CylinderSkelton], description: np.ndarray) -> bool:
        hmap_now = create_heightmap_z_slice(self._region_box, obstacles, 112)
        is_est_feasible, _ = self._checker_single.infer_single(description, hmap_now)
        return is_est_feasible

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


class SharedContext:
    pr2_spec: PR2LarmSpec
    pr2: RobotModel
    planner: MotionPlanerBase
    relocation_order: Tuple[int, ...]  # indices of relocation obstacles
    base_pose: np.ndarray
    final_target_pose: np.ndarray

    def __init__(
        self,
        relocation_order: Tuple[int, ...],
        base_pose: np.ndarray,
        final_target_pose: np.ndarray,
    ):
        pr2_spec = PR2LarmSpec(use_fixed_uuid=False)
        pr2 = pr2_spec.get_robot_model(deepcopy=False)
        pr2.angle_vector(AV_INIT)
        pr2_spec.reflect_kin_to_skrobot_model(pr2)
        self.pr2_spec = pr2_spec
        self.pr2 = pr2
        self.planner = CoverlibMotionPlanner()
        self.relocation_order = relocation_order
        self.base_pose = base_pose
        self.final_target_pose = final_target_pose
        larm_reach_clf.set_base_pose(base_pose)

    def solve_grasp_plan(
        self, q_now: np.ndarray, obstacles_remain: List[CylinderSkelton]
    ) -> Optional[np.ndarray]:
        # remove taget cylinder from the collision obstacls and check if the reach toward the
        # grasp position is feasible
        model = self.pr2_spec.get_robot_model(deepcopy=False)
        self.pr2_spec.set_skrobot_model_state(model, q_now)
        self.pr2_spec.reflect_skrobot_model_to_kin(model)
        co = model.l_gripper_tool_frame.copy_worldcoords()
        co.translate([CYLINDER_PREGRASP_OFFSET, 0.0, 0.0])

        sdf = get_fridge_model_sdf()
        for i, obstacle in enumerate(obstacles_remain):
            sdf.add(CylinderSDF(obstacle.radius, obstacle.height, Pose(obstacle.worldpos())))
        coll_cst = self.pr2_spec.create_collision_const()
        coll_cst.set_sdf(sdf)
        yaw = rpy_angle(co.worldrot())[0][0]
        pose_goal = np.hstack([co.worldpos(), 0.0, 0.0, yaw])
        pose_cst = self.pr2_spec.create_gripper_pose_const(pose_goal)
        lb, ub = self.pr2_spec.angle_bounds()
        ret = solve_ik(pose_cst, None, lb, ub, q_seed=q_now, max_trial=1)
        if ret.success:
            return ret.q
        return None


class Action:
    ...


@dataclass
class Node(ABC):
    shared_context: SharedContext
    remaining_relocations: int
    q: np.ndarray  # configuration
    obstacles: List[CylinderSkelton]
    failure_count: int = 0

    # children nodes
    _generator: Optional[Generator] = None
    children: List[Tuple[Action, "Node"]] = field(default_factory=list)

    def __post_init__(self):
        self._generator = self._get_action_gen()

    def extend(self) -> Optional[Tuple[Action, "Node"]]:
        assert self._generator is not None, "This node is already invalidated."
        try:
            action, next_node = next(self._generator)
            self.children.append((action, next_node))
            return action, next_node
        except StopIteration:
            self._generator = None
            return None

    @abstractmethod
    def _get_action_gen(self) -> Generator[Optional[Tuple[Action, "Node"]], None, None]:
        ...


@dataclass
class ReachAndGrasp(Action):
    path_to_pre_grasp: Trajectory
    q_grasp: np.ndarray


@dataclass
class BeforeGraspNode(Node):
    def _get_action_gen(self) -> Generator[Tuple[Action, Node], None, None]:
        get_fridge_model().regions[1].box
        pre_grasp_pose_gen = self._get_pre_grasp_pose_gen()

        for _ in range(20):
            pre_grasp_pose = next(pre_grasp_pose_gen)
            if pre_grasp_pose is None:
                # do not consider it as a failure
                continue

            description = np.hstack([pre_grasp_pose, self.shared_context.base_pose])
            solution_relocation = self.shared_context.planner.solve_motion_plan(
                self.obstacles, description
            )
            if solution_relocation is None:
                self.failure_count += 1
                return None
            q_grasp = self.shared_context.solve_grasp_plan(self.q, self.obstacles)
            if q_grasp is None:
                self.failure_count += 1
                return None
            action = ReachAndGrasp(solution_relocation, q_grasp)
            next_node = BeforeRelocationNode(
                self.shared_context,
                self.remaining_relocations - 1,
                q_grasp,
                copy.deepcopy(self.obstacles),
            )
            yield action, next_node
        return

    def _get_pre_grasp_pose_gen(self) -> Generator[Optional[np.ndarray], None, None]:
        obstacle_idx = len(self.shared_context.relocation_order) - self.remaining_relocations
        obstacle_remove = self.obstacles[obstacle_idx]

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

        co_baseline = obstacle_remove.copy_worldcoords()
        z_offset = z - obstacle_remove.worldpos()[2]
        co_baseline.translate([0.0, 0.0, z_offset])
        while True:
            co_cand = co_baseline.copy_worldcoords()
            pos = co_cand.worldpos()
            if np.any(pos[:2] < lb) or np.any(pos[:2] > ub):
                yield None
            yaw = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
            co_cand.rotate(yaw, "z")
            co_cand.translate([-CYLINDER_PREGRASP_OFFSET, 0.0, 0.0])
            is_reachable = larm_reach_clf.predict(
                co_cand
            )  # assuming that base pose is already set in solve()
            if is_reachable:
                if is_valid_target_pose(co_cand, self.obstacles, is_grasping=False):
                    yield np.hstack([co_cand.worldpos(), yaw])
        # << DEPEND ON rpbench JSKFridgeTaskBase.sample_pose() method!!
        # ============================================================


class BeforeRelocationNode(Node):
    def _get_action_gen(self) -> Generator[Optional[Tuple[Action, "Node"]], None, None]:
        ...


if __name__ == "__main__":
    np_seed = 0
    set_random_seed(0)
    tamp_problem = problem_single_object_blocking()
    task_param = tamp_problem.to_param()
    task = JskFridgeReachingTask.from_task_param(task_param)
    base_pose = task.description[-3:]
    final_target_pose = task.description[:3]
    context = SharedContext([1], base_pose, final_target_pose)
    node = BeforeGraspNode(context, 1, Q_INIT, copy.deepcopy(task.world.get_obstacle_list()))

    for _ in range(100):
        ret = node.extend()
        print(ret)
