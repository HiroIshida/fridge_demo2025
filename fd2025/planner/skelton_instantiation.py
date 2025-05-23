import time
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache

import graphviz
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, set_random_seed

warnings.filterwarnings("ignore")

import copy
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
from hifuku.domain import JSKFridge
from hifuku.script_utils import load_library
from plainmp.constraint import SphereAttachmentSpec
from plainmp.ik import solve_ik
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig
from plainmp.problem import Problem
from plainmp.psdf import CylinderSDF, Pose
from plainmp.robot_spec import PR2LarmSpec
from plainmp.trajectory import Trajectory
from rpbench.articulated.pr2.jskfridge import (
    AV_INIT,
    Q_INIT,
    JskFridgeReachingTask,
    create_cylinder_points,
    larm_reach_clf,
)
from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.jskfridge import get_fridge_model, get_fridge_model_sdf
from rpbench.articulated.world.utils import CylinderSkelton
from skrobot.coordinates import Coordinates, rpy_angle
from skrobot.model.robot_model import RobotModel
from skrobot.models import PR2
from skrobot.viewers import PyrenderViewer

from fd2025.planner.inference import FeasibilityCheckerBatchImageJit
from fd2025.planner.problem_set import problem_triple_object_blocking2

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
    if sdf.evaluate(co.worldpos()) < 0.02:
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


class NaiveMotionPlanner(MotionPlanerBase):
    def __init__(self, timeout: float = 0.5):
        super().__init__()
        conf = OMPLSolverConfig(timeout=timeout)
        self._mp_solver = OMPLSolver(conf)
        self._region_box = get_fridge_model().regions[1].box

    def is_feasible(self, obstacles: List[CylinderSkelton], description: np.ndarray) -> bool:
        traj = self.solve_motion_plan(obstacles, description)
        return traj is not None

    def solve_motion_plan(
        self, obstacles: List[CylinderSkelton], description: np.ndarray
    ) -> Optional[Trajectory]:
        obstacles_param = self.obstacles_to_obstacles_param(obstacles, self._region_box)
        world = JskFridgeReachingTask.get_world_type()(obstacles_param[: len(obstacles) * 4])
        task = JskFridgeReachingTask(world, description)
        problem = task.export_problem()
        ret = self._mp_solver.solve(problem)
        return ret.traj


@lru_cache(maxsize=None)
def get_motion_planner(use_coverlib: bool, timeout: Optional[float]) -> MotionPlanerBase:
    if use_coverlib:
        return CoverlibMotionPlanner()
    else:
        return NaiveMotionPlanner(timeout=timeout)


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
        use_coverlib: bool = True,
        timeout: Optional[float] = None,
    ):
        pr2_spec = PR2LarmSpec(spec_id="rpbench-pr2-jskfridge")
        pr2 = self.get_pr2()
        pr2.angle_vector(AV_INIT)

        base_co = Coordinates([base_pose[0], base_pose[1], 0.0])
        base_co.rotate(base_pose[2], "z")
        pr2.newcoords(base_co)  # This must come after reflect_kin_to_skrobot_model!!!
        pr2_spec.reflect_skrobot_model_to_kin(pr2)
        self.pr2 = pr2
        self.pr2_spec = pr2_spec
        self.planner = get_motion_planner(use_coverlib, timeout)
        self.relocation_order = relocation_order
        self.base_pose = base_pose
        self.final_target_pose = final_target_pose
        larm_reach_clf.set_base_pose(base_pose)

    @staticmethod
    @lru_cache(maxsize=1)
    def get_pr2() -> RobotModel:
        pr2_spec = PR2LarmSpec(spec_id="rpbench-pr2-jskfridge")
        pr2 = pr2_spec.get_robot_model(deepcopy=True, with_mesh=True)
        return pr2

    def solve_grasp_plan(
        self, q_now: np.ndarray, obstacles_fixed: List[CylinderSkelton]
    ) -> Optional[np.ndarray]:
        self.pr2_spec.set_skrobot_model_state(self.pr2, q_now)
        self.pr2_spec.reflect_skrobot_model_to_kin(self.pr2)
        co = self.pr2.l_gripper_tool_frame.copy_worldcoords()
        co.translate([CYLINDER_PREGRASP_OFFSET, 0.0, 0.0])

        sdf = get_fridge_model_sdf()
        for obstacle in obstacles_fixed:
            sdf.add(CylinderSDF(obstacle.radius, obstacle.height, Pose(obstacle.worldpos())))
        coll_cst = self.pr2_spec.create_collision_const()
        coll_cst.set_sdf(sdf)
        yaw = rpy_angle(co.worldrot())[0][0]
        pose_goal = np.hstack([co.worldpos(), 0.0, 0.0, yaw])
        pose_cst = self.pr2_spec.create_gripper_pose_const(pose_goal)
        lb, ub = self.pr2_spec.angle_bounds()
        ret = solve_ik(pose_cst, coll_cst, lb, ub, q_seed=q_now, max_trial=1)

        if ret.success:
            return ret.q
        return None

    def solve_relocation_plan(
        self,
        obstacle_relocate,
        obstacles_fixed: List[CylinderSkelton],
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> Optional[Trajectory]:

        self.pr2_spec.set_skrobot_model_state(self.pr2, q_start)
        co_gripper_start = self.pr2.l_gripper_tool_frame.copy_worldcoords()
        assert isinstance(co_gripper_start, Coordinates)

        # compute cylinder attachment
        cylinder_pos = obstacle_relocate.worldpos()
        relative_pos = co_gripper_start.inverse_transform_vector(cylinder_pos)
        offset = 0.025  # assuming that robot slightly lifted it up
        relative_pos[2] += offset
        pts = (
            create_cylinder_points(obstacle_relocate.height, obstacle_relocate.radius, 8)
            + relative_pos
        )
        radii = np.ones(pts.shape[0]) * 0.005
        attachement = SphereAttachmentSpec("l_gripper_tool_frame", pts.T, radii, False)

        # setup sdf
        sdf = get_fridge_model_sdf()
        for obstacle in obstacles_fixed:
            sdf.add(CylinderSDF(obstacle.radius, obstacle.height, Pose(obstacle.worldpos())))

        # setup problem
        coll_cst = self.pr2_spec.create_collision_const(
            attachements=(attachement,), reorder_spheres=False
        )
        coll_cst.set_sdf(sdf)

        if not coll_cst.is_valid(q_start):
            # NOTE: this should not happen, but just unfortunately sometimes happens
            # I'll return None for now
            return None

        if not coll_cst.is_valid(q_goal):
            # NOTE: this sometimes happens, because collision between attachment against
            # other is not cheked in the previous IK step
            return None
        lb, ub = self.pr2_spec.angle_bounds()

        # confine the search space
        q_min = np.maximum(np.minimum(q_start, q_goal) - 0.3, lb)
        q_max = np.minimum(np.maximum(q_start, q_goal) + 0.3, ub)

        resolution = np.ones(7) * 0.03
        problem = Problem(q_start, q_min, q_max, q_goal, coll_cst, None, resolution)
        solver_config = OMPLSolverConfig(algorithm_range=0.1, shortcut=True, timeout=0.03)
        solver = OMPLSolver(solver_config)
        ret = solver.solve(problem)
        return ret.traj


class Action:
    ...


class CompType(Enum):
    MP = 0
    IK = 1
    RELOC = 2


@dataclass
class Node(ABC):
    shared_context: SharedContext
    remaining_relocations: int
    q: np.ndarray  # configuration
    obstacles: List[CylinderSkelton]
    depth: int = 0
    failure_count: int = 0
    _generator: Optional[Generator] = None
    parent: Optional[Tuple[Action, "Node"]] = None
    failure_info_list: List[List[CompType]] = field(default_factory=list)

    def recored_failure(self, failure_info: Any):  # shoud I rename it?
        self.failure_info_list.append(failure_info)
        self.failure_count += 1
        # TODO: propagate the failure count to parent node??

    @property
    def is_open(self) -> bool:
        return self._generator is not None

    def get_relocate_idx(self) -> int:
        tmp = len(self.shared_context.relocation_order) - self.remaining_relocations
        return self.shared_context.relocation_order[tmp]

    def __post_init__(self):
        self._generator = self._get_action_gen()

    def extend(self) -> Optional["Node"]:
        assert self._generator is not None
        try:
            ret = next(self._generator)
            if ret is None:
                return None
            action, next_node = ret
            next_node.parent = (action, self)
            return next_node
        except StopIteration:
            self._generator = None
            return None

    @abstractmethod
    def _get_action_gen(self) -> Generator[Optional[Tuple[Action, "Node"]], None, None]:
        ...

    @staticmethod
    def _sample_possible_pre_grasp_pose(
        obstacle_relocate: CylinderSkelton, obstacles: List[CylinderSkelton]
    ) -> Generator[Optional[np.ndarray], None, None]:
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
        lb[0] -= 0.05  # gripper x-pos can be smaller than object
        ub = center + 0.5 * width_effective
        z = 1.07  # the value is fixed for task (check by actually sample task!)

        co_baseline = obstacle_relocate.copy_worldcoords()
        z_offset = z - obstacle_relocate.worldpos()[2]
        co_baseline.translate([0.0, 0.0, z_offset])
        n_max_iter = 10
        for _ in range(n_max_iter):
            co_cand = co_baseline.copy_worldcoords()
            pos = co_cand.worldpos()

            yaw = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
            co_cand.rotate(yaw, "z")
            co_cand.translate([-CYLINDER_PREGRASP_OFFSET, 0.0, 0.0])

            if np.any(pos[:2] < lb) or np.any(pos[:2] > ub):
                yield None
                continue

            is_reachable = larm_reach_clf.predict(
                co_cand
            )  # assuming that base pose is already set in solve()

            if is_reachable:
                if is_valid_target_pose(co_cand, obstacles, is_grasping=False):
                    yield np.hstack([co_cand.worldpos(), yaw])
        # << DEPEND ON rpbench JSKFridgeTaskBase.sample_pose() method!!
        # ============================================================
        return None


@dataclass
class ReachAndGrasp(Action):
    obj_idx: int
    path_to_pre_grasp: Trajectory
    q_grasp: np.ndarray


@dataclass
class RelocateAndHome(Action):
    obj_idx: int
    path_relocate: Trajectory
    q_pregrasp: np.ndarray
    path_to_home: Trajectory


@dataclass
class FinalReach(Action):
    path_to_reach: Trajectory


@dataclass
class BeforeGraspNode(Node):
    def _get_action_gen(self) -> Generator[Tuple[Action, Node], None, None]:
        get_fridge_model().regions[1].box
        pre_grasp_pose_gen = self._get_pre_grasp_pose_gen()

        for _ in range(20):
            try:
                pre_grasp_pose = next(pre_grasp_pose_gen)
            except StopIteration:
                self._generator = None
                return None
            if pre_grasp_pose is None:
                # do not consider it as a failure
                continue

            comp_history = []

            description = np.hstack([pre_grasp_pose, self.shared_context.base_pose])
            solution = self.shared_context.planner.solve_motion_plan(self.obstacles, description)
            comp_history.append(CompType.MP)
            if solution is None:
                self.recored_failure(comp_history)
                yield None
                continue

            obstacles_fixed = [
                o for i, o in enumerate(self.obstacles) if i != self.get_relocate_idx()
            ]
            q_pregrasp = solution.numpy()[-1]
            q_grasp = self.shared_context.solve_grasp_plan(q_pregrasp, obstacles_fixed)
            comp_history.append(CompType.IK)
            if q_grasp is None:
                self.recored_failure(comp_history)
                yield None
                continue
            action = ReachAndGrasp(self.get_relocate_idx(), solution, q_grasp)
            next_node = BeforeRelocationNode(
                self.shared_context,
                self.remaining_relocations,
                q_grasp,
                copy.deepcopy(self.obstacles),
                self.depth + 1,
            )
            yield action, next_node
        return None

    def _get_pre_grasp_pose_gen(self) -> Generator[Optional[np.ndarray], None, None]:
        obstacle_relocate = self.obstacles[self.get_relocate_idx()]
        return self._sample_possible_pre_grasp_pose(obstacle_relocate, self.obstacles)


class BeforeRelocationNode(Node):
    def _get_action_gen(self) -> Generator[Optional[Tuple[Action, "Node"]], None, None]:

        gen_reloc = self._get_reloc_gen()

        while True:
            try:
                reloc = next(gen_reloc)
            except StopIteration:
                self._generator = None
                return None
            if reloc is None:
                yield None  # don't consider it as a failure
                continue

            obstacles_new, pre_grasp_pose = reloc
            comp_history = []

            # 1. check if relocation's pre-grasp pose is reachable
            description = np.hstack([pre_grasp_pose, self.shared_context.base_pose])
            solution = self.shared_context.planner.solve_motion_plan(obstacles_new, description)
            comp_history.append(CompType.MP)
            if solution is None:
                self.recored_failure(comp_history)
                yield None
                continue
            traj_to_go_home = Trajectory(solution.numpy()[::-1])

            # 2. solve IK
            obstacles_fixed = [
                o for i, o in enumerate(obstacles_new) if i != self.get_relocate_idx()
            ]
            q_pregrasp = solution.numpy()[-1]
            q_grasp = self.shared_context.solve_grasp_plan(q_pregrasp, obstacles_fixed)
            comp_history.append(CompType.IK)
            if q_grasp is None:
                self.recored_failure(comp_history)
                yield None
                continue

            # 3. solve relocation
            obstacle_relocate = self.obstacles[self.get_relocate_idx()]
            assert not np.allclose(self.q, Q_INIT)
            solution = self.shared_context.solve_relocation_plan(
                obstacle_relocate,
                obstacles_fixed,
                self.q,
                q_grasp,
            )
            comp_history.append(CompType.RELOC)
            if solution is None:
                self.recored_failure(comp_history)
                yield None
                continue

            if self.remaining_relocations == 1:
                node_type = BeforeFinalReachNode
            else:
                node_type = BeforeGraspNode

            node_new = node_type(
                self.shared_context,
                self.remaining_relocations - 1,
                Q_INIT,  # assuming that go back to the initial pose
                obstacles_new,
                self.depth + 1,
            )
            yield RelocateAndHome(
                self.get_relocate_idx(), solution, q_pregrasp, traj_to_go_home
            ), node_new

    def _get_reloc_gen(
        self,
    ) -> Generator[Optional[Tuple[List[CylinderSkelton], np.ndarray]], None, None]:
        obstacle_relocate = self.obstacles[self.get_relocate_idx()]
        obstacles_relocate_later = self.obstacles[self.get_relocate_idx() + 1 :]
        obstacles_fixed = [o for i, o in enumerate(self.obstacles) if i != self.get_relocate_idx()]

        n_budget = 1000
        region_box = get_fridge_model().regions[1].box

        center2d = region_box.worldpos()[:2]
        radius = obstacle_relocate.radius
        lb = center2d - 0.5 * region_box.extents[:2] + radius
        ub = center2d + 0.5 * region_box.extents[:2] - radius

        # NOTE: obstacles_fixed for checking collision between rellocation target and other
        other_obstacles_pos = np.array([obs.worldpos()[:2] for obs in obstacles_fixed])
        other_obstacles_radius = np.array([obs.radius for obs in obstacles_fixed])
        pos2d_original = obstacle_relocate.worldpos()[:2]
        pos2d_cands = np.random.uniform(lb, ub, (n_budget, 2))
        z = obstacle_relocate.worldpos()[2]
        dists_from_original = np.linalg.norm(pos2d_cands - pos2d_original, axis=1)
        # sorted_indices = np.argsort(dists_from_original)
        # pos2d_cands = pos2d_cands[sorted_indices]

        co_final_reach_target = Coordinates(self.shared_context.final_target_pose[:3])
        co_final_reach_target.rotate(self.shared_context.final_target_pose[3], "z")

        for i, pos2d in enumerate(pos2d_cands):
            if other_obstacles_pos.size > 0:
                distances = np.linalg.norm(other_obstacles_pos - pos2d, axis=1)
                min_distances = distances - other_obstacles_radius - radius
                is_any_collision = np.any(min_distances < 0)
                if is_any_collision:
                    continue

            new_obs_co = Coordinates(np.hstack([pos2d, z]))
            obstacle_relocate_new = copy.deepcopy(obstacle_relocate)  # do we really need deepcopy?
            obstacle_relocate_new.newcoords(new_obs_co)
            obstacles_new = copy.deepcopy(self.obstacles)  # TODO: inefficient
            obstacles_new[self.get_relocate_idx()] = obstacle_relocate_new

            # NOTE: obstacles_to_check for confirming that at least with this rellocation
            # except for future relocation, the target pose is valid.
            # So obstacles_relocate_later is not included in the check.
            future_relocate_indices = self.shared_context.relocation_order[
                -self.remaining_relocations + 1 :
            ]
            obstacles_to_check = [
                o for i, o in enumerate(obstacles_new) if i not in future_relocate_indices
            ]
            if not is_valid_target_pose(
                co_final_reach_target, obstacles_to_check, is_grasping=False
            ):
                yield None
                continue

            # maybe creating gen here is not efficient, but for now
            gen = self._sample_possible_pre_grasp_pose(obstacle_relocate_new, obstacles_new)
            try:
                pre_grasp_pose = next(gen)
            except StopIteration:
                yield None
                continue
            if pre_grasp_pose is None:
                yield None
                continue
            yield (copy.deepcopy(obstacles_new), pre_grasp_pose)

        return None


class BeforeFinalReachNode(Node):
    def _get_action_gen(self) -> Generator[Optional[Tuple[Action, "Node"]], None, None]:
        description = np.hstack(
            [self.shared_context.final_target_pose, self.shared_context.base_pose]
        )
        solution = self.shared_context.planner.solve_motion_plan(self.obstacles, description)
        if solution is None:
            yield None
            return  # close the generator
        q_final = solution.numpy()[-1]
        yield FinalReach(solution), GoalNode(self.shared_context, 0, q_final, self.obstacles)
        return None  # close the generator


class GoalNode(Node):
    def _get_action_gen(self) -> Generator[Optional[Tuple[Action, "Node"]], None, None]:
        yield None


def setup_cache() -> None:
    get_motion_planner(True, None)  # cache
    SharedContext.get_pr2()  # cache
    spec = PR2LarmSpec(spec_id="rpbench-pr2-jskfridge")
    spec.get_kin()
    spec.get_robot_model(deepcopy=False)


def instantiate_skelton(
    obstacles: List[CylinderSkelton],
    base_pose: np.ndarray,
    final_target_pose: np.ndarray,
    reloc_order_list: List[Tuple[int, ...]],
    p_exploit: float = 0.5,
    max_iter: int = 1000,
    use_coverlib: bool = True,
) -> Optional[Tuple[Tuple[Action, ...], Node, Node, List[Node], Tuple[int, ...]]]:
    forest = []  # [(node_init, [node_init], relocation_order), ...]
    all_nodes = []
    for reloc_order in reloc_order_list:
        context = SharedContext(
            reloc_order,
            base_pose,
            final_target_pose,
            use_coverlib=use_coverlib,
        )
        remaining_relocations = len(reloc_order)

        if remaining_relocations == 0:
            node_init = BeforeFinalReachNode(context, 0, Q_INIT, obstacles)
        else:
            node_init = BeforeGraspNode(context, remaining_relocations, Q_INIT, obstacles)

        forest.append((node_init, [node_init], reloc_order))
        all_nodes.append(node_init)

    for i_iter in range(max_iter):
        open_nodes = []
        for (init_node, nodes, reloc_order) in forest:
            open_nodes.extend([n for n in nodes if n.is_open])

        if len(open_nodes) == 0:
            print("No open nodes remain. All search trees are exhausted.")
            return None

        do_exploit = np.random.uniform() < p_exploit
        if do_exploit:
            max_depth = max(n.depth for n in open_nodes)
            max_depth_nodes = [n for n in open_nodes if n.depth == max_depth]
            best_failure_count = min(n.failure_count for n in max_depth_nodes)
            candidate_nodes = [n for n in max_depth_nodes if n.failure_count == best_failure_count]
            node = np.random.choice(candidate_nodes)
        else:
            node = np.random.choice(open_nodes)

        child = node.extend()
        if child is None:
            continue

        for (init_node, nodes, reloc_order) in forest:
            if child.shared_context is node.shared_context:
                nodes.append(child)
                break
        all_nodes.append(child)

        if isinstance(child, GoalNode):
            print("Found a solution!!")
            actions = []
            goal_node = child
            current = goal_node
            while current.parent is not None:
                action, parent_node = current.parent
                actions.append(action)
                current = parent_node
            actions.reverse()

            used_reloc_order = child.shared_context.relocation_order
            node_init_for_this = None
            for (init_node, nodes, reloc_order) in forest:
                if reloc_order == used_reloc_order:
                    node_init_for_this = init_node
                    break

            return (tuple(actions), node_init_for_this, goal_node, all_nodes)

    print("No solution found under max_iter limit.")
    return None


def visualize_search_graph(nodes, filename="search_graph", view=True):

    COMP_TYPE_COLOR_MAP = {
        "MP": "red",
        "IK": "green",
        "RELOC": "orange",
    }

    def get_comp_types_from_action(action) -> list:
        if isinstance(action, ReachAndGrasp):
            return [CompType.MP, CompType.IK]
        elif isinstance(action, RelocateAndHome):
            return [CompType.MP, CompType.IK, CompType.RELOC]
        elif isinstance(action, FinalReach):
            return [CompType.MP]
        else:
            return []

    dot = graphviz.Digraph(name="search_graph", format="pdf")
    dot.attr(rankdir="LR")
    num_mp = 0

    node_id_map = {}
    for i, node in enumerate(nodes):
        node_id_map[id(node)] = i

    for node in nodes:
        i = node_id_map[id(node)]
        label = f"Node[{i}]"
        dot.node(
            f"node_{i}",
            label=label,
            shape="box",
            style="filled",
            color="lightgray",
        )

    comp_node_global_idx = 0
    fail_node_global_idx = 0

    for node in nodes:
        start_node_id = f"node_{node_id_map[id(node)]}"
        for comp_seq in node.failure_info_list:
            prev_id = start_node_id
            for comp_type in comp_seq:
                if comp_type == CompType.MP:
                    num_mp += 1

                comp_node_id = f"comp_{comp_node_global_idx}"
                comp_node_global_idx += 1
                dot.node(
                    comp_node_id,
                    shape="circle",
                    style="filled",
                    fixedsize="true",
                    width="0.4",
                    height="0.4",
                    label="",
                    color=COMP_TYPE_COLOR_MAP[comp_type.name],
                )
                dot.edge(prev_id, comp_node_id)
                prev_id = comp_node_id
            fail_node_id = f"fail_{fail_node_global_idx}"
            fail_node_global_idx += 1
            dot.node(fail_node_id, label="", shape="diamond", color="black")
            dot.edge(prev_id, fail_node_id)

    for child in nodes:
        if child.parent is None:
            continue
        action, parent_node = child.parent
        comp_types = get_comp_types_from_action(action)
        if not comp_types:
            dot.edge(f"node_{node_id_map[id(parent_node)]}", f"node_{node_id_map[id(child)]}")
            continue

        prev_id = f"node_{node_id_map[id(parent_node)]}"
        for comp_type in comp_types:
            if comp_type == CompType.MP:
                num_mp += 1
            comp_node_id = f"comp_{comp_node_global_idx}"
            comp_node_global_idx += 1
            dot.node(
                comp_node_id,
                shape="circle",
                style="filled",
                color=COMP_TYPE_COLOR_MAP[comp_type.name],
                fixedsize="true",
                width="0.4",
                height="0.4",
                label="",
            )
            dot.edge(prev_id, comp_node_id)
            prev_id = comp_node_id
        dot.edge(prev_id, f"node_{node_id_map[id(child)]}")

    dot.render(filename, view=view)
    print(f"num_mp: {num_mp}")
    return dot


def solve_tamp(
    obstacles: List[CylinderSkelton],
    base_pose: np.ndarray,
    final_target_pose: np.ndarray,
    p_exploit: float = 0.5,
    max_iter: int = 1000,
    use_coverlib: bool = True,
    timeout: Optional[float] = None,
) -> Optional[
    Tuple[Tuple[Action, ...], Node, Node, List[Node]]
]:  # solution, initial node, goal node, nodes

    planner = get_motion_planner(use_coverlib, timeout)

    for n_relocate in range(len(obstacles) + 1):
        indices = np.arange(len(obstacles))
        for relocate_indices_comb in combinations(indices, n_relocate):
            print(f"relocate_indices_comb: {relocate_indices_comb}")
            obstacles_hypo = [obstacles[i] for i in indices if i not in relocate_indices_comb]
            est_feasible = planner.is_feasible(
                obstacles_hypo,
                np.hstack([final_target_pose, base_pose]),
            )
            if est_feasible:
                relocate_indices_list = list(permutations(relocate_indices_comb))
                ret = instantiate_skelton(
                    obstacles,
                    base_pose,
                    final_target_pose,
                    relocate_indices_list,
                    p_exploit=p_exploit,
                    max_iter=max_iter,
                    use_coverlib=use_coverlib,
                )
                if ret is not None:
                    actions, node_init, node_goal, nodes = ret
                    return actions, node_init, node_goal, nodes
                print(f"failed to find a solution")
            else:
                print(f"estimated infeasible")
    return None


if __name__ == "__main__":
    np_seed = 0
    np.random.seed(5)
    set_random_seed(0)
    tamp_problem = problem_triple_object_blocking2()

    # tamp_problem = problem_triple_object_blocking()
    task_param = tamp_problem.to_param()
    task = JskFridgeReachingTask.from_task_param(task_param)
    base_pose = task.description[4:]
    final_target_pose = task.description[:4]
    setup_cache()

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    ret = solve_tamp(
        task.world.get_obstacle_list(),
        base_pose,
        final_target_pose,
    )

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=False))

    actions, node_init, node_goal, nodes = ret

    visualize_search_graph(nodes, "search_graph", view=True)

    # visualize goal
    v = PyrenderViewer()
    obj_handle_list = task.world.visualize(v)
    pr2 = PR2(use_tight_joint_limit=False)
    base_pose = task.description[-3:]
    pr2.translate(np.hstack([base_pose[:2], 0.0]))
    pr2.rotate(base_pose[2], "z")
    pr2.angle_vector(AV_INIT)
    v.add(pr2)
    v.show()
    pr2_spec = PR2LarmSpec(use_fixed_spec_id=False)

    input("press enter to continue")
    for action in actions:
        if isinstance(action, ReachAndGrasp):
            for q in action.path_to_pre_grasp.resample(100):
                pr2_spec.set_skrobot_model_state(pr2, q)
                time.sleep(0.01)
                v.redraw()
            input("press enter to continue")
            pr2_spec.set_skrobot_model_state(pr2, action.q_grasp)
            v.redraw()
            input("press enter to continue")
            obj_handle = obj_handle_list[action.obj_idx]
            pr2.larm_end_coords.assoc(obj_handle)

        if isinstance(action, RelocateAndHome):
            for q in action.path_relocate.resample(100):
                pr2_spec.set_skrobot_model_state(pr2, q)
                time.sleep(0.01)
                v.redraw()
            input("press enter to continue")
            obj_handle = obj_handle_list[action.obj_idx]
            pr2.larm_end_coords.dissoc(obj_handle)

            pr2_spec.set_skrobot_model_state(pr2, action.q_pregrasp)
            v.redraw()
            input("press enter to continue")

            for q in action.path_to_home.resample(100):
                pr2_spec.set_skrobot_model_state(pr2, q)
                time.sleep(0.01)
                v.redraw()
            input("press enter to continue")

        if isinstance(action, FinalReach):
            for q in action.path_to_reach.resample(100):
                pr2_spec.set_skrobot_model_state(pr2, q)
                time.sleep(0.01)
                v.redraw()
            input("press enter to continue")
