import time
import warnings
from abc import ABC, abstractmethod

from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, set_random_seed

warnings.filterwarnings("ignore")

import copy
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

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
        pr2 = pr2_spec.get_robot_model(deepcopy=True, with_mesh=True)
        pr2.angle_vector(AV_INIT)

        base_co = Coordinates([base_pose[0], base_pose[1], 0.0])
        base_co.rotate(base_pose[2], "z")
        pr2.newcoords(base_co)  # This must come after reflect_kin_to_skrobot_model!!!
        pr2_spec.reflect_skrobot_model_to_kin(pr2)
        self.pr2 = pr2
        self.pr2_spec = pr2_spec
        self.planner = CoverlibMotionPlanner()
        self.relocation_order = relocation_order
        self.base_pose = base_pose
        self.final_target_pose = final_target_pose
        larm_reach_clf.set_base_pose(base_pose)

    def solve_grasp_plan(
        self, q_now: np.ndarray, obstacles_remain: List[CylinderSkelton]
    ) -> Optional[np.ndarray]:
        self.pr2_spec.set_skrobot_model_state(self.pr2, q_now)
        self.pr2_spec.reflect_skrobot_model_to_kin(self.pr2)
        co = self.pr2.l_gripper_tool_frame.copy_worldcoords()
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
        ret = solve_ik(pose_cst, coll_cst, lb, ub, q_seed=q_now, max_trial=1)

        if ret.success:
            return ret.q
        return None

    def solve_relocation_plan(
        self,
        obstacle_remove,
        obstacles_remain: List[CylinderSkelton],
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> Optional[Trajectory]:

        self.pr2_spec.set_skrobot_model_state(self.pr2, q_start)
        co_gripper_start = self.pr2.l_gripper_tool_frame.copy_worldcoords()
        assert isinstance(co_gripper_start, Coordinates)

        # compute cylinder attachment
        cylinder_pos = obstacle_remove.worldpos()
        relative_pos = co_gripper_start.inverse_transform_vector(cylinder_pos)
        offset = 0.025  # assuming that robot slightly lifted it up
        relative_pos[2] += offset
        pts = (
            create_cylinder_points(obstacle_remove.height, obstacle_remove.radius, 8) + relative_pos
        )
        radii = np.ones(pts.shape[0]) * 0.005
        attachement = SphereAttachmentSpec("l_gripper_tool_frame", pts.T, radii, False)

        # setup sdf
        sdf = get_fridge_model_sdf()
        for i, obstacle in enumerate(obstacles_remain):
            sdf.add(CylinderSDF(obstacle.radius, obstacle.height, Pose(obstacle.worldpos())))

        # setup problem
        coll_cst = self.pr2_spec.create_collision_const(attachements=(attachement,))
        coll_cst.set_sdf(sdf)
        assert coll_cst.is_valid(q_start)
        assert coll_cst.is_valid(q_goal)
        lb, ub = self.pr2_spec.angle_bounds()

        # confine the search space
        q_min = np.maximum(np.minimum(q_start, q_goal) - 0.3, lb)
        q_max = np.minimum(np.maximum(q_start, q_goal) + 0.3, ub)

        resolution = np.ones(7) * 0.03
        problem = Problem(q_start, q_min, q_max, q_goal, coll_cst, None, resolution)
        solver_config = OMPLSolverConfig(algorithm_range=0.1, shortcut=True, timeout=0.05)
        solver = OMPLSolver(solver_config)
        ret = solver.solve(problem)
        return ret.traj


class Action:
    ...


@dataclass
class Node(ABC):
    shared_context: SharedContext
    remaining_relocations: int
    q: np.ndarray  # configuration
    obstacles: List[CylinderSkelton]
    failure_count: int = 0

    _generator: Optional[Generator] = None
    parent: Optional[Tuple[Action, "Node"]] = None

    @property
    def is_open(self) -> bool:
        return self._generator is not None

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
        obstacle_remove: CylinderSkelton, obstacles: List[CylinderSkelton]
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
        ub = center + 0.5 * width_effective
        z = 1.07  # the value is fixed for task (check by actually sample task!)

        co_baseline = obstacle_remove.copy_worldcoords()
        z_offset = z - obstacle_remove.worldpos()[2]
        co_baseline.translate([0.0, 0.0, z_offset])
        n_max_iter = 10
        for _ in range(n_max_iter):
            co_cand = co_baseline.copy_worldcoords()
            pos = co_cand.worldpos()
            if np.any(pos[:2] < lb) or np.any(pos[:2] > ub):
                yield None
                continue
            yaw = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
            co_cand.rotate(yaw, "z")
            co_cand.translate([-CYLINDER_PREGRASP_OFFSET, 0.0, 0.0])
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
    path_to_pre_grasp: Trajectory
    q_grasp: np.ndarray


@dataclass
class RelocateAndHome(Action):
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

            description = np.hstack([pre_grasp_pose, self.shared_context.base_pose])
            solution = self.shared_context.planner.solve_motion_plan(self.obstacles, description)
            if solution is None:
                self.failure_count += 1
                yield None
                continue

            obstacle_idx = len(self.shared_context.relocation_order) - self.remaining_relocations
            obstacles_remain = [o for i, o in enumerate(self.obstacles) if i != obstacle_idx]

            q_pregrasp = solution.numpy()[-1]
            q_grasp = self.shared_context.solve_grasp_plan(q_pregrasp, obstacles_remain)
            if q_grasp is None:
                self.failure_count += 1
                yield None
                continue
            action = ReachAndGrasp(solution, q_grasp)
            next_node = BeforeRelocationNode(
                self.shared_context,
                self.remaining_relocations,
                q_grasp,
                copy.deepcopy(self.obstacles),
            )
            yield action, next_node
        return None

    def _get_pre_grasp_pose_gen(self) -> Generator[Optional[np.ndarray], None, None]:
        obstacle_idx = len(self.shared_context.relocation_order) - self.remaining_relocations
        obstacle_remove = self.obstacles[obstacle_idx]
        return self._sample_possible_pre_grasp_pose(obstacle_remove, self.obstacles)


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

            # 1. check if relocation's pre-grasp pose is reachable
            description = np.hstack([pre_grasp_pose, self.shared_context.base_pose])
            solution = self.shared_context.planner.solve_motion_plan(obstacles_new, description)
            if solution is None:
                print("motion plan failed")
                self.failure_count += 1
                yield None
                continue
            traj_to_go_home = Trajectory(solution.numpy()[::-1])

            # 2. solve IK
            obstacle_idx = len(self.shared_context.relocation_order) - self.remaining_relocations
            obstacles_remain = [o for i, o in enumerate(obstacles_new) if i != obstacle_idx]
            q_pregrasp = solution.numpy()[-1]
            q_grasp = self.shared_context.solve_grasp_plan(q_pregrasp, obstacles_remain)
            if q_grasp is None:
                self.failure_count += 1
                print("solve ik failed")
                yield None
                continue

            # 3. solve relocation
            obstacle_idx = len(self.shared_context.relocation_order) - self.remaining_relocations
            obstacle_pick = self.obstacles[obstacle_idx]
            obstacles_other = [o for i, o in enumerate(obstacles_new) if i != obstacle_idx]
            assert not np.allclose(self.q, Q_INIT)
            solution = self.shared_context.solve_relocation_plan(
                obstacle_pick,
                obstacles_other,
                self.q,
                q_grasp,
            )
            if solution is None:
                self.failure_count += 1
                print("solve relocation failed")
                yield None
                continue

            if self.remaining_relocations == 1:
                node_type = BeforeFinalReachNode
            else:
                node_type = BeforeRelocationNode

            node_new = node_type(
                self.shared_context,
                self.remaining_relocations - 1,
                Q_INIT,  # assuming that go back to the initial pose
                obstacles_new,
            )
            yield RelocateAndHome(solution, q_pregrasp, traj_to_go_home), node_new

    def _get_reloc_gen(
        self,
    ) -> Generator[Optional[Tuple[List[CylinderSkelton], np.ndarray]], None, None]:
        obstacle_idx = len(self.shared_context.relocation_order) - self.remaining_relocations
        obstacle_pick = self.obstacles[obstacle_idx]
        obstacles_remove_later = self.obstacles[obstacle_idx + 1 :]
        obstacles_remain = [
            o for i, o in enumerate(self.obstacles) if i not in self.shared_context.relocation_order
        ]

        n_budget = 1000
        region_box = get_fridge_model().regions[1].box

        center2d = region_box.worldpos()[:2]
        radius = obstacle_pick.radius
        lb = center2d - 0.5 * region_box.extents[:2] + radius
        ub = center2d + 0.5 * region_box.extents[:2] - radius

        # NOTE: obstacles_fixed for checking collision between rellocation target and other
        obstacles_fixed = obstacles_remove_later + obstacles_remain
        other_obstacles_pos = np.array([obs.worldpos()[:2] for obs in obstacles_fixed])
        other_obstacles_radius = np.array([obs.radius for obs in obstacles_fixed])
        pos2d_original = obstacle_pick.worldpos()[:2]
        pos2d_cands = np.random.uniform(lb, ub, (n_budget, 2))
        z = obstacle_pick.worldpos()[2]
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
            obstacle_pick_new = copy.deepcopy(obstacle_pick)  # do we really need deepcopy?
            obstacle_pick_new.newcoords(new_obs_co)

            # NOTE: obstacles_to_check for confirming that at least with this rellocation
            # except for future relocation, the target pose is valid.
            # So obstacles_remove_later is not included in the check.
            obstacles_to_check = [obstacle_pick_new] + obstacles_remain
            if not is_valid_target_pose(
                co_final_reach_target, obstacles_to_check, is_grasping=False
            ):
                print("target pose is not valid")
                yield None
                continue

            obstacles_new = [obstacle_pick_new] + obstacles_remove_later + obstacles_remain
            assert len(obstacles_new) == len(self.obstacles)

            # maybe creating gen here is not efficient, but for now
            gen = self._sample_possible_pre_grasp_pose(obstacle_pick_new, obstacles_new)
            try:
                pre_grasp_pose = next(gen)
            except StopIteration:
                print("pre-grasp pose generator failed")
                yield None
                continue
            if pre_grasp_pose is None:
                print("found a valid pre-grasp pose")
                yield None
                continue
            yield (copy.deepcopy(obstacles_new), pre_grasp_pose)

        assert False, "fuakc"
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


if __name__ == "__main__":
    np_seed = 0
    # np.random.seed(3)
    set_random_seed(0)
    tamp_problem = problem_single_object_blocking()
    task_param = tamp_problem.to_param()
    task = JskFridgeReachingTask.from_task_param(task_param)
    base_pose = task.description[4:]
    final_target_pose = task.description[:4]
    context = SharedContext([0], base_pose, final_target_pose)
    node_init = BeforeGraspNode(context, 1, Q_INIT, copy.deepcopy(task.world.get_obstacle_list()))

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    nodes = [node_init]
    goal = None
    for i in range(1000):
        print(f"iteration {i}")
        node_types = [n.__class__ for n in nodes]
        failure_counts = [n.failure_count for n in nodes]

        # select a node randomly that is open
        open_nodes = [n for n in nodes if n.is_open]
        node = np.random.choice(open_nodes)
        child = node.extend()
        if child is None:
            continue
        nodes.append(child)
        if isinstance(child, GoalNode):
            print("found a solution")
            goal = child
            break

    # backtrack
    assert goal is not None
    reverse_actions = []
    node = goal
    while True:
        if node.parent is None:
            break
        action, parent_node = node.parent
        reverse_actions.append(action)
        node = parent_node
    actions = reverse_actions[::-1]

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=False))

    # visualize goal
    v = PyrenderViewer()
    task.world.visualize(v)
    pr2 = PR2(use_tight_joint_limit=False)
    base_pose = task.description[-3:]
    pr2.translate(np.hstack([base_pose[:2], 0.0]))
    pr2.rotate(base_pose[2], "z")
    pr2.angle_vector(AV_INIT)
    v.add(pr2)
    v.show()

    input("press enter to continue")
    for action in actions:
        if isinstance(action, ReachAndGrasp):
            for q in action.path_to_pre_grasp.resample(100):
                context.pr2_spec.set_skrobot_model_state(pr2, q)
                time.sleep(0.01)
                v.redraw()
            input("press enter to continue")
            context.pr2_spec.set_skrobot_model_state(pr2, action.q_grasp)
            v.redraw()
            input("press enter to continue")

        if isinstance(action, RelocateAndHome):
            for q in action.path_relocate.resample(100):
                context.pr2_spec.set_skrobot_model_state(pr2, q)
                time.sleep(0.01)
                v.redraw()
            input("press enter to continue")

            context.pr2_spec.set_skrobot_model_state(pr2, action.q_pregrasp)
            v.redraw()
            input("press enter to continue")

            for q in action.path_to_home.resample(100):
                context.pr2_spec.set_skrobot_model_state(pr2, q)
                time.sleep(0.01)
                v.redraw()
            input("press enter to continue")

        if isinstance(action, FinalReach):
            for q in action.path_to_reach.resample(100):
                context.pr2_spec.set_skrobot_model_state(pr2, q)
                time.sleep(0.01)
                v.redraw()
            input("press enter to continue")
