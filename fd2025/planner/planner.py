import contextlib
import warnings

warnings.filterwarnings("ignore")

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations
from typing import ClassVar, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from hifuku.domain import JSKFridge
from hifuku.script_utils import load_library
from matplotlib.patches import Circle, Rectangle
from plainmp.constraint import SphereAttachmentSpec
from plainmp.ik import solve_ik
from plainmp.ompl_solver import OMPLSolver, OMPLSolverConfig, set_random_seed
from plainmp.problem import Problem
from plainmp.psdf import CylinderSDF, Pose
from plainmp.robot_spec import PR2LarmSpec
from plainmp.trajectory import Trajectory
from rpbench.articulated.pr2.jskfridge import (
    AV_INIT,
    JskFridgeReachingTask,
    create_cylinder_points,
    larm_reach_clf,
)
from rpbench.articulated.vision import create_heightmap_z_slice
from rpbench.articulated.world.jskfridge import get_fridge_model, get_fridge_model_sdf
from rpbench.articulated.world.utils import CylinderSkelton
from skrobot.coordinates import Coordinates, rpy_angle
from skrobot.model.primitives import Axis
from skrobot.models import PR2
from skrobot.viewers import PyrenderViewer

from fd2025.planner.inference import FeasibilityCheckerBatchImageJit
from fd2025.planner.problem_set import problem_double_object2_blocking


def debug_plot_container(
    container_size: Tuple[float, float],
    container_pos: Tuple[float, float],
    obstacles_remove: List[CylinderSkelton],
    obstacles_remain: List[CylinderSkelton],
    target: np.ndarray,  #  [x, y, z, yaw]
) -> plt.Figure:
    fig, ax = plt.subplots()
    cx, cy = container_pos
    w, h = container_size
    ax.add_patch(
        Rectangle((cx - 0.5 * w, cy - 0.5 * h), w, h, fill=False, edgecolor="black", linewidth=2)
    )
    for cyl in obstacles_remove:
        px, py, _ = cyl.worldpos()
        ax.add_patch(Circle((px, py), cyl.radius, fill=False, edgecolor="blue", linewidth=1.5))
    for cyl in obstacles_remain:
        px, py, _ = cyl.worldpos()
        ax.add_patch(Circle((px, py), cyl.radius, fill=True, alpha=0.5))
    x, y, _, yaw = target
    dx, dy = np.cos(yaw) * 0.1, np.sin(yaw) * 0.1
    ax.arrow(x, y, dx, dy, head_width=0.01, length_includes_head=True, color="r")
    ax.arrow(x, y, -dy, dx, head_width=0.01, length_includes_head=True, color="g")
    ax.set_aspect("equal")
    eps = 0.1
    ax.set_xlim(cx - 0.5 * w - eps, cx + 0.5 * w + eps)
    ax.set_ylim(cy - 0.5 * h - eps, cy + 0.5 * h + eps)
    return fig


@contextlib.contextmanager
def temp_newcoords(obj, new_co):
    saved = obj.copy_worldcoords()
    obj.newcoords(new_co)
    try:
        yield
    finally:
        obj.newcoords(saved)


@dataclass
class TampSolution:
    @dataclass
    class RelocationPlan:
        traj_to_pregrasp: Optional[Trajectory] = None  # home -> pregrasp
        q_grasp: Optional[np.ndarray] = None
        traj_grasp_to_relocate: Optional[Trajectory] = None  # pregrasp -> q_relocate
        q_relocate: Optional[np.ndarray] = None
        traj_to_home: Optional[Trajectory] = None  # q_relocate -> home

    traj_final_reach: Optional[Trajectory] = None
    relocation_seq: List[RelocationPlan] = field(default_factory=list)


class RelocationPlanningStage(Enum):
    # assuming that relocate obstacle from A to B
    # stages for starting each planning (not finish them)
    MP_PREGRASP_A = auto()
    IK_GRASP_A = auto()
    MP_FINAL_REACh = auto()
    MP_PREGRASP_B = auto()
    IK_GRASP_B = auto()
    MP_RELOCATE = auto()
    SUCCESS = auto()


class TampSolverBase:
    CYLINDER_PREGRASP_OFFSET: ClassVar[float] = 0.06

    def __init__(self):
        # Initialize region box as a member variable
        self._region_box = get_fridge_model().regions[1].box

        # setup specs
        self._pr2_spec = PR2LarmSpec(use_fixed_uuid=False)
        pr2 = self._pr2_spec.get_robot_model(deepcopy=False)
        pr2.angle_vector(AV_INIT)
        self._pr2_spec.reflect_kin_to_skrobot_model(pr2)

        # others
        self.reloc_stage_history: List[
            Tuple[int, RelocationPlanningStage]
        ] = []  # int for recursion depth
        self.verbose = False

    def print(self, something):
        if self.verbose:
            print(something)

    def solve(self, task_param: np.ndarray) -> Optional[TampSolution]:
        task_init = JskFridgeReachingTask.from_task_param(task_param)

        target_pose, base_pose = task_init.description[:4], task_init.description[4:7]

        # FIXME: currently solve function assume that base_pose is fixed, so I
        # first set the base pose here and assume that the base pose is fixed
        base_co = Coordinates([base_pose[0], base_pose[1], 0.0])
        base_co.rotate(base_pose[2], "z")
        pr2 = self._pr2_spec.get_robot_model(deepcopy=False)
        pr2.newcoords(base_co)
        self._pr2_spec.reflect_skrobot_model_to_kin(pr2)

        larm_reach_clf.set_base_pose(base_pose)

        target_pose_xyzrpy = np.hstack([target_pose[:3], 0.0, 0.0, target_pose[3]])
        if not larm_reach_clf.predict(target_pose_xyzrpy):
            return None

        # check if solvable without any replacement
        co = Coordinates(task_init.description[:3])
        co.rotate(task_init.description[3], "z")
        obstacle_list = task_init.world.get_obstacle_list()
        if self.is_valid_target_pose(co, obstacle_list, is_grasping=False):
            self.print("valid target pose without any replacement")
            if self.is_feasible(obstacle_list, task_init.description):
                self.print("solvable without any replacement")
                assert False, "unmaintained branch"

        return self._hypothetical_obstacle_delete_check(task_init)

    def _hypothetical_obstacle_delete_check(
        self, task: JskFridgeReachingTask
    ) -> Optional[TampSolution]:
        self.reloc_stage_history = []  # reset

        obstacles = task.world.get_obstacle_list()
        tamp_solution = TampSolution()

        # consider removing single obstacle
        for i in range(len(obstacles)):
            indices_remain = list(set(range(len(obstacles))) - {i})
            obstacles_remain = [obstacles[i] for i in indices_remain]
            is_est_feasible = self.is_feasible(obstacles_remain, task.description)
            if is_est_feasible:
                obstacles_remove = [obstacles[i]]
                is_success = self._plan_obstacle_relocation(
                    task.description, obstacles_remove, obstacles_remain, tamp_solution
                )
                if is_success:
                    return tamp_solution

        # consider removing two obstacles (use heuristic that)
        for remove_pair in combinations(range(len(obstacles)), 2):
            indices_remain = list(set(range(len(obstacles))) - set(remove_pair))
            obstacles_remain = [obstacles[i] for i in indices_remain]
            is_est_feasible = self.is_feasible(obstacles_remain, task.description)
            if is_est_feasible:
                obstacles_remove = [obstacles[i] for i in remove_pair]
                is_success = self._plan_obstacle_relocation(
                    task.description, obstacles_remove, obstacles_remain, tamp_solution
                )
                if is_success:
                    return tamp_solution

        return None

    def _plan_obstacle_relocation(
        self,
        description: np.ndarray,
        obstacles_remove: List[CylinderSkelton],
        obstacles_remain: List[CylinderSkelton],
        tamp_solution: TampSolution,
    ) -> bool:

        obstacles = obstacles_remain + obstacles_remove
        obstacle_remove_here = obstacles_remove[0]  # pass obstacles_remove[1:] to recursion

        pose_final_reaching_target = description[:4]
        co_final_reaching_target = Coordinates(pose_final_reaching_target[:3])
        co_final_reaching_target.rotate(pose_final_reaching_target[3], "z")

        reloc_plan = TampSolution.RelocationPlan()

        # 1. Determine how to reach and grasp remove_idx obstacle
        pregrasp_pose = None
        create_heightmap_z_slice(self._region_box, obstacles, 112)
        for pregrasp_pose_cand in self._sample_possible_pre_grasp_pose(
            obstacle_remove_here, obstacles
        ):
            description_tweak = description.copy()
            description_tweak[:4] = pregrasp_pose_cand
            solution_relocation = self.solve_motion_plan(
                obstacles, description_tweak
            )  # check if reachable
            if solution_relocation is not None:
                pregrasp_pose = pregrasp_pose_cand
                q_final = solution_relocation._points[-1]
                q_grasp = self.solve_grasp_plan(q_final, obstacles_remain)
                if q_grasp is not None:
                    reloc_plan.traj_to_pregrasp = solution_relocation
                    reloc_plan.q_grasp = q_grasp
                    break
        if pregrasp_pose is None:
            self.print("1. not found good pre-grasp pose for remove_idx")
            return False

        obstacles_remove_later = obstacles_remove[1:]
        for relocation_target in self._sample_possible_relocation_target_pose(
            obstacle_remove_here, obstacles_remove_later, obstacles_remain, co_final_reaching_target
        ):

            # check relocation target is valid
            if len(obstacles_remove_later) > 0:
                cylinder = obstacles_remove_later[0]
                pos, r = cylinder.worldpos()[:2], cylinder.radius
                r_this = obstacle_remove_here.radius
                pos_target = relocation_target[:2]
                dist = np.linalg.norm(pos - pos_target)
                assert dist >= r + r_this, "relocation target is not valid"

            with temp_newcoords(obstacle_remove_here, Coordinates(relocation_target)):
                # 2. post-relocate feasibility check (inherently recursive)

                if len(obstacles_remove) == 1:
                    # The base case of the recursion
                    solution_relocation = self.solve_motion_plan(obstacles, description)
                    if solution_relocation is None:
                        self.print("2. (base case) post relocation motion planning is not feasible")
                        continue
                    tamp_solution.traj_final_reach = solution_relocation
                else:
                    # The recursive case
                    obstacles_remain_hypo = obstacles_remain + [obstacle_remove_here]
                    # debug_plot_container(
                    #     self._region_box.extents[:2],
                    #     self._region_box.worldpos()[:2],
                    #     obstacles_remove,
                    #     obstacles_remain_hypo,
                    #     pose_final_reaching_target,
                    # )
                    # plt.show()

                    is_success = self._plan_obstacle_relocation(
                        description, obstacles_remove_later, obstacles_remain_hypo, tamp_solution
                    )

                    if not is_success:
                        self.print(
                            f"2. (recursive case) post relocation motion planning is not feasible"
                        )
                        continue

                for pregrasp_cand_pose in self._sample_possible_pre_grasp_pose(
                    obstacle_remove_here, obstacles
                ):
                    # 3. post-relocate final-pregrasp reachability check
                    description_tweak = description.copy()
                    description_tweak[:4] = pregrasp_cand_pose
                    solution_gohome_reversed = self.solve_motion_plan(obstacles, description_tweak)
                    if solution_gohome_reversed is None:
                        self.print("3. post relocation motion plan is not feasible")
                        continue

                    # 3.1. post-relocate final-grasp check
                    q_grasp = self.solve_grasp_plan(
                        solution_gohome_reversed._points[-1], obstacles_remain
                    )
                    if q_grasp is None:
                        self.print("3.1 post relocatoin final grasp is not feasible")
                        continue

                    # 4. check relocation plan is feasible
                    solution_relocation = self.solve_relocation_plan(
                        obstacle_remove_here, obstacles_remain, reloc_plan.q_grasp, q_grasp
                    )
                    if solution_relocation is None:
                        self.print("4. relocation motion plan is not feasible")
                        continue

                    # OK! now, pack the solutions and return
                    reloc_plan.q_relocate = q_grasp
                    reloc_plan.traj_to_home = solution_gohome_reversed
                    reloc_plan.traj_grasp_to_relocate = solution_relocation
                    tamp_solution.relocation_seq.append(reloc_plan)
                    return True
        return False

    def _sample_possible_pre_grasp_pose(
        self, obstacle_remove: CylinderSkelton, obstacles: List[CylinderSkelton]
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

        co_baseline = obstacle_remove.copy_worldcoords()
        z_offset = z - obstacle_remove.worldpos()[2]
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
        obstacle_pick: CylinderSkelton,
        obstacles_remove_later: List[CylinderSkelton],
        obstacles_remain: List[CylinderSkelton],
        co_final_reaching_target,
        n_budget: int = 100,
    ) -> Iterator[np.ndarray]:
        center2d = self._region_box.worldpos()[:2]
        radius = obstacle_pick.radius
        lb = center2d - 0.5 * self._region_box.extents[:2] + radius
        ub = center2d + 0.5 * self._region_box.extents[:2] - radius

        # NOTE: obstacles_fixed for checking collision between rellocation target and other
        obstacles_fixed = obstacles_remove_later + obstacles_remain
        other_obstacles_pos = np.array([obs.worldpos()[:2] for obs in obstacles_fixed])
        other_obstacles_radius = np.array([obs.radius for obs in obstacles_fixed])
        pos2d_original = obstacle_pick.worldpos()[:2]
        pos2d_cands = np.random.uniform(lb, ub, (n_budget, 2))
        z = obstacle_pick.worldpos()[2]
        dists_from_original = np.linalg.norm(pos2d_cands - pos2d_original, axis=1)
        sorted_indices = np.argsort(dists_from_original)
        pos2d_cands = pos2d_cands[sorted_indices]

        # NOTE: obstacles_to_check for confirming that at least with this rellocation
        # except for future relocation, the target pose is valid.
        # So obstacles_remove_later is not included in the check.
        obstacles_to_check = [obstacle_pick] + obstacles_remain

        for pos2d in pos2d_cands:
            if other_obstacles_pos.size > 0:
                distances = np.linalg.norm(other_obstacles_pos - pos2d, axis=1)
                min_distances = distances - other_obstacles_radius - radius
                is_any_collision = np.any(min_distances < 0)
                if is_any_collision:
                    continue
            new_obs_co = Coordinates(np.hstack([pos2d, z]))
            with temp_newcoords(obstacle_pick, new_obs_co):
                if not self.is_valid_target_pose(
                    co_final_reaching_target, obstacles_to_check, is_grasping=False
                ):
                    continue
            yield new_obs_co.worldpos()

    def solve_relocation_plan(
        self,
        obstacle_remove,
        obstacles_remain: List[CylinderSkelton],
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> Optional[Trajectory]:
        # compute gripper coords
        model = self._pr2_spec.get_robot_model(deepcopy=False)
        self._pr2_spec.set_skrobot_model_state(model, q_start)
        co_gripper_start = model.l_gripper_tool_frame.copy_worldcoords()
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
        coll_cst = self._pr2_spec.create_collision_const(attachements=(attachement,))
        coll_cst.set_sdf(sdf)
        lb, ub = self._pr2_spec.angle_bounds()

        # confine the search space
        q_min = np.maximum(np.minimum(q_start, q_goal) - 0.3, lb)
        q_max = np.minimum(np.maximum(q_start, q_goal) + 0.3, ub)

        resolution = np.ones(7) * 0.03
        problem = Problem(q_start, q_min, q_max, q_goal, coll_cst, None, resolution)
        solver_config = OMPLSolverConfig(algorithm_range=0.1, shortcut=True, timeout=0.05)
        solver = OMPLSolver(solver_config)
        ret = solver.solve(problem)
        return ret.traj

    def solve_grasp_plan(
        self, q_now: np.ndarray, obstacles_remain: List[CylinderSkelton]
    ) -> Optional[np.ndarray]:
        # remove taget cylinder from the collision obstacls and check if the reach toward the
        # grasp position is feasible
        model = self._pr2_spec.get_robot_model(deepcopy=False)
        self._pr2_spec.set_skrobot_model_state(model, q_now)
        self._pr2_spec.reflect_skrobot_model_to_kin(model)
        co = model.l_gripper_tool_frame.copy_worldcoords()
        co.translate([self.CYLINDER_PREGRASP_OFFSET, 0.0, 0.0])

        sdf = get_fridge_model_sdf()
        for i, obstacle in enumerate(obstacles_remain):
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


class TampSolverCoverLib(TampSolverBase):
    def __init__(self):
        super().__init__()
        self._lib = load_library(JSKFridge, "cuda", postfix="0.1")
        conf = copy.deepcopy(JSKFridge.solver_config)
        conf.n_max_call *= 2  # ensure enough time
        self._mp_solver = OMPLSolver(conf)
        self._checker = FeasibilityCheckerBatchImageJit(self._lib, 30, 7)
        self._checker_single = FeasibilityCheckerBatchImageJit(self._lib, 1, 7)

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
            self.print("inference engine expected that the motion plan is not feasible!")
            return None

        obstacles_param = self.obstacles_to_obstacles_param(obstacles, self._region_box)
        world = JskFridgeReachingTask.get_world_type()(obstacles_param[: len(obstacles) * 4])
        task = JskFridgeReachingTask(world, description)
        problem = task.export_problem()
        ret = self._mp_solver.solve(problem, self._lib.init_solutions[traj_idx])
        if ret.traj is None:
            self.print(f"FF>> {task.to_task_param()}")
        return ret.traj


class TampSolverNaive(TampSolverBase):
    def __init__(self, timeout: float = 0.1):
        super().__init__()
        conf = OMPLSolverConfig(timeout=timeout)
        self._mp_solver = OMPLSolver(conf)

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


if __name__ == "__main__":
    np_seed = 3
    np.random.seed(np_seed)
    set_random_seed(0)
    # tamp_problem = problem_single_object_blocking_hard()
    tamp_problem = problem_double_object2_blocking()
    task_param = tamp_problem.to_param()

    task = JskFridgeReachingTask.from_task_param(task_param)
    solver = TampSolverCoverLib()
    # solver = TampSolverNaive(timeout=0.3)

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    ret = solver.solve(task_param)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=False))

    import hashlib
    import pickle

    print(ret)

    hash_value = hashlib.sha256(pickle.dumps(ret)).hexdigest()
    print(f"hash_value: {hash_value}")

    debug = False
    if debug:
        # v = detection.visualize()
        task = JskFridgeReachingTask.from_task_param(task_param)
        v = PyrenderViewer()
        task.world.visualize(v)
        pr2 = PR2(use_tight_joint_limit=False)
        pr2.angle_vector(AV_INIT)
        base_pose = task.description[-3:]
        pr2.translate(np.hstack([base_pose[:2], 0.0]))
        pr2.rotate(base_pose[2], "z")
        spec = PR2LarmSpec()
        ax = Axis()
        ax.newcoords(tamp_problem.target_co)
        v.add(pr2)
        v.add(ax)
        v.show()
        import time

        # replay relocation trajectory
        for reloc_plan in ret.relocation_seq:
            print("reach to obstacle")
            for q in reloc_plan.traj_to_pregrasp.resample(100):
                spec.set_skrobot_model_state(pr2, q)
                v.redraw()
                time.sleep(0.03)
            input("press enter to continue")

            print("consider IK offset")
            spec.set_skrobot_model_state(pr2, reloc_plan.q_grasp)
            v.redraw()
            input("press enter to continue")

            print("relocation planning")
            for q in reloc_plan.traj_grasp_to_relocate.resample(100):
                spec.set_skrobot_model_state(pr2, q)
                v.redraw()
                time.sleep(0.03)

            print("q reloc")
            spec.set_skrobot_model_state(pr2, reloc_plan.q_relocate)
            v.redraw()
            input("press enter to continue")

            print("go back to home")
            for q in reloc_plan.traj_to_home.resample(100)[::-1]:
                spec.set_skrobot_model_state(pr2, q)
                v.redraw()
                time.sleep(0.03)
            input("press enter to continue")

        # finally reach
        print("final reach")
        for q in ret.traj_final_reach.resample(100):
            spec.set_skrobot_model_state(pr2, q)
            v.redraw()
            time.sleep(0.03)

        time.sleep(1000)
