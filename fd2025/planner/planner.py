import copy
from dataclasses import dataclass, field
from typing import ClassVar, Iterator, List, Optional

import numpy as np
from hifuku.domain import JSKFridge
from hifuku.script_utils import load_library
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

from fd2025.perception.perception_node import PerceptionDebugNode
from fd2025.planner.bridge import create_task_param
from fd2025.planner.inference import FeasibilityCheckerBatchImageJit


@dataclass
class TampSolution:
    @dataclass
    class RelocationPlan:
        traj_to_pregrasp: Optional[Trajectory] = None  # home -> pregrasp
        q_grasp: Optional[np.ndarray] = None
        q_relocate: Optional[np.ndarray] = None
        traj_to_home: Optional[Trajectory] = None  # q_relocate -> home

    traj_final_reach: Optional[Trajectory] = None
    relocation_seq: List[RelocationPlan] = field(default_factory=list)


class TampSolver:
    CYLINDER_PREGRASP_OFFSET: ClassVar[float] = 0.06

    def __init__(self):
        self._lib = load_library(JSKFridge, "cuda", postfix="0.2")
        conf = copy.deepcopy(JSKFridge.solver_config)
        conf.n_max_call *= 2  # ensure enough time
        self._mp_solver = OMPLSolver(conf)
        self._checker = FeasibilityCheckerBatchImageJit(self._lib, 30, 7)
        self._checker_single = FeasibilityCheckerBatchImageJit(self._lib, 1, 7)
        # Initialize region box as a member variable
        self._region_box = get_fridge_model().regions[1].box

        # setup specs
        self._pr2_spec = PR2LarmSpec(use_fixed_uuid=False)
        pr2 = self._pr2_spec.get_robot_model(deepcopy=False)
        pr2.angle_vector(AV_INIT)
        self._pr2_spec.reflect_kin_to_skrobot_model(pr2)

    def solve(self, task_param: np.ndarray) -> Optional[TampSolution]:
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
                assert False, "unmaintained branch"

        return self._hypothetical_obstacle_delete_check(task_init)

    def _hypothetical_obstacle_delete_check(
        self, task: JskFridgeReachingTask
    ) -> Optional[TampSolution]:

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
    ) -> Optional[TampSolution]:
        n_obs = len(obstacles)
        indices_move = list(set(range(n_obs)) - set(indices_remain))
        assert len(indices_move) == 1  # TODO: support multiple
        obstacles = copy.deepcopy(obstacles)
        remove_idx = indices_move[0]
        obstacle_remove = obstacles[remove_idx]

        pose_final_reaching_target = description[:4]
        co_final_reaching_target = Coordinates(pose_final_reaching_target[:3])
        co_final_reaching_target.rotate(pose_final_reaching_target[3], "z")

        tamp_solution = TampSolution()
        reloc_plan = TampSolution.RelocationPlan()

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
                    reloc_plan.traj_to_pregrasp = solution
                    reloc_plan.q_grasp = q_grasp
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
            if solution is not None:
                tamp_solution.traj_final_reach = solution

                # 2. post-relocate reachability check
                for pregrasp_cand_pose in self._sample_possible_pre_grasp_pose(
                    remove_idx, obstacles
                ):
                    description_tweak = description.copy()
                    description_tweak[:4] = pregrasp_cand_pose
                    solution = self.solve_motion_plan(obstacles, description_tweak)
                    if solution is not None:
                        q_grasp = self.solve_grasp_plan(solution._points[-1], remove_idx, obstacles)
                        if q_grasp is not None:
                            reloc_plan.q_relocate = q_grasp
                            reloc_plan.traj_to_home = solution
                            tamp_solution.relocation_seq.append(reloc_plan)
                            return tamp_solution
                        print("grasp pose is not reachable")
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

    def solve_relocation_plan(
        self, i_pick: int, obstacles: List[CylinderSkelton], q_start: np.ndarray, q_goal: np.ndarray
    ) -> Optional[Trajectory]:
        # compute gripper coords
        model = self._pr2_spec.get_robot_model(deepcopy=False)
        self._pr2_spec.set_skrobot_model_state(model, q_start)
        co_gripper_start = model.l_gripper_tool_frame.copy_worldcoords()
        assert isinstance(co_gripper_start, Coordinates)

        # compute cylinder attachment
        cylinder_pos = obstacles[i_pick].worldpos()
        relative_pos = co_gripper_start.inverse_transform_vector(cylinder_pos)
        offset = 0.025  # assuming that robot slightly lifted it up
        relative_pos[2] += offset
        pts = create_cylinder_points(cylinder_move.height, cylinder_move.radius, 8) + relative_pos
        radii = np.ones(pts.shape[0]) * 0.005
        attachement = SphereAttachmentSpec("l_gripper_tool_frame", pts.T, radii, False)

        # setup sdf
        sdf = get_fridge_model_sdf()
        for i, obstacle in enumerate(obstacles):
            if i == i_pick:
                continue
            sdf.add(CylinderSDF(obstacle.radius, obstacle.height, Pose(obstacle.worldpos())))

        # setup problem
        coll_cst = self._pr2_spec.create_collision_const(attachements=(attachement,))
        coll_cst.set_sdf(sdf)
        lb, ub = self._pr2_spec.angle_bounds()
        resolution = np.ones(7) * 0.03
        problem = Problem(q_start, lb, ub, q_goal, coll_cst, None, resolution)
        solver_config = OMPLSolverConfig(algorithm_range=0.3, shortcut=True, timeout=0.05)
        solver = OMPLSolver(solver_config)
        ret = solver.solve(problem)
        return ret.traj

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
        model = self._pr2_spec.get_robot_model(deepcopy=False)
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
    ret = solver.solve(task_param)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True, show_all=False))
    print(ret)

    import hashlib
    import pickle

    hash_value = hashlib.sha256(pickle.dumps(ret)).hexdigest()
    print(f"hash_value: {hash_value}")

    debug = True
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
        v.add(pr2)
        v.add(ax)
        v.show()
        import time

        # replay relocation trajectory
        for reloc_plan in ret.relocation_seq:
            for q in reloc_plan.traj_to_pregrasp.resample(100):
                spec.set_skrobot_model_state(pr2, q)
                v.redraw()
                time.sleep(0.03)
            input("press enter to continue")

            spec.set_skrobot_model_state(pr2, reloc_plan.q_grasp)
            v.redraw()
            input("press enter to continue")

            spec.set_skrobot_model_state(pr2, reloc_plan.q_relocate)
            v.redraw()
            input("press enter to continue")

            for q in reloc_plan.traj_to_home.resample(100)[::-1]:
                spec.set_skrobot_model_state(pr2, q)
                v.redraw()
                time.sleep(0.03)
            input("press enter to continue")

        # finally reach
        for q in ret.traj_final_reach.resample(100):
            spec.set_skrobot_model_state(pr2, q)
            v.redraw()
            time.sleep(0.03)

        time.sleep(1000)
