import warnings

warnings.filterwarnings("ignore")

import time

import numpy as np
from plainmp.ompl_solver import set_random_seed
from rpbench.articulated.pr2.jskfridge import JskFridgeReachingTask

from fd2025.planner.problem_set import (
    problem_double_object2_blocking,
    problem_single_object_blocking,
    problem_triple_object_blocking,
)
from fd2025.planner.skelton_instantiation import instantiate_skelton, build_shared_context

if __name__ == "__main__":
    use_coverlib = True
    np_seed = 0
    set_random_seed(0)
    tamp_problem = problem_single_object_blocking()
    tamp_problem = problem_double_object2_blocking()
    tamp_problem = problem_triple_object_blocking()
    task_param = tamp_problem.to_param()
    task = JskFridgeReachingTask.from_task_param(task_param)
    base_pose = task.description[4:]
    final_target_pose = task.description[:4]
    build_shared_context((0, 1, 2), base_pose, final_target_pose, use_coverlib=use_coverlib)

    elapsed_list = []
    for _ in range(100):
        try:
            print("start")
            ts = time.time()
            actions, node_init, node_goal = instantiate_skelton(
                task.world.get_obstacle_list(),
                base_pose,
                final_target_pose,
                relocation_order=(0, 1, 2),
                use_coverlib=use_coverlib
            )
            elapsed = time.time() - ts
            print(f"elapsed time: {elapsed:.4f} seconds")
            elapsed_list.append(elapsed)
        except:
            pass

    median = np.median(elapsed_list)
    print(f"median elapsed time: {median:.4f} seconds")
    percentile_75 = np.percentile(elapsed_list, 75)
    print(f"75th percentile elapsed time: {percentile_75:.4f} seconds")
