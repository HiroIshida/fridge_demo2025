import time
import warnings

warnings.filterwarnings("ignore")


import numpy as np
from plainmp.ompl_solver import set_random_seed

from fd2025.planner.planner import TampSolverCoverLib, TampSolverNaive
from fd2025.planner.problem_set import problem_single_object_blocking

if __name__ == "__main__":
    np_seed = 0
    set_random_seed(0)
    task_param = problem_single_object_blocking().to_param()

    use_coverlib = True
    if use_coverlib:
        solver = TampSolverCoverLib()
        filename = "result-coverlib.txt"
    else:
        solver = TampSolverNaive(timeout=0.5)
        filename = "result-naive.txt"

    results = []
    for _ in range(100):
        ts = time.time()
        ret = solver.solve(task_param)
        elapsed = np.inf if ret is None else time.time() - ts
        results.append(elapsed)
    success_rate = len([r for r in results if r < 1000.0]) / len(results)
    print(f"Success rate: {success_rate:.2f}")
    median = np.median(results)
    print(f"Median: {median:.2f}")

    with open(filename, "w") as f:
        f.write(f"Success rate: {success_rate:.2f}\n")
        f.write(f"Median: {median:.2f}\n")
        f.write("Elapsed times:\n")
        for r in results:
            f.write(f"{r:.2f}\n")
