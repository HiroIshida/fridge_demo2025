import numpy as np
from hifuku.domain import JSKFridge
from hifuku.script_utils import load_library
from rpbench.articulated.pr2.jskfridge import JskFridgeReachingTask

from fd2025.planner.planner import FeasibilityCheckerBatchImageJit, Transform2d


def test_transform2d():
    t12 = Transform2d(np.array([1, 2]), 0.8)
    t21 = t12.inv()
    ident = t12 * t21
    assert np.allclose(ident.trans, [0, 0])
    assert np.allclose(ident.rot, 0)

    ident = t21 * t12
    assert np.allclose(ident.trans, [0, 0])
    assert np.allclose(ident.rot, 0)

    t23 = Transform2d(np.array([3, 4]), 0.5)
    t13 = t12 * t23
    t31 = t13.inv()

    t32 = t23.inv()
    t31_other = t32 * t21

    assert np.allclose(t31.trans, t31_other.trans)
    assert np.allclose(t31.rot, t31_other.rot)


def test_feasibility_checker():
    checker = FeasibilityCheckerBatchImageJit(10)
    for _ in range(100):
        task = JskFridgeReachingTask.sample()
        exp = task.export_task_expression(use_matrix=True)
        mat = exp.get_matrix()
        vec = exp.get_vector()
        mats = [mat for _ in range(10)]

        booleans, indices = checker.infer(vec, mats)
        # all booleans and indices must have save values
        for b, i in zip(booleans, indices):
            assert b == booleans[0]
            assert i == indices[0]

        lib = load_library(JSKFridge, "cuda", postfix="0.2")
        ret = lib.infer(task)
        assert ret.idx == indices[0]
        assert (ret.cost < lib.max_admissible_cost) == booleans[0]
