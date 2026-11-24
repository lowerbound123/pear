"""Run with LIBERO's simulator Python and Mesa environment. All 120 task/start pairs must be solvable."""

import pytest

from pear.envs.libero.easy_tasks import SUITES, TASKS, MotionTask


@pytest.mark.parametrize("trial", range(3))
def test_reference_action_reaches_every_goal_from_this_start(trial):
    pytest.importorskip("robosuite")
    from pear.envs.libero.easy_env import EasyMotionEnv
    from pear.envs.libero.easy_scene import EasyScene

    scene = EasyScene(trial, render=False)
    try:
        for suite in SUITES:
            for tid in range(len(TASKS)):
                env = object.__new__(EasyMotionEnv)
                env.task = MotionTask(suite, tid, trial)
                scene.prepare(env.task)
                assert not env.task.reached(scene.origin, scene.origin)
                result = env.advance(scene, env.task.reference_action(), env.STEP_BUDGET)
                assert result["outcome"] == "success" and sum(result["rewards"]) == 1, (env.task, result["state"])
    finally:
        scene.close()


def test_changing_only_the_wrong_side_never_scores():
    import numpy as np

    from pear.envs.libero.easy_tasks import AXES, SIDES

    assert len(TASKS) == 20
    for suite in SUITES:
        for tid in range(len(TASKS)):
            task = MotionTask(suite, tid, 0)
            assert 1 <= len(task.changes) <= 2
            target = SIDES.index(task.side)
            start = np.zeros((2, 7))
            start[:, 6] = 0 if task.changes.get("width", 0) > 0 else 8
            for moved_side in (0, 1):
                now = start.copy()
                for key, change in task.changes.items():
                    if key == "width":
                        now[moved_side, 6] = 8 if change > 0 else 0
                    else:
                        now[moved_side, AXES.index(key)] += change
                assert task.reached(start, now) == (target == moved_side)
