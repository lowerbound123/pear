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
