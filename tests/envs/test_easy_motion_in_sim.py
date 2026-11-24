"""Run with LIBERO's simulator Python and Mesa environment. All 120 task/start pairs must be solvable."""

import numpy as np
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


def _finger_direction(scene, i):
    positions = scene.sim.data.body_xpos
    tips = [scene.sim.model.body_name2id(f"gripper{i}_finger_joint{n}_tip") for n in (1, 2)]
    palm = scene.sim.model.body_name2id(f"gripper{i}_right_gripper")
    vector = positions[tips].mean(axis=0) - positions[palm]
    return vector / np.linalg.norm(vector)


def test_workspace_rejection_and_robot_frame_directions():
    pytest.importorskip("robosuite")
    from pear.envs.libero.easy_env import EasyMotionEnv
    from pear.envs.libero.easy_scene import EasyScene

    scene = EasyScene(2, render=False)
    env = object.__new__(EasyMotionEnv)
    try:
        env.task = MotionTask(SUITES[0], 5, 2)
        scene.prepare(env.task)
        assert scene.sim.data.site_xpos[scene.robots[0].eef_site_id, 1] > scene.sim.data.site_xpos[
            scene.robots[1].eef_site_id, 1
        ]
        action = env.task.reference_action()
        action["left_arm"]["z"] = -30
        result = env.advance(scene, action, env.STEP_BUDGET)
        assert result["outcome"] == "rejected" and result["feedback"]
        assert result["actions"] == [[0.] * 14] and not any(result["rewards"])
        for i, suite in enumerate(SUITES):
            for task_id, axis, sign in [(6, 1, 1), (7, 1, -1), (8, 0, -1), (9, 0, 1)]:
                env.task = MotionTask(suite, task_id, 2)
                scene.prepare(env.task)
                direction = _finger_direction(scene, i)
                assert direction[2] < -.98
                assert env.advance(scene, env.task.reference_action(), env.STEP_BUDGET)["outcome"] == "success"
                assert sign * (_finger_direction(scene, i) - direction)[axis] > .08
    finally:
        scene.close()
