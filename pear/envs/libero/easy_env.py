"""PEAR's optional two-Panda basic-motion tasks, using the LIBERO simulator environment."""

import numpy as np
from PIL import Image

from pear.envs.libero.easy_tasks import AXES, SCHEMA, SIDES, MotionTask
from pear.envs.libero.env import LiberoEnv


class EasyMotionEnv(LiberoEnv):
    SCHEMA = SCHEMA
    CAMERAS = ("front", "left wrist", "right wrist")
    ENVIRONMENT_PROMPT = (
        "Two Panda arms share an empty table and face forward. Left and right are from the robot's perspective. "
        "In both base frames +x is forward, +y is left, and +z is up. "
        "Angles are fixed-axis roll, pitch and yaw relative to the starting orientation, with the fingertips down. "
        "In each gripper a positive width change opens fully, a negative change closes fully and zero holds. "
        "Requested changes in the task are measured from the starting pose. Use zero changes for idle parts."
    )
    STEP_BUDGET = 80
    ACTION_STEPS = 30
    # Local envelope relative to the settled start; cm for position, radians for orientation.
    ENVELOPE = np.array([6., 6., 6., .35, .35, .35])

    def __init__(self, suite, task_id, trial_id):
        self.task = MotionTask(suite, task_id, trial_id)
        self._bind_rendering()  # bind the same EGL software renderer as LIBERO before importing robosuite

    def rollout(self, history, action):
        from pear.envs.libero.easy_scene import EasyScene

        scene = EasyScene(self.task.trial_id, render=True)
        try:
            scene.prepare(self.task)
            if len(history) > self.STEP_BUDGET:
                raise ValueError("History exceeds easy-motion step budget")
            for native in history:
                scene.step(np.asarray(native))
            result = self.advance(scene, action, self.STEP_BUDGET - len(history))
            if not result["done"]:
                result["images"] = [
                    Image.fromarray(np.ascontiguousarray(scene.sim.render(self.FRAME_SIZE, self.FRAME_SIZE,
                                                                          camera_name=camera)[::-1]))
                    for camera in scene.CAMERAS
                ]
            return result
        finally:
            scene.close()

    def advance(self, scene, action, remaining):
        current = scene.quantities()
        actions = []
        success = False
        done = remaining == 0
        outcome = "budget" if done else None
        feedback = None
        if action is not None and not done:
            delta = np.array([[action[f"{s}_arm"][k] for k in AXES] for s in SIDES])
            widths = np.array([action[f"{s}_gripper"]["width"] for s in SIDES])
            targets = current[:, :6] + delta
            if not np.all(np.abs(targets - scene.origin[:, :6]) <= self.ENVELOPE):
                # A hold advances the episode clock and makes retries replayable.
                native = np.zeros(14)
                scene.step(native)
                actions.append(native.tolist())
                current = scene.quantities()
                outcome = "rejected"
                feedback = "Requested pose outside the local workspace. No movement was commanded. Try a smaller change."
            else:
                for _ in range(min(remaining, self.ACTION_STEPS)):
                    native = scene.command(targets, current, widths)
                    scene.step(native)
                    actions.append(native.tolist())
                    current = scene.quantities()
                    if scene.unsafe() or not np.all(np.abs(current[:, :6] - scene.origin[:, :6]) <= self.ENVELOPE + .01):
                        done, outcome = True, "safety_stop"
                        break
                    if scene.at_rest():
                        if self.task.reached(scene.origin, current):
                            success, done, outcome = True, True, "success"
                            break
                        fingers = (widths == 0) | ((widths > 0) & (current[:, 6] >= 7)) | (
                            (widths < 0) & (current[:, 6] <= .8))
                        if np.all(fingers) and np.all(np.abs(targets - current[:, :6]) < [.3, .3, .3, .02, .02, .02]):
                            outcome = "reached"
                            break
                outcome = outcome or "timeout"
            if scene.unsafe():
                done, success, outcome = True, False, "safety_stop"
            if len(actions) == remaining and not done:
                done, outcome = True, "budget"
        rewards = [0.] * len(actions)
        if success:
            rewards[-1] = 1.
        state = {}
        for i, side in enumerate(SIDES):
            state[f"{side}_arm"] = dict(zip(AXES, current[i, :6].round(3).tolist()))
            state[f"{side}_gripper"] = {"width": round(float(current[i, 6]), 2)}
        return dict(images=None, state=state, actions=actions, rewards=rewards, done=done, executed=len(actions),
                    outcome=outcome, instruction=self.task.instruction, feedback=feedback)
