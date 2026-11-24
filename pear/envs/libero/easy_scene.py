"""Simulator-only imports for the two parallel Pandas in an empty LIBERO tabletop arena."""

from pathlib import Path

import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs.arenas import TableArena
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils import transform_utils as T

from pear.envs.libero.controller import MotionController
from pear.envs.libero.easy_tasks import SIDES


class EasyScene(TwoArmEnv):
    TABLE_Z = .8
    # Joint-space resets with a vertical hand. Shoulder, elbow and wrist pitch move together to preserve orientation.
    STARTS = ((.45, -2.45), (.55, -2.25), (.50, -2.35))
    CAMERAS = ("agentview", "robot0_eye_in_hand", "robot1_eye_in_hand")

    def __init__(self, trial_id, render):
        self.trial_id = trial_id
        super().__init__(
            robots=["Panda", "Panda"], env_configuration="single-arm-parallel",
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            initialization_noise=None, has_offscreen_renderer=render, use_camera_obs=False,
            has_renderer=False, hard_reset=False, ignore_done=True, control_freq=20,
        )

    def _load_model(self):
        super()._load_model()
        shoulder, elbow = self.STARTS[self.trial_id]
        for robot, y in zip(self.robots, (.30, -.30)):
            robot.robot_model.set_base_xpos([-.55, y, 0])
            robot.init_qpos = np.array([0., shoulder, 0., elbow, 0., shoulder-elbow, np.pi/4])
        arena = TableArena(table_full_size=(.8, 1.2, .05), table_offset=(0, 0, self.TABLE_Z),
                           xml=str(Path(get_libero_path("assets")) / "scenes/libero_tabletop_base_style.xml"))
        # From the front, looking down and toward the two bases.
        arena.set_camera(camera_name="agentview", pos=[1.05, 0, 1.65], quat=[.60, .37, .37, .60])
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots])

    def reward(self, action=None):
        return 0.0  # the curriculum scores the settled pose after the control steps

    def prepare(self, task):
        self.reset()
        hold = np.zeros((2, 7))
        hold[:, 6] = -1.0
        if task.changes.get("width", 0) > 0:
            hold[SIDES.index(task.side), 6] = 1.0
        for robot in self.robots:
            robot.gripper.current_action[:] = 0
        self.finger_geoms = []
        for i in range(2):
            self.finger_geoms.extend(self.sim.model.geom_name2id(f"gripper{i}_finger{n}_pad_collision") for n in (1, 2))
        for _ in range(30):
            self.step(hold.ravel())
            if self.unsafe():
                raise RuntimeError(f"Unsafe reset for {task}")
        self.neutral = [self.rotation(r) for r in self.robots]
        self.origin = self.quantities()
        if not .105 <= self.clearance() <= .15:
            raise RuntimeError(f"Initial finger clearance outside the calibrated band for {task}")
        if not self.at_rest() or task.reached(self.origin, self.origin):
            raise RuntimeError(f"Unsettled or already solved reset for {task}")
        i = SIDES.index(task.side)
        if ((hold[i, 6] > 0 and self.origin[i, 6] > .8)
                or (hold[i, 6] < 0 and self.origin[i, 6] < 7.0)):
            raise RuntimeError(f"Incorrect initial finger gap for {task}")

    def rotation(self, robot):
        return self.sim.data.site_xmat[robot.eef_site_id].reshape(3, 3).copy()

    def quantities(self):
        data = self.sim.data
        result = []
        for i, r in enumerate(self.robots):
            base = data.body_xpos[self.sim.model.body_name2id(f"robot{i}_base")]
            pos = (data.site_xpos[r.eef_site_id] - base) * 100
            rpy = MotionController._matrix_to_rpy(self.rotation(r) @ self.neutral[i].T)
            fingers = data.qpos[r._ref_gripper_joint_pos_indexes]
            result.append([*pos, *rpy, 100 * (fingers[0] - fingers[1])])
        return np.array(result)

    def at_rest(self):
        data = self.sim.data
        return all(np.linalg.norm(data.get_site_xvelp(r.controller.eef_name)) < .03
                   and np.linalg.norm(data.get_site_xvelr(r.controller.eef_name)) < .15
                   and np.max(np.abs(data.qvel[r._ref_gripper_joint_vel_indexes])) < .02 for r in self.robots)

    def clearance(self):
        return min(self.sim.data.geom_xpos[g, 2] - self.sim.model.geom_rbound[g] - self.TABLE_Z
                   for g in self.finger_geoms)

    def unsafe(self):
        model, data = self.sim.model, self.sim.data
        for r in self.robots:
            q = data.qpos[r._ref_joint_pos_indexes]
            limits = model.jnt_range[r._ref_joint_indexes]
            if not np.all((limits[:, 0] + .04 < q) & (q < limits[:, 1] - .04)):
                return True
        if self.clearance() < .045:
            return True
        for contact in data.contact[:data.ncon]:
            a, b = (model.body_id2name(model.geom_bodyid[g]) or "" for g in (contact.geom1, contact.geom2))
            # Opposing pads of the same empty gripper can meet at full closure.
            if any(a.startswith(f"gripper{i}_") and b.startswith(f"gripper{i}_")
                   and "finger" in a and "finger" in b for i in range(2)):
                continue
            if any(s.startswith(("robot0_", "robot1_", "gripper0_", "gripper1_")) for s in (a, b)):
                return True
        return False

    def command(self, targets, current, width_changes):
        native = np.zeros((2, 7))
        for i, r in enumerate(self.robots):
            # Both bases have identity orientation; OSC consumes world-frame position and axis-angle deltas.
            site = r.controller.eef_name
            native[i, :3] = (np.clip(.14 * (targets[i, :3] - current[i, :3]), -.28, .28)
                             - .045 * self.sim.data.get_site_xvelp(site) / .07)
            goal = MotionController._rpy_to_matrix(targets[i, 3:6]) @ self.neutral[i]
            rot_error = MotionController._rotation(T.mat2quat(goal), T.mat2quat(self.rotation(r)))
            native[i, 3:6] = (np.clip(1.8 * rot_error, -.36, .36)
                              - .045 * self.sim.data.get_site_xvelr(site) / .55)
            native[i, 6] = -np.sign(width_changes[i])
        return np.clip(native, -1., 1.).ravel()
