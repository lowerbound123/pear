"""Empty-hand curriculum in the robot base frame. Translations are cm and angles are rad."""

from dataclasses import dataclass

import numpy as np

SIDES = ("left", "right")
SUITES = tuple(f"libero_easy_{side}" for side in SIDES)
AXES = ("x", "y", "z", "roll", "pitch", "yaw")
SCHEMA = {f"{side}_{part}": axes for side in SIDES for part, axes in (("arm", AXES), ("gripper", ("width",)))}
# Changes for one-step reference actions. A width change selects fully open (+) or closed (-).
TASKS = (
    ("Move the {side} gripper straight forward by 3 cm.", {"x": 3}),
    ("Pull the {side} gripper back toward the base by 3 cm.", {"x": -3}),
    ("Move the {side} gripper 3 cm to the robot's left.", {"y": 3}),
    ("Move the {side} gripper 3 cm to the robot's right.", {"y": -3}),
    ("Raise the {side} gripper straight up by 3 cm.", {"z": 3}),
    ("Lower the {side} gripper straight down by 3 cm.", {"z": -3}),
    ("Tilt the {side} gripper about nine degrees so its fingertips lean to the robot's left.", {"roll": .15}),
    ("Tilt the {side} gripper about nine degrees so its fingertips lean to the robot's right.", {"roll": -.15}),
    ("Tip the {side} fingertips about nine degrees toward the robot's base.", {"pitch": .15}),
    ("Tip the {side} fingertips about nine degrees away from the robot's base.", {"pitch": -.15}),
    ("Turn the {side} gripper counterclockwise by about nine degrees as seen from above.", {"yaw": .15}),
    ("Turn the {side} gripper clockwise by about nine degrees as seen from above.", {"yaw": -.15}),
    ("Open the {side} gripper all the way.", {"width": 1}),
    ("Close the {side} gripper all the way.", {"width": -1}),
    ("Move the {side} gripper diagonally forward and left, 2 cm in each direction.", {"x": 2, "y": 2}),
    ("Pull the {side} gripper back 2 cm while raising it 2 cm.", {"x": -2, "z": 2}),
    ("Move the {side} gripper 2 cm to the right while lowering it 2 cm.", {"y": -2, "z": -2}),
    ("Tilt the {side} fingertips left and away from the base, about nine degrees each way.",
     {"roll": .15, "pitch": -.15}),
    ("Open the {side} gripper fully while turning it counterclockwise by about nine degrees as seen from above.",
     {"width": 1, "yaw": .15}),
    ("Raise the {side} gripper by 2 cm while closing it fully.", {"z": 2, "width": -1}),
)


@dataclass(frozen=True)
class MotionTask:
    suite: str
    task_id: int
    trial_id: int

    def __post_init__(self):
        if self.suite not in SUITES or self.task_id not in range(len(TASKS)) or self.trial_id not in range(3):
            raise ValueError(f"Unknown easy LIBERO episode: {self}")

    @property
    def side(self):
        return self.suite.removeprefix("libero_easy_")

    @property
    def instruction(self):
        return TASKS[self.task_id][0].format(side=self.side) + " Keep the other arm and gripper still."

    @property
    def changes(self):
        return TASKS[self.task_id][1]

    def reference_action(self):
        action = {part: dict.fromkeys(axes, 0.0) for part, axes in SCHEMA.items()}
        for axis, change in self.changes.items():
            part = "gripper" if axis == "width" else "arm"
            action[f"{self.side}_{part}"][axis] = change
        return action

    def reached(self, start, now):
        """Rows are left/right; columns are xyz, relative roll/pitch/yaw and finger gap."""
        i = SIDES.index(self.side)
        other_delta = np.abs(now[1-i] - start[1-i])
        if np.any(other_delta > [.6, .6, .6, .04, .04, .04, .8]):
            return False
        for axis, target in self.changes.items():
            if axis == "width":
                if (target > 0 and now[i, 6] < 7.0) or (target < 0 and now[i, 6] > .8):
                    return False
            else:
                j = AXES.index(axis)
                # The checked workspace is within 0.4 rad of neutral on each axis.
                if abs(now[i, j] - start[i, j] - target) > (.6 if j < 3 else .04):
                    return False
        return True
