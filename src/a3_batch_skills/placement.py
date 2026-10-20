"""Measured pre-release placement and bounded post-release regrasp/place.

All corrections are robot actions. Unsupported scattered or heavily tilted
batches are rejected instead of being teleported or blindly pushed.
"""
import numpy as np
from .motion import average_rotation, group_width
from .planning import rectangle_overlap


class AlignPickEntry:
    """Move to a measured pre-grasp frame before restarting a short policy."""
    canonical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])

    def __init__(self, servo, config):
        self.servo, self.config = servo, config

    def start(self, goal, scene):
        self.goal = goal
        self.steps, self.stable = 0, 0
        self.done, self.failure = False, None
        self.stage = "CLEAR_TARGET"
        self.clear_position = scene.tool_position + np.array([0., 0., .025])
        self.clear_rotation = scene.tool_rotation.copy()
        self.initial_opening = float(self.servo.env.last_applied_action[7])

    def act(self, scene):
        self.steps += 1
        if self.steps > 220 or scene.safety_stop:
            self.failure = "Could not establish the policy pre-grasp entry pose"
        if self.failure or self.done:
            return self.servo.hold()
        if self.stage == "CLEAR_TARGET":
            command, reached = self.servo.move(self.clear_position, self.clear_rotation,
                self.initial_opening, speed=.004)
            if reached:
                self.stage = "ALIGN_SOURCE"
            return command
        rotations = [scene.cookies[i].rotation for i in self.goal.cookie_ids]
        mean_rotation = average_rotation(rotations)
        mean = np.asarray([scene.cookies[i].position for i in self.goal.cookie_ids]).mean(0)
        rotation = mean_rotation @ self.canonical
        # Keep the current main expert's pitch convention for farther columns.
        source = np.asarray(self.servo.env.SOURCE_POSITIONS)
        columns = np.unique(source[:, 0])
        rank = int(np.argmin(np.abs(columns - source[self.goal.cookie_ids[0], 0])))
        pitch = np.deg2rad((0, -15, -35, -35)[min(rank, 3)])
        c, s = np.cos(pitch), np.sin(pitch)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ rotation
        pinch = mean + mean_rotation[:, 2] * .010
        position = pinch - rotation @ self.servo.pad_offset() + np.array([0., 0., .082])
        opening = self.servo.opening_for_width(group_width(scene, self.goal.cookie_ids, rotation[:, 0]), margin=.015)
        command, reached = self.servo.move(position, rotation, opening, speed=.005)
        self.stable = self.stable + 1 if reached else 0
        if self.stable >= 5:
            self.done = True
        return command


class PrecisePlacement:
    def __init__(self, servo, config):
        self.servo, self.config = servo, config

    def start(self, goal, scene):
        self.goal = goal
        mean = np.asarray([scene.cookies[i].position for i in goal.cookie_ids]).mean(0)
        object_rotation = average_rotation([scene.cookies[i].rotation for i in goal.cookie_ids])
        self.offset = scene.tool_rotation.T @ (mean - scene.tool_position)
        self.relative_rotation = scene.tool_rotation.T @ object_rotation
        self.opening = float(self.servo.env.last_applied_action[7])
        self.stage, self.steps, self.stage_steps = "ABOVE", 0, 0
        self.done = False
        self.failure = None
        self.jam_steps = 0

    def _transition(self, stage):
        self.stage, self.stage_steps = stage, 0

    def _target(self, scene, clearance):
        rotation = scene.box_rotation @ self.relative_rotation.T
        center = self.goal.target_positions.mean(0).copy()
        center[2] += .0015 + clearance
        return scene.world(center) - rotation @ self.offset, rotation

    def act(self, scene):
        self.steps += 1
        self.stage_steps += 1
        if self.steps > self.config.placement_timeout_steps:
            self.failure = "Precise placement exceeded its step budget"
        if scene.safety_stop:
            self.failure = "Safety stop during placement"
        # Release can have short finger-object transients; force protection is
        # active during insertion and alignment, where continued descent jams.
        self.jam_steps = self.jam_steps + 1 if max(scene.pad_forces) > self.config.max_pad_force_n else 0
        if self.stage in ("INSERT", "ALIGN") and self.jam_steps >= 3:
            self.failure = "Placement blocked by excessive pad contact force"
        if self.failure or self.done:
            return self.servo.hold(self.opening)
        if self.stage in ("ABOVE", "INSERT", "ALIGN"):
            clearance = .080 if self.stage == "ABOVE" else 0.
            pos, rot = self._target(scene, clearance)
            command, reached = self.servo.move(pos, rot, self.opening,
                                               speed=.005 if self.stage == "ABOVE" else .0012)
            if reached:
                if self.stage == "ABOVE":
                    self._transition("INSERT")
                elif self.stage == "INSERT":
                    self._transition("ALIGN")
                elif self.stage_steps >= 5:
                    width = group_width(scene, self.goal.cookie_ids, scene.tool_rotation[:, 0])
                    self.release_opening = max(self.opening, self.servo.opening_for_width(width))
                    self._transition("OPEN")
            return command
        if self.stage == "OPEN":
            pos, rot = self._target(scene, 0.)
            self.opening = min(self.release_opening, self.opening + .015)
            command, _ = self.servo.move(pos, rot, self.opening, speed=.0008)
            if self.opening >= self.release_opening and self.stage_steps >= 10:
                self.retract_position = scene.tool_position + np.array([0., 0., self.config.retreat_clearance_m])
                self.retract_rotation = scene.tool_rotation.copy()
                self._transition("RETRACT")
            return command
        command, reached = self.servo.move(self.retract_position, self.retract_rotation,
                                          self.opening, speed=.004)
        if reached:
            self.done = True
        return command


class AdjustPlacement:
    """Regrasp an intact, locally misplaced batch, then run measured placement.

    A safe rejection is a real outcome. This first implementation deliberately
    does not claim to recover arbitrary collapsed or interlocked piles.
    """
    canonical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])

    def __init__(self, servo, config):
        self.servo, self.config = servo, config

    def start(self, goal, scene):
        self.goal, self.steps, self.stage_steps = goal, 0, 0
        self.done, self.failure = False, None
        self.stage = "APPROACH"
        actual = scene.local(goal.cookie_ids)
        offset = np.linalg.norm(actual.mean(0)[:2] - goal.target_positions.mean(0)[:2])
        rotations = [scene.cookies[i].rotation for i in goal.cookie_ids]
        average = average_rotation(rotations)
        up = scene.box_rotation[:, 2]
        tilt = max(np.rad2deg(np.arccos(np.clip(abs(up @ r[:, 2]), -1, 1))) for r in rotations)
        spread = max(np.linalg.norm(r - average) for r in rotations)
        formation = actual - actual.mean(0)
        desired = goal.target_positions - goal.target_positions.mean(0)
        scattered = np.max(np.linalg.norm(formation - desired, axis=1)) > .008
        if (offset > self.config.adjustment_max_offset_m or tilt > self.config.adjustment_max_tilt_deg
                or spread > .35 or scattered or set(goal.cookie_ids) & scene.held_ids):
            self.failure = "Adjustment requires an intact, mildly displaced five-cookie batch"
            return
        for other_id, obj in scene.cookies.items():
            if other_id in goal.cookie_ids:
                continue
            other = scene.box_rotation.T @ (obj.position - scene.box_position)
            other_rotation = scene.box_rotation.T @ obj.rotation
            for i in goal.cookie_ids:
                current = scene.box_rotation.T @ (scene.cookies[i].position - scene.box_position)
                rotation = scene.box_rotation.T @ scene.cookies[i].rotation
                if rectangle_overlap(current, rotation, scene.half_size,
                                     other, other_rotation, scene.half_size, margin=.001):
                    self.failure = "Corrective regrasp is obstructed by a neighbouring object"
                    return
        mean = np.asarray([scene.cookies[i].position for i in goal.cookie_ids]).mean(0)
        self.rotation = average @ self.canonical
        pad_center = mean + average[:, 2] * .010
        self.pick_position = pad_center - self.rotation @ self.servo.pad_offset()
        self.high_position = self.pick_position + np.array([0., 0., .070])
        self.opening = self.servo.opening_for_width(group_width(scene, goal.cookie_ids,
                                                                self.rotation[:, 0]))
        self.initial_heights = np.asarray([scene.cookies[i].position[2] for i in goal.cookie_ids])

    def _transition(self, stage):
        self.stage, self.stage_steps = stage, 0

    def act(self, scene):
        self.steps += 1
        self.stage_steps += 1
        if self.steps > self.config.placement_timeout_steps:
            self.failure = "Post-release adjustment exceeded its step budget"
        if scene.safety_stop:
            self.failure = "Safety stop during adjustment"
        if self.failure or self.done:
            return self.servo.hold()
        if self.stage == "REPLACE":
            command = self.placer.act(scene)
            self.failure, self.done = self.placer.failure, self.placer.done
            return command
        if self.stage in ("APPROACH", "DESCEND"):
            pos = self.high_position if self.stage == "APPROACH" else self.pick_position
            command, reached = self.servo.move(pos, self.rotation, self.opening,
                                               speed=.004 if self.stage == "APPROACH" else .001)
            if max(scene.pad_forces) > self.config.max_pad_force_n:
                self.failure = "Adjustment approach obstructed by contact"
            if reached:
                self._transition("DESCEND" if self.stage == "APPROACH" else "CLOSE")
            return command
        if self.stage == "CLOSE":
            if set(self.goal.cookie_ids) <= scene.held_ids and min(scene.pad_forces) >= .35:
                self._transition("LIFT")
                self.lift_position = scene.tool_position + np.array([0., 0., .055])
            elif max(scene.pad_forces) > self.config.max_pad_force_n:
                self.failure = "Regrasp force limit exceeded without a secure five-cookie grasp"
            else:
                self.opening = max(.10, self.opening - .004)
            command, _ = self.servo.move(self.pick_position, self.rotation, self.opening, speed=.0007)
            return command
        command, reached = self.servo.move(self.lift_position, self.rotation, self.opening, speed=.0012)
        rises = np.asarray([scene.cookies[i].position[2] for i in self.goal.cookie_ids]) - self.initial_heights
        if self.stage_steps > 15 and min(rises) < -.005:
            self.failure = "An object slipped during corrective lift"
        if reached and min(rises) > .025:
            self.placer = PrecisePlacement(self.servo, self.config)
            self.placer.start(self.goal, scene)
            self._transition("REPLACE")
        return command
