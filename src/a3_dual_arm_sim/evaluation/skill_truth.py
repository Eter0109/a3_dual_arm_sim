"""Privileged skill labels and image-only temporal verification samples.

Simulator truth is intentionally confined to this evaluator and its audit file.
The planner, learned action policy and visual verifier consume no cookie IDs,
positions, contacts or truth predicates from this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from a3_dual_arm_sim.core.skills import CookieSkill, SkillOutcome, SkillRequest

Diagnosis = Literal["StillTrying", "Stuck"]


@dataclass(frozen=True)
class SkillTruth:
    """Evaluation/labeling result, never a visual-verifier input."""

    complete: bool
    reason: str
    cookie_indices: tuple[int, ...]
    stable_steps: int = 0
    progress: float = 0.0
    diagnosis: Diagnosis | None = None


@dataclass(frozen=True)
class SkillTruthConfig:
    lift_height_m: float = 0.070
    pick_hold_steps: int = 5
    place_hold_steps: int = 6
    relative_drift_m: float = 0.006
    held_speed_m_s: float = 0.12
    stalled_steps: int = 40
    stationary_drift_m: float = 0.002
    progress_delta: float = 0.01
    contact_force_n: float = 0.005
    target_column_margin_m: float = 0.005

    def __post_init__(self) -> None:
        if min(self.pick_hold_steps, self.place_hold_steps, self.stalled_steps) < 2:
            raise ValueError("temporal hold windows must contain at least two steps")
        if min(
            self.lift_height_m, self.relative_drift_m, self.held_speed_m_s,
            self.stationary_drift_m, self.progress_delta,
        ) <= 0:
            raise ValueError("skill truth tolerances must be positive")


def request_cookie_indices(env: Any, request: SkillRequest) -> tuple[int, ...]:
    """Resolve original IDs from configured layout, not current moved positions."""
    source = np.asarray(env.SOURCE_POSITIONS, dtype=np.float64)
    columns = np.unique(source[:, 0])
    if not 1 <= request.source_column_number <= len(columns):
        raise ValueError("source column is outside the configured source layout")
    if request.batch_index not in (0, 1):
        raise ValueError("single-box five-cookie skills require batch index 0 or 1")
    if request.target_slot_number not in (1, 2):
        raise ValueError("single-box target column must be 1 or 2")
    candidates = np.flatnonzero(
        np.isclose(source[:, 0], columns[request.source_column_number - 1])
    )
    ordered = sorted(candidates.tolist(), key=lambda index: source[index, 1])
    start = request.batch_index * 5
    selected = tuple(ordered[start : start + 5])
    if len(selected) != 5:
        raise ValueError("configured source column has too few cookies for this batch")
    return selected


class GroundTruthSkillEvaluator:
    """Independent simulator oracle for evaluation and verifier supervision only.

    Call reset at a skill boundary and update after each environment step. PICK
    requires a pad--five-cookie--pad contact chain, relative-to-tool stability,
    lift and moderate world speed. It does NOT require both pads to touch every
    cookie: interior cookies are physically supported through their neighbors.
    PLACE checks the actual requested five IDs in the moving target-bin frame.
    """

    def __init__(self, env: Any, *, config: SkillTruthConfig | None = None) -> None:
        self.env = env
        self.config = config or SkillTruthConfig()
        self.request: SkillRequest | None = None
        self.latest: SkillTruth | None = None
        self._pick_references: dict[tuple[int, ...], np.ndarray] = {}

    def reset(self, request: SkillRequest) -> None:
        self.request = request
        self.cookie_indices = request_cookie_indices(self.env, request)
        now = float(self.env.data.time)
        if now < getattr(self, "_previous_time", -1.0):
            self._pick_references.clear()  # environment reset starts a new episode
        initial_positions = self._positions()
        if request.skill is CookieSkill.PICK_FIVE:
            # Retrying a visually rejected but already-held batch must not demand
            # an additional 70 mm lift relative to its retry-time position.
            self._initial_positions = self._pick_references.setdefault(
                self.cookie_indices, initial_positions.copy()
            ).copy()
        else:
            self._initial_positions = initial_positions
        self._stable_steps = 0
        self._held_offsets: np.ndarray | None = None
        self._previous_positions = initial_positions.copy()
        self._previous_time = now
        self._history: deque[tuple[np.ndarray, float]] = deque(
            maxlen=self.config.stalled_steps
        )
        self.latest = SkillTruth(False, "skill_started", self.cookie_indices)

    def _positions(self) -> np.ndarray:
        return np.asarray([
            self.env.privileged_cookie_position(index) for index in self.cookie_indices
        ], dtype=np.float64)

    def _tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        site = self.env._eef_sites[0]
        return (
            self.env.data.site_xpos[site].copy(),
            self.env.data.site_xmat[site].reshape(3, 3).copy(),
        )

    def _contact_chain(self) -> bool:
        """Force-bearing graph connects two finger pads and all five cookies."""
        import mujoco

        fingers = tuple(int(index) for index in self.env._left_finger_geoms)
        groups = self.env._cookie_collision_geoms
        # Negative nodes prevent cookie IDs from colliding with MuJoCo geom IDs.
        mapping = {
            int(geom): -(index + 1)
            for index in self.cookie_indices for geom in groups[index]
        }
        nodes = set(mapping.values()) | set(fingers)
        graph: dict[int, set[int]] = {node: set() for node in nodes}
        for index in range(self.env.data.ncon):
            contact = self.env.data.contact[index]
            first = mapping.get(int(contact.geom1), int(contact.geom1))
            second = mapping.get(int(contact.geom2), int(contact.geom2))
            if first not in nodes or second not in nodes or first == second:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.env.model, self.env.data, index, force)
            if force[0] <= self.config.contact_force_n:
                continue
            graph[first].add(second)
            graph[second].add(first)
        reached: set[int] = set()
        pending = [fingers[0]]
        while pending:
            node = pending.pop()
            if node not in reached:
                reached.add(node)
                pending.extend(graph[node] - reached)
        return nodes <= reached

    def _pick(self, positions: np.ndarray, speed: float) -> tuple[bool, str, float]:
        lifted = positions[:, 2] - self._initial_positions[:, 2]
        progress = float(np.clip(np.min(lifted) / self.config.lift_height_m, 0, 1))
        if any(self.env.privileged_cookie_in_source(i) for i in self.cookie_indices):
            self._held_offsets = None
            return False, "cookies_still_in_source", progress
        if not np.all(lifted >= self.config.lift_height_m):
            self._held_offsets = None
            return False, "not_all_five_lifted", progress
        if not self._contact_chain():
            self._held_offsets = None
            return False, "broken_grasp_contact_chain", progress
        tool_position, tool_rotation = self._tool_pose()
        offsets = (positions - tool_position) @ tool_rotation
        if self._held_offsets is None:
            self._held_offsets = offsets.copy()
        drift = float(np.max(np.linalg.norm(offsets - self._held_offsets, axis=1)))
        if drift > self.config.relative_drift_m:
            self._held_offsets = offsets.copy()
            return False, "cookies_slipping_relative_to_tool", progress
        if speed > self.config.held_speed_m_s:
            return False, "lift_not_yet_settled", progress
        return True, "five_cookies_held", progress

    def _place(self) -> tuple[bool, str, float]:
        assert self.request is not None
        local_positions = np.asarray([
            self.env.privileged_cookie_target_position(i) for i in self.cookie_indices
        ])
        columns = np.unique(np.asarray(self.env.TARGET_SLOTS_LOCAL)[:, 0])
        if len(columns) != 2:
            raise ValueError("single-box verifier requires two target columns")
        x = columns[self.request.target_slot_number - 1]
        tolerance = float(self.env.TARGET_SLOT_TOLERANCE[0]) + self.config.target_column_margin_m
        in_column = np.abs(local_positions[:, 0] - x) <= tolerance
        settled = np.asarray([
            self.env.privileged_cookie_in_target(i) for i in self.cookie_indices
        ])
        released = not any(
            any(self.env.privileged_left_finger_contacts(i)) for i in self.cookie_indices
        )
        progress = float(np.count_nonzero(in_column & settled) / 5)
        if not np.all(in_column):
            return False, "cookies_in_wrong_target_column", progress
        if not np.all(settled):
            return False, "cookies_not_settled_in_target", progress
        if not released:
            return False, "cookies_still_contacting_gripper", progress
        return True, "five_cookies_released_in_target_column", progress

    def update(self) -> SkillTruth:
        if self.request is None:
            raise RuntimeError("reset a skill before updating its truth evaluator")
        positions = self._positions()
        now = float(self.env.data.time)
        elapsed = now - self._previous_time
        if elapsed <= 0:
            # Multiple verifier checks on one simulator frame cannot earn hold time.
            assert self.latest is not None
            return self.latest
        speed = float(np.max(np.linalg.norm(
            positions - self._previous_positions, axis=1
        )) / elapsed)
        if self.request.skill is CookieSkill.PICK_FIVE:
            condition, reason, progress = self._pick(positions, speed)
            required = self.config.pick_hold_steps
        else:
            condition, reason, progress = self._place()
            required = self.config.place_hold_steps
            if self.request.batch_index == 1:
                required = max(required, getattr(
                    getattr(self.env, "task_config", None), "success_hold_steps", required
                ))
        self._stable_steps = self._stable_steps + 1 if condition else 0
        complete = self._stable_steps >= required
        tool_position, tool_rotation = self._tool_pose()
        grip = float(self.env.current_joint_action[7]) * float(
            getattr(self.env, "GRIPPER_RANGE_M", 0.0425)
        )
        # A rotating/reorienting or closing gripper is still trying even if the
        # cookie centroid and end-effector origin have not yet moved.
        state = np.vstack([
            positions, tool_position[None, :], tool_rotation * 0.03,
            np.asarray([[grip, 0.0, 0.0]]),
        ])
        self._history.append((state.copy(), progress))
        diagnosis: Diagnosis | None = None if complete else "StillTrying"
        if not complete and len(self._history) == self.config.stalled_steps:
            states = np.stack([entry[0] for entry in self._history])
            drift = float(np.max(np.linalg.norm(states - states[0], axis=2)))
            progress_range = np.ptp([entry[1] for entry in self._history])
            if drift <= self.config.stationary_drift_m and progress_range <= self.config.progress_delta:
                diagnosis = "Stuck"
        self.latest = SkillTruth(
            complete, reason, self.cookie_indices, self._stable_steps, progress, diagnosis
        )
        self._previous_positions = positions.copy()
        self._previous_time = now
        return self.latest

    def check(self, request: SkillRequest) -> SkillOutcome:
        if request != self.request:
            raise ValueError("verification request does not match the active skill")
        result = self.update()
        return SkillOutcome(result.complete, result.reason)
