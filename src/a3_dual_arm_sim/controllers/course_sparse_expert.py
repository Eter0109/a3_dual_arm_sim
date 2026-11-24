"""Contact-driven, privileged one-object-at-a-time sparse-scene expert."""

from __future__ import annotations

import mujoco
import numpy as np

from a3_dual_arm_sim.contracts import LEFT_JOINTS

from .phases import CookiePhase
from .same_column_batch_expert import A3SameColumnBatchExpert

_PITCHES = (0, -15, 15, -35, 35, -40, 40, -45, 45, -50, 50, -55, 55)
_OPENINGS = (.02, .03, .04, .05, .1, .3, .5, .7, 1.)
_BRANCH = np.array([1.28, 1.119, -1.127, -.999, -.252, .478, -1.535])
_CANON = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def _collisions(env, work):
    for contact in work.contact:
        if contact.dist >= -1e-5:
            continue
        a = env.model.geom(int(contact.geom1)).name or ""
        b = env.model.geom(int(contact.geom2)).name or ""
        if a.startswith("L_") or b.startswith("L_"):
            other = b if a.startswith("L_") else a
            if other.startswith(("L_", "R_", "cookie_", "source_bin", "target_bin", "spare_target_bin")) or other == "table_top":
                return True
    return False


def _pose_candidate(env, work, qids, site, pad, frame, offset, rotation):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    pick = pad - rotation @ offset
    pre = pick + frame[:, 2] * .082
    qpre = env.solve_ik(pre, quat, _BRANCH)
    qpick = env.solve_ik(pick, quat, qpre)
    for q, desired in ((qpre, pre), (qpick, pick)):
        work.qpos[qids] = q
        mujoco.mj_kinematics(env.model, work)
        actual, error = np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(actual, work.site_xmat[site])
        mujoco.mju_subQuat(error, quat, actual)
        if np.linalg.norm(work.site_xpos[site] - desired) > .0008 or np.linalg.norm(error) > .015:
            return None
    for opening in _OPENINGS:
        work.qpos[:] = env.data.qpos
        for jid in env._finger_joints["L"]:
            work.qpos[int(env.model.jnt_qposadr[jid])] = (1 - opening) * env.GRIPPER_RANGE_M
        for q in (qpre, qpick):
            work.qpos[qids] = q
            mujoco.mj_forward(env.model, work)
            if _collisions(env, work):
                break
        else:
            return pre.copy(), pick.copy()
    return None


def _carrying_route(env, cookie, position, rotation, relative_p, relative_r):
    """Find a sampled collision-free joint interpolation for the held object."""
    qids = [env._qpos_ids[n] for n in LEFT_JOINTS]
    site = env._eef_sites[0]
    body = env._cookie_bodies[cookie]
    adr = int(env.model.jnt_qposadr[env._cookie_joints[cookie]])
    work = mujoco.MjData(env.model)
    base = env.data.qpos.copy()
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    q0 = env.current_joint_action[:7].copy()
    limits = env._arm_ranges[:7]
    rng = np.random.default_rng(12)
    seeds = [q0, _BRANCH, env.DEPLOYMENT_HOME[:7]]
    seeds += [np.clip(q0 + rng.uniform(-.7, .7, 7), limits[:, 0] + .002,
                      limits[:, 1] - .002) for _ in range(12)]
    for seed in seeds:
        q = env.solve_ik(position, quat, seed)
        work.qpos[:] = base
        work.qpos[qids] = q
        mujoco.mj_kinematics(env.model, work)
        actual, error = np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(actual, work.site_xmat[site])
        mujoco.mju_subQuat(error, quat, actual)
        if np.linalg.norm(work.site_xpos[site] - position) > .002 or np.linalg.norm(error) > .04:
            continue
        valid = True
        for fraction in np.linspace(.025, 1., 40):
            work.qpos[:] = base
            work.qpos[qids] = q0 + (q - q0) * fraction
            mujoco.mj_kinematics(env.model, work)
            tool_r = work.site_xmat[site].reshape(3, 3)
            work.qpos[adr:adr + 3] = work.site_xpos[site] + tool_r @ relative_p
            mujoco.mju_mat2Quat(work.qpos[adr + 3:adr + 7], (tool_r @ relative_r).ravel())
            mujoco.mj_forward(env.model, work)
            for contact in work.contact:
                if contact.dist >= -1e-5:
                    continue
                g1, g2 = int(contact.geom1), int(contact.geom2)
                b1, b2 = int(env.model.geom_bodyid[g1]), int(env.model.geom_bodyid[g2])
                a = env.model.geom(g1).name or ""
                b = env.model.geom(g2).name or ""
                if body in (b1, b2):
                    other = b if b1 == body else a
                    if other.startswith("L_finger_"):
                        continue
                elif a.startswith("L_") or b.startswith("L_"):
                    other = b if a.startswith("L_") else a
                else:
                    continue
                if other.startswith(("cookie_", "source_bin", "target_bin", "spare_target_bin", "L_", "R_")) or other == "table_top":
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return q
    return None


class A3CourseSparseExpert(A3SameColumnBatchExpert):
    """Use real gripper contacts and target-aware IK for each sparse object."""

    def _validate_target_workspace(self):
        # Per-object candidate checks include both target and approach poses.
        pass

    def reset(self, context=None):
        super().reset(context)
        rows = self.env.config.cookie_transfer.target_rows
        self._slots = list(range(0, rows * 2, 2)) + list(range(1, rows * 2, 2))
        farthest = [i for i in np.flatnonzero(self.env._active)
                    if np.isclose(self.env.SOURCE_POSITIONS[i][0],
                                  max(x for x, _ in self.env.SOURCE_POSITIONS))]
        self._hard_first = bool(farthest and max(i % 20 for i in farthest) >= 15)
        self._early_priority = []
        if self._hard_first and rows == 10:
            source = np.asarray(self.env.SOURCE_POSITIONS)
            farthest.sort(key=lambda i: source[i, 1])
            hardest = farthest[-1]
            # A rear object in the adjacent column can block the gripper body.
            # Clear that object early, then take the hard-to-reach far object
            # while there is still ample room in the target.
            xs = np.unique(source[:, 0])
            adjacent = [int(i) for i in np.flatnonzero(self.env._active)
                        if np.isclose(source[i, 0], xs[-2])]
            blocker = min(adjacent, key=lambda i: abs(source[i, 1] - source[hardest, 1])) if adjacent else None
            self._early_priority = ([farthest[-2]] if len(farthest) > 1 else [])
            if blocker is not None:
                self._early_priority.append(blocker)
            self._early_priority.append(hardest)
        if self._hard_first:
            self._slots.remove(rows * 2 - 1)
            self._slots.insert(0, rows * 2 - 1)
        self._slot_cursor = 0
        self._used = set()
        self._route = []
        self._route_attempted = False
        self._route_found = False
        self._stage_steps = 0
        self.phase = CookiePhase.SELECT_COOKIE

    def _next_phase(self, phase):
        self.phase = phase
        self._stage_steps = 0

    def _plan(self):
        env = self.env
        site = self._l_site
        qids = self._l_qpos.tolist()
        current = self.data.site_xmat[site].reshape(3, 3)
        offset = current.T @ (self.data.geom_xpos[list(env._left_finger_geoms)].mean(axis=0)
                              - self.data.site_xpos[site])
        work = mujoco.MjData(env.model)
        work.qpos[:] = self.data.qpos
        body = env._target_bin_body
        box_r = self.data.xmat[body].reshape(3, 3).copy()
        slot = self._slots[self._slot_cursor]
        xy = np.asarray(env.TARGET_SLOTS_LOCAL[slot], dtype=float)
        scene = env.config.cookie_transfer
        z = scene.target_floor_z_m + scene.bin_wall_thickness_m / 2 + env.COOKIE_HALF_SIZE[2] + .001
        destination = self.data.xpos[body] + box_r @ np.array([*xy, z])
        candidates = [int(i) for i in np.flatnonzero(env._active)
                      if i not in self._used and env.privileged_cookie_in_source(int(i))]
        if self._hard_first:
            candidates.sort(key=lambda i: (-int(i // 20) if slot % 2 else 0,
                                           -env.cookie_positions[i, 1] if slot % 2
                                           else env.cookie_positions[i, 1]))
        else:
            candidates.sort(key=lambda i: env.cookie_positions[i, 1])
        if self._early_priority:
            ranks = {cookie: rank for rank, cookie in enumerate(self._early_priority)}
            candidates.sort(key=lambda i: ranks.get(i, len(ranks)))
        for cookie in candidates:
            center = env.cookie_positions[cookie].copy()
            cookie_r = self.data.xmat[env._cookie_bodies[cookie]].reshape(3, 3).copy()
            pad = center.copy()
            pad[2] += env.COOKIE_HALF_SIZE[2] - .002
            for pitch in _PITCHES:
                angle = np.deg2rad(pitch)
                c, s = np.cos(angle), np.sin(angle)
                ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                for flip in (np.eye(3), np.diag([-1., 1., -1.])):
                    rotation = ry @ _CANON @ flip
                    source = _pose_candidate(env, work, qids, site, pad, np.eye(3), offset, rotation)
                    if source is None:
                        continue
                    pre, pick = source
                    rel_r = rotation.T @ cookie_r
                    rel_p = rotation.T @ (center - pick)
                    goal_r = box_r @ rel_r.T
                    goal_pad = destination - goal_r @ rel_p + goal_r @ offset
                    if _pose_candidate(env, work, qids, site, goal_pad, box_r, offset, goal_r) is not None:
                        self._cookie, self._pre, self._pick = cookie, pre, pick
                        self._pick_r, self._xy = rotation, xy
                        self._route_attempted = False
                        self._route_found = False
                        return True
        return False

    def _plan_carrying_route(self):
        self._route_attempted = True
        q = _carrying_route(self.env, self._cookie, self._above, self._place_r,
                            self._held_p, self._held_r)
        self._route_found = q is not None
        if q is not None:
            q0 = self.env.current_joint_action[:7].copy()
            self._route_base = self.env.current_joint_action.copy()
            n = int(np.ceil(np.max(np.abs(q - q0)) / .008))
            self._route = [q0 + (q - q0) * f for f in np.linspace(0, 1, n + 1)[1:]] + [q] * 50

    def _act_step(self, observation=None, task=""):
        del observation, task
        if self.finished:
            return self._hold_command(0.1)
        if self.phase is CookiePhase.SELECT_COOKIE:
            if not self._plan():
                self._fail("no matching collision-free source and target endpoints; "
                           f"slot={self._slots[self._slot_cursor]}")
                return self._hold_command(1.0)
            self._next_phase(CookiePhase.APPROACH)
        self._stage_steps += 1
        phase = self.phase
        specs = {
            CookiePhase.APPROACH: (self._pre, self._pick_r, .5, .004, 350),
            CookiePhase.PRE_CLOSE: (self._pre, self._pick_r, .02, .001, 20),
            CookiePhase.DESCEND: (self._pick, self._pick_r, .02, .0008, 240),
            CookiePhase.CLOSE: (self._pick, self._pick_r, 0., .0005, 25),
            CookiePhase.LIFT: (self._pre, self._pick_r, 0., .0015, 250),
        }
        if phase in (CookiePhase.MOVE_TO_SLOT, CookiePhase.DESCEND_TO_PLACE, CookiePhase.OPEN,
                     CookiePhase.RETRACT, CookiePhase.VERIFY_RELEASE):
            specs.update({
                CookiePhase.MOVE_TO_SLOT: (self._above, self._place_r, 0., .003, 600),
                CookiePhase.DESCEND_TO_PLACE: (self._place, self._place_r, 0., .001, 300),
                CookiePhase.OPEN: (self._place, self._place_r, .1, .0005, 25),
                CookiePhase.RETRACT: (self._above, self._place_r, .1, .0015, 200),
                CookiePhase.VERIFY_RELEASE: (self._above, self._place_r, .1, .001, 10),
            })
        goal, rotation, opening, speed, limit = specs[phase]
        if phase is CookiePhase.MOVE_TO_SLOT and self._route:
            command = self._route_base.copy()
            command[:7] = self._route.pop(0)
            command[7] = 0.
            return command
        action, reached = self._servo(goal, rotation, opening, speed=speed)
        opened = abs(self.env.current_joint_action[7] - opening) < .015
        fixed = phase in (CookiePhase.PRE_CLOSE, CookiePhase.CLOSE, CookiePhase.OPEN,
                          CookiePhase.VERIFY_RELEASE)
        done = self._stage_steps >= limit if fixed else reached and opened
        if not done and not fixed and self._stage_steps >= limit:
            self._fail(f"sparse expert timed out in {phase.name}; cookie={self._cookie}, "
                       f"slot={self._slots[self._slot_cursor]}, route_found={self._route_found}")
            return action
        if phase is CookiePhase.MOVE_TO_SLOT and not done and self._stage_steps >= 200 and not self._route_attempted:
            self._plan_carrying_route()
        if not done:
            return action
        transitions = {
            CookiePhase.APPROACH: CookiePhase.PRE_CLOSE,
            CookiePhase.PRE_CLOSE: CookiePhase.DESCEND,
            CookiePhase.DESCEND: CookiePhase.CLOSE,
            CookiePhase.CLOSE: CookiePhase.LIFT,
            CookiePhase.LIFT: CookiePhase.MOVE_TO_SLOT,
            CookiePhase.MOVE_TO_SLOT: CookiePhase.DESCEND_TO_PLACE,
            CookiePhase.DESCEND_TO_PLACE: CookiePhase.OPEN,
            CookiePhase.OPEN: CookiePhase.RETRACT,
            CookiePhase.RETRACT: CookiePhase.VERIFY_RELEASE,
        }
        if (phase in (CookiePhase.CLOSE, CookiePhase.LIFT, CookiePhase.MOVE_TO_SLOT)
                and not all(self.env.privileged_left_finger_contacts(self._cookie))):
            self._fail(f"cookie {self._cookie} lost gripper contact in {phase.name}")
            return action
        if phase is CookiePhase.LIFT:
            body = self.env._cookie_bodies[self._cookie]
            tool = self.data.site_xmat[self._l_site].reshape(3, 3).copy()
            obj = self.data.xmat[body].reshape(3, 3).copy()
            self._held_r = tool.T @ obj
            self._held_p = tool.T @ (self.data.xpos[body] - self.data.site_xpos[self._l_site])
            target = self.env._target_bin_body
            box_r = self.data.xmat[target].reshape(3, 3).copy()
            scene = self.env.config.cookie_transfer
            z = scene.target_floor_z_m + scene.bin_wall_thickness_m / 2 + self.env.COOKIE_HALF_SIZE[2] + .001
            pos = self.data.xpos[target] + box_r @ np.array([*self._xy, z])
            self._place_r = box_r @ self._held_r.T
            self._place = pos - self._place_r @ self._held_p
            self._above = self._place + box_r[:, 2] * .07
            rows = self.env.config.cookie_transfer.target_rows
            late_far_source = (rows == 5 and not self._hard_first
                               and self._slot_cursor >= len(self._slots) - 2
                               and self.env.SOURCE_POSITIONS[self._cookie][0]
                               > self.env.config.cookie_transfer.source_bin_center_m[0])
            if ((self._hard_first and (self._slot_cursor == len(self._slots) - 1
                                      or (rows == 10 and self._slot_cursor >= rows)))
                    or late_far_source):
                # With the box nearly full, choose a checked carrying path
                # before committing to a Cartesian motion through it.
                self._plan_carrying_route()
        if phase is CookiePhase.VERIFY_RELEASE:
            slot = self._slots[self._slot_cursor]
            if not self.env.privileged_cookie_in_slot(self._cookie, slot) or not all(
                self.env.privileged_cookie_in_target(i) for i in self._used
            ):
                self._fail(f"sparse placement {slot} was not stable")
                return action
            self._used.add(self._cookie)
            if self._cookie in self._early_priority:
                self._early_priority.remove(self._cookie)
            self.batch_reports.append({"cookies": [self._cookie], "slot": slot, "released": True,
                                       "carrying_route_found": self._route_found})
            self._slot_cursor += 1
            self._next_phase(CookiePhase.DONE if self._slot_cursor == len(self._slots)
                             else CookiePhase.SELECT_COOKIE)
        else:
            self._next_phase(transitions[phase])
        return action
