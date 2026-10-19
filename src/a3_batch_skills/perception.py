"""Replaceable scene estimation; simulator truth is named and recorded explicitly."""
from typing import Protocol
import numpy as np
from .contracts import CookieState, SceneState


class SceneProvider(Protocol):
    def observe(self, observation: dict) -> SceneState: ...


class ExternalSceneProvider:
    """Connect a calibrated visual pose estimator without changing the controller.

    No image-to-pose estimator is silently substituted with simulator truth.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def observe(self, observation):
        state = self.estimator(observation)
        if not isinstance(state, SceneState) or state.source == "simulator_truth":
            raise ValueError("External perception must return its own named SceneState")
        return state


class SimulatorSceneProvider:
    source = "simulator_truth"

    def __init__(self, env):
        self.env = env
        self._mapping = {int(geom): i for i, group in enumerate(env._cookie_collision_geoms)
                         for geom in group}

    def observe(self, observation=None):
        import mujoco
        env = self.env
        fingers = tuple(int(g) for g in env._left_finger_geoms)
        graph = {}
        forces = [0.0, 0.0]
        finger_contacts = set()
        for k in range(env.data.ncon):
            contact = env.data.contact[k]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            force = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, k, force)
            for side, finger in enumerate(fingers):
                if finger in (g1, g2):
                    forces[side] += max(0.0, float(force[0]))
                    other = g2 if g1 == finger else g1
                    if other in self._mapping:
                        finger_contacts.add(self._mapping[other])
            if force[0] < 0.005:
                continue
            nodes = []
            for geom in (g1, g2):
                if geom in fingers:
                    nodes.append(-(fingers.index(geom) + 1))
                elif geom in self._mapping:
                    nodes.append(self._mapping[geom])
            if len(nodes) == 2:
                a, b = nodes
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
        reachable, pending = set(), [-1]
        while pending:
            node = pending.pop()
            if node not in reachable:
                reachable.add(node)
                pending.extend(graph.get(node, set()) - reachable)
        held = frozenset(i for i in reachable if i >= 0) if -2 in reachable else frozenset()
        cookies = {}
        for i, body in enumerate(env._cookie_bodies):
            linear, angular = env._cookie_speed(i, relative_to_target=True)
            cookies[i] = CookieState(i, env.data.xpos[body].copy(),
                env.data.xmat[body].reshape(3, 3).copy(), linear, angular,
                bool(env.privileged_cookie_in_source(i)), i in finger_contacts)
        scene = env.config.cookie_transfer
        site = env._eef_sites[0]
        return SceneState(cookies, env.data.xpos[env._target_bin_body].copy(),
            env.data.xmat[env._target_bin_body].reshape(3, 3).copy(),
            env.data.site_xpos[site].copy(), env.data.site_xmat[site].reshape(3, 3).copy(),
            np.asarray(env.COOKIE_HALF_SIZE).copy(), np.asarray(env.TARGET_INNER_HALF_SIZE).copy(),
            scene.target_floor_z_m + scene.bin_wall_thickness_m / 2,
            env.TARGET_WALL_HEIGHT, held, tuple(forces), bool(env._safety_stop), self.source)
