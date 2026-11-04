"""Optional projected target cues, shared by recording and adapted-policy inference.

The current simulator camera calibration and truth object poses supply these
cues. They are privileged goal annotations, not a new RGB pose detector.
"""
from itertools import product
import numpy as np

MARKER_VERSION = "projected_batch_targets_v1"
MARKED_INSTRUCTION = ("Pick the five cookies outlined in cyan, place them in the green target region, "
                      "release, and retreat.")
CAMERAS = ("front", "left_wrist", "right_wrist")


def project_points(points, camera_position, camera_rotation, fovy_deg, height, width):
    local = (np.asarray(points)-camera_position) @ camera_rotation
    depth = -local[:, 2]
    focal = height / (2*np.tan(np.deg2rad(fovy_deg)/2))
    safe_depth = np.maximum(depth, 1e-8)
    uv = np.c_[width/2 + focal*local[:, 0]/safe_depth,
               height/2 - focal*local[:, 1]/safe_depth]
    return uv, depth > .001


def draw_box(image, uv, visible, color):
    result = image.copy()
    uv = uv[visible]
    if len(uv) < 2 or not np.isfinite(uv).all():
        return result
    h, w = image.shape[:2]
    lower, upper = np.floor(uv.min(0)).astype(int)-3, np.ceil(uv.max(0)).astype(int)+3
    if upper[0] < 0 or upper[1] < 0 or lower[0] >= w or lower[1] >= h:
        return result
    x0, y0 = np.maximum(lower, 0)
    x1, y1 = np.minimum(upper, [w-1, h-1])
    if x1-x0 < 2 or y1-y0 < 2:
        return result
    result[y0:min(y0+2,h), x0:x1+1] = color
    result[max(y1-1,0):y1+1, x0:x1+1] = color
    result[y0:y1+1, x0:min(x0+2,w)] = color
    result[y0:y1+1, max(x1-1,0):x1+1] = color
    return result


def annotate_goals(observation, env, scene, goal):
    signs = np.asarray(list(product((-1, 1), repeat=3)))
    corners = signs * scene.half_size
    source = np.concatenate([scene.cookies[i].position + corners @ scene.cookies[i].rotation.T
                             for i in goal.cookie_ids])
    target = np.concatenate([scene.world(p) + corners @ scene.box_rotation.T
                             for p in goal.target_positions])
    result = dict(observation)
    for name in CAMERAS:
        key = "observation.images." + name
        image = np.asarray(observation[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Goal cues require the original uint8 RGB images")
        cid = env.model.camera(name).id
        args = (env.data.cam_xpos[cid], env.data.cam_xmat[cid].reshape(3, 3),
                env.model.cam_fovy[cid], *image.shape[:2])
        uv, visible = project_points(source, *args)
        marked = draw_box(image, uv, visible, (0, 220, 255))
        uv, visible = project_points(target, *args)
        result[key] = draw_box(marked, uv, visible, (80, 255, 80))
    return result


class GoalMarkedPolicy:
    """Opt-in wrapper for a checkpoint actually adapted with these same cues."""
    def __init__(self, policy, env, provider, metadata):
        if metadata.get("goal_conditioning") != MARKER_VERSION:
            raise ValueError("This checkpoint was not declared trained with the projected goal cues")
        self.policy, self.env, self.provider = policy, env, provider
        self.action_mode = policy.action_mode
        self.config = policy.config
        self.name = "goal_marked_" + policy.name
        self.goal = None

    def set_batch_goal(self, goal):
        self.goal = goal

    def reset(self, context=None):
        self.policy.reset(context)

    def act(self, observation, task=""):
        if self.goal is None:
            raise RuntimeError("The adapted policy has no selected goal")
        scene = self.provider.observe(observation)
        return self.policy.act(annotate_goals(observation, self.env, scene, self.goal), MARKED_INSTRUCTION)

    def close(self):
        self.policy.close()
