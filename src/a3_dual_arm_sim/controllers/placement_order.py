"""Fixed policy-front-camera ordering; never uses the interactive viewer."""
import numpy as np


def front_camera_column_order(env):
    slots = np.asarray(env.TARGET_SLOTS_LOCAL)
    columns = np.unique(slots[:, 0])
    rotation = env.data.xmat[env._target_bin_body].reshape(3, 3)
    camera = env._camera_ids["front"]
    camera_rotation = env.data.cam_xmat[camera].reshape(3, 3)
    projected = []
    for index, x in enumerate(columns):
        center = np.array([x, slots[np.isclose(slots[:, 0], x), 1].mean(), env._target_cookie_center_z])
        world = env.data.xpos[env._target_bin_body] + rotation @ center
        local = camera_rotation.T @ (world - env.data.cam_xpos[camera])
        if local[2] >= -1e-6:
            raise ValueError("Target column is behind the policy front camera")
        # Image y points down; larger y is closer to the bottom of the image.
        projected.append((-local[1] / -local[2], index))
    return tuple(index for _, index in sorted(projected, reverse=True))


def ordered_column(env, index):
    return getattr(env, "demonstration_column_order", (0, 1))[index]
