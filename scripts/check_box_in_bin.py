"""Is the box actually *inside* the source bin, or resting on its rim?

Projects the centres of both containers and, more usefully, diffs the front frame of the
box-on-the-table scene against the box-in-the-bin scene: every pixel that changes is a pixel
the move affected, and the Cookie's 23 yellow pixels are either inside that change or not.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import scene_by_name

SIZE = 256.0


def front(name):
    env = scene_by_name(name).build_env(0.0, 0.0)
    observation, _ = env.reset(seed=0, options=None)
    model, data = env.model, env.data
    frame = np.asarray(observation["observation.images.front"]).astype(np.float64)
    out = {}
    for body in ("source_bin", "target_bin", "cookie_0"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        point = np.asarray(data.xpos[index])
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        rot = np.asarray(data.cam_xmat[cam]).reshape(3, 3)
        local = rot.T @ (point - np.asarray(data.cam_xpos[cam]))
        f = 0.5 * SIZE / np.tan(np.deg2rad(float(model.cam_fovy[cam])) / 2.0)
        px = (SIZE / 2 - f * local[0] / (-local[2]), SIZE / 2 - f * local[1] / (-local[2]))
        out[body] = (point, px, data, model)
        # The bin's own rim: the highest point of any of its geoms.
        if body == "source_bin":
            tops = []
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == index:
                    tops.append(float(data.geom_xpos[g][2] + model.geom_size[g][2]))
            out["source_bin_rim_z"] = max(tops)
    env.close()
    return frame, out


a_frame, a = front("a3_probe_home_over_cookie")   # box on the table
b_frame, b = front("a3_probe_box_in_bin")         # box in the bin

print("body                  box on table            box in the bin")
for body in ("source_bin", "target_bin", "cookie_0"):
    pa, xa, _, _ = a[body]
    pb, xb, _, _ = b[body]
    print(f"  {body:12s} xpos {np.round(xa, 4)}  ->  xpos {np.round(xb, 4)}")
    print(f"  {'':12s}  px  ({xa[0]:6.1f}, {xa[1]:6.1f})       px  ({xb[0]:6.1f}, {xb[1]:6.1f})")
print(f"  source_bin rim top z {a['source_bin_rim_z']:.4f}"
      f"   box bottom z {a['target_bin'][0][2]:.4f} (table) / {b['target_bin'][0][2]:.4f} (bin)")

diff = np.abs(a_frame - b_frame).max(axis=2) > 8
ys, xs = np.nonzero(diff)
print(f"\nfront frame, box on table vs box in the bin: {int(diff.sum())} px differ"
      f"   x {xs.min()}..{xs.max()}  y {ys.min()}..{ys.max()}")


def yellow(frame):
    r, g, bl = frame[..., 0], frame[..., 1], frame[..., 2]
    return (r > 110) & (g > 95) & (bl < 90) & (r - bl > 45) & (g - bl > 35)


ya, yb = yellow(a_frame), yellow(b_frame)
print(f"Cookie's yellow pixels   box on table {int(ya.sum())}   box in the bin {int(yb.sum())}")
print(f"  of those, changed by the move: {int((ya & diff).sum())} / {int((yb & diff).sum())}")
