"""What LIBERO's policy actually reads as its end-effector pose, and what this arm should send.

Two facts that are not visible from the field names, and each of which silently produces a
180-degree error if it is assumed the other way.

**The orientation is a body, and the position is a site.**  robosuite's `robot0_eef_pos` is
`site_xpos[gripper0_grip_site]`, but `robot0_eef_quat` is
`get_body_xquat(robot_model.eef_name)` -- and `eef_name` is `robot0_right_hand`, a *different*
frame 97 mm above the grip point whose axes are a 180-degree rotation of the world's.  The two
halves of one state vector are expressed in two different frames.

**The quaternion is (x, y, z, w).**  `LiberoProcessorStep._quat2axisangle` documents and reads
`quat[:, 3]` as w.  Read as (w, x, y, z) -- which is MuJoCo's own order, and this project's --
LIBERO's starting orientation looks like "identity, 3.2 degrees off".  Read correctly it is
**180.03 degrees about world +x**.  A state that is a half turn away from every training
sample is not a subtle error, and it is invisible: three plausible numbers either way.

So this prints, at the state an episode starts from, the eight-wide `observation.state` LIBERO
builds, and the constant that makes this arm's three rotation numbers equal LIBERO's.

Usage::

    python scripts/measure_state_convention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

EEF_SITE = "L_eef"


def libero_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The eight numbers LIBERO's policy is handed, and the two frames they come from."""

    import mujoco
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env
    from lerobot.processor.env_processor import LiberoProcessorStep

    config = LiberoEnvConfig(task="libero_spatial", task_ids=[0])
    envs = make_env(config, n_envs=1)
    vec = envs["libero_spatial"][0]
    sub = vec.envs[0]
    inner = getattr(sub._env, "env", sub._env)  # noqa: SLF001
    inner.hard_reset = False
    observation, _ = vec.reset(seed=[0])
    model, data = inner.sim.model._model, inner.sim.data._data  # noqa: SLF001

    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
    rotation = np.asarray(data.xmat[hand], dtype=np.float64).reshape(3, 3).copy()

    # The pipeline step needs a whole transition to run through, so the one method that
    # matters is called directly -- it is the function that turns the environment's
    # quaternion into the three numbers the policy reads, and using it rather than a
    # reimplementation is the point of this script.
    step = LiberoProcessorStep()
    quaternion = np.asarray(observation["robot_state"]["eef"]["quat"], dtype=np.float64).ravel()
    turned = np.asarray(
        step._quat2axisangle(torch.as_tensor(quaternion).float().view(1, 4))  # noqa: SLF001
    ).ravel()
    position = np.asarray(observation["robot_state"]["eef"]["pos"], dtype=np.float64).ravel()
    gripper = np.asarray(observation["robot_state"]["gripper"]["qpos"], dtype=np.float64).ravel()
    state = np.r_[position, turned, gripper]
    vec.close()
    return state, rotation, position


def our_frame(scene: str) -> np.ndarray:
    """This arm's end-effector site frame at its home."""

    import mujoco

    from a3_dual_arm_sim.collection import scene_by_name

    env = scene_by_name(scene).build_env(0.0, 0.0)
    env.reset(seed=0, options=None)
    site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    return np.asarray(env.data.site_xmat[site], dtype=np.float64).reshape(3, 3).copy()


def main() -> int:
    import mujoco

    state, hand_rotation, grip_position = libero_state()
    print("LIBERO's `observation.state` at the start of an episode (8 wide)")
    print(f"  eef position      {np.round(state[0:3], 4)}   (site `gripper0_grip_site`)")
    print(f"  eef axis-angle    {np.round(state[3:6], 4)}   (body `robot0_right_hand`)")
    print(f"  gripper qpos      {np.round(state[6:8], 4)}")
    if not np.allclose(state[0:3], grip_position, atol=1e-4):
        raise RuntimeError("the state's position is not the grip site's position")
    angle = np.degrees(np.linalg.norm(state[3:6]))
    print(f"  -> the rotation is {angle:.2f} degrees about "
          f"{np.round(state[3:6] / max(angle, 1e-9), 4)}, not the identity")

    ours = our_frame("a3_libero_props_matched")
    print("\nthis arm's end-effector site frame at its own home")
    print(np.round(ours, 4))

    constant = ours.T @ hand_rotation
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, constant.reshape(9))
    turn = np.degrees(np.arccos(np.clip((np.trace(constant) - 1) / 2, -1, 1)))
    print(f"\nconvention constant C = R_ours^T R_hand  ({turn:.2f} degrees)")
    print(np.round(constant, 4))
    print(f"  this arm's tool axis (+y) in the hand's frame: "
          f"{np.round(constant.T @ np.array([0.0, 1.0, 0.0]), 4)}  "
          f"(LIBERO's approach direction is +z)")
    print("\nthe manifest line that makes the two agree:")
    print(f"    - state_rotation_reference={','.join(f'{v:.6f}' for v in quaternion)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
