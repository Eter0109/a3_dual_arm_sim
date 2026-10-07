"""Where LIBERO's own policy expects to find the hand, solved onto this arm's joints.

Why this exists
---------------
The state a LIBERO policy is handed is the end-effector pose *in LIBERO's world*, and the
pictures it is handed are LIBERO's two cameras.  Those two halves of one observation agree
only while the arm is where LIBERO's arm is.  This arm's home put the tool 360 mm away from
LIBERO's -- most of it in y, in front of the bowl where LIBERO's Panda is and behind it
where this one was -- so the policy was reading a hand position that its own image
contradicted, and every command it emitted began from that lie.

The offset in the adapter used to paper over this: a fitted translation that mapped this
arm's home onto LIBERO's hover *in the state only*.  Moving the arm itself is the honest
version, and it makes the offset collapse to the one thing it should always have been --
the map between the two worlds -- `-LIBERO_WORLD_OFFSET_M`.

What is solved
--------------
Position: LIBERO's own end-effector point at the start of an episode -- the one behind the
`robot_state.eef.pos` the policy reads, measured as (-0.2112, -0.0111, 1.1742) -- put
through the world offset measured at the bowl.

Orientation: **unchanged**, and that is a measurement rather than a simplification.  Both
arms already stand with the tool pointing straight down, which is the pose the state is
about; LIBERO's own reported quaternion at the start of an episode is (0.9996, -0.0009,
-0.0279, -0.0003), i.e. 3.20 degrees from the identity, and this arm's adapter already
reports exactly zero there by referencing its own home.  "Match LIBERO's quaternion" would
have been the wrong instruction twice over: the *reported* quaternion is not the end-effector
frame's own orientation -- that frame is a 180-degree rotation of the world in LIBERO's model
-- so copying the one that is visible would have rotated the tool by 180 degrees while
looking like precision.

Usage::

    python scripts/solve_libero_home.py            # measure and solve
    python scripts/solve_libero_home.py --write    # ...and print the YAML block
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from a3_dual_arm_sim.collection import LIBERO_WORLD_OFFSET_M  # noqa: E402
from a3_dual_arm_sim.contracts import LEFT_JOINTS  # noqa: E402

EEF_SITE = "L_eef"


def libero_home_pose() -> tuple[np.ndarray, np.ndarray]:
    """LIBERO's end-effector point, at the state an episode actually starts from.

    Read off the site robosuite calls the end-effector (`gripper0_grip_site`) *and* checked
    against the `robot_state.eef.pos` the policy is handed, because the two are the same
    point here and only one of them is the thing being matched.
    """

    import mujoco
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env

    config = LiberoEnvConfig(task="libero_spatial", task_ids=[0])
    envs = make_env(config, n_envs=1)
    vec = envs["libero_spatial"][0]
    sub = vec.envs[0]
    inner = getattr(sub._env, "env", sub._env)  # noqa: SLF001
    inner.hard_reset = False
    observation, _ = vec.reset(seed=[0])
    model, data = inner.sim.model._model, inner.sim.data._data  # noqa: SLF001
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    if site < 0:
        raise RuntimeError("LIBERO's model has no `gripper0_grip_site` site")
    rotation = np.asarray(data.site_xmat[site], dtype=np.float64).reshape(3, 3).copy()
    position = np.asarray(data.site_xpos[site], dtype=np.float64).copy()
    reported = np.asarray(observation["robot_state"]["eef"]["pos"]).ravel()
    if not np.allclose(position, reported, atol=1e-4):
        raise RuntimeError(
            f"the site is at {position} and the policy is handed {reported}; the home has to "
            "be solved against the point the policy actually reads"
        )
    vec.close()
    return position, rotation


def our_home_pose(scene: str) -> tuple[np.ndarray, np.ndarray, object, np.ndarray]:
    """This arm's end-effector site at its configured home, and the environment."""

    import mujoco

    from a3_dual_arm_sim.collection import scene_by_name

    env = scene_by_name(scene).build_env(0.0, 0.0)
    env.reset(seed=0, options=None)
    site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    rotation = np.asarray(env.data.site_xmat[site], dtype=np.float64).reshape(3, 3).copy()
    position = np.asarray(env.data.site_xpos[site], dtype=np.float64).copy()
    joints = np.asarray(env.config.home.left, dtype=np.float64)
    return position, rotation, env, joints


def solve_position(env, target_position: np.ndarray, target_rotation: np.ndarray,
                   start: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Seven left-arm joint angles reaching for LIBERO's home with the tool pointing down.

    **The spin about the tool axis is free, and the position is not held exactly.**  That is
    forced, not chosen.  Solved as `position + full orientation` this arm lands 102 mm short
    of LIBERO's home and as `position + tool axis` it lands 47 mm short, while `position
    alone` reaches to 1.3 mm only by tilting the tool **55 degrees** off vertical.  The gate
    is this arm's wrist: `L_WRIST_P` spans +/-40 degrees, and the target stands at the height
    of the shoulder 0.365 m in front of it, which to point a tool straight down needs more
    bend than the joint has.

    So the class of poses LIBERO starts from does not exist on this arm, and the honest
    target is the nearest pose that does: tool down (0.7 degrees off), which is the only way
    a grasp ever happens, and position as close as the kinematics allow.  The residual is
    returned rather than hidden -- see the printed comparison, which is the measurement the
    decision to look at a different arm rests on.

    The orientation term is a straight axis difference rather than a quaternion error
    because only the tool's *direction* is wanted; the damping and the joint limits are the
    same damped least squares `CartesianDeltaAdapter.convert` uses.
    """

    import mujoco
    from scipy.optimize import least_squares

    model = env.model
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    qids = np.asarray(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
         for name in LEFT_JOINTS],
        dtype=np.int32,
    )
    limits = np.asarray(
        [model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
         for name in LEFT_JOINTS],
        dtype=np.float64,
    )
    work = mujoco.MjData(model)
    #: This arm's tool axis is the end-effector frame's **+y** (the wrist camera was moved to
    #: `(0, 0.045, 0)` to sit on it).  LIBERO's gripper hangs straight down, so the target
    #: direction is world -z.
    tool_local = np.array([0.0, 1.0, 0.0])
    down = np.array([0.0, 0.0, -1.0])

    def residual(qpos: np.ndarray) -> np.ndarray:
        work.qpos[:] = env.data.qpos
        work.qpos[qids] = qpos
        mujoco.mj_forward(model, work)
        rotation = work.site_xmat[site].reshape(3, 3)
        return np.r_[
            work.site_xpos[site] - target_position,
            0.25 * (rotation @ tool_local - down),
            0.02 * (qpos - start),
        ]

    result = least_squares(
        residual,
        np.clip(start, limits[:, 0] + 1e-4, limits[:, 1] - 1e-4),
        bounds=(limits[:, 0] + 1e-4, limits[:, 1] - 1e-4),
        max_nfev=600, ftol=1e-13, xtol=1e-13, gtol=1e-13,
    )
    work.qpos[:] = env.data.qpos
    work.qpos[qids] = result.x
    mujoco.mj_forward(model, work)
    position_error = float(np.linalg.norm(work.site_xpos[site] - target_position)) * 1000
    achieved = work.site_xmat[site].reshape(3, 3)
    tool_error = float(np.degrees(np.arccos(np.clip((achieved @ tool_local) @ down, -1, 1))))
    cosine = np.clip((np.trace(achieved.T @ target_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return (
        np.asarray(result.x, dtype=np.float64),
        position_error,
        tool_error,
        float(np.degrees(np.arccos(cosine))),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="a3_libero_props_matched")
    parser.add_argument("--write", action="store_true",
                        help="print the `home:` block to paste into the scene config")
    args = parser.parse_args()

    libero_position, libero_rotation = libero_home_pose()
    offset = np.asarray(LIBERO_WORLD_OFFSET_M, dtype=np.float64)
    ours_position, ours_rotation, env, ours_joints = our_home_pose(args.scene)

    print("LIBERO's own `robot0_eef` at the start of an episode")
    print(f"  position {np.round(libero_position, 4)}")
    print(f"  rotation\n{np.round(libero_rotation, 4)}")
    print()
    print(f"this arm's `{EEF_SITE}` at its configured home")
    print(f"  position {np.round(ours_position, 4)}   (offset from LIBERO's: "
          f"{np.round(ours_position - libero_position, 4)})")
    print(f"  rotation\n{np.round(ours_rotation, 4)}")
    convention = libero_rotation.T @ ours_rotation
    angle = np.degrees(np.arccos(np.clip((np.trace(convention) - 1) / 2, -1, 1)))
    print(f"\n  the constant relating the two end-effector frames: {angle:.2f} degrees")
    print("  (this is why the orientation is held rather than copied -- matching LIBERO's")
    print("   quaternion outright would rotate the tool by that much)")

    target = libero_position + offset
    print(f"\nsolving for the tool at {np.round(target, 4)} "
          f"(LIBERO's home through the world offset), orientation unchanged")

    joints, position_error, tool_error, orientation_error = solve_position(
        env, target, ours_rotation, ours_joints
    )
    print(f"  position error       {position_error:.3f} mm")
    print(f"  tool axis off vertical {tool_error:.4f} degrees")
    print(f"  full orientation error {orientation_error:.2f} degrees (free -- spin is unused)")
    print(f"  joints               {[round(float(v), 6) for v in joints]}")
    print(f"  home it replaces     {[round(float(v), 6) for v in ours_joints]}")
    print(f"  largest joint change {np.max(np.abs(joints - ours_joints)):.4f} rad")
    print()
    print(f"  a residual position error of {position_error:.0f} mm is the measurement, not a")
    print("  tolerance: it says how far this arm's reachable poses are from the pose LIBERO")
    print("  starts every episode in.")
    if args.write:
        body = ", ".join(f"{float(v):.6f}" for v in joints)
        print(f"\nhome:\n  left: [{body}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
