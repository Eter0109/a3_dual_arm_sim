"""Where does the ridge rollout leave the manifold, and how far?

`compare` reported the ridge rollout as `saturated 0.96`, `corrected 0.96`,
`motion mean 0.002 rad` -- the arm pinned against a joint stop with its command overruled
on nearly every step, and no Cookie moved.  Those four numbers are consistent with at
least two very different stories, and the aggregate cannot tell them apart:

1. **The features are wrong at rollout time.**  If the live camera frames do not match the
   frames the coefficients were fitted on -- a different render pass, a different resize,
   a mirrored camera -- then the linear read-out is being handed inputs from another
   distribution and its output is meaningless from step one.
2. **The model extrapolates.**  A linear map fitted near a trajectory is only valid near
   that trajectory.  The first few steps are sane, the state drifts a little, the map
   amplifies the drift, the target leaves the joint range, the clip pins the arm, and the
   state that results is one the fit has never seen.  Fixed point: the arm stops, 0.05 rad
   of correction per step against a limit.

The two are separated by *when* the first saturation happens and *what the state was*
just before it, so this script prints the first steps one at a time: the state, the
target, the distance between them, what the environment actually applied, and which joint
is pinned.  It also compares the live wrist frame against the feature cache, because
story 1 has a cheap test and story 2 makes a specific prediction about step one.

Usage::

    MUJOCO_GL=egl python scripts/probe_ridge_divergence.py --steps 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from a3_dual_arm_sim.contracts import LEFT_WRIST_IMAGE, STATE  # noqa: E402

import adapter_ridge_cookie  # noqa: E402

SCENE = "a3_cookie_single_piece"
COEFFS = Path("/root/autodl-tmp/logs/ridge_cookie_policy.npz")
TOWER = "/root/autodl-tmp/pi05-runs/a3_one_base/checkpoints/004000/pretrained_model"
CACHE = Path("/root/autodl-tmp/logs/vision_features.npz")

JOINT_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "夹爪"]


class _Request:
    """The two accessors the adapter uses, over a plain observation dict.

    The adapter is written against the *manifest's* image names (``image2``), because
    that is what the bridge hands it; the environment names the same camera
    ``observation.images.left_wrist``.  This is the one place the two vocabularies meet
    outside the bridge, and it is why the rollout goes through `manifest.observation`
    at all.
    """

    #: Manifest image name -> A3 observation key.  Same pairing as
    #: `configs/policies/ridge_cookie_one.yaml`.
    NAMES = {"image2": LEFT_WRIST_IMAGE}
    #: Width each vector is cut to, mirroring the manifest's `slice: [0, N]`.  The
    #: environment is bimanual and reports sixteen joint values; this interface commands
    #: eight, and the coefficients were fitted on eight.  Passing all sixteen raises
    #: "operands could not be broadcast together with shapes (16,) (8,)" from inside the
    #: standardisation, which points at the arithmetic rather than at the missing slice.
    SLICES = {"observation.state": 8}

    def __init__(self, observation):
        self._observation = observation
        self.chunk = 1
        self.task = ""

    def image(self, name):
        return self._observation[self.NAMES.get(name, name)]

    def vector(self, name):
        values = np.asarray(self._observation[name], dtype=np.float64)
        return values[: self.SLICES[name]] if name in self.SLICES else values


class _Dataset:
    """Nearest-neighbour distance from a live joint state to the training states.

    The one number that says whether a step is still *on* the demonstration manifold.
    A behaviour-cloning rollout fails legitimately when it leaves the states the fit was
    trained on, because the fit has no way to be right there -- so "it saturated at step
    88" is only interpretable next to "and step 87 was already 0.4 rad away from anything
    in the dataset".
    """

    def __init__(self, root: Path):
        import pandas as pd

        table = pd.concat(
            [pd.read_parquet(f, columns=["observation.state"])
             for f in sorted(root.glob("data/**/*.parquet"))],
            ignore_index=True,
        )
        self.states = np.stack(
            [np.asarray(v, dtype=np.float64) for v in table["observation.state"]]
        )[:, :8]
        # Spacing of the dataset itself: the distance from one training state to its own
        # nearest neighbour.  A live state whose nearest-neighbour distance is comparable
        # to this is inside the data; tens of times larger is outside it.
        #
        # Done through the ||a-b||^2 = |a|^2 + |b|^2 - 2a.b identity rather than by
        # subtraction, because broadcasting (chunk, N, 8) against (N, 8) materialises
        # chunk*N*8 float64 -- 11 GB at chunk 2000, which is how a diagnostic turns into
        # an OOM.
        norms = (self.states ** 2).sum(axis=1)
        spacing = []
        for start in range(0, len(self.states), 500):
            block = self.states[start:start + 500]
            squared = norms[start:start + len(block)][:, None] + norms[None, :] - 2.0 * (
                block @ self.states.T
            )
            np.maximum(squared, 0.0, out=squared)
            squared[np.arange(len(block)), np.arange(start, start + len(block))] = np.inf
            spacing.append(np.sqrt(squared.min(axis=1)))
        self.spacing = float(np.concatenate(spacing).mean())
        print(f"数据集：{len(self.states)} 帧状态，平均最近邻间距 {self.spacing:.4f} rad")

    def nearest(self, state: np.ndarray) -> tuple[float, np.ndarray]:
        """Distance to the closest training state, and the difference vector."""

        index = int(np.argmin(np.linalg.norm(self.states - state, axis=1)))
        difference = state - self.states[index]
        return float(np.linalg.norm(difference)), difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root", type=Path,
                        default=Path("/root/autodl-tmp/datasets/a3_cookie_one_pi05"))
    return parser.parse_args()


def feature_probe(model, observation) -> None:
    """Is the live wrist frame in the same distribution as the cached features?

    The coefficients were fitted on cached features, so this compares one live frame
    against the cached ones rather than against images -- a resize or render difference
    shows up as a feature-space distance even when the two frames look alike, which is
    exactly the failure this is looking for.

    The frame goes through `adapter_ridge_cookie._tower_features` rather than straight
    into the tower, and that is not a convenience: the tower is the fixed-resolution
    SigLIP, so anything that is not 224x224 raises "The size of tensor a (324) must match
    the size of tensor b (256)" from inside its position embedding.  Calling the tower
    directly here would test a code path the rollout does not use.
    """

    live = adapter_ridge_cookie._tower_features(  # noqa: SLF001
        [observation[LEFT_WRIST_IMAGE]], model["torch"], model["F"],
        model["tower"], model["device"],
    )[0].float().cpu().numpy().astype(np.float64)

    cached = np.load(CACHE)["left_wrist_0_rgb"].astype(np.float64)
    # The cache is the tower's own output, so both sides are in the same space and the
    # distance is comparable to the spread of the cache itself.
    radius = float(np.linalg.norm(cached - cached.mean(0), axis=1).mean())
    distances = np.linalg.norm(cached - live, axis=1)
    nearest = int(np.argmin(distances))
    print(f"活体腕部帧 vs 缓存特征：最近邻距离 {distances[nearest]:.3f}   "
          f"缓存平均半径 {radius:.3f}   比值 {distances[nearest]/radius:.3f}")
    print("  （比值接近或小于 1 ⇒ 活体渲染与训练帧同分布；远大于 1 ⇒ 特征层面前置条件不成立）")


def main() -> int:
    args = parse_args()

    from a3_dual_arm_sim.bridge.action_bridge import ActionBridge
    from a3_dual_arm_sim.bridge.manifest import load_manifest
    from a3_dual_arm_sim.cli import _make_env

    env = _make_env(SCENE, None, action_mode="joint_position", render_cameras=True)
    env.use_fast_render()
    observation, _ = env.reset(seed=args.seed, options=None)

    # The *real* bridge, not a reimplementation of it.  This is the only place the
    # diagnostic could silently stop measuring the thing that ran: a hand-written
    # 8 -> 16 pad would have to reproduce `joint_map`, the latch of the unclaimed
    # right arm, and the clip, and any difference would be a difference in the
    # rollout being diagnosed.
    manifest = load_manifest(Path(PROJECT / "configs/policies/ridge_cookie_one.yaml"))
    bridge = ActionBridge(manifest.action, target_mode=env.action_mode, env=env)
    bridge.latch_hold_values()
    held = bridge._held_native(  # noqa: SLF001
        len(np.asarray(env._last_applied_action))  # noqa: SLF001
    )
    print(f"被保持的右臂（动作 8..15）: {np.round(held[8:], 3)}")

    model = adapter_ridge_cookie.load(
        {"coeffs": str(COEFFS), "tower": TOWER, "device": args.device, "images": "image2"}
    )
    limits = np.asarray(env._arm_ranges, dtype=np.float64)  # noqa: SLF001

    feature_probe(model, observation)

    print()
    print(f"{'step':>4} {'|目标-状态|':>10} {'|施加-目标|':>10} {'离数据集':>9} {'倍':>5} {'顶限位':>10}  "
          f"{'状态（7 关节 + 夹爪）'}")
    dataset = _Dataset(args.root)
    first_saturated = None
    info: dict = {}
    for step in range(args.steps):
        action = adapter_ridge_cookie.infer(model, _Request(observation))["actions"][0]
        state = np.asarray(observation[STATE], dtype=np.float64)[:8]
        native = bridge.convert_row(action)
        distance, difference = dataset.nearest(state)
        observation, _, terminated, truncated, info = env.step(native)
        applied = np.asarray(info["applied_action"], dtype=np.float64)

        asked = float(np.abs(action - state).max())
        overruled = float(np.abs(applied[:8] - action).max())
        at_limit = [
            f"{JOINT_NAMES[j]}{'-' if abs(applied[j] - limits[j, 0]) < 1e-3 else '+'}"
            for j in range(7)
            if abs(applied[j] - limits[j, 0]) < 1e-3 or abs(applied[j] - limits[j, 1]) < 1e-3
        ]
        if at_limit and first_saturated is None:
            first_saturated = step
        flag = ",".join(at_limit) or "-"
        if step < 25 or step % 20 == 0 or (
            first_saturated is not None and step < first_saturated + 5
        ):
            print(f"{step:>4} {asked:>10.4f} {overruled:>10.4f} {distance:>9.4f} "
                  f"{distance/dataset.spacing:>5.1f} {flag:>10}  "
                  + " ".join(f"{v:+.3f}" for v in state))
            # Which joint is carrying the deviation.  A trajectory that tracks the
            # expert in six joints and is wrong in one is a different failure from a
            # trajectory that is uniformly a little off, and the aggregate cannot tell
            # them apart.
            worst = int(np.argmax(np.abs(difference)))
            print(f"     偏离最大的关节 {JOINT_NAMES[worst]}: {difference[worst]:+.3f} rad "
                  f"（离最近训练帧的差："
                  + " ".join(f"{v:+.3f}" for v in difference) + "）")
        if terminated or truncated:
            print(f"  回合在第 {step} 步结束：terminated={terminated} truncated={truncated}")
            break

    print()
    print(f"首次触限步数：{first_saturated}")
    print(f"目标框内饼干：{info.get('cookies_in_target')}   源盒内：{info.get('cookies_in_source')}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
