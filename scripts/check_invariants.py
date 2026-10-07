#!/usr/bin/env python
"""Check the observation/action interface against a recorded dataset, with no policy.

Why this exists
---------------
Every bug in `docs/bug-catalogue.md` was silent: the simulation ran, the trajectory was
smooth, the images rendered, the gripper closed.  Four of them were individually fatal and
none would have been caught by watching a rollout.  What caught them was comparing one channel
of the interface against an external reference.

For A3's own scenes that reference is **the dataset**.  This script makes the comparison, and
it runs without a checkpoint, without a GPU, and in seconds -- so it can run on every commit,
and the policy only runs when it is green.

What it checks, and against what
--------------------------------
1. **Fingerprint** (channel 1, images).  Renders each camera at the episode's first state and
   scores it against the dataset's own frame for `{as-is, flip H, flip W, flip both}`.  The
   signal is not the absolute error -- the scene randomises -- it is that **as-is must win, by
   a margin**.  A flip winning has meant an upside-down policy twice.
2. **State convention** (channel 2).  Loads the dataset's first state into the environment,
   asks the environment for its own `observation.state`, and compares element by element.  Two
   conventions live here: a `(x, y, z, w)` quaternion read as `(w, x, y, z)` is a half-turn
   that looks like a small tilt, and the two grippers disagree about whether zero is open.
3. **Gripper round trip** (channel 5).  Commanded open and close, read back.
4. **Tracking** (channel 4).  One joint step, and how much of it arrives in one control step.
   The A3 arm delivers 43% at the default damping and the number is a property of the scene,
   so it belongs in the record.

What it does **not** check: the dynamics (channel 6) needs an open-loop replay, and the task
layer (channel 7) needs the success predicate -- see
`docs/scene-transition-and-bug-triage.md` §2.2 and §7.

Usage::

    MUJOCO_GL=egl python scripts/check_invariants.py \
        --scene a3_cookie_batch --dataset /root/autodl-tmp/datasets/a3_cookie_adv_1000_merged \
        --episode 0 --output /tmp/invariants.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="a3_cookie_batch")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/root/autodl-tmp/datasets/a3_cookie_adv_1000_merged"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0,
                        help="which frame of the episode to compare against")
    parser.add_argument("--cameras", default="front,left_wrist,right_wrist")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="how much better the correct orientation must score than the "
                             "next best, as a fraction; the absolute error is not the signal "
                             "because the scene randomises")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def episode_meta(dataset: Path, episode: int) -> tuple[dict, dict, int]:
    """`(meta row, camera locators, dataset_from_index)` for one episode."""

    import pyarrow.parquet as pq

    meta = None
    for path in sorted((dataset / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.ParquetFile(path).read().to_pylist():
            if int(row["episode_index"]) == episode:
                meta = row
                break
        if meta is not None:
            break
    if meta is None:
        raise SystemExit(f"episode {episode} is not in {dataset}/meta/episodes")

    #: short camera name -> (feature name, chunk, file, timestamp).  The metadata calls it
    #: `observation.images.front` and the scene calls it `front`; both are needed, one for the
    #: path and one for the command line.
    locators: dict[str, tuple[str, int, int, float]] = {}
    for key, value in meta.items():
        if key.startswith("videos/") and key.endswith("/chunk_index"):
            feature = key[len("videos/"):-len("/chunk_index")]
            camera = feature.rsplit(".", 1)[-1]
            locators[camera] = (
                feature,
                int(value),
                int(meta[f"videos/{feature}/file_index"]),
                float(meta[f"videos/{feature}/from_timestamp"]),
            )
    return meta, locators, int(meta["dataset_from_index"])


def episode_bounds(dataset: Path, episode: int) -> tuple[int, int]:
    """`(first, one past last)` row index of one episode inside `dataset`."""

    meta, _locators, start = episode_meta(dataset, episode)
    return start, int(meta["dataset_to_index"])


def read_episode_frames(dataset: Path, episode: int, first: int, count: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(states, actions, velocities)`, each `(count, width)`, from `first`.

    One parquet read for the whole range rather than one per frame: a teacher-forced replay
    wants hundreds of consecutive frames and pyarrow re-reads the file each time otherwise.

    The velocities are returned because **a teacher-forced replay is a second-order initial
    value problem and the dataset records both halves of the initial condition.**  Starting
    each step from rest is not a neutral simplification: measured on `a3_cookie_one` it put a
    0.0033 rad residual on a scene that reproduces the recording bit-exactly from the
    recorded velocity, and 0.0033 rad is three quarters of a recorded step there -- an error
    of the same order as the signal, in the direction of "the dynamics differ".

    **A window may not cross an episode boundary, and this refuses to.**  Episode 0 of
    `a3_cookie_one` is 190 frames and asking for 200 silently appended the first ten frames of
    episode 1: the gripper error jumped to 0.75 where the recording steps its jaw from 0.25 to
    1.0, and the *arm* channel showed nothing at all, because teacher forcing teleports the arm
    onto the recorded pose at every frame by construction.  A discontinuity that hides in one
    channel and screams in another is exactly the kind of false finding this tool exists to
    prevent, so the boundary is an error rather than a clamp.
    """

    import pyarrow.parquet as pq

    _meta, _locators, start = episode_meta(dataset, episode)
    _first, stop = episode_bounds(dataset, episode)
    if first < 0 or count < 1:
        raise SystemExit(f"cannot read {count} frames from frame {first}")
    if start + first + count > stop:
        raise SystemExit(
            f"episode {episode} of {dataset} has {stop - start} frames; frames "
            f"{first}..{first + count - 1} cross into the next episode "
            f"(the boundary is at frame {stop - start})")
    wanted = np.arange(start + first, start + first + count)
    wanted_set = set(wanted.tolist())
    states: dict[int, np.ndarray] = {}
    actions: dict[int, np.ndarray] = {}
    velocities: dict[int, np.ndarray] = {}
    for path in sorted((dataset / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches():
            rows = batch.to_pylist()
            if not rows:
                continue
            if int(rows[-1]["index"]) < wanted[0] or int(rows[0]["index"]) > wanted[-1]:
                continue
            for row in rows:
                index = int(row["index"])
                if index in wanted_set:
                    states[index] = np.asarray(row["observation.state"], dtype=np.float64)
                    actions[index] = np.asarray(row["action"], dtype=np.float64)
                    velocities[index] = np.asarray(row["observation.velocity"],
                                                   dtype=np.float64)
        if len(states) == len(wanted):
            break
    missing = [int(i) for i in wanted if int(i) not in states]
    if missing:
        raise SystemExit(f"{len(missing)} frames of episode {episode} start at {first} are "
                         f"missing from {dataset}/data (first missing index {missing[0]})")
    ordered = [int(i) for i in wanted]
    return (np.stack([states[i] for i in ordered]),
            np.stack([actions[i] for i in ordered]),
            np.stack([velocities[i] for i in ordered]))


def read_episode(dataset: Path, episode: int, frame: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """`(state, action, video_frame_locators)` for one frame of one episode."""

    import pyarrow.parquet as pq

    meta, locators, start = episode_meta(dataset, episode)

    data_files = sorted((dataset / "data").rglob("*.parquet"))
    index = start + frame
    state = action = None
    for path in data_files:
        parquet = pq.ParquetFile(path)
        rows = parquet.read().to_pylist()
        if not rows or int(rows[0]["index"]) > index or int(rows[-1]["index"]) < index:
            continue
        for row in rows:
            if int(row["index"]) == index:
                state = np.asarray(row["observation.state"], dtype=np.float64)
                action = np.asarray(row["action"], dtype=np.float64)
                break
        if state is not None:
            break
    if state is None or action is None:
        raise SystemExit(f"frame {frame} of episode {episode} is not in {dataset}/data")
    return state, action, locators


def fingerprint(rendered: np.ndarray, recorded: np.ndarray, noise: float,
                margin: float) -> dict:
    """Which orientation of `rendered` matches `recorded`, and is the comparison meaningful.

    **The margin alone is not enough**, and this function exists because of the day it was:
    an old dataset's wrist camera scored `flip_hw` 3x better than `as_is`, which looked like
    a rotated camera until the absolute numbers were read -- the *best* orientation was still
    MSE 2490, where a matching scene scores 42.  The content did not agree in any
    orientation, so the ordering was noise and the "finding" was an artifact of comparing two
    different scenes.  That is the same trap as measuring two quantities and calling them one.

    Comparability is calibrated from the dataset itself: `noise` is the error between two
    *adjacent recorded frames*, i.e. what "the same scene, a moment later" costs.  A
    comparison is meaningful when the winner is within `k` times that.  `k` is a policy about
    how much scene change is tolerable, and it is stated rather than hidden.
    """

    scores = {
        "as_is": mse(rendered, recorded),
        "flip_h": mse(rendered[::-1, :], recorded),
        "flip_w": mse(rendered[:, ::-1], recorded),
        "flip_hw": mse(rendered[::-1, ::-1], recorded),
    }
    ordered = sorted(scores.items(), key=lambda pair: pair[1])
    best, runner_up = ordered[0], ordered[1]
    ratio = (runner_up[1] - best[1]) / best[1] if best[1] else float("inf")
    comparable = best[1] <= max(noise, 1.0)
    return {
        "scores": {k: round(v, 1) for k, v in scores.items()},
        "best": best[0],
        "margin": round(ratio, 4),
        "noise": round(noise, 1),
        "comparable": bool(comparable),
        "passes": bool(comparable and best[0] == "as_is" and ratio >= margin),
    }


def dataset_frame(dataset: Path, camera: str, locators: dict, fps: float, frame: int) -> np.ndarray:
    """The recorded frame, decoded and in the same orientation the dataset stores."""

    import av

    feature, chunk, file_index, from_timestamp = locators[camera]
    path = (dataset / "videos" / feature / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target = float(from_timestamp) + frame / fps
        container.seek(int(target / float(stream.time_base)), stream=stream)
        for frame_obj in container.decode(stream):
            if frame_obj.time is None or frame_obj.time + 1e-6 >= target:
                return frame_obj.to_ndarray(format="rgb24")
    raise SystemExit(f"could not decode {path} at {target:.3f}s")


def main() -> int:
    args = parse_args()

    import mujoco

    from a3_dual_arm_sim.collection import scene_by_name

    report: dict = {"scene": args.scene, "dataset": str(args.dataset), "episode": args.episode}

    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 20))
    report["fps"] = fps

    state, action, locators = read_episode(args.dataset, args.episode, args.frame)
    report["state_width"] = int(state.size)
    report["action_width"] = int(action.size)

    env = scene_by_name(args.scene).build_env(0.0, 0.0)
    obs, _ = env.reset(seed=args.episode, options=None)
    model, data = env.model, env.data

    # ---- channel 2: state convention, by loading the recorded state in ----
    from a3_dual_arm_sim.contracts import ARM_JOINTS

    dof_for = {}
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name:
            dof_for[name] = (int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]))

    arm_names = [name for name in ARM_JOINTS if name in dof_for]
    if len(arm_names) != 14:
        raise SystemExit(f"{args.scene} drives {len(arm_names)} arm joints, expected 14")

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for index, name in enumerate(arm_names):
        address = dof_for[name][0]
        data.qpos[address] = state[index if index < 7 else index + 1]
    for side, index in (("L", 7), ("R", 15)):
        opening = float(state[index])
        fraction = env._finger_travel_fraction(side, opening)  # noqa: SLF001
        travel = env._finger_travel[side]  # noqa: SLF001
        for sign, joint_id in zip(env._finger_sign[side], env._finger_joints[side]):  # noqa: SLF001
            data.qpos[model.jnt_qposadr[joint_id]] = sign * fraction * travel
    mujoco.mj_forward(model, data)

    loaded = np.asarray(env._observation()["observation.state"], dtype=np.float64)  # noqa: SLF001
    diff = loaded - state
    report["state"] = {
        "max_abs_diff": float(np.abs(diff).max()),
        "left_arm_max": float(np.abs(diff[:7]).max()),
        "right_arm_max": float(np.abs(diff[8:15]).max()),
        "grippers": [float(diff[7]), float(diff[15])],
        "ours": loaded.round(5).tolist(),
        "recorded": state.round(5).tolist(),
    }

    # ---- channel 1: fingerprint, per camera ----
    # A second episode of the same task, same frame index: the yardstick for "a different draw
    # of the same scene".
    cross_episode = args.episode + 1
    _state2, _action2, cross_locators = read_episode(args.dataset, cross_episode, args.frame)

    orientation = {}
    for camera in args.cameras.split(","):
        if camera not in locators or camera not in cross_locators:
            continue
        recorded = dataset_frame(args.dataset, camera, locators, fps, args.frame)
        other = dataset_frame(args.dataset, camera, cross_locators, fps, args.frame)
        rendered = np.asarray(env.render_view(camera, height=recorded.shape[0],
                                             width=recorded.shape[1]))
        orientation[camera] = fingerprint(rendered, recorded, mse(recorded, other),
                                          args.margin)
    report["images"] = orientation
    report["cross_episode"] = cross_episode

    # ---- channel 5: gripper round trip ----
    gripper = {}
    home = env._current_joint_action().copy()  # noqa: SLF001
    for label, command in (("open", 1.0), ("close", 0.0)):
        row = home.copy()
        row[7] = command
        row[15] = command
        for _ in range(12):
            env.step(row)
        gripper[label] = round(float(env._gripper_state("L")[0]), 4)  # noqa: SLF001
    report["gripper"] = {
        **gripper,
        "ordered": bool(gripper.get("open", 0) > gripper.get("close", 1) + 0.5),
    }

    # ---- channel 4: how much of one joint step arrives ----
    env.reset(seed=args.episode, options=None)
    name = arm_names[0]
    address = dof_for[name][0]
    before = float(data.qpos[address])
    target = before + 0.05
    row = env._current_joint_action().copy()  # noqa: SLF001
    row[0] = target
    env.step(row)
    achieved = float(data.qpos[address]) - before
    report["tracking"] = {
        "joint": name,
        "requested_rad": 0.05,
        "achieved_rad": round(achieved, 4),
        "fraction": round(achieved / 0.05, 3),
    }

    ok = (
        report["state"]["max_abs_diff"] < 1e-6
        and all(entry["passes"] for entry in orientation.values())
        and report["gripper"]["ordered"]
    )
    report["verdict"] = "INVARIANTS HOLD" if ok else "INVARIANT FAILED -- treat as a bug"

    print(f"scene      {args.scene}   episode {args.episode}   frame {args.frame}")
    print(f"state      max |ours - recorded| = {report['state']['max_abs_diff']:.2e}"
          f"   (grippers {report['state']['grippers']})")
    for camera, entry in orientation.items():
        verdict = "ok" if entry["passes"] else (
            "NOT COMPARABLE" if not entry["comparable"] else "WRONG ORIENTATION")
        print(f"image {camera:<12} best {entry['best']:<6} margin {entry['margin']:+.1%}"
              f"  noise {entry['noise']:>7.1f}  {verdict}")
        print(f"{'':18}scores {entry['scores']}")
    print(f"gripper    open {gripper.get('open')}  close {gripper.get('close')}"
          f"   ordered {report['gripper']['ordered']}")
    print(f"tracking   {report['tracking']['joint']}: "
          f"{report['tracking']['fraction']:.0%} of a step in one control step")
    print(f"\n{report['verdict']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"json       {args.output}")
    return 0 if ok else 1


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())


if __name__ == "__main__":
    raise SystemExit(main())
