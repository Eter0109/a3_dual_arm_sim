"""Does the policy actually see the object? The Cookie's pixel footprint in every A3 camera.

The question
------------
The perturbation sweep over `libero_spatial` puts a price on each rendering axis: every
material flattened to one grey costs 2 points, the camera headlight costs 0, the front
camera's field of view costs 12-14 either way -- and **the wrist camera's field of view costs
all 50**.  A policy that intolerant of what its cameras show is one whose cameras are carrying
the task, so "is the task object actually visible, and how large is it" stops being a question
that can be answered by reading a scene file.

This measures it, from the stored videos of the dataset the checkpoints were fine-tuned on.
No simulation and no rendering: these are the frames the model was trained on.

The hypothesis this file was written to test, and how it died
------------------------------------------------------------
A3's left wrist camera is declared as

    left_wrist_position_m: [0.0, 0.045, 0.085]
    left_wrist_target_m:   [0.0, 0.245, 0.085]

on the flange, with `mode="targetbody"`.  The Robotiq is mounted with `xyaxes="0 0 1 1 0 0"`,
which makes the flange's +y the tool axis -- so the camera is 45 mm out along that axis (the
same as LIBERO's `eye_in_hand`, `pos="0.05 0 0"`) but **85 mm off it**, aimed at a target body
that shares its lateral offset.  Its optical axis is *parallel* to the tool axis rather than
through it; LIBERO's is on the axis.  At `fovy = 70` a point 100 mm out and 85 mm to the side
is 40.4 degrees off axis against a 35-degree half-angle, which reads like a defect: a wrist
camera looking past the thing the gripper is reaching for.

Projecting the recorded `observation.eef_pose` agreed -- median off-axis angle 32 degrees,
43 degrees over the 20 frames before the gripper shuts.

**The images say otherwise.**  Masking the Cookie by colour in the stored wrist frames puts
its centroid at x = 125 of 256, i.e. **off-axis by -0.5 degrees, dead centre, for the whole
episode.**  The camera was never looking past anything.  The projection was what was wrong,
and it was wrong for an instructive reason: it placed the Cookie at
`cookie_source_positions_m[0]` of `configs/cookie_same_column.yaml`, and the data puts the
Cookie 80 mm away from there.  (Found by back-projecting: the left eef site sits at
(0.080, 0.157, 0.790) while the gripper holds, against a config value of
(0.0994, 0.2311, 0.7685).)

So the file kept its measurements, dropped its conclusion, and the measurement that survives
is the one below.  The projection is still printed -- not as a result, but because its
disagreement with the images is what exposed the config error, and that is worth keeping
reproducible.

What it measures
----------------
The Cookie's pixel footprint, per camera, over the approach phase of many episodes.  A3's
Cookie is a 50 x 6.3 x 25 mm plate standing upright, so a camera looking along the tool axis
sees its 6.3 mm edge: an object that is large in the world can be a handful of pixels in the
image, and no amount of camera tuning fixes a silhouette that thin.

Frames stop at the gripper's hold, because after that the Cookie is being carried and a
projection against a fixed world position stops meaning anything.  See `GRIPPER_HOLD_ABOVE`
for why "holding" reads 0.25 rather than 1.0, and why the first version of this selection
silently reduced 60 episodes to 60 frames.

Usage::

    /root/autodl-tmp/a3-collect/.venv/bin/python scripts/probe_wrist_framing.py --episodes 60
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]

#: The three cameras the A3 observation declares, in its own order.
CAMERAS = ("observation.images.front", "observation.images.left_wrist",
           "observation.images.right_wrist")

#: The Cookie's two colours are `1.0 0.78 0.20` and `0.88 0.50 0.04` (`model.py`, alternating
#: by column and row parity).  The other large surfaces in the frame are a brown table and
#: black gripper fingers, so the mask can be tight on "much more red than blue": a loose one
#: would count table pixels, and a Cookie that measures five pixels cannot afford that.
COOKIE_RED_ABOVE = 200
COOKIE_GREEN_ABOVE = 150
COOKIE_BLUE_BELOW = 130

#: Gripper readings marking the two events an episode is divided by.  Neither is 0.5, which is
#: where the first version put the hold, on the assumption that a closed gripper reads near
#: 1.0.  It does not: `_gripper_state` returns `1 - mean(finger position) / RANGE`, so 1.0 is
#: the fingers touching with nothing between them, and **holding a 50 mm Cookie stops them
#: around 0.25**.  Measured on episode 0, the trace is 1.00 at reset, 0.01 once the gripper
#: opens, 0.25 at the end -- it never crosses 0.5.  A 0.5 threshold therefore found no hold at
#: all, quietly fell back to "keep the whole episode", and "the last 20 frames before the hold"
#: became the last 20 frames of the episode, by which time the Cookie has been carried away.
#: The symptom was a physically impossible result, not an exception.
GRIPPER_OPEN_BELOW = 0.05
GRIPPER_HOLD_ABOVE = 0.15

#: The eef site's offset along the tool axis in the flange frame (`model.py`), with
#: `direction = +1` for the left arm.
EEF_SITE_ALONG_M = 0.145

#: The left eef site's position while the gripper is holding, read off the data.  Used only to
#: state how far the config's Cookie position is from the real one.
OBSERVED_HOLD_POSITION = np.array([0.080, 0.157, 0.790])


def _scene_offsets() -> dict:
    """Camera geometry, read from the config the scenes are built from."""

    import sys

    sys.path.insert(0, str(PROJECT / "src"))
    from a3_dual_arm_sim.config import load_config

    cameras = load_config(PROJECT / "configs" / "default.yaml").cameras
    return {
        "wrist_pos": np.asarray(cameras.left_wrist_position_m, dtype=np.float64),
        "wrist_fovy": float(cameras.wrist_fovy_deg),
    }


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrix from a `(w, x, y, z)` quaternion, the inverse of `mju_mat2Quat`."""

    w, x, y, z = (float(v) for v in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _config_cookie() -> np.ndarray:
    """The Cookie's nominal world position -- kept because it is the wrong one."""

    import sys

    sys.path.insert(0, str(PROJECT / "src"))
    from a3_dual_arm_sim.config import load_config

    config = load_config(PROJECT / "configs" / "cookie_same_column.yaml")
    x, y = config.cookie_transfer.cookie_source_positions_m[0][:2]
    return np.array([float(x), float(y), float(config.cookie_transfer.cookie_model_z_m)])


def _approach_frames(root: Path, episodes: int, stride: int) -> dict:
    """Global frame indices of the approach phase, plus the eef poses on them.

    Global, not per-episode: the stored videos concatenate every episode of the dataset into
    one or two files, so a frame's position in the stream is its row number in the parquet.
    Counting per file with `enumerate` restarts the count and silently pairs frames with the
    wrong episode -- the trap `probe_observation_information.py` records.
    """

    table = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    episode = np.asarray(table["episode_index"])
    state = np.stack(table["observation.state"].to_pylist()).astype(np.float64)
    pose = np.stack(table["observation.eef_pose"].to_pylist()).astype(np.float64)

    rows: list[int] = []
    frame_index: list[int] = []
    episode_index: list[int] = []
    for index in range(episodes):
        where = np.flatnonzero(episode == index)
        if not len(where):
            continue
        grip = state[where, 7]
        opened = np.flatnonzero(grip < GRIPPER_OPEN_BELOW)
        stop = int(where[-1])
        if len(opened):
            after = where[opened[0] :]
            hold = np.flatnonzero(state[after, 7] > GRIPPER_HOLD_ABOVE)
            if len(hold):
                stop = int(after[hold[0]])
        rows.extend(range(int(where[0]), stop, stride))
        frame_index.extend(range(0, stop - int(where[0]), stride))
        episode_index.extend([index] * len(range(int(where[0]), stop, stride)))

    selection = np.asarray(rows, dtype=np.int64)
    return {
        "global": selection,
        "episode": np.asarray(episode_index, dtype=np.int64),
        "frame": np.asarray(frame_index, dtype=np.int64),
        "pos": pose[selection, 0:3],
        # The left arm's eef: position then a `(w, x, y, z)` quaternion.
        "quat": pose[selection, 3:7],
        "gripper": state[selection, 7],
    }


def _cookie_pixels(image: np.ndarray) -> tuple[int, float, int]:
    """(pixel count, centroid x, bounding width) of the Cookie in one 256x256 frame."""

    red = image[:, :, 0].astype(np.int32)
    green = image[:, :, 1].astype(np.int32)
    blue = image[:, :, 2].astype(np.int32)
    mask = (red > COOKIE_RED_ABOVE) & (green > COOKIE_GREEN_ABOVE) & (blue < COOKIE_BLUE_BELOW)
    count = int(mask.sum())
    if count < 5:
        return count, float("nan"), 0
    columns = np.nonzero(mask)[1]
    return count, float(columns.mean()), int(columns.max() - columns.min() + 1)


def _scan(root: Path, wanted: dict[int, int], camera: str) -> dict:
    """Measure the Cookie on every wanted frame of one camera's stored video.

    Decoding stops as soon as the stream passes the last wanted frame, so scanning the first
    few thousand frames never touches the second file.
    """

    import av

    chunk = root / "videos" / camera / "chunk-000"
    last = max(wanted)
    counts = np.full(len(wanted), -1, dtype=np.int64)
    centroids = np.full(len(wanted), np.nan, dtype=np.float64)
    widths = np.zeros(len(wanted), dtype=np.int64)

    counter = 0
    for path in sorted(chunk.glob("*.mp4")):
        for frame in av.open(str(path)).decode(video=0):
            if counter > last:
                break
            slot = wanted.get(counter)
            if slot is not None:
                counts[slot], centroids[slot], widths[slot] = _cookie_pixels(
                    frame.to_ndarray(format="rgb24")
                )
            counter += 1
        if counter > last:
            break
    return {"count": counts, "centroid": centroids, "width": widths}


def _project(frames: dict, offsets: dict, cookie: np.ndarray) -> np.ndarray:
    """Off-axis angle of the Cookie from the recorded eef poses -- the refuted hypothesis."""

    angles = []
    for position, quaternion in zip(frames["pos"], frames["quat"], strict=True):
        rotation = _quat_matrix(quaternion)
        flange = position - rotation @ np.array([0.0, EEF_SITE_ALONG_M, 0.0])
        camera = flange + rotation @ offsets["wrist_pos"]
        axis = rotation @ np.array([0.0, 1.0, 0.0])
        vector = cookie - camera
        depth = float(np.dot(vector, axis))
        lateral = float(np.linalg.norm(vector - depth * axis))
        angles.append(math.degrees(math.atan2(lateral, depth)))
    return np.asarray(angles)


def _plot(measured: dict, frames: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    colours = {"observation.images.front": "tab:blue",
               "observation.images.left_wrist": "tab:red",
               "observation.images.right_wrist": "tab:grey"}
    edges = np.arange(0, 404, 4)
    for camera, stats in measured.items():
        axes[0].hist(stats["count"].clip(0, 399), bins=edges, histtype="step",
                     linewidth=1.6, color=colours[camera], label=camera.split(".")[-1])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Cookie pixels in a 256x256 frame (65536 total)")
    axes[0].set_ylabel("frames (log)")
    axes[0].set_title("object footprint per camera, approach phase")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    wrist = measured["observation.images.left_wrist"]
    axes[1].scatter(frames["frame"], wrist["count"], s=2, alpha=0.15, color="tab:red")
    axes[1].set_xlabel("frame within episode")
    axes[1].set_ylabel("Cookie pixels")
    axes[1].set_ylim(0, 400)
    axes[1].set_title("left wrist: footprint vs approach")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/root/autodl-tmp/datasets/a3_cookie_one"))
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--stride", type=int, default=4,
                        help="keep every Nth frame of the approach phase")
    parser.add_argument("--figure", type=Path,
                        default=Path("/root/autodl-tmp/logs/cookie_footprint.png"))
    args = parser.parse_args()

    offsets = _scene_offsets()
    cookie = _config_cookie()
    frames = _approach_frames(args.root, args.episodes, args.stride)
    wanted = {int(g): slot for slot, g in enumerate(frames["global"])}
    measured = {camera: _scan(args.root, wanted, camera) for camera in CAMERAS}

    print(f"{args.root.name}: {args.episodes} episodes, approach phase, every "
          f"{args.stride}th frame => {len(frames['global'])} frames x {len(CAMERAS)} cameras")
    print()
    print("object: A3 Cookie = 50 x 6.3 x 25 mm plate, upright; frame 256x256 = 65536 px")
    print()
    print(f"{'camera':<14} {'median px':>10} {'p90':>6} {'max':>6} {'median w':>9} "
          f"{'<10px':>7} {'<100px':>7} {'median x':>9} {'seen':>6}")
    for camera, stats in measured.items():
        seen = stats["count"] >= 0
        counts = stats["count"][seen]
        centroid = stats["centroid"][seen & ~np.isnan(stats["centroid"])]
        width = stats["width"][seen & (stats["width"] > 0)]
        print(f"{camera.split('.')[-1]:<14} {np.median(counts):>10.0f} "
              f"{np.percentile(counts, 90):>6.0f} {counts.max():>6.0f} "
              f"{np.median(width) if len(width) else 0:>9.0f} "
              f"{np.mean(counts < 10)*100:>6.0f}% {np.mean(counts < 100)*100:>6.0f}% "
              f"{np.median(centroid) if len(centroid) else float('nan'):>9.1f} "
              f"{seen.mean()*100:>5.0f}%")

    wrist = measured["observation.images.left_wrist"]
    ok = ~np.isnan(wrist["centroid"])
    wrist_centroid = float(np.median(wrist["centroid"][ok])) if ok.any() else float("nan")
    offset_deg = (wrist_centroid - 127.5) / 128.0 * offsets["wrist_fovy"] / 2
    angles = _project(frames, offsets, cookie)
    print()
    print("The refuted hypothesis, kept because it exposed a wrong position")
    print(f"  the Cookie is found in {ok.mean()*100:.0f}% of the wrist frames; "
          f"its median centroid is x = {wrist_centroid:.1f}/255, "
          f"off-axis {offset_deg:+.1f} deg (half-FOV {offsets['wrist_fovy']/2:.0f}) -- dead centre.")
    print(f"  projecting the config's Cookie {np.round(cookie, 4)} instead gives "
          f"off-axis median {np.median(angles):.1f} deg -- apparently 'outside the frame'.")
    print(f"  the config's Cookie is {np.linalg.norm(OBSERVED_HOLD_POSITION - cookie):.3f} m from "
          f"where the recorded eef poses put it ({np.round(OBSERVED_HOLD_POSITION, 3)}), which is "
          f"the whole of the disagreement.")

    _plot(measured, frames, args.figure)
    print(f"\nfigure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
