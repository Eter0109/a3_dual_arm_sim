"""Derive a single-arm A3 cookie dataset from the dual-arm one.

Why this exists
---------------
The right arm in the cookie scene never moves.  Measured over all 537,765 frames of
``a3_cookie_adv_1000_merged``, ``observation.state`` and ``action`` dimensions 8-15 each have
``min == max`` and ``std == 0.00000``: seven joints pinned at one pose and a gripper parked at
1.0, never closed.  The right-wrist camera is static too -- 0 of 119 consecutive frames differ
by more than 0.5 of a grey level, against 84/119 for the front camera and 112/119 for the left
wrist, because it is bolted to an arm that does not move.

So half of the interface carried no information.  Worse, it blocked the one starting
checkpoint that is a *working* single-arm policy rather than a base: with the arm gone, the A3
task has exactly the shape ``lerobot/pi05_libero_finetuned`` was trained on -- one arm, a
third-person camera plus a wrist camera, an 8-dimensional state -- and its third image slot
(``empty_camera_0``) stays empty, which is how it was trained.  Feeding a real camera into that
slot was the argument against using those weights at all.

Renaming, not reshaping
-----------------------
The features are named here *exactly* as that checkpoint expects:

    observation.images.front      -> observation.images.image      (agentview)
    observation.images.left_wrist -> observation.images.image2     (wrist)

so no ``--rename_map`` is needed, and ``make_policy`` (factory.py:509) keeps the checkpoint's
``input_features`` unchanged because they are non-empty.  The missing ``empty_camera_0`` never
appears in a batch and ``modeling_pi05.py:1200`` creates it as ``-1`` filler with a zero
attention mask, so the dataset does not have to carry it.

What is dropped, and why it is safe to drop
-------------------------------------------
* ``observation.images.right_wrist`` -- a static image (above).
* ``observation.velocity`` / ``observation.eef_pose`` / ``observation.force`` -- never listed
  in the policy's ``input_features``, so they were loaded and carried through the batch
  without ever being read.
* dimensions 8-15 of ``observation.state`` and ``action`` -- constant, and because the stored
  ``q99 - q01`` for them is exactly 0 the quantile transform maps them to a fixed -1 anyway.
  Their loss was identically zero, so removing them removes no learning signal.

Videos are symlinked, not copied, so this costs a few hundred kilobytes rather than 26 GB, and
the dual-arm dataset is left untouched for the runs that still use it.

    python scripts/make_single_arm_dataset.py [--source DIR] [--dest DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE = Path("/root/autodl-tmp/datasets/a3_cookie_adv_1000_merged")
DEFAULT_DEST = Path("/root/autodl-tmp/datasets/a3_cookie_single_arm")

#: Camera renames.  The destination names are not a style choice: they are the keys
#: ``lerobot/pi05_libero_finetuned`` declares, and matching them is what removes the need for a
#: rename map.
CAMERAS = {
    "observation.images.front": "observation.images.image",
    "observation.images.left_wrist": "observation.images.image2",
}
#: Static, or never read by the policy.
DROP = (
    "observation.images.right_wrist",
    "observation.velocity",
    "observation.eef_pose",
    "observation.force",
)
#: The single arm is the first eight dimensions: seven joints then the gripper.
ARM_DIMS = 8


def _slice_stats(values, dims: int):
    """Trim per-dimension statistics, leaving scalars alone.

    ``count`` is a scalar; ``min``/``max``/``mean``/``std``/``q01``/... are per-dimension lists
    of length 16.  Slicing everything would fail on the first, and slicing nothing would leave
    a statistics block whose width disagrees with the feature it describes -- which is the kind
    of mismatch that surfaces much later as a tensor broadcast error.
    """

    return values[:dims] if isinstance(values, list) else values


def _rewrite_frames(source: Path, dest: Path, dims: int) -> tuple[int, int]:
    """Copy the data parquets with the arm dims trimmed and the unused columns gone.

    Only the numeric columns are touched.  The frames themselves live in ``videos/``, which
    this file never reads -- that separation is what lets the whole derivation cost megabytes.
    """

    frames = 0
    files = 0
    for src_file in sorted((source / "data").rglob("*.parquet")):
        rel = src_file.relative_to(source / "data")
        out = dest / "data" / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        table = pd.read_parquet(src_file)
        table = table.drop(columns=[c for c in DROP if c in table.columns])

        for column in ("observation.state", "action"):
            if column not in table.columns:
                continue
            stacked = np.stack(table[column].to_numpy())[:, :dims]
            table[column] = [row.astype(np.float32) for row in stacked]

        table.to_parquet(out, index=False)
        frames += len(table)
        files += 1
    return frames, files


def _rewrite_episodes(
    source: Path, dest: Path, dims: int, cameras: dict[str, str] | None = None
) -> None:
    """Episode metadata: drop the static camera's video columns, rename the rest, trim stats."""

    cameras = cameras or CAMERAS

    for src_file in sorted((source / "meta" / "episodes").rglob("*.parquet")):
        rel = src_file.relative_to(source / "meta" / "episodes")
        out = dest / "meta" / "episodes" / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        table = pd.read_parquet(src_file)
        table = table.drop(
            columns=[c for c in table.columns if c.startswith(f"videos/{DROP[0]}/")]
            + [c for c in table.columns if c.startswith(f"stats/{DROP[0]}/")]
            + [c for c in table.columns if any(c.startswith(f"stats/{e}/") for e in DROP[1:])]
        )

        renamed = {}
        for old, new in cameras.items():
            for column in table.columns:
                if column.startswith(f"videos/{old}/") or column.startswith(f"stats/{old}/"):
                    renamed[column] = column.replace(old, new)
        table = table.rename(columns=renamed)

        for column in table.columns:
            if column.startswith("stats/observation.state/") or column.startswith("stats/action/"):
                table[column] = table[column].map(lambda v: _slice_stats(v, dims))

        table.to_parquet(out, index=False)


def _delta_action_stats(source: Path, dims: int) -> dict:
    """The ``action`` block's statistics, recomputed on ``action - state``.

    Needed because ``--policy.use_relative_actions=true`` inserts a
    ``delta_actions_processor`` *before* the normaliser, so the tensor being normalised is the
    delta -- but the statistics handed to the normaliser are ``dataset.meta.stats``, which
    describe the *absolute* actions.  The two are not the same distribution and the mismatch is
    silent.

    Measured on this dataset: absolute actions span 0.51-1.19 rad while the deltas span
    0.031-0.085, so the normalised delta is not centred on zero.  It is centred on
    ``2 * (0 - q01_abs) / (q99_abs - q01_abs) - 1``, which for these joints is **-4.23, -3.50,
    +2.92, +2.24, +1.52, -1.70, +1.79, -1.02** -- outside the [-1, 1] the transform is defined
    on -- and it has a spread of only 0.025-0.056.  So the model is asked to produce a nearly
    constant offset far outside its range and resolve about 1% of that offset's own spread.

    A 6,000-step run with that configuration did exactly what it implies: 20% of its commands
    sat at joint limits against 0% for the absolute-action run, `corrected_fraction` was 0.245
    against 0.050, and it never grasped the Cookie.  Its weights are therefore not evidence
    about relative actions; they are evidence about this mismatch.

    Writing delta statistics here is what makes the normalised target land in [-1, 1] again,
    and it is the caller's job because lerobot cannot know: it reads the dataset's statistics
    and those describe whatever the columns hold, which before this transform is absolute.
    """

    frames = pd.concat(
        [
            pd.read_parquet(f, columns=["action", "observation.state"])
            for f in sorted((source / "data").rglob("*.parquet"))
        ],
        ignore_index=True,
    )
    action = np.stack(frames["action"].to_numpy()).astype(np.float64)[:, :dims]
    state = np.stack(frames["observation.state"].to_numpy()).astype(np.float64)[:, :dims]
    delta = action - state

    percentiles = np.percentile(delta, [1, 10, 50, 90, 99], axis=0)
    return {
        "min": delta.min(axis=0).tolist(),
        "max": delta.max(axis=0).tolist(),
        "mean": delta.mean(axis=0).tolist(),
        "std": delta.std(axis=0).tolist(),
        "count": int(delta.shape[0]),
        "q01": percentiles[0].tolist(),
        "q10": percentiles[1].tolist(),
        "q50": percentiles[2].tolist(),
        "q90": percentiles[3].tolist(),
        "q99": percentiles[4].tolist(),
    }


def build(
    source: Path, dest: Path, cameras: dict[str, str] | None = None, *, delta_stats: bool = False
) -> dict:
    cameras = cameras or CAMERAS
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "meta").mkdir(parents=True)
    (dest / "videos").mkdir(parents=True)

    info = json.loads((source / "meta" / "info.json").read_text())
    stats = json.loads((source / "meta" / "stats.json").read_text())

    features = {}
    for key, spec in info["features"].items():
        if key in cameras:
            features[cameras[key]] = spec
        elif key in DROP:
            continue
        elif key in ("observation.state", "action"):
            spec = dict(spec)
            spec["shape"] = [ARM_DIMS]
            spec["names"] = [f"{key.split('.')[-1]}_{i}" for i in range(ARM_DIMS)]
            features[key] = spec
        else:
            features[key] = spec

    info["features"] = features
    (dest / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    trimmed = {}
    for key, block in stats.items():
        if key in cameras:
            trimmed[cameras[key]] = block
        elif key in DROP:
            continue
        elif key in ("observation.state", "action"):
            trimmed[key] = {name: _slice_stats(value, ARM_DIMS) for name, value in block.items()}
        else:
            trimmed[key] = block
    if delta_stats:
        # Replaces the absolute-action block wholesale.  Only `action` changes: the state is
        # not transformed by the relative step, so its statistics still describe what arrives.
        trimmed["action"] = _delta_action_stats(source, ARM_DIMS)
    (dest / "meta" / "stats.json").write_text(json.dumps(trimmed, indent=2))

    shutil.copy2(source / "meta" / "tasks.parquet", dest / "meta" / "tasks.parquet")

    for name, spec in cameras.items():
        (dest / "videos" / spec).symlink_to(source / "videos" / name, target_is_directory=True)

    frames, files = _rewrite_frames(source, dest, ARM_DIMS)
    _rewrite_episodes(source, dest, ARM_DIMS, cameras)

    return {
        "episodes": info["total_episodes"],
        "frames": frames,
        "parquet_files": files,
        "features": list(features),
    }


def _camera_names(raw: str | None) -> dict[str, str]:
    """The destination camera keys, defaulting to the LIBERO checkpoint's.

    Two checkpoints in use here declare *different* camera names:
    ``lerobot/pi05_libero_finetuned`` uses ``observation.images.image`` and ``image2``, while
    ``lerobot/pi05_base`` -- and therefore every fine-tune of it -- uses openpi's
    ``base_0_rgb`` and ``left_wrist_0_rgb``.  The derived dataset has to match whichever one
    it will be trained against.

    A dataset whose names already match needs no ``rename_map``, and that matters more than it
    sounds.  ``lerobot_train.py`` sets ``processor_pretrained_path = None`` whenever
    ``--policy.use_relative_actions=true``, which skips the entire block that installs the
    ``rename_observations_processor`` override -- so with relative actions on, a dataset that
    needs a rename fails in ``_preprocess_images`` with "All image features are missing from
    the batch", and a dataset that needs none trains.  Renaming the columns here is simpler
    than patching around that, and it leaves the dataset self-describing.
    """

    if not raw:
        return dict(CAMERAS)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if len(names) != len(CAMERAS):
        raise SystemExit(
            f"--camera-names expects {len(CAMERAS)} comma-separated names, got {len(names)}"
        )
    return dict(zip(CAMERAS, names, strict=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--camera-names",
        default=None,
        help=(
            "Destination keys for the two kept cameras, in the order "
            f"{list(CAMERAS.values())} -- e.g. "
            "'observation.images.base_0_rgb,observation.images.left_wrist_0_rgb' for a "
            "pi05_base fine-tune.  Default: the LIBERO checkpoint's names."
        ),
    )
    parser.add_argument(
        "--delta-stats",
        action="store_true",
        help=(
            "Write the `action` statistics as those of `action - state` rather than of the "
            "absolute action.  Required for `--policy.use_relative_actions=true`, which "
            "inserts a delta transform before the normaliser; without it the normaliser "
            "applies absolute-action statistics to deltas and the normalised target does not "
            "land in [-1, 1].  See `_delta_action_stats` for the measurements."
        ),
    )
    args = parser.parse_args()

    report = build(
        args.source, args.dest, _camera_names(args.camera_names), delta_stats=args.delta_stats
    )
    print(f"wrote {args.dest}")
    for key, value in report.items():
        print(f"  {key:16s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
