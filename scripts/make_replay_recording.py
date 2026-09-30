#!/usr/bin/env python
"""Write a replay recording for the bridge's ``replay`` transport.

Two sources, and both are useful for different reasons:

``--source hold``
    The environment's own reset pose, repeated.  Proves the pipeline runs and the
    action contract holds, with no model and no dataset.

``--source dataset``
    A recorded episode's actions, read straight from the parquet files.  Replaying
    an expert through the bridge is the only way to check that normalisation,
    chunking and dimensionality reproduce a known-good trajectory rather than
    merely producing finite numbers.

Usage::

    python scripts/make_replay_recording.py --source hold --steps 200 \\
        --output configs/policies/replay_hold.json

    python scripts/make_replay_recording.py --source dataset --root /data/a3-cookie \\
        --episode 0 --output /tmp/expert_episode0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def hold_actions(steps: int) -> np.ndarray:
    """The A3 home pose as a 16-dimensional action, repeated ``steps`` times."""

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from a3_dual_arm_sim.config import load_config

    config = load_config(None)
    home = np.array(
        [*config.home.left, config.home.grippers[0], *config.home.right, config.home.grippers[1]],
        dtype=np.float64,
    )
    return np.tile(home, (steps, 1))


def dataset_actions(root: Path, episode: int) -> np.ndarray:
    """Read one episode's recorded actions without decoding any video."""

    try:
        import pyarrow.dataset as pa_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("Reading a dataset needs pyarrow: pip install pyarrow") from exc
    dataset = pa_dataset.dataset(root / "data", format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "action"],
        filter=pa_dataset.field("episode_index") == episode,
    )
    rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
    if not rows:
        raise SystemExit(f"episode {episode} has no actions under {root}")
    return np.asarray([row["action"] for row in rows], dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("hold", "dataset"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200, help="Only for --source hold")
    parser.add_argument(
        "--root", type=Path, default=None, help="Dataset root, for --source dataset"
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--space",
        default="joint_position",
        help="Action space the recording is in; must match the manifest",
    )
    args = parser.parse_args(argv)

    if args.source == "hold":
        actions = hold_actions(args.steps)
        name = f"hold pose x{args.steps}"
    else:
        if args.root is None:
            parser.error("--source dataset needs --root")
        actions = dataset_actions(args.root, args.episode)
        name = f"{args.root.name} episode {args.episode}"

    if actions.shape[1] != 16:
        raise SystemExit(
            f"recorded actions are {actions.shape[1]}-dimensional; the A3 joint contract is "
            "16. A recording in another space needs the matching manifest entry."
        )
    payload = {"name": name, "space": args.space, "actions": actions.tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "name": name,
                "frames": int(actions.shape[0]),
                "dim": int(actions.shape[1]),
                "bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
