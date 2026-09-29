# Explicit same-name imports preserve the legacy public API.
# ruff: noqa: PLC0414
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from a3_dual_arm_sim.learning.training import default_base_model

from .workflows.cli_handlers import _collect_grasp as _collect_grasp
from .workflows.cli_handlers import _default_task as _default_task
from .workflows.cli_handlers import _inspect as _inspect
from .workflows.cli_handlers import _make_env as _make_env
from .workflows.cli_handlers import _read_episode_actions as _read_episode_actions
from .workflows.cli_handlers import _read_episode_context as _read_episode_context
from .workflows.cli_handlers import _recorder as _recorder
from .workflows.cli_handlers import _replay as _replay
from .workflows.cli_handlers import _run as _run
from .workflows.cli_handlers import _smoke as _smoke
from .workflows.cli_handlers import _teleop as _teleop
from .workflows.cli_handlers import _train_smolvla as _train_smolvla
from .workflows.cli_handlers import _validate_camera_recording as _validate_camera_recording


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)


def _add_recording(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record", type=Path, default=None, help="New LeRobot dataset root")
    parser.add_argument("--repo-id", default="local/a3-dual-arm")
    parser.add_argument("--videos", action="store_true", help="Encode camera streams as video")


def _add_camera_rendering(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera-render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render the three Policy RGB observations; use --no-camera-render "
            "for responsive viewer-only debugging"
        ),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="a3-sim", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Compile and describe the A3 model")
    inspect_parser.add_argument("--config", type=Path, default=None)
    inspect_parser.add_argument(
        "--scene", choices=("sandbox", "cookie_transfer"), default="sandbox"
    )
    inspect_parser.add_argument("--write-xml", type=Path, default=None)
    inspect_parser.set_defaults(function=_inspect)

    smoke = commands.add_parser("smoke", help="Run a deterministic headless hold episode")
    _add_common(smoke)
    smoke.add_argument("--scene", choices=("sandbox", "cookie_transfer"), default="sandbox")
    smoke.add_argument("--steps", type=int, default=1000)
    smoke.set_defaults(function=_smoke)

    run = commands.add_parser("run", help="Run a replaceable Python policy plugin")
    _add_common(run)
    _add_recording(run)
    _add_camera_rendering(run)
    run.add_argument("--scene", choices=("sandbox", "cookie_transfer"), default="sandbox")
    run.add_argument("--policy", required=True, help="Python module:factory")
    run.add_argument("--task", default=None)
    run.add_argument("--steps", type=int, default=None)
    run.add_argument("--render", action="store_true")
    run.set_defaults(function=_run)

    teleop = commands.add_parser("teleop", help="Control both arms with the keyboard")
    _add_common(teleop)
    _add_recording(teleop)
    _add_camera_rendering(teleop)
    teleop.add_argument("--scene", choices=("sandbox", "cookie_transfer"), default="sandbox")
    teleop.add_argument("--task", default=None)
    teleop.add_argument("--steps", type=int, default=None)
    teleop.set_defaults(function=_teleop)

    replay = commands.add_parser("replay", help="Replay canonical actions from a dataset episode")
    replay.add_argument("--config", type=Path, default=None)
    replay.add_argument("--root", type=Path, required=True)
    replay.add_argument("--repo-id", default="local/a3-dual-arm")
    replay.add_argument("--episode", type=int, default=0)
    replay.add_argument("--render", action="store_true")
    replay.set_defaults(function=_replay)

    collect = commands.add_parser(
        "collect-grasp", help="Collect successful A3 grasp expert episodes"
    )
    _add_common(collect)
    collect.add_argument("--root", type=Path, required=True)
    collect.add_argument("--repo-id", default="local/a3-grasp")
    collect.add_argument("--episodes", type=int, default=10)
    collect.add_argument("--max-attempts", type=int, default=None)
    collect.set_defaults(function=_collect_grasp)

    from a3_dual_arm_sim.learning.download import main as download_main

    download = commands.add_parser("download-smolvla", help="Download A3 data, base model and VLM")
    download.add_argument(
        "--dataset-root", type=Path, default=Path("datasets/a3_front_close_left_100")
    )
    download.add_argument("--model-root", type=Path, default=Path("models/smolvla_base"))
    download.add_argument("--manifest", type=Path, default=Path("outputs/download_manifest.json"))
    download.add_argument("--dataset-revision", default="main")
    download.add_argument("--model-revision", default="main")
    download.add_argument("--vlm-revision", default="main")
    download.set_defaults(function=download_main)

    train = commands.add_parser(
        "train-smolvla", help="Fine-tune SmolVLA on a validated A3 grasp dataset"
    )
    train.add_argument("--root", type=Path, required=True, help="LeRobot v3 dataset root")
    train.add_argument("--repo-id", default="Eter0109/a3-front-close-left-100")
    train.add_argument(
        "--model", default=default_base_model(), help="Hub repo ID or local checkpoint"
    )
    train.add_argument("--model-revision", default=None)
    train.add_argument("--offline", action="store_true")
    train.add_argument("--save-freq", type=int, default=2000)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=10_000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--seed", type=int, default=1000)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(function=_train_smolvla)
    from a3_dual_arm_sim.learning.act_training import main as act_main

    act = commands.add_parser("train-act", help="Train ACT on a validated A3 dataset")
    act.add_argument("--root", type=Path, required=True)
    act.add_argument("--repo-id", default="Eter0109/a3-front-close-left-100")
    act.add_argument("--output", type=Path, required=True)
    act.add_argument("--steps", type=int, default=20_000)
    act.add_argument("--batch-size", type=int, default=8)
    act.add_argument("--seed", type=int, default=1000)
    act.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    act.add_argument("--lr", type=float, default=1e-5)
    act.add_argument("--chunk-size", type=int, default=50)
    act.add_argument("--n-action-steps", type=int, default=8)
    act.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    act.add_argument("--save-freq", type=int, default=2000)
    act.add_argument("--num-workers", type=int, default=2)
    act.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    act.add_argument("--dry-run", action="store_true")
    act.set_defaults(function=act_main)
    return root


def main() -> int:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        _validate_camera_recording(args)
    except ValueError as exc:
        argument_parser.error(str(exc))
    exit_code = int(args.function(args))
    uses_human_viewer = args.command == "teleop" or bool(getattr(args, "render", False))
    if uses_human_viewer:
        # On Python 3.13, combining mujoco.viewer with mujoco.Renderer can finish
        # explicit cleanup successfully and then SIGSEGV during native GLFW module
        # finalization.  At this point runner.close() has already flushed recordings
        # and closed both contexts, so bypass only the faulty second teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
