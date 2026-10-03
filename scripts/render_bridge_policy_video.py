"""Record an external model's rollout to an mp4.

The bridge's numbers say whether a model's output could drive the robot -- how far
the environment had to correct each command, whether joints sat on their limits --
but they cannot say what the arm actually did.  A video can, and for a zero-shot
model that is the whole question: the task metric is expected to be near zero, so
the informative thing to watch is *how* it fails.

Layout and encoding are shared with ``render_cookie_policy_video.py``: the scene
view on the left, the three policy-camera insets on the right, and a text plate
carrying the step, the model's own numbers and the behaviour metrics accumulated so
far.  The metrics are recomputed here with the same ``BehaviorTrace`` the comparison
uses, so a video and a comparison table cannot disagree about what happened.

Usage::

    python scripts/render_bridge_policy_video.py \\
        --manifest configs/policies/gr00t_bimanual.yaml \\
        --output outputs/videos/gr00t_sandbox.mp4 --steps 200

Encoding needs libx264, provided by PyAV; ffmpeg itself is not on PATH.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from render_cookie_policy_video import FrameComposer

from a3_dual_arm_sim.config import VISUAL_STYLES

POLICY_CAMERAS = ("front", "left_wrist", "right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="A policy manifest.  Omit it with --expert to record the scene's own expert.",
    )
    parser.add_argument(
        "--expert",
        action="store_true",
        help=(
            "Record the scene's own scripted expert instead of a model.  The scene has to be "
            "one of the registered ones -- the sandbox has no expert.  This is the reference a "
            "model's recording is read against: same scene, same camera framing, same "
            "compositing, same metrics, so the only difference left is which controller "
            "drives the arm."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene",
        default="sandbox",
        help=(
            "'sandbox', or a registered collection scene: "
            "a3_cookie_same_column (one source box to one target box), "
            "a3_cookie_batch, a3_cookie_two_box, a3_cookie_transfer, a3_grasp"
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--task",
        default=None,
        help="Task string sent to the model; defaults to the scene's own",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help=(
            "Let the scene randomise its layout.  Off by default so two models are "
            "recorded on the same scene; a comparison that showed each model a "
            "different arrangement would not be one."
        ),
    )
    parser.add_argument(
        "--view-size",
        type=int,
        default=640,
        help="Scene-view render size; 0 records only the policy cameras",
    )
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=0, help="0 follows the env control rate")
    parser.add_argument("--crf", type=int, default=20, help="lower is higher quality")
    parser.add_argument(
        "--fast-render",
        action="store_true",
        help="Skip shadow and reflection passes; much faster without a GPU",
    )
    parser.add_argument(
        "--visual-style",
        choices=VISUAL_STYLES,
        default="plain",
        help=(
            "Renderer appearance. 'plain' is the original look and stays the "
            "default because the rendered pixels are a dataset contract; "
            "'detailed' gives the Cookies their own materials so packed blades "
            "read as separate objects. It changes no physical parameter."
        ),
    )
    return parser.parse_args()


def build_environment(
    scene: str, config: Path | None, visual_style: str = "plain"
) -> tuple[Any, str, dict[str, Any]]:
    """Build a scene's environment, its task prompt and its reset options.

    Registered collection scenes are built through their own builders rather than by
    repeating their task config here.  That config decides what counts as success --
    ``require_exact_slots``, ``require_released``, ``terminate_on_success`` -- and a
    copy in this script would drift from the one the datasets were collected with,
    which would make a video disagree with the data it is supposed to illustrate.

    The layout is left unrandomised unless asked otherwise: two models recorded on
    different arrangements could not be compared, and a video is usually watched to
    see what a model does with a known scene.
    """

    from a3_dual_arm_sim.collection import registered_scenes, scene_by_name
    from a3_dual_arm_sim.config import load_config
    from a3_dual_arm_sim.env import A3DualArmEnv

    def styled(path: Path | None) -> Any:
        loaded = load_config(path)
        if visual_style == loaded.visual_style:
            return loaded
        from dataclasses import replace

        return replace(loaded, visual_style=visual_style)

    if scene == "sandbox":
        return (
            A3DualArmEnv(styled(config), scene="sandbox", render_cameras=True),
            "move the A3 dual-arm robot",
            {"randomize_objects": False},
        )

    if scene not in registered_scenes():
        raise SystemExit(
            f"unknown scene {scene!r}; use 'sandbox' or one of "
            f"{sorted(registered_scenes())}"
        )

    built = scene_by_name(scene, visual_style=visual_style)
    # Zero noise: the scene's own builder takes the jitter as an argument, and a
    # video wants the canonical layout rather than a sampled one.
    env = built.build_env(0.0, 0.0)
    return (
        env,
        built.task.prompt,
        {"randomize_scene": False, "randomize_cookies": False},
    )


def progress_line(info: dict[str, Any]) -> str | None:
    """The scene's own task counters, when it publishes any.

    A cookie scene reports how many pieces are in the target box and how many are
    left in the source, which is the one number a viewer needs to read a rollout: a
    model that never moves a piece and one that moves pieces and drops them look
    identical without it.
    """

    if "cookies_in_target" not in info:
        return None
    placed = int(info.get("cookies_in_target", 0))
    required = int(info.get("required_cookies", 0))
    source = int(info.get("cookies_in_source", 0))
    return f"in target {placed}/{required}   in source {source}"


class _SceneExpertDriver:
    """The scene's own scripted expert behind the gateway's three-member interface.

    The recording loop only ever calls ``reset`` and ``act`` and reads ``metrics``, so three
    members are enough to put an expert and a model through one compositing path.  Sharing
    that path is the point: the two recordings then differ in the controller and in nothing
    else -- same scene, same camera framing, same overlay, same ``BehaviorTrace`` limits -- so
    a difference in the video is a difference in behaviour rather than in how it was filmed.
    """

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    @property
    def metrics(self) -> dict[str, Any]:
        # No self-report.  An expert has no value estimate or token count to publish, and the
        # overlay's model line is meant to stay empty for exactly that case.
        return {}

    def reset(self, context: Any) -> None:
        self._policy.reset(context)

    def act(self, observation: dict[str, Any], task: str) -> Any:
        return self._policy.act(observation, task)

    def close(self) -> None:
        self._policy = None


def main() -> int:
    args = parse_args()

    import av

    from a3_dual_arm_sim.bridge.compare import BehaviorTrace, _arm_limits
    from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway
    from a3_dual_arm_sim.bridge.manifest import load_manifest, validate_manifest
    from a3_dual_arm_sim.collection import registered_scenes, scene_by_name
    from a3_dual_arm_sim.contracts import EpisodeContext

    manifest = None
    if args.manifest is not None:
        manifest = load_manifest(args.manifest)
        problems = validate_manifest(manifest)
        if problems:
            raise SystemExit(
                f"{args.manifest} is not usable:\n  - " + "\n  - ".join(problems)
            )
    elif not args.expert:
        raise SystemExit("--manifest is required unless --expert is given")

    env, scene_task, reset_options = build_environment(
        args.scene, args.config, args.visual_style
    )
    if args.fast_render:
        env.use_fast_render()
    if args.randomize:
        reset_options = {}

    task = args.task or scene_task

    if args.expert:
        # The scene's own `build_policy`: the same expert a collection run would drive, on the
        # same environment object the recording is about to step.
        if args.scene not in registered_scenes():
            raise SystemExit(
                f"--expert needs a registered scene; {args.scene!r} has no scripted expert. "
                f"Available: {sorted(registered_scenes())}"
            )
        gateway: Any = _SceneExpertDriver(
            scene_by_name(args.scene, visual_style=args.visual_style).build_policy(env)
        )
    else:
        gateway = ExternalPolicyGateway(manifest, env=env)
    # The same trace the comparison uses, so the numbers on the video and the
    # numbers in a report are computed by one implementation.
    trace = BehaviorTrace()
    trace.bind_limits(_arm_limits(env), getattr(env.config, "max_joint_step_rad", None))

    fps = args.fps or int(getattr(env.config, "control_hz", 20))
    composer = FrameComposer(args.view_size, args.panel_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.expert:
        title = (
            f"scripted expert  (scene {args.scene})",
            "the reference that says this scene can be solved at all",
        )
    else:
        title = (
            f"{manifest.name}  ({manifest.evaluation.regime}, {manifest.evaluation.prompt_kind})",
            (
                f"transport {manifest.transport.kind}   action {manifest.action.space} "
                f"{manifest.action.dim}d   chunk {manifest.action.chunk}/{manifest.action.execute}"
            ),
        )

    container = av.open(str(args.output), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = composer.width
    stream.height = composer.height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf), "preset": "medium"}

    print(f"manifest   : {args.manifest if manifest is not None else '<scripted expert>'}")
    print(
        "model      : "
        + (
            "scripted expert"
            if manifest is None
            else f"{manifest.name}  regime={manifest.evaluation.regime}"
        )
    )
    print(f"scene      : {args.scene}")
    print(f"task       : {task!r}")
    print(f"output     : {args.output}")
    print(f"frame      : {composer.width}x{composer.height} at {fps} fps")
    print(f"steps      : {args.steps} (seed {args.seed})")
    print()

    frames = 0
    started = time.perf_counter()
    info: dict[str, Any] = {}
    terminated = truncated = False
    step = 0
    try:
        observation, _ = env.reset(seed=args.seed, options=reset_options)
        gateway.reset(EpisodeContext(seed=args.seed, task=task, action_mode=env.action_mode))

        while not terminated and not truncated and step < args.steps:
            step += 1
            view = (
                env.render_view("front", height=args.view_size, width=args.view_size)
                if args.view_size
                else None
            )
            cameras = {
                name: observation[f"observation.images.{name}"] for name in POLICY_CAMERAS
            }
            summary = trace.summary()
            # The model's own reporting lives on the gateway, not on the environment:
            # this loop drives the rollout directly rather than through
            # ``EpisodeRunner``, so nothing merges the policy's ``metrics`` into the
            # env's ``info``.  Reading only ``info`` is why an earlier version of this
            # script printed "no model self-report" for a model that was reporting a
            # value estimate every step.
            model_info = gateway.metrics
            lines = [
                (
                    f"step {step}   corrected {summary.get('corrected_fraction', 0.0):.3f}"
                    f"   rate limited {summary.get('rate_limited_fraction', 0.0):.3f}"
                    f"   saturated {summary.get('saturated_fraction', 0.0):.3f}"
                ),
                (
                    f"correction max {summary.get('correction_max_rad', 0.0):.3f} rad"
                    f"   motion mean {summary.get('motion_mean_rad', 0.0):.3f} rad"
                ),
                _model_line(model_info),
            ]
            # The scene's own counters go first when it has any: on a cookie scene
            # the piece count is what a viewer reads the rollout by, and a model that
            # never moves a piece looks the same as one that moves and drops them
            # without it.
            progress = progress_line(info)
            if progress is not None:
                lines.insert(0, progress)
            frame = composer.compose(view, cameras, lines, title=title)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
            frames += 1

            action = gateway.act(observation, task)
            observation, _, terminated, truncated, info = env.step(action)
            # Recorded after the step, because the correction is the difference
            # between what was asked for and what the environment applied.
            trace.record(requested=action, applied=info["applied_action"])

            if step % 50 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  step {step:>4}  corrected {summary.get('corrected_fraction', 0.0):.3f}"
                    f"  saturated {summary.get('saturated_fraction', 0.0):.3f}"
                    f"  ({step / max(elapsed, 1e-6):.1f} steps/s)",
                    flush=True,
                )

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
        gateway.close()
        env.close()

    elapsed = time.perf_counter() - started
    summary = trace.summary()
    print()
    print(f"frames     : {frames} in {elapsed:.1f}s ({frames / max(elapsed, 1e-6):.1f} fps)")
    print(f"terminated : {terminated}   truncated: {truncated}")
    print(f"behaviour  : {summary}")
    print(f"output     : {args.output}")
    return 0


def _model_line(info: dict[str, Any]) -> str:
    """The model's own numbers, when it reports any.

    Kept separate from the behaviour line because the two answer different
    questions: behaviour is what the environment measured, this is what the model
    claimed.  A disagreement between them is worth seeing.

    The suffixes match ``bridge.compare._self_report`` on purpose, so a video and a
    comparison table describe the same fields.  ``_calls`` is included: a model that
    reports only a call count is still reporting something, and showing "no
    self-report" for it made the two disagree about GR00T.
    """

    parts: list[str] = []
    for key in sorted(info):
        value = info[key]
        if key.endswith(("_value", "_calls")) and isinstance(value, (int, float)):
            parts.append(f"{key} {float(value):.3f}")
        elif key.endswith("_task") and isinstance(value, str):
            parts.append(f"prompt {value!r}")
    return "   ".join(parts) if parts else "no model self-report"


if __name__ == "__main__":
    raise SystemExit(main())
