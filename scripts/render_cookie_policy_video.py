"""Record a trained checkpoint's closed-loop rollout to an mp4.

The environment already renders the three policy cameras on every step, so a
recording is a matter of compositing those frames and encoding them. A larger
render of the scene camera is added as the main view because the policy cameras
are pinned to the 256x256 training resolution and are too small to judge what the
arm is actually doing.

The overlay carries the step index, the cookie counts, and the phase index - the
demonstrated frame the arm's joint state currently matches. Phase makes the
qualitative behaviour legible: a healthy rollout tracks it 1:1, and the failure
mode this checkpoint exhibits is the phase jumping backwards when the policy
starts replaying an earlier cycle. Without it a viewer cannot tell replaying the
demonstration apart from acting correctly.

``--expert`` records the scripted expert through the same code path instead of a
checkpoint, which gives a reference video that differs only in which policy drives
the arm.

Encoding needs libx264, provided by PyAV; ffmpeg itself is not on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

TASK = "transfer exactly ten upright square cookie blocks into the 2x5 box"
POLICY_CAMERAS = ("front", "left_wrist", "right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained policy to record; ignored when --expert is given",
    )
    parser.add_argument(
        "--expert",
        action="store_true",
        help="Record the scripted expert instead, as a reference to compare against",
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/videos/rollout.mp4")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT / "outputs/datasets/a3_cookie_overnight",
        help="Supplies the action contract and the reference trajectory for the phase readout",
    )
    parser.add_argument("--repo-id", default="local/a3-cookie-overnight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help=(
            "Control steps to record. 0 (the default) runs until the environment "
            "ends the episode, i.e. to the 2500-step cookie horizon or to a "
            "success/safety termination, so the recording covers the whole task "
            "attempt including however it fails. The demonstration needs ~126 "
            "steps per cookie."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Safety cap used when --steps is 0, so a stuck rollout still ends",
    )
    parser.add_argument(
        "--view-size",
        type=int,
        default=480,
        help="Pixel size of the main scene view; 0 uses only the policy cameras",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=160,
        help="Pixel size of each policy-camera inset",
    )
    parser.add_argument("--fps", type=int, default=0, help="0 follows the env control rate")
    parser.add_argument("--crf", type=int, default=20, help="lower is higher quality")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_font(size: int):
    """A scalable font for the overlay, with a bundled fallback."""

    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


MIN_FONT_SIZE = 10


def fit_text(text: str, max_width: int, cap_size: int):
    """The font and text, shortened only if necessary, that fit ``max_width``.

    Measuring at a reference size and scaling by the ratio avoids reloading the
    TrueType file once per candidate size, which matters because this runs for
    every frame of a several-thousand-step recording. Shrinking has a floor:
    below it the text is clipped instead, since unreadable type is worse than a
    truncated label.
    """

    available = max(40, max_width - 20)
    measured = load_font(100).getlength(text)
    if measured <= available:
        return load_font(cap_size), text

    size = int(100 * available / measured)
    if size >= MIN_FONT_SIZE:
        return load_font(min(cap_size, size)), text

    font = load_font(min(cap_size, MIN_FONT_SIZE))
    clipped = text
    while clipped and font.getlength(clipped + "...") > available:
        clipped = clipped[:-1]
    return font, (clipped.rstrip() + "..." if len(clipped) < len(text) else text)


class ExpertPolicy:
    """Adapts the scripted expert to the policy interface the recorder uses.

    Recording the demonstration with the same code path makes the two videos
    directly comparable: the only difference left is which policy drives the arm.
    The environment's own success termination ends the episode, so the expert needs
    no separate stopping rule.
    """

    def __init__(self, env) -> None:
        from a3_dual_arm_sim.expert import make_cookie_transfer_expert

        self._expert = make_cookie_transfer_expert(env)

    def reset(self, context) -> None:
        self._expert.reset()

    def act(self, observation: dict, task: str) -> np.ndarray:
        return self._expert.act(observation)

    def close(self) -> None:
        pass


class FrameComposer:
    """Lays each step's renders out into one video frame.

    Layout is the scene view on the left and the three policy-camera insets
    stacked on the right, sized so the two columns are the same height. Insets are
    labelled because three similar-looking robot views are otherwise ambiguous.
    """

    def __init__(self, view_size: int, panel_size: int) -> None:
        from PIL import Image, ImageDraw

        self._image = Image
        self._draw = ImageDraw
        self.view_size = view_size
        self.panel_size = panel_size
        self.width = view_size + panel_size if view_size else panel_size
        self.height = view_size if view_size else panel_size * len(POLICY_CAMERAS)
        self._label_font = load_font(max(12, panel_size // 11))
        self._hud_font = load_font(max(14, self.height // 30))
        self._panel_labels = ("policy: front", "policy: left wrist", "policy: right wrist")

    def compose(
        self,
        view: np.ndarray | None,
        cameras: dict[str, np.ndarray],
        lines: list[str],
        title: tuple[str, ...] = (),
    ) -> np.ndarray:
        Image, ImageDraw = self._image, self._draw
        frame = Image.new("RGB", (self.width, self.height), (18, 18, 18))

        if view is not None:
            frame.paste(Image.fromarray(view), (0, 0))

        column = self.view_size if self.view_size else 0
        for index, name in enumerate(POLICY_CAMERAS):
            top = index * self.panel_size
            panel = Image.fromarray(np.asarray(cameras[name], dtype=np.uint8))
            # The insets are square in practice, but resize defensively so a
            # different camera aspect ratio cannot distort the layout.
            if panel.size != (self.panel_size, self.panel_size):
                panel = panel.resize((self.panel_size, self.panel_size))
            frame.paste(panel, (column, top))
            label = self._panel_labels[index]
            draw = ImageDraw.Draw(frame)
            box = draw.textbbox((0, 0), label, font=self._label_font)
            draw.rectangle(
                (column, top, column + (box[2] - box[0]) + 8, top + (box[3] - box[1]) + 6),
                fill=(0, 0, 0),
            )
            draw.text((column + 4, top + 2), label, font=self._label_font, fill=(255, 230, 120))

        draw = ImageDraw.Draw(frame)
        # The label identifies which policy produced the video, so a recording is
        # self-describing once it is separated from its log file. It is allowed to
        # use the full width, while the numeric lines stay within the scene view so
        # they never cover the policy-camera insets.
        rows: list[tuple[str, Any, tuple[int, int, int]]] = []
        for index, line in enumerate(title):
            font, text = fit_text(line, self.width, self._hud_font.size)
            colour = (120, 255, 160) if index == 0 else (150, 220, 170)
            rows.append((text, font, colour))
        if lines:
            body_width = self.view_size or self.width
            body_font, _ = fit_text(max(lines, key=len), body_width, self._hud_font.size)
            rows.extend((line, body_font, (255, 255, 255)) for line in lines)

        heights = [int(font.size * 1.35) for _, font, _ in rows]
        plate_width = min(
            max(draw.textbbox((0, 0), text, font=font)[2] for text, font, _ in rows) + 20,
            self.width,
        )
        plate_height = sum(heights) + 8
        # A translucent plate keeps the text readable over any scene content.
        shade = Image.new("RGB", (plate_width, plate_height), (0, 0, 0))
        blended = Image.blend(frame.crop((0, 0, plate_width, plate_height)), shade, 0.6)
        frame.paste(blended, (0, 0))
        top = 4
        for (text, font, colour), height in zip(rows, heights, strict=True):
            draw.text((10, top), text, font=font, fill=colour)
            top += height

        return np.asarray(frame, dtype=np.uint8)


def main() -> int:
    args = parse_args()

    # These must be set before lerobot is imported anywhere.
    os.environ.setdefault("HF_HOME", str(PROJECT / ".runtime/hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import av
    import pyarrow.parquet as pq

    from a3_dual_arm_sim.contracts import STATE, EpisodeContext
    from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv
    from a3_dual_arm_sim.lerobot_policy import (
        LeRobotPolicyAdapter,
        PolicyRuntimeConfig,
    )

    active = list(range(8))  # only the left arm moves in this dataset

    # Reference trajectory for the phase readout: episode 0 of the recording,
    # which is the demonstration closest to the deterministic evaluation scene.
    table = pq.read_table(
        next((args.dataset_root / "data").rglob("*.parquet")),
        columns=[STATE, "episode_index"],
    )
    states = np.stack(table[STATE].to_numpy(zero_copy_only=False)).astype(np.float64)
    episodes = table["episode_index"].to_numpy(zero_copy_only=False).reshape(-1)
    reference = states[episodes == 0][:, active]

    env = A3CookieTransferEnv(render_cameras=True)
    env.use_fast_render()

    # The expert plans against live simulator state, so it can only be built once
    # the environment it drives exists.
    if args.expert:
        driver: Any = ExpertPolicy(env)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --expert is given")
        driver = LeRobotPolicyAdapter(
            PolicyRuntimeConfig(
                checkpoint=args.checkpoint,
                dataset_root=args.dataset_root,
                repo_id=args.repo_id,
                device=args.device,
            )
        )

    fps = args.fps or int(getattr(env.config, "control_hz", 20))
    composer = FrameComposer(args.view_size, args.panel_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # --steps 0 means "record the whole attempt", so the cap becomes the
    # environment's own horizon unless the caller asks for something shorter.
    step_limit = args.steps if args.steps > 0 else min(args.max_steps, env.config.horizon)
    title = describe_policy(args, env)

    container = av.open(str(args.output), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = composer.width
    stream.height = composer.height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf), "preset": "medium"}

    print(f"source     : {'scripted expert' if args.expert else args.checkpoint}")
    for index, line in enumerate(title):
        print(f"{'policy     ' if index == 0 else '             '}: {line}")
    print(f"output     : {args.output}")
    print(f"frame      : {composer.width}x{composer.height} at {fps} fps")
    print(f"steps      : {step_limit} (seed {args.seed})")
    if step_limit == env.config.horizon:
        print("             = the environment horizon, so the whole attempt is recorded")
    print()

    frames = 0
    started = time.perf_counter()
    info: dict = {}
    terminated = truncated = False
    step = 0
    peak = 0
    knockouts = 0
    previous_target = 0
    # Steps since the bin last gained a cookie. The rollout's failure is that it
    # stops making progress long before the horizon, so the stall length is the
    # number worth reading off the video.
    stall = 0
    peak_step = 0
    first_knockout_step: int | None = None
    timeline: list[tuple[int, int, int]] = []
    try:
        observation, _ = env.reset(seed=args.seed, options={"randomize_cookies": False})
        driver.reset(EpisodeContext(seed=args.seed, task=TASK, action_mode="joint_position"))

        while not terminated and not truncated and step < step_limit:
            step += 1

            view = (
                env.render_view("front", height=args.view_size, width=args.view_size)
                if args.view_size
                else None
            )
            cameras = {name: observation[f"observation.images.{name}"] for name in POLICY_CAMERAS}
            # `info` and `observation` describe the same instant: env.step returns
            # both for the state it just produced, so they stay in step here.
            distances = np.linalg.norm(
                reference - np.asarray(observation[STATE], dtype=np.float64)[active], axis=1
            )
            phase = int(np.argmin(distances))
            distance = float(distances[phase])

            placed = int(info.get("cookies_in_target", 0))
            if placed > peak:
                peak = placed
                peak_step = step
                stall = 0
                timeline.append((step, placed, int(info.get("cookies_in_source", 30))))
                print(f"  step {step:>4}: in_bin {placed}/10 (new peak)", flush=True)
            else:
                stall += 1
            if placed < previous_target and first_knockout_step is None:
                first_knockout_step = step
            if placed < previous_target:
                knockouts += 1
            previous_target = placed

            source = int(info.get("cookies_in_source", 30))
            frame = composer.compose(
                view,
                cameras,
                [
                    (f"step {step}   in bin {placed}/10   peak {peak}   in source {source}/30"),
                    (
                        f"phase {phase}/{len(reference)}   distance {distance:.3f} rad   "
                        f"knocked out {knockouts}"
                    ),
                    f"stalled {stall} steps since the last new placement",
                ],
                title=title,
            )
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
            frames += 1

            action = driver.act(observation, TASK)
            observation, _, terminated, truncated, info = env.step(action)

            if step % 100 == 0:
                elapsed = time.perf_counter() - started
                remaining = (step_limit - step) / max(step / elapsed, 1e-6)
                print(
                    f"  step {step:>4} in_bin {placed:>2} peak {peak:>2} "
                    f"stalled {stall:>4} phase {phase:>4} dist {distance:.3f} "
                    f"({elapsed:.0f}s, ~{remaining:.0f}s left)",
                    flush=True,
                )

        # The final state is one step ahead of the last recorded frame, so its
        # numbers are not what the last frame shows; encode nothing extra and
        # report it in the console summary instead.
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
        driver.close()
        env.close()

    elapsed = time.perf_counter() - started
    print()
    print(f"frames   : {frames} ({frames / fps:.1f} s of video) in {elapsed:.0f}s")
    print(f"ended on : terminated={terminated} truncated={truncated} after {step} steps")
    print(
        f"final    : in_bin {int(info.get('cookies_in_target', 0))}/10 "
        f"in_source {int(info.get('cookies_in_source', 0))}/30"
    )
    print(f"result   : peak {peak}/10 in the bin, {knockouts} knockouts")
    print(f"stall    : last new placement at step {peak_step}, then {stall} steps of none")
    if first_knockout_step is not None:
        print(f"first loss of a placed cookie at step {first_knockout_step}")
    print()
    print("placements (step, in_bin, in_source):")
    for placed_step, bin_count, source_count in timeline:
        print(f"  step {placed_step:>4}  in_bin {bin_count:>2}  in_source {source_count:>2}")
    print(f"written  : {args.output}")
    return 0


def describe_policy(args: argparse.Namespace, env: Any) -> tuple[str, ...]:
    """The overlay lines naming the policy.

    A recording is usually watched away from its log file, and "is this the
    trained model or the expert?" is the first question it has to answer. For a
    checkpoint that means the policy type, the optimisation steps it saw, and
    which directory it came from, all read from the checkpoint rather than assumed
    from the path.
    """

    if args.expert:
        return ("expert: scripted planner, not a learned policy",)

    checkpoint = Path(args.checkpoint)
    policy_type = "unknown"
    config_path = checkpoint / "config.json"
    if config_path.exists():
        try:
            policy_type = json.loads(config_path.read_text()).get("type", "unknown")
        except (OSError, ValueError):
            pass

    # `train_config.json` is written by the training run; its `steps` is the total
    # number of optimisation steps, which is what separates a fine-tuned model from
    # the untouched base.
    steps = "trained, step count unknown"
    train_config = checkpoint / "train_config.json"
    if train_config.exists():
        try:
            total = json.loads(train_config.read_text()).get("steps")
            if total:
                steps = f"fine-tuned {int(total)} steps"
        except (OSError, ValueError):
            pass

    # Show the path as given and its target when they differ: `last` is a symlink
    # into a numbered step directory, and which step a video came from is exactly
    # what a viewer needs to know.
    resolved = Path(checkpoint).resolve()
    shown = _relative(Path(checkpoint))
    if resolved != Path(checkpoint).absolute():
        shown = f"{shown} -> {resolved.parent.name}"

    horizon = getattr(env.config, "horizon", 0)
    return (
        f"policy: {policy_type} ({steps})   env horizon {horizon} steps",
        f"checkpoint: {shown}",
    )


def _relative(path: Path) -> Path:
    """``path`` relative to the project when possible, else unchanged.

    Deliberately does not resolve symlinks: the caller wants to show the path it
    was given, and resolving here would make the symlink annotation redundant.
    """

    try:
        return path.absolute().relative_to(PROJECT)
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main())
