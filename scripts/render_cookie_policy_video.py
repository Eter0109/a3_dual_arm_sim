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
        default=800,
        help="Control steps to record; the demonstration needs ~126 per cookie",
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
        line_height = int(self._hud_font.size * 1.35)
        boxes = [draw.textbbox((0, 0), line, font=self._hud_font) for line in lines]
        plate_width = min(max(box[2] for box in boxes) + 20, self.width)
        plate_height = line_height * len(lines) + 8
        # A translucent plate keeps the numbers readable over any scene content.
        shade = Image.new("RGB", (plate_width, plate_height), (0, 0, 0))
        frame.paste(Image.blend(frame.crop((0, 0, plate_width, plate_height)), shade, 0.6), (0, 0))
        for row, line in enumerate(lines):
            draw.text((10, 4 + line_height * row), line, font=self._hud_font, fill=(255, 255, 255))

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

    container = av.open(str(args.output), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = composer.width
    stream.height = composer.height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf), "preset": "medium"}

    print(f"source     : {'scripted expert' if args.expert else args.checkpoint}")
    print(f"output     : {args.output}")
    print(f"frame      : {composer.width}x{composer.height} at {fps} fps")
    print(f"steps      : {args.steps} (seed {args.seed})")
    print()

    frames = 0
    started = time.perf_counter()
    info: dict = {}
    terminated = truncated = False
    step = 0
    peak = 0
    knockouts = 0
    previous_target = 0
    try:
        observation, _ = env.reset(seed=args.seed, options={"randomize_cookies": False})
        driver.reset(EpisodeContext(seed=args.seed, task=TASK, action_mode="joint_position"))

        while not terminated and not truncated and step < args.steps:
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
            peak = max(peak, placed)
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
                ],
            )
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
            frames += 1

            action = driver.act(observation, TASK)
            observation, _, terminated, truncated, info = env.step(action)

            if step % 50 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  step {step:>4} in_bin {placed:>2} peak {peak:>2} "
                    f"phase {phase:>4} dist {distance:.3f} "
                    f"({elapsed:.0f}s, {step / elapsed:.2f} steps/s)",
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
    print(f"written  : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
