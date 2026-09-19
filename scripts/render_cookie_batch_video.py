"""Record the five-at-a-time batch expert to an mp4.

The batch runner only writes diagnostic PNGs, so this drives the same expert and
composites the scene camera plus the three policy cameras into a video, with an
overlay showing the phase and the live Cookie counts.  The counts matter for
judging the run: a healthy batch confirms five Cookies together, and the failure
mode worth watching for is a Cookie riding along that never gets confirmed.

Rendering is by far the dominant cost (software rasterisation), so the frame rate
is decoupled from the control rate by ``--every``: the expert still runs every
control step, but only every Nth step is rendered.  ``--fast-render`` drops the
shadow and reflection passes, which is a ~5x saving and the difference between a
few minutes and an hour.

Encoding needs libx264, provided by PyAV; ffmpeg itself is not on PATH.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from a3_dual_arm_sim.batch_expert import A3CookieBatchExpert
from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv

PROJECT = Path(__file__).resolve().parents[1]
TASK = "transfer exactly ten upright square cookie blocks into the 2x5 box"
PANEL_CAMERAS = ("left_wrist", "right_wrist")


def load_font(size: int):
    for path in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class Encoder:
    """Minimal PyAV h264 writer."""

    def __init__(self, path: Path, *, fps: int, crf: int, size: tuple[int, int]):
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = size
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": "medium"}
        self.count = 0

    def add(self, frame: np.ndarray) -> None:
        import av

        video = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self.stream.encode(video):
            self.container.mux(packet)
        self.count += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def build_frame(
    view: np.ndarray,
    panels: dict[str, np.ndarray],
    *,
    step: int,
    status: str,
    info: dict,
    font_big,
    font_small,
) -> np.ndarray:
    """Scene view with the policy cameras and a stats strip composited below."""
    width = view.shape[1]
    panel_w = panels["left_wrist"].shape[1] if panels else 0
    strip_h = panels["left_wrist"].shape[0] if panels else 0

    rows = [view]
    if panels:
        strip = Image.new("RGB", (width, strip_h), (18, 18, 20))
        x = 0
        for name in PANEL_CAMERAS:
            if name in panels:
                strip.paste(Image.fromarray(panels[name]), (x, 0))
                x += panel_w
        rows.append(np.asarray(strip, dtype=np.uint8))

    canvas = Image.fromarray(np.concatenate(rows, axis=0))
    draw = ImageDraw.Draw(canvas)

    # A translucent band keeps the text legible over both bright table and dark
    # background without hiding the scene behind it.
    band = Image.new("RGBA", (width, 46), (0, 0, 0, 140))
    canvas.paste(Image.alpha_composite(
        canvas.crop((0, 0, width, 46)).convert("RGBA"), band
    ).convert("RGB"), (0, 0))

    in_target = info.get("cookies_in_target", 0)
    in_source = info.get("cookies_in_source", 0)
    draw.text((10, 6), status, font=font_big, fill=(255, 255, 255))
    draw.text(
        (10, 28),
        f"step {step:5d}    target {in_target:2d}/10    source {in_source:2d}",
        font=font_small,
        fill=(210, 210, 215),
    )
    return np.asarray(canvas, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/cookie_batch.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/videos/batch_5plus5.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--view-size", type=int, default=640)
    parser.add_argument("--panel-size", type=int, default=200)
    parser.add_argument("--every", type=int, default=3, help="render every Nth control step")
    parser.add_argument("--fps", type=int, default=0, help="0 derives fps from --every")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--no-fast-render", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    env = A3CookieTransferEnv(config=config, render_cameras=False)
    if not args.no_fast_render:
        env.use_fast_render()

    control_hz = int(getattr(config, "control_hz", 20))
    fps = args.fps or max(1, control_hz // max(args.every, 1))

    observation, _ = env.reset(seed=args.seed, options={"randomize_cookies": False})
    expert = A3CookieBatchExpert(env)
    expert.reset()

    font_big = load_font(19)
    font_small = load_font(15)
    encoder = Encoder(args.output, fps=fps, crf=args.crf,
                      size=(args.view_size, args.view_size + args.panel_size))

    print(f"config   : {args.config}")
    print(f"output   : {args.output}")
    print(f"frame    : {args.view_size}x{args.view_size + args.panel_size} at {fps} fps "
          f"(rendering every {args.every} of {control_hz} Hz control steps)")
    print(f"max steps: {args.max_steps}")

    started = time.time()
    info: dict = {}
    step = 0
    rendered = 0
    try:
        for step in range(1, args.max_steps + 1):
            action = expert.act(observation, TASK)
            observation, _, terminated, truncated, info = env.step(action)

            if step % args.every == 0:
                view = env.render_view("front", height=args.view_size, width=args.view_size)
                panels = {
                    name: env.render_view(name, height=args.panel_size, width=args.panel_size)
                    for name in PANEL_CAMERAS
                }
                encoder.add(build_frame(
                    view, panels, step=step, status=expert.status, info=info,
                    font_big=font_big, font_small=font_small,
                ))
                rendered += 1
                if rendered % 40 == 0:
                    done = time.time() - started
                    print(f"  step {step:5d} {expert.status} "
                          f"({done:.0f}s, {rendered} frames)", flush=True)

            if terminated or truncated or expert.failed:
                break
    finally:
        encoder.close()
        env.close()

    elapsed = time.time() - started
    print()
    print(f"stopped  : step {step}, failed={expert.failed} "
          f"{expert.failure_reason or ''}")
    print(f"frames   : {encoder.count}  ({elapsed:.0f}s wall, "
          f"{elapsed / max(encoder.count, 1):.1f}s per rendered frame)")
    print(f"cookies  : target {info.get('cookies_in_target')}  "
          f"source {info.get('cookies_in_source')}  "
          f"success={info.get('success')}")
    print(f"written  : {args.output}")
    return 0 if info.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
