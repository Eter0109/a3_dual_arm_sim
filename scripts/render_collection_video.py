"""Turn a collected dataset into watchable videos, one per episode.

Collection writes the three policy cameras as video streams, because that is what a
policy trains on. Those are 256x256 and stored per camera, with every episode
concatenated into one file, so they are awkward to review by hand -- and reviewing
by hand is the point: a dataset of a contact-heavy task cannot be judged from its
summary JSON alone.

This composites the collected frames into one file per episode plus a single
combined file, with an overlay carrying the episode, its seed, the outcome, and the
time through the episode. The seed and outcome matter as much as the pictures: a run
of six successful episodes collected at six seeds is a different artifact from six
successes at one seed, and only the overlay tells them apart on screen.

Frames are streamed, not buffered: the three camera files are decoded in lockstep
and composed one frame at a time, so memory stays flat regardless of episode length.
Episode boundaries come from ``a3_episode_metadata.jsonl``, which records the frame
count and outcome of each episode in the order they were written.

Usage:
    python scripts/render_collection_video.py --root outputs/datasets/x --output outputs/videos/x
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]

#: Drawn side by side under the main view, in this order.
PANEL_CAMERAS = ("left_wrist", "right_wrist")
EPISODE_METADATA = "a3_episode_metadata.jsonl"
CHUNK = "chunk-000"


def load_font(size: int):
    for candidate in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


class Encoder:
    """One mp4, fed frames in order."""

    def __init__(self, path: Path, *, fps: int, crf: int, size: tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = size
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf)}
        self.frames = 0

    def add(self, picture: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(picture, format="rgb24")
        frame.pts = self.frames
        self.frames += 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def episode_rows(root: Path) -> list[dict]:
    """One row per recorded episode, in the order the frames were written."""

    path = root / EPISODE_METADATA
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; this is not an A3 collection root"
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} has no episodes")
    return rows


def camera_path(root: Path, camera: str) -> Path:
    name = f"observation.images.{camera}"
    return root / "videos" / name / CHUNK / "file-000.mp4"


def open_cameras(root: Path, cameras: list[str]) -> dict[str, object]:
    """Decode every requested camera in one pass, yielding bundled frames."""

    decoders = {}
    for camera in cameras:
        path = camera_path(root, camera)
        if not path.is_file():
            raise FileNotFoundError(f"missing {camera} video: {path}")
        container = av.open(str(path))
        decoders[camera] = (container, container.decode(container.streams.video[0]))
    return decoders


def compose(
    view: np.ndarray,
    panels: list[np.ndarray],
    *,
    title: str,
    subtitle: str,
    font_big,
    font_small,
) -> np.ndarray:
    """Main view with the wrist cameras below it and a text band on top."""

    height, width = view.shape[:2]
    strip = 0
    if panels:
        panel_h = panels[0].shape[0]
        strip = panel_h
    canvas = Image.new("RGB", (width, height + strip), (18, 18, 20))
    canvas.paste(Image.fromarray(view), (0, 0))
    x = 0
    for panel in panels:
        canvas.paste(Image.fromarray(panel), (x, height))
        x += panel.shape[1]

    draw = ImageDraw.Draw(canvas)
    band = Image.new("RGBA", (width, 52), (0, 0, 0, 155))
    canvas.paste(
        Image.alpha_composite(
            canvas.crop((0, 0, width, 52)).convert("RGBA"), band
        ).convert("RGB"),
        (0, 0),
    )
    draw.text((10, 6), title, font=font_big, fill=(255, 255, 255))
    draw.text((10, 30), subtitle, font=font_small, fill=(212, 212, 218))
    return np.asarray(canvas, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Collected dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the videos")
    parser.add_argument(
        "--view-size", type=int, default=512, help="Main view size in pixels"
    )
    parser.add_argument(
        "--panel-size", type=int, default=256, help="Wrist panel size in pixels"
    )
    parser.add_argument(
        "--every", type=int, default=2, help="Encode every Nth collected frame"
    )
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument(
        "--max-episodes", type=int, default=0, help="0 means all of them"
    )
    parser.add_argument(
        "--no-combined", action="store_true", help="Skip the single all-episode file"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.every < 1:
        raise SystemExit("--every must be positive")

    root = args.root.expanduser().resolve()
    rows = episode_rows(root)
    if args.max_episodes:
        rows = rows[: args.max_episodes]

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(round(info.get("fps", 20) / args.every))
    cameras = ["front", *PANEL_CAMERAS]

    font_big = load_font(21)
    font_small = load_font(16)
    size = (args.view_size, args.view_size + args.panel_size)

    print(f"root     : {root}")
    print(f"episodes : {len(rows)}")
    for row in rows:
        outcome = "success" if row.get("success") else "failure"
        print(
            f"  episode {row['episode_index']:2d}  seed {row.get('seed')}  "
            f"{row.get('frames')} frames  {outcome}"
        )
    print(f"encoding : every {args.every}th frame at {fps} fps, {size[0]}x{size[1]}")

    decoders = open_cameras(root, cameras)
    combined = None if args.no_combined else Encoder(
        args.output / "all_episodes.mp4", fps=fps, crf=args.crf, size=size
    )
    total_written = 0
    try:
        for row in rows:
            index = int(row["episode_index"])
            frames = int(row["frames"])
            outcome = "SUCCESS" if row.get("success") else "FAILED"
            path = args.output / f"episode_{index:02d}_{outcome.lower()}.mp4"
            encoder = Encoder(path, fps=fps, crf=args.crf, size=size)
            written = 0
            try:
                for step in range(frames):
                    pictures = {}
                    for camera in cameras:
                        container, stream = decoders[camera]
                        try:
                            frame = next(stream)
                        except StopIteration:
                            raise RuntimeError(
                                f"{camera} video ended at frame {step} of episode "
                                f"{index}, which claims {frames} frames"
                            ) from None
                        pictures[camera] = frame.to_ndarray(format="rgb24")
                    if step % args.every:
                        continue
                    view = np.asarray(
                        Image.fromarray(pictures["front"]).resize(
                            (args.view_size, args.view_size), Image.BILINEAR
                        )
                    )
                    panels = [
                        np.asarray(
                            Image.fromarray(pictures[name]).resize(
                                (args.panel_size, args.panel_size), Image.BILINEAR
                            )
                        )
                        for name in PANEL_CAMERAS
                    ]
                    seconds = step / float(info.get("fps", 20))
                    picture = compose(
                        view,
                        panels,
                        title=(
                            f"episode {index + 1}/{len(rows)}   seed {row.get('seed')}"
                            f"   {outcome}"
                        ),
                        subtitle=(
                            f"t={seconds:5.1f}s   frame {step:5d}/{frames}"
                            f"   collected at {info.get('fps', 20)} Hz"
                        ),
                        font_big=font_big,
                        font_small=font_small,
                    )
                    encoder.add(picture)
                    if combined is not None:
                        combined.add(picture)
                    written += 1
            finally:
                encoder.close()
            total_written += written
            print(f"  wrote {path.name}  {written} frames  ({written / fps:.1f}s)", flush=True)
    finally:
        for container, _ in decoders.values():
            container.close()
        if combined is not None:
            combined.close()

    print(f"\n{len(rows)} episode video(s), {total_written} frames total")
    if combined is not None:
        print(f"combined: {args.output / 'all_episodes.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
