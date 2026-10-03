"""Contact sheets from the single-arm cookie rollouts, so the behaviour can be looked at.

Reading a report tells you a model scored zero; it does not tell you whether the arm reached
for the cookies, went somewhere else, or moved at all.  Six evenly spaced frames per video,
stacked as one row per model, makes that visible in a single image -- and it is the only way
to review four 25-second clips without watching four 25-second clips.

The scene view is the left of each recorded frame and the policy cameras are the insets on
the right; only the scene view is kept, because that is where the arm is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAMES = 6
CROP = (0, 0, 0.62)  # left fraction of the composite, which is the scene view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=300)
    return parser.parse_args()


def frames_of(path: Path, count: int) -> list[np.ndarray]:
    """``count`` frames spread across the clip, decoded and cropped to the scene view."""

    container = av.open(str(path))
    stream = container.streams.video[0]
    total = stream.frames or 0
    wanted = (
        [int(round(i * (total - 1) / (count - 1))) for i in range(count)] if total else []
    )
    out: list[np.ndarray] = []
    seen = 0
    for frame in container.decode(stream):
        if wanted and seen in wanted:
            image = frame.to_ndarray(format="rgb24")
            height, width = image.shape[:2]
            out.append(image[:, : int(width * CROP[2])])
        seen += 1
        if wanted and seen > wanted[-1]:
            break
    container.close()
    if not out:  # a stream whose frame count is unknown: take what evenly spaced ones we can
        container = av.open(str(path))
        grabbed = [f for f in container.decode(video=0)]
        container.close()
        picks = np.linspace(0, len(grabbed) - 1, count).astype(int)
        for index in picks:
            image = grabbed[int(index)].to_ndarray(format="rgb24")
            out.append(image[:, : int(image.shape[1] * CROP[2])])
    return out


def font(size: int):
    for candidate in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def main() -> int:
    args = parse_args()
    rows = []
    for path, label in zip(args.videos, args.labels, strict=True):
        cells = frames_of(path, FRAMES)
        tiles = [
            Image.fromarray(cell).resize(
                (args.width, int(args.width * cell.shape[0] / cell.shape[1])),
                Image.LANCZOS,
            )
            for cell in cells
        ]
        strip = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), "black")
        x = 0
        for tile in tiles:
            strip.paste(tile, (x, 0))
            x += tile.width
        rows.append((label, strip))

    label_width = 24 + max(
        int(font(22).getlength(label)) for label, _ in rows
    )
    total_w = label_width + max(strip.width for _, strip in rows)
    total_h = 34 + sum(strip.height for _, strip in rows)
    sheet = Image.new("RGB", (total_w, total_h), "#101014")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 6), "single-arm cookie: 6 frames across a 500-step attempt", fill="white", font=font(20))

    y = 34
    for label, strip in rows:
        draw.text((12, y + strip.height // 2 - 12), label, fill="#7fd4ff", font=font(22))
        sheet.paste(strip, (label_width, y))
        y += strip.height + 4

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"wrote {args.output}  {sheet.width}x{sheet.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
