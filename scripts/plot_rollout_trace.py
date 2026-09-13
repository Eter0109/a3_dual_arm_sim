"""Chart the phase trace of a recorded rollout, using only Pillow.

The phase readout is the arm's joint state matched by nearest neighbour against
the demonstrated trajectory, so plotting it against the step index shows at a
glance whether a rollout is following the demonstration or cycling through it. A
healthy rollout keeps phase and step in lockstep and climbs to the end of the
demonstration; this policy instead oscillates across the early cycles while the
step count climbs, which is exactly what its failures look like.

The renderer logs one sample every 100 steps, which is what this parses. Matplotlib
is not installed in this environment and the chart is simple enough for Pillow.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
SAMPLE = re.compile(
    r"step\s+(\d+)\s+in_bin\s+(\d+)\s+peak\s+(\d+)\s+stalled\s+(\d+)"
    r"\s+phase\s+(\d+)\s+dist\s+([\d.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Renderer log to read")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference-steps", type=int, default=1261)
    return parser.parse_args()


def load_font(size: int):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def main() -> int:
    args = parse_args()
    output = args.output or args.log.with_suffix(".png")

    samples = [
        tuple(
            int(value) if index < 5 else float(value) for index, value in enumerate(match.groups())
        )
        for match in SAMPLE.finditer(args.log.read_text())
    ]
    if not samples:
        raise SystemExit(f"no per-100-step samples found in {args.log}")

    steps = [row[0] for row in samples]
    in_bin = [row[1] for row in samples]
    phase = [row[4] for row in samples]
    distance = [row[5] for row in samples]
    span = max(steps)

    # Layout: three stacked panels sharing an x axis.
    width, height = 1280, 780
    left, right, top, bottom = 100, 40, 78, 64
    plot_width = width - left - right
    gap = 54
    panel_height = (height - top - bottom - 2 * gap) // 3

    image = Image.new("RGB", (width, height), (250, 250, 252))
    draw = ImageDraw.Draw(image)
    title_font = load_font(26)
    label_font = load_font(17)
    small_font = load_font(14)

    draw.text((left, 18), f"rollout trace - {args.log.name}", font=title_font, fill=(20, 20, 30))

    def to_x(step: int) -> float:
        return left + plot_width * step / max(1, span)

    def panel(index: int) -> tuple[int, int]:
        top_y = top + index * (panel_height + gap)
        return top_y, top_y + panel_height

    def frame(index: int, title: str, ymax: float, ticks: int, fmt: str = ".0f") -> tuple[int, int]:
        top_y, bottom_y = panel(index)
        draw.rectangle(
            (left, top_y, left + plot_width, bottom_y),
            fill=(255, 255, 255),
            outline=(200, 200, 210),
        )
        draw.text((left, top_y - 26), title, font=label_font, fill=(40, 40, 60))
        for tick in range(ticks + 1):
            y = bottom_y - (bottom_y - top_y) * tick / ticks
            draw.line((left, y, left + plot_width, y), fill=(235, 235, 240))
            text = format(ymax * tick / ticks, fmt)
            draw.text(
                (left - 12 - draw.textlength(text, font=small_font), y - 8),
                text,
                font=small_font,
                fill=(90, 90, 100),
            )
        return top_y, bottom_y

    # Panel 1: phase against step. The diagonal is where a rollout that tracks the
    # demonstration exactly would sit, so it is the reference to judge against.
    top_y, bottom_y = frame(
        0,
        "phase (demonstrated frame the arm matches) vs step - a healthy rollout hugs the diagonal",
        args.reference_steps,
        4,
    )
    draw.line(
        [(to_x(0), bottom_y), (to_x(args.reference_steps), top_y)],
        fill=(120, 190, 120),
        width=2,
    )
    draw.text(
        (to_x(args.reference_steps * 0.62), bottom_y - panel_height * 0.72),
        "perfect tracking",
        font=small_font,
        fill=(90, 160, 90),
    )
    points = [
        (to_x(step), bottom_y - panel_height * min(1.0, value / args.reference_steps))
        for step, value in zip(steps, phase, strict=True)
    ]
    draw.line(points, fill=(40, 90, 200), width=3)
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(40, 90, 200))

    # Panel 2: cookies in the bin. A step function, since it only changes on a
    # placement or a knockout.
    top_y, bottom_y = frame(1, "cookies in the bin (task requires an exact 2x5 fill of 10)", 10, 5)
    line = []
    for step, count in zip(steps, in_bin, strict=True):
        line.extend([(to_x(step), bottom_y - panel_height * count / 10)] * 2)
    draw.line(line, fill=(200, 60, 60), width=3)

    # Panel 3: tracking distance, which flags where the arm leaves the demonstration.
    limit = max(distance) * 1.15
    top_y, bottom_y = frame(
        2, "distance from the nearest demonstrated state (rad)", limit, 4, fmt=".2f"
    )
    points = [
        (to_x(step), bottom_y - panel_height * min(1.0, value / limit))
        for step, value in zip(steps, distance, strict=True)
    ]
    draw.line(points, fill=(150, 90, 200), width=3)
    mean_step = 0.00824
    y = bottom_y - panel_height * min(1.0, mean_step / limit)
    draw.line((left, y, left + plot_width, y), fill=(230, 180, 60), width=2)
    draw.text(
        (left + 8, y - 18),
        f"one expert step = {mean_step} rad",
        font=small_font,
        fill=(200, 140, 20),
    )

    draw.text((left, height - 34), "step", font=label_font, fill=(40, 40, 60))
    for step in range(0, span + 1, max(1, span // 10)):
        x = to_x(step)
        draw.text(
            (x - draw.textlength(str(step), font=small_font) / 2, height - 56),
            str(step),
            font=small_font,
            fill=(90, 90, 100),
        )

    image.save(output)
    peak_index = in_bin.index(max(in_bin))
    print(f"samples    : {len(samples)} at {span // max(1, len(samples) - 1)}-step intervals")
    print(f"peak       : {max(in_bin)}/10 in the bin, first reached at step {steps[peak_index]}")
    print(f"final      : {in_bin[-1]}/10 in the bin")
    print(
        f"phase      : min {min(phase)}, max {max(phase)} of {args.reference_steps} demonstrated frames"
    )
    print(f"distance   : mean {sum(distance) / len(distance):.3f}, max {max(distance):.3f} rad")
    print(f"written    : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
