"""Put our recording next to LIBERO's own, frame for frame, on the same camera.

Both panels are the **agentview** camera.  Ours is the `front` camera, which was copied from
LIBERO's own and verified to project six known points to within 0.01 px at 256, so the two
are the same view of the same scene with the same robot -- the only thing that differs is
which simulator drew them.  That is what makes the comparison worth looking at.

Our recording is a composite: the scene view (480x480) on the left, the three policy cameras
in a column beside it, and a chart strip underneath.  Only the scene view is kept, so the two
panels are the same field of view and the same size, and the text overlay never lands on the
arm (it is laid along the bottom of the scene view -- see `render_libero_props_video.py`).

Usage:

    python scripts/splice_libero_videos.py \
        --ours /root/autodl-tmp/logs/videos/panda_vs_libero2.mp4 \
        --ours-seed 0 \
        --libero /root/autodl-tmp/logs/eval_libero_spatial0/videos/libero_spatial_0/eval_episode_0.mp4 \
        --output /root/autodl-tmp/logs/videos/side_by_side_0.mp4

Both sides are resampled to the same frame count (they run at different rates -- ours at the
20 Hz control rate, LIBERO's written out at 80 fps), so the two panels stay in step.
"""

from __future__ import annotations

import argparse

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
BAR_HEIGHT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, help="our composite recording")
    parser.add_argument("--ours-seed", type=int, default=0,
                        help="which seed to lift out of the composite")
    parser.add_argument("--ours-steps", type=int, default=300,
                        help="steps per seed in the composite")
    parser.add_argument("--ours-view", type=int, default=480,
                        help="size of the square scene view inside our composite")
    parser.add_argument("--libero", required=True, help="LIBERO's own eval episode video")
    parser.add_argument("--output", required=True)
    parser.add_argument("--still", default="", help="also write the middle frame as a PNG")
    parser.add_argument("--frames", type=int, default=0,
                        help="frames to emit (0 = as many as the shorter side has)")
    parser.add_argument("--height", type=int, default=480, help="panel height of the output")
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


def read_frames(path: str) -> list[np.ndarray]:
    """Every frame of `path` as RGB uint8, in order."""
    frames: list[np.ndarray] = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise SystemExit(f"{path} decoded to nothing")
    return frames


def resample(frames: list[np.ndarray], count: int) -> list[np.ndarray]:
    """`count` frames picked evenly out of `frames`, first and last included.

    The two recordings are not the same length -- LIBERO's episode ended where it ended --
    so one of them has to be re-timed.  Nearest-index rather than interpolation keeps every
    emitted frame a real frame from the recording, so nothing in the picture is invented.
    """
    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    return [frames[index] for index in indices]


def label_bar(width: int, left: str, right: str) -> Image.Image:
    """The caption strip across the top, one label centred over each panel."""
    bar = Image.new("RGB", (width, BAR_HEIGHT), (18, 18, 18))
    draw = ImageDraw.Draw(bar)
    font = ImageFont.truetype(FONT_PATH, 15)
    half = width // 2
    for index, text in enumerate((left, right)):
        box = draw.textbbox((0, 0), text, font=font)
        x = index * half + (half - (box[2] - box[0])) // 2
        draw.text((x, 6), text, font=font,
                  fill=(120, 255, 160) if index == 0 else (255, 230, 120))
    draw.line((half, 0, half, BAR_HEIGHT), fill=(90, 90, 90))
    return bar


def main() -> int:
    args = parse_args()

    ours = read_frames(args.ours)
    start = args.ours_seed * args.ours_steps
    segment = ours[start:start + args.ours_steps]
    if len(segment) != args.ours_steps:
        raise SystemExit(f"seed {args.ours_seed} is not in {args.ours}: the composite only "
                         f"has {len(ours)} frames")
    view = args.ours_view
    # The scene view is the square in the top-left corner; the policy-camera column and the
    # chart strip are dropped so the panel is nothing but the agentview camera.
    segment = [frame[:view, :view] for frame in segment]
    theirs = read_frames(args.libero)

    count = args.frames or min(len(segment), len(theirs))
    left = resample(theirs, count)
    right = resample(segment, count)

    height = args.height
    panels = []
    for frames in (left, right):
        panels.append([
            np.asarray(Image.fromarray(frame).resize((height, height), Image.LANCZOS))
            for frame in frames
        ])
    bar = np.asarray(label_bar(height * 2, "LIBERO (robosuite)  agentview",
                               "ours (A3 sim, Panda)  front"))

    with av.open(args.output, "w") as container:
        stream = container.add_stream("libx264", rate=int(args.fps))
        stream.width = height * 2
        stream.height = height + BAR_HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for index in range(count):
            grid = np.concatenate([panels[0][index], panels[1][index]], axis=1)
            composite = np.concatenate([bar, grid], axis=0)
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(composite), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    if args.still:
        index = count // 2
        grid = np.concatenate([panels[0][index], panels[1][index]], axis=1)
        Image.fromarray(np.concatenate([bar, grid], axis=0)).save(args.still)
        print(f"still  : {args.still}")

    print(f"frames : {count}  ({height + BAR_HEIGHT}x{height * 2})  "
          f"ours seed {args.ours_seed} vs {args.libero.split('/')[-1]}")
    print(f"output : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
