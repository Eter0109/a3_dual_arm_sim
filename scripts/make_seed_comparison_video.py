"""Tile several rollouts into one video, so seeds can be compared side by side.

Why this exists
---------------
A sweep's numbers say *how many* Cookies landed; they cannot say *why* one seed reached
ten and another stopped at five.  Reading that needs the seeds next to each other, at the
same moment of the episode, and a rollout video is a tall strip (a scene panel above the
policy cameras), so the comparison is a grid of strips rather than a single row.

The first ``--seconds`` of each input, decoded in lock step: frame *i* of the output is
frame *i* of every input.  That is the point of building it this way rather than with a
concat -- the episodes start from the same scene state, so column-for-column is the only
arrangement in which "this one is ahead" means anything.

Empty cells stay black.  A grid is filled row-major, so five inputs in a 3-wide grid leave
the last cell empty; padding rather than refusing keeps "the seeds I have" expressible.

Usage
-----
::

    python scripts/make_seed_comparison_video.py \\
        --output outputs/videos/full_0_4_8.mp4 \\
        --columns 3 --seconds 30 \\
        a_seed0.mp4 a_seed4.mp4 a_seed8.mp4

Every input must have the same resolution -- the tiling is a stack of frames, not a
scaling.  A mismatch is reported with both sizes rather than silently resized: two rollouts
recorded with different camera settings are not the same picture, and scaling one to match
the other would hide exactly that.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


class Reader:
    """One decoded input, one frame at a time, in the same order for every reader."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._container = None
        self._frames: Any = None
        self.size: tuple[int, int] | None = None

    def open(self) -> tuple[int, int]:
        import av

        self._container = av.open(str(self.path))
        stream = self._container.streams.video[0]
        stream.thread_type = "AUTO"
        self._frames = self._container.decode(stream)
        return int(stream.codec_context.width), int(stream.codec_context.height)

    def next_frame(self) -> np.ndarray | None:
        if self._frames is None:
            return None
        try:
            return next(self._frames).to_ndarray(format="rgb24")
        except StopIteration:
            return None

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
        self._frames = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="Rollout videos, in grid order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="How much of each input to keep"
    )
    parser.add_argument("--rate", type=float, default=None, help="Defaults to the input's rate")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args(argv)

    import av

    readers = [Reader(path) for path in args.inputs]
    sizes = [reader.open() for reader in readers]
    try:
        distinct = sorted(set(sizes))
        if len(distinct) != 1:
            detail = ", ".join(f"{reader.path.name}={size}" for reader, size in zip(readers, sizes))
            raise SystemExit(f"inputs differ in size, so they cannot be tiled: {detail}")
        tile_width, tile_height = distinct[0]
        columns = max(1, args.columns)
        rows = (len(readers) + columns - 1) // columns

        rate = args.rate
        if rate is None:
            with av.open(str(readers[0].path)) as probe:
                rate = float(probe.streams.video[0].average_rate or 20.0)
        frames = max(1, int(round(args.seconds * rate)))

        width = tile_width * columns
        height = tile_height * rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(args.output), mode="w")
        stream = container.add_stream("libx264", rate=int(round(rate)))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(args.crf), "preset": args.preset}

        print(
            f"tiling {len(readers)} inputs into {columns}x{rows} "
            f"({width}x{height}) for {args.seconds:g}s at {rate:g} fps -> {args.output}"
        )
        written = 0
        for index in range(frames):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            alive = 0
            for position, reader in enumerate(readers):
                tile = reader.next_frame()
                if tile is None:
                    # Shorter than the others: leave its cell black rather than repeating a
                    # frame, which would read as the rollout continuing.
                    continue
                row, column = divmod(position, columns)
                canvas[
                    row * tile_height : (row + 1) * tile_height,
                    column * tile_width : (column + 1) * tile_width,
                ] = tile
                alive += 1
            if alive == 0:
                break
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                container.mux(packet)
            written += 1
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        print(f"wrote {written} frames ({written / rate:.1f}s)")
    finally:
        for reader in readers:
            reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
