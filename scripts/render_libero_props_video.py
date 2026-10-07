"""Record the LIBERO-interface policy driving the A3 arm, so the failure can be watched.

Every number this investigation has produced is a projection of a motion, and a projection
that says "the bowl moved 40 mm" cannot be told apart from one that would have been 140 mm
had the arm gone the other way.  A recording is the thing that says which.  It is also the
only check on the two errors the trace probe was found to have been making: a bowl centre
26 mm low and a rim 17 mm narrow would have been obvious in a video within seconds, and
were invisible in a table for a week.

The frame is the scene view on the left -- the policy cameras are pinned to the 256x256
training resolution and are too small to judge an arm by -- the three policy cameras
stacked on the right, and a strip chart along the bottom.  Both cameras and chart carry the
numbers the trace probe records, so a clip and the trace of the same seed can be read
against each other.

The chart is over the two distances that decide the outcome: the nearest fingertip's
distance from the bowl's axis, and the bowl's distance from the plate's centre, each against
the measured radius it has to beat.  A rollout that closes the second curve is doing the
task; one that only dips the first is knocking the bowl about.

Usage::

    MUJOCO_GL=egl python scripts/render_libero_props_video.py --seeds 0 --steps 600
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

from a3_dual_arm_sim.props import (  # noqa: E402
    BOWL_MESH_BODY,
    EEF_SITE,
    FINGERTIPS,
    PLATE_MESH_BODY,
    prop_frame,
    prop_geoms,
    prop_points,
    robot_bodies,
    touching,
)

PROJECT = Path(__file__).resolve().parents[1]

POLICY_CAMERAS = ("front", "left_wrist", "right_wrist")

#: One unit of action, under the scene's `cartesian_translation_scale_m`.
ACTION_UNIT_M = 0.05

#: How near its starting height the bowl has to be to count as resting on the table.
RESTING_TOL_M = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="a3_libero_props_matched")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT / "configs/policies/pi05_libero_a3_cartesian.yaml")
    parser.add_argument("--task",
                        default="pick up the black bowl between the plate and the ramekin and "
                                "place it on the plate")
    parser.add_argument("--seeds", default="0",
                        help="Comma separated.  Each becomes one section of the video, "
                             "sharing one model process; the sections are not comparable "
                             "to each other beyond what n=1 ever is.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--output", type=Path,
                        default=Path("/root/autodl-tmp/logs/videos/libero_props_policy.mp4"))
    parser.add_argument("--view-size", type=int, default=480,
                        help="Pixel size of the scene view; 0 shows only the policy cameras")
    parser.add_argument("--panel-size", type=int, default=160)
    parser.add_argument("--chart-height", type=int, default=170)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--crf", type=int, default=20, help="lower is higher quality")
    parser.add_argument("--trace", type=Path, default=None,
                        help="Optional .npz of the same series the trace probe records.  "
                             "A success shows up in about one rollout in fifteen, so when "
                             "the camera catches one the numbers have to be kept too -- "
                             "re-recording the same seed will not reproduce it.")
    parser.add_argument("--shadows", action="store_true",
                        help="Keep the shadow and reflection passes.  They cost about four "
                             "times the frame time and only change how it looks.")
    return parser.parse_args()


_FONTS: dict[int, object] = {}


def load_font(size: int):
    """A scalable font, cached: this is called per frame and reading a TrueType file is
    far more expensive than drawing the text it is used for."""

    from PIL import ImageFont

    if size not in _FONTS:
        font = None
        for path in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
            if Path(path).exists():
                font = ImageFont.truetype(path, size)
                break
        _FONTS[size] = font or ImageFont.load_default(size=size)
    return _FONTS[size]


def fits(lines: list[str], size: int, width: int):
    """The largest size at or below ``size`` at which every line fits ``width``."""

    font = load_font(size)
    longest = max(font.getlength(line) for line in lines)
    if longest <= width:
        return font
    return load_font(max(10, int(size * width / longest)))


class Chart:
    """The two distances that decide the outcome, as a strip chart.

    Both series are millimetres against one axis, with the bowl's mouth radius and the
    plate's radius drawn in as limits, so the picture answers "did it get to the bowl" and
    "did it get to the plate" without the viewer holding either number in their head.
    """

    def __init__(self, width: int, height: int, steps: int, mouth_mm: float, plate_mm: float):
        self.width = width
        self.height = height
        self.steps = max(1, steps)
        self.mouth_mm = mouth_mm
        self.plate_mm = plate_mm
        # The ceiling only ever grows, so the curves do not rescale under the viewer.
        self.top_mm = max(120.0, mouth_mm * 1.8, plate_mm * 1.8)
        self.history: list[tuple[float, float]] = []
        self._font = load_font(13)

    def add(self, radial_mm: float, bowl_plate_mm: float) -> None:
        self.history.append((radial_mm, bowl_plate_mm))
        self.top_mm = max(self.top_mm, radial_mm * 1.05, bowl_plate_mm * 1.05)

    def _y(self, value: float) -> float:
        margin = 16
        span = self.height - 2 * margin
        return margin + span * (1.0 - min(1.0, max(0.0, value / self.top_mm)))

    def render(self):
        from PIL import Image, ImageDraw

        frame = Image.new("RGB", (self.width, self.height), (26, 26, 26))
        draw = ImageDraw.Draw(frame)
        plot_left, plot_right = 58, self.width - 8

        def x(index: int) -> float:
            return plot_left + (plot_right - plot_left) * index / self.steps

        for value, colour, label, style in (
            (self.mouth_mm, (255, 210, 120), f"bowl mouth {self.mouth_mm:.0f}", "dashed"),
            (self.plate_mm, (150, 200, 255), f"plate edge {self.plate_mm:.0f}", "dashed"),
        ):
            if value > self.top_mm:
                continue
            y = self._y(value)
            step = 10 if style == "dashed" else 1
            for start in range(plot_left, plot_right, step * 2):
                draw.line((start, y, min(start + step, plot_right), y), fill=colour)
            draw.text((plot_left + 6, y - 15), label, font=self._font, fill=colour)

        draw.text((4, 2), f"{self.top_mm:.0f} mm", font=self._font, fill=(150, 150, 150))
        draw.text((4, self.height - 18), "0", font=self._font, fill=(150, 150, 150))

        for index in (0, 1):
            colour = (255, 120, 120) if index == 0 else (120, 255, 160)
            points = [(x(step), self._y(values[index]))
                      for step, values in enumerate(self.history)]
            if len(points) > 1:
                draw.line(points, fill=colour, width=2)
            if points:
                draw.ellipse((points[-1][0] - 3, points[-1][1] - 3,
                              points[-1][0] + 3, points[-1][1] + 3), fill=colour)
            label = ("fingertip to the bowl's axis" if index == 0
                     else "bowl to the plate's centre")
            draw.text((plot_right - 250, 4 + index * 16), label, font=self._font, fill=colour)

        draw.rectangle((plot_left, 0, plot_right, self.height - 1), outline=(60, 60, 60))
        return frame


class Composer:
    """Scene view, policy cameras and the chart, laid out into one frame."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.view_size = args.view_size
        self.panel_size = args.panel_size
        self.chart_height = args.chart_height
        top_height = self.view_size if self.view_size else self.panel_size * len(POLICY_CAMERAS)
        self.width = (self.view_size + self.panel_size) if self.view_size else self.panel_size
        self.height = top_height + self.chart_height
        self._label_font = load_font(max(11, self.panel_size // 12))
        self._hud_font = load_font(max(14, top_height // 26))
        self._panel_labels = ("policy: front", "policy: left wrist",
                              "policy: right wrist")

    def compose(self, view, cameras, chart, title: tuple[str, ...], lines: list[str]):
        from PIL import Image, ImageDraw

        frame = Image.new("RGB", (self.width, self.height), (18, 18, 18))
        if view is not None:
            frame.paste(Image.fromarray(view), (0, 0))
        column = self.view_size if self.view_size else 0
        for index, name in enumerate(POLICY_CAMERAS):
            top = index * self.panel_size
            panel = Image.fromarray(np.asarray(cameras[name], dtype=np.uint8))
            if panel.size != (self.panel_size, self.panel_size):
                panel = panel.resize((self.panel_size, self.panel_size))
            frame.paste(panel, (column, top))
            label = self._panel_labels[index]
            draw = ImageDraw.Draw(frame)
            box = draw.textbbox((0, 0), label, font=self._label_font)
            draw.rectangle((column, top, column + (box[2] - box[0]) + 8,
                            top + (box[3] - box[1]) + 6), fill=(0, 0, 0))
            draw.text((column + 4, top + 2), label, font=self._label_font,
                      fill=(255, 230, 120))

        frame.paste(chart.render(), (0, self.height - self.chart_height))

        # The text sits over the scene view only, so it never covers the policy cameras or
        # the chart, and each line gets its own plate rather than one shared sheet, so a short
        # line does not dim scene it does not use.  One size for all of them, chosen from the
        # longest, so the block reads as a block.
        # Lay the text out along the **bottom** of the scene view, over a semi-transparent
        # plate.
        #
        draw = ImageDraw.Draw(frame)
        body_width = (self.view_size or self.width) - 16
        heading = fits(list(title), self._hud_font.size, body_width)
        body = fits(lines, max(11, self._hud_font.size - 6), body_width)
        entries = [
            (line, heading, (120, 255, 160) if index == 0 else (215, 215, 215))
            for index, line in enumerate(title)
        ] + [(line, body, (255, 255, 255)) for line in lines]
        heights = [draw.textbbox((0, 0), text, font=font)[3] + 8 for text, font, _ in entries]
        block = sum(heights)
        top = max(0, (self.view_size or self.height) - block - 6)

        sheets = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        brush = ImageDraw.Draw(sheets)
        y = top
        for (text, font, _), height in zip(entries, heights, strict=True):
            width = min(body_width + 16, draw.textbbox((0, 0), text, font=font)[2] + 14)
            brush.rectangle((0, y, width, y + height), fill=(0, 0, 0, 135))
            y += height
        frame = Image.alpha_composite(frame.convert("RGBA"), sheets).convert("RGB")

        draw = ImageDraw.Draw(frame)
        y = top
        for (text, font, colour), height in zip(entries, heights, strict=True):
            draw.text((8, y + 2), text, font=font, fill=colour)
            y += height
        return np.asarray(frame, dtype=np.uint8)


def main() -> int:
    args = parse_args()

    import av
    import mujoco

    from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway
    from a3_dual_arm_sim.bridge.manifest import load_manifest
    from a3_dual_arm_sim.collection import scene_by_name
    from a3_dual_arm_sim.contracts import STATE, EpisodeContext

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    scene = scene_by_name(args.scene)
    composer = Composer(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fps = args.fps
    container = av.open(str(args.output), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = composer.width
    stream.height = composer.height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf), "preset": "medium"}

    print(f"scene   : {args.scene}  ({scene.details['cameras']})")
    print(f"manifest: {args.manifest}")
    print(f"output  : {args.output}  {composer.width}x{composer.height} at {fps} fps")
    print(f"seeds   : {seeds}   steps per seed: {args.steps}")
    print()

    gateway = None
    written = 0
    traces: list[dict[str, np.ndarray]] = []
    started = time.perf_counter()
    try:
        for seed in seeds:
            env = scene.build_env(0.0, 0.0)
            if not args.shadows:
                env.use_fast_render()
            observation, _ = env.reset(seed=seed, options=None)

            model = env.model
            bowl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOWL_MESH_BODY)
            plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PLATE_MESH_BODY)
            tips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                    for name in FINGERTIPS]
            eef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
            if min(bowl, plate, eef, *tips) < 0:
                raise SystemExit(f"{args.scene} does not carry the LIBERO props")

            if gateway is None:
                gateway = ExternalPolicyGateway(load_manifest(args.manifest), env=env)
            else:
                gateway.rebind(env)
            gateway.reset(EpisodeContext(seed=seed, task=args.task,
                                         action_mode="joint_position"))

            mujoco.mj_forward(model, env.data)
            bowl_start, bowl_radius_m, bowl_size_m = prop_frame(model, env.data, bowl)
            plate_centre, plate_radius_m, _ = prop_frame(model, env.data, plate)
            bowl_geoms = set(prop_geoms(model, bowl))
            arm_bodies = robot_bodies(model)
            bowl_top_z = float(bowl_start[2] + bowl_size_m[2] / 2.0)
            start_gap_mm = float(np.linalg.norm(bowl_start[:2] - plate_centre[:2])) * 1000.0

            chart = Chart(composer.width, args.chart_height, args.steps,
                          bowl_radius_m * 1000.0, plate_radius_m * 1000.0)
            title = (
                f"pi05_libero_finetuned on the A3 arm -- seed {seed}",
                f"{args.scene}: bowl mouth {bowl_radius_m * 1000:.0f} mm, plate edge "
                f"{plate_radius_m * 1000:.0f} mm, and {start_gap_mm:.0f} mm between them",
            )
            print(f"seed {seed}: bowl {np.round(bowl_start, 4)} r={bowl_radius_m * 1000:.1f} mm"
                  f"  plate {np.round(plate_centre, 4)} r={plate_radius_m * 1000:.1f} mm"
                  f"  gap {start_gap_mm:.1f} mm", flush=True)

            touched = 0
            rows: list[dict[str, float]] = []
            for step in range(args.steps):
                view = (env.render_view("front", height=args.view_size, width=args.view_size)
                        if args.view_size else None)
                cameras = {name: observation[f"observation.images.{name}"]
                           for name in POLICY_CAMERAS}

                points = prop_points(model, env.data, bowl)
                bowl_centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
                fingertips = [np.asarray(env.data.site_xpos[t]) for t in tips]
                radial_mm = min(float(np.linalg.norm(tip[:2] - bowl_centre[:2]))
                                for tip in fingertips) * 1000.0
                above_rim_mm = min(float(tip[2] - bowl_top_z) for tip in fingertips) * 1000.0
                bowl_plate_mm = float(np.linalg.norm(
                    bowl_centre[:2] - plate_centre[:2])) * 1000.0
                bowl_height_mm = float(bowl_centre[2] - bowl_start[2]) * 1000.0
                on_bowl = touching(model, env.data, bowl_geoms, arm_bodies)
                on_table = touching(model, env.data, bowl_geoms, {0})
                touched += int(on_bowl)
                eef_position = np.asarray(env.data.site_xpos[eef])
                gripper = float(np.asarray(observation[STATE], dtype=np.float64)[7])
                command = gateway.metrics.get("libero_command") or [0.0] * 7
                chart.add(radial_mm, bowl_plate_mm)
                # The same keys the trace probe writes, so one analysis reads either.
                rows.append({
                    "step": step,
                    "radial_mm": radial_mm,
                    "above_rim_mm": above_rim_mm,
                    "gap_mm": float(np.linalg.norm(
                        np.asarray(fingertips[0]) - bowl_centre)) * 1000.0,
                    "bowl_mm": float(np.linalg.norm(bowl_centre - bowl_start)) * 1000.0,
                    "bowl_plate_mm": bowl_plate_mm,
                    "bowl_height_mm": bowl_height_mm,
                    "eef_x_mm": float(eef_position[0] - bowl_start[0]) * 1000.0,
                    "eef_y_mm": float(eef_position[1] - bowl_start[1]) * 1000.0,
                    "eef_z_mm": float(eef_position[2] - bowl_start[2]) * 1000.0,
                    "touch_arm": float(on_bowl),
                    "touch_table": float(on_table),
                    "gripper": gripper,
                })

                frame = composer.compose(
                    view, cameras, chart, title,
                    [
                        f"step {step + 1}/{args.steps}",
                        f"fingertip {radial_mm:.0f} mm from the bowl's axis, "
                        f"{above_rim_mm:+.0f} mm vs the rim",
                        f"bowl {bowl_plate_mm:.0f} mm from the plate's centre "
                        f"({bowl_plate_mm - start_gap_mm:+.0f} mm since the start), "
                        f"height {bowl_height_mm:+.0f} mm",
                        f"gripper {gripper:.2f} (0 shut, 1 open)   "
                        f"asking for {np.abs(command[:3]).max() * ACTION_UNIT_M * 1000:.1f} mm",
                        f"arm on the bowl: {'YES' if on_bowl else 'no'} "
                        f"({touched} steps)   bowl on the table: {'yes' if on_table else 'NO'}",
                    ],
                )
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
                written += 1

                action = gateway.act(observation, args.task)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"  episode ended at step {step + 1}", flush=True)
                    break
                if (step + 1) % 100 == 0:
                    print(f"  step {step + 1:>4}  fingertip {radial_mm:6.0f} mm  "
                          f"bowl->plate {bowl_plate_mm:6.0f} mm  "
                          f"touched {touched:>3}", flush=True)
            env.close()
            traces.append({key: np.asarray([row[key] for row in rows]) for key in rows[0]})

        if args.trace is not None:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.trace, **{f"{key}_s{index}": trace[key]
                                    for index, trace in enumerate(traces)
                                    for key in trace})
            print(f"trace  : {args.trace}")

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
        if gateway is not None:
            gateway.close()

    elapsed = time.perf_counter() - started
    print()
    print(f"frames : {written} ({written / fps:.1f} s of video) in {elapsed:.0f}s")
    print(f"output : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
