"""Check that inference renders the cameras the way the training data was captured.

The cookie dataset was collected with `--fast-render`, and every evaluation and
recording path calls `use_fast_render()` unconditionally, so training and inference
agree. That agreement only matters if the flag actually changes the pixels, which
this quantifies: it renders the same simulator state twice, once with the per-light
shadow and reflection passes and once without, and reports the difference.

It also reports each variant against the recorded dataset's camera statistics, so
"which rendering was the policy trained on" is answered by measurement rather than
by reading flags. Run it after changing a camera pose, the lighting, or a default in
`configs/default.yaml`, because a mismatch here degrades a vision policy silently
rather than raising.

    python scripts/measure_render_consistency.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from a3_dual_arm_sim.contracts import FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv

CAMERAS = (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE)


def capture(with_shadows: bool) -> dict[str, np.ndarray]:
    """Render the deterministic seed-0 scene with or without the light passes."""

    env = A3CookieTransferEnv(render_cameras=True)
    if not with_shadows:
        env.use_fast_render()
    try:
        observation, _ = env.reset(seed=0, options={"randomize_cookies": False})
        return {key: np.asarray(observation[key], dtype=np.uint8).copy() for key in CAMERAS}
    finally:
        env.close()


print(f"mujoco {mujoco.__version__}")
print()

with_shadows = capture(with_shadows=True)
without = capture(with_shadows=False)

print("=" * 78)
print("Same simulator state, rendered with and without the shadow/reflection passes")
print("=" * 78)
print(
    f"  {'camera':>34} {'mean':>7} {'mean_noshadow':>14} {'mean|diff|':>11} {'p95|diff|':>10} {'max|diff|':>10}"
)
for key in CAMERAS:
    a = with_shadows[key].astype(np.int16)
    b = without[key].astype(np.int16)
    difference = np.abs(a - b)
    print(
        f"  {key:>34} {a.mean():>7.2f} {b.mean():>14.2f} "
        f"{difference.mean():>11.2f} {np.percentile(difference, 95):>10.0f} {difference.max():>10.0f}"
    )

print()
print("=" * 78)
print("Against the recorded dataset")
print("=" * 78)
try:
    import av

    video = (
        PROJECT / "outputs/datasets/a3_cookie_overnight/videos/observation.images.front/"
        "chunk-000/file-000.mp4"
    )
    container = av.open(str(video))
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    sample = np.stack(frames[:60]).astype(np.float64)
    print(
        f"  dataset front camera, first 60 frames : mean={sample.mean():.2f} std={sample.std():.2f}"
    )
except (OSError, ValueError) as exc:  # pragma: no cover - diagnostic
    print(f"  dataset read failed: {type(exc).__name__}: {exc}")

front_with = with_shadows[FRONT_IMAGE].astype(np.float64)
front_without = without[FRONT_IMAGE].astype(np.float64)
print(
    f"  live scene, shadows ON                 : mean={front_with.mean():.2f} std={front_with.std():.2f}"
)
print(
    f"  live scene, shadows OFF (--fast-render): mean={front_without.mean():.2f} std={front_without.std():.2f}"
)
print()
print("  The dataset and the shadows-OFF render are the ones that must agree; the")
print("  absolute means differ anyway because the robot and cookies are in a")
print("  different pose after reset than in the middle of a recorded episode.")
