"""Gate: does this branch now reproduce main's benchmark scene to the pixel and the millimetre?

Run before collecting anything on the aligned scene -- a millimetre-scale offset is exactly what
turned a 5/10 policy into 0/20 on main's benchmark, so "looks about right" is not a standard here.

Checks, against a dump of main's own environment (`dump_main_scene.py`):

  * both bins, all 80 Cookies, all 18 arm joints (by name),
  * every camera's world position, local orientation and fovy,
  * and the three 256x256 renders, against main's own renders, in mean absolute pixel error.

    MUJOCO_GL=egl python scripts/verify_main_alignment.py --scene /tmp/main_scene.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv  # noqa: E402
from a3_dual_arm_sim.main_alignment import (  # noqa: E402
    CAMERAS,
    apply_scene,
    diff_scene,
    load_scene,
    scene_fingerprint,
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--scene", default="/tmp/main_scene.json")
parser.add_argument("--config", default="configs/cookie_batch_second_batch.yaml")
parser.add_argument("--references", default="/tmp/bench_shots")
parser.add_argument("--cookie-tolerance-mm", type=float, default=0.5)
parser.add_argument("--joint-tolerance-rad", type=float, default=1e-6)
parser.add_argument("--camera-tolerance-mm", type=float, default=0.5)
parser.add_argument("--mae-tolerance", type=float, default=1.0, help="mean |pixel| difference")
args = parser.parse_args()

reference = load_scene(args.scene)
env = A3CookieTransferEnv(config=args.config, render_cameras=True)
env.reset(seed=0, options={"randomize_cookies": False})
apply_scene(env, reference)
candidate = scene_fingerprint(env)

print(f"scene reference: {reference['source']}")
print(f"{'quantity':32s} {'unit':>6s} {'difference':>12s}")
worst: dict[str, float] = {}
for field, unit, value in diff_scene(reference, candidate):
    print(f"{field:32s} {unit:>6s} {value:>12.6f}")
    worst[unit] = max(worst.get(unit, 0.0), value)

mae = {}
for name in CAMERAS:
    path = Path(args.references) / f"mainref_{name}.npy"
    if not path.exists():
        continue
    reference_image = np.load(path).astype(float)
    image = np.asarray(env._render_camera(name), dtype=float)
    mae[name] = float(np.abs(image - reference_image).mean())
env.close()

print()
for name, value in mae.items():
    print(f"render MAE {name:11s} = {value:7.3f} / 255")
print()

failures = []
if worst.get("mm", 0.0) > max(args.cookie_tolerance_mm, args.camera_tolerance_mm):
    failures.append(f"position difference {worst['mm']:.3f} mm")
if worst.get("rad", 0.0) > args.joint_tolerance_rad:
    failures.append(f"joint difference {worst['rad']:.2e} rad")
if mae and max(mae.values()) > args.mae_tolerance:
    failures.append(f"render MAE {max(mae.values()):.3f} > {args.mae_tolerance}")
print("VERDICT:", "PASS" if not failures else "FAIL -- " + "; ".join(failures))
raise SystemExit(0 if not failures else 1)
