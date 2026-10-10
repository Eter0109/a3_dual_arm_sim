"""Why a Cookie that is inside the box is not counted as inside the box.

`_cookie_inside_target` decides the dataset's success contract, and it has five clauses.
When it returns False the report says only that, so a rollout that put both pieces in the
box and a rollout that dropped one on the floor look the same from outside.  This prints
the clauses separately, in the box's own frame, for every Cookie at every checkpoint of a
rollout -- which is the instrument whose absence let a stacked placement read as a
failure.

The numbers come from the environment's own attributes, not from a copy of the formula, so
what is printed is what the verdict is actually made of.  The same reason
`render_bridge_policy_video.py` builds its scene through the scene's own builder.

    python scripts/probe_two_cookie_containment.py \
        --manifest configs/policies/pi05_a3_two_cookie_15k.yaml --seed 0 --steps 400
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from render_bridge_policy_video import build_environment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", default="a3_libero_panda_two_cookies")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default=None,
        help="A range or list, e.g. '0-3' or '0,2,5'.  One gateway, so the model loads once.",
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--fast-render", action="store_true")
    return parser.parse_args()


def clauses(env: Any, index: int) -> dict[str, Any]:
    """Every clause of `_cookie_inside_target`, in the target box's frame."""

    tb_pos = env.data.xpos[env._target_bin_body]
    tb_mat = env.data.xmat[env._target_bin_body].reshape(3, 3)
    position = env.data.xpos[env._cookie_bodies[index]]
    rotation = env._cookie_rotation(index)
    p_rel = tb_mat.T @ (position - tb_pos)
    r_rel = tb_mat.T @ rotation
    half_extent = np.abs(r_rel) @ env.COOKIE_HALF_SIZE

    floor_top = (
        env.config.cookie_transfer.target_floor_z_m
        + env.config.cookie_transfer.bin_wall_thickness_m / 2
    )
    bottom = float(p_rel[2] - half_extent[2])  # the Cookie's lowest face, in box frame
    wall_top = float(env.TARGET_WALL_HEIGHT)

    footprint_inside = bool(
        np.all(
            np.abs(p_rel[:2]) + half_extent[:2]
            <= env.TARGET_INNER_HALF_SIZE + env.CONTACT_CONTAINMENT_TOLERANCE_M
        )
    )
    above_floor = bool(bottom >= floor_top - 0.003)
    below_rim = bool(bottom < wall_top)
    #: The clause this probe exists for: the Cookie must be down *on the box floor*.
    on_the_floor = bool(bottom <= floor_top + 0.004)
    upright = bool(
        abs(float(r_rel[2, 2])) >= np.cos(env.task_config.max_tilt_rad)
        or abs(float(rotation[2, 2])) >= np.cos(env.task_config.max_tilt_rad)
    )
    linear_speed, angular_speed = env._cookie_average_speed(index)
    settled = bool(
        linear_speed <= max(env.task_config.max_linear_speed_m_s, 0.09)
        and angular_speed <= env.task_config.max_angular_speed_rad_s
    )
    contacts = env.privileged_left_finger_contacts(index)
    released = bool(not env.task_config.require_released or not any(contacts))
    return {
        "xy": np.round(p_rel[:2] * 1000.0, 2),  # mm, box frame
        "bottom_mm": round(bottom * 1000.0, 2),
        "height_mm": round(float(2 * half_extent[2]) * 1000.0, 2),
        "floor_mm": round(floor_top * 1000.0, 2),
        "wall_mm": round(wall_top * 1000.0, 2),
        "footprint": footprint_inside,
        "above_floor": above_floor,
        "below_rim": below_rim,
        "on_the_floor": on_the_floor,
        "upright": upright,
        "settled": settled,
        "released": released,
        "in_target": env._cookie_inside_target(index),
    }


def blockers(values: dict[str, Any]) -> str:
    """Which clauses are False, in the order `_cookie_inside_target` tests them.

    Printed because the verdict alone cannot distinguish the failures a reader cares
    about: a Cookie outside the box and a Cookie inside it that is one layer too high have
    the same answer, and only one of them is a policy that did the task.
    """

    failed = [
        name
        for name in ("footprint", "above_floor", "below_rim", "on_the_floor", "upright", "settled", "released")
        if not values[name]
    ]
    return ",".join(failed) if failed else "none"


def report(env: Any, step: int, label: str) -> None:
    count = env.task_config.cookie_count
    inside = [env._cookie_inside_target(index) for index in range(count)]
    print(f"  step {step:>4}  {label}  in target {sum(inside)}/{count}")
    for index in range(count):
        values = clauses(env, index)
        # The one clause that decides the verdict when everything else passes, printed as
        # a distance so a reader can see how far from the floor the piece actually is.
        gap = values["bottom_mm"] - values["floor_mm"]
        print(
            f"    cookie {index}: xy {values['xy']} mm  bottom {values['bottom_mm']:>6.2f} mm"
            f" ({gap:+.2f} above floor)  height {values['height_mm']:.2f} mm"
            f"  wall {values['wall_mm']:.1f} mm"
        )
        print(
            f"               footprint={values['footprint']}"
            f"  above_floor={values['above_floor']}"
            f"  below_rim={values['below_rim']}"
            f"  ON_THE_FLOOR={values['on_the_floor']}"
            f"  upright={values['upright']}"
            f"  settled={values['settled']}"
            f"  released={values['released']}"
            f"  -> {values['in_target']}  failed: {blockers(values)}"
        )


def _seeds(args: argparse.Namespace) -> list[int]:
    """`--seeds 0-3` or `0,2,5`, falling back to the single `--seed`.

    Several seeds in one process because the model load dominates: on this box the bridge
    takes about fourteen seconds to come up and each episode is around a hundred, so four
    seeds cost little more than one.
    """

    if args.seeds is None:
        return [args.seed]
    text = str(args.seeds).strip()
    if "-" in text:
        low, _, high = text.partition("-")
        return list(range(int(low), int(high) + 1))
    return [int(part) for part in text.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway
    from a3_dual_arm_sim.bridge.manifest import load_manifest, validate_manifest
    from a3_dual_arm_sim.contracts import EpisodeContext

    manifest = load_manifest(args.manifest)
    problems = validate_manifest(manifest)
    if problems:
        raise SystemExit(f"{args.manifest} is not usable:\n  - " + "\n  - ".join(problems))

    env, task, reset_options = build_environment(args.scene, args.config)
    if args.fast_render:
        env.use_fast_render()
    gateway = ExternalPolicyGateway(manifest, env=env)
    seeds = _seeds(args)
    print(f"scene  : {args.scene}")
    print(f"task   : {task!r}")
    print(f"seeds  : {seeds}   steps: {args.steps}")

    terminated = truncated = False
    step = 0
    try:
        for seed in seeds:
            print(f"\n===== seed {seed} =====")
            observation, _ = env.reset(seed=seed, options=reset_options)
            gateway.reset(EpisodeContext(seed=seed, task=task, action_mode=env.action_mode))
            terminated = truncated = False
            step = 0
            while not terminated and not truncated and step < args.steps:
                step += 1
                action = gateway.act(observation, task)
                observation, _, terminated, truncated, info = env.step(action)
                if step % args.report_every == 0:
                    report(env, step, "during")
            report(env, step, f"final (terminated={terminated} truncated={truncated})")
            print(f"  scene counters: {info.get('cookies_in_target')} in target, "
                  f"{info.get('cookies_in_source')} in source, success={info.get('success')}")
    finally:
        gateway.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
