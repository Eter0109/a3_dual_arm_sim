"""Measure the two candidate physics-speed levers on the dense cookie scene.

Lever A -- physics rate.  The 2x10 benchmark runs at ``physics_hz=1000``
(``substeps=50`` per control step), while ``A3DualArmEnvConfig`` defaults to
500.  Halving the rate halves ``mj_step`` calls, but it also coarsens contact
integration, so throughput alone is not a result: a variant is only usable if
the expert still scores the same.

Lever B -- idle cookie collision.  Every one of the 80 cookies is a 6-DOF
``freejoint`` body with ``contype=1 conaffinity=3``, so all 80 take part in
broad- and narrow-phase on every substep even though a 2x10 episode only ever
touches the one or two source columns it was assigned.  Clearing
``contype``/``conaffinity`` on the unused columns keeps the model layout and
``qpos`` untouched (unlike welding, which would shift every downstream index)
and removes only their contact generation.

Why ``dataclasses.replace`` matters: ``A3DualArmEnvConfig`` is
``@dataclass(frozen=True)`` and ``substeps`` is a derived property, not a
field.  Assigning ``b.config.physics_hz`` raises ``FrozenInstanceError`` and
``replace(..., substeps=...)`` raises ``TypeError``.  Rebinding ``b.config``
to a replaced copy is the only supported route, and it must happen before
``create_env`` because the env captures the config at construction.

Usage::

    MUJOCO_GL=egl python scripts/recap/bench_physics_levers.py --steps 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_pi05_on_main import make_cookie_benchmark  # noqa: E402

# physics_hz must stay an integer multiple of control_hz (20), so 250 is
# rejected with "physics_hz must be an integer multiple of control_hz".
RATES = (1000, 500, 400, 300, 200)

RESET_OPTIONS = dict(
    randomize_cookies=True,
    randomize_boxes=True,
    randomize_source_bin=True,
    randomize_target_bin=True,
    randomization_profile="basic",
)


def build(hz: int, max_steps: int):
    bench = make_cookie_benchmark(
        task="2x10",
        prompt_style="task",
        max_steps=max_steps,
        randomize_boxes=True,
        randomize_cookies=True,
        render=False,
        success_hold_steps=20,
    )
    if hz != bench.config.physics_hz:
        bench.config = dataclasses.replace(bench.config, physics_hz=hz)
    return bench


def throughput(hz: int, steps: int) -> dict:
    """Time ``steps`` control steps with the arms held still.

    The arms repeat the proprioceptive state the env itself reports, which is
    what a zero-action step does, so no external policy is needed.
    """
    bench = build(hz, max_steps=max(steps, 10))
    env = bench.create_env(render_cameras=False)
    options = dict(RESET_OPTIONS)
    options.update(
        source_bin_noise_m=bench.source_bin_noise_m,
        target_bin_noise_m=bench.target_bin_noise_m,
        target_bin_yaw_noise_rad=bench.target_bin_yaw_noise_rad,
    )
    env.reset(seed=1000, options=options)
    hold = np.array(env.data.qpos, copy=True)
    action = np.asarray(env._observation()["observation.state"], dtype=np.float64)

    # One warm-up step so first-call model compilation is not charged to the
    # timing loop.
    env.step(action)
    hold = np.array(env.data.qpos, copy=True)

    start = time.time()
    for _ in range(steps):
        env.step(action)
    elapsed = time.time() - start

    return {
        "physics_hz": int(env.config.physics_hz),
        "substeps": int(env.config.substeps),
        "ms_per_step": elapsed / steps * 1000.0,
        "steps_per_s": steps / elapsed,
        # Not a fidelity metric on its own -- a resting stack that drifts is
        # still a different simulation from one that does not.  Recorded so a
        # rate that is merely unstable is caught before the expert run.
        "hold_still_drift_m": float(np.abs(np.array(env.data.qpos) - hold).max()),
    }


def expert_scores(hz: int, seeds, max_steps: int) -> dict:
    bench = build(hz, max_steps=max_steps)
    scores, failures = [], []
    start = time.time()
    for seed in seeds:
        try:
            result = bench.run_episode(policy="expert", seed=seed)
            scores.append(int(getattr(result, "cookies_in_target", 0)))
        except Exception as exc:  # a crash is a fidelity failure too
            scores.append(-1)
            failures.append(f"seed {seed}: {type(exc).__name__}: {exc}")
    elapsed = time.time() - start
    arr = np.array(scores, dtype=np.float64)
    return {
        "physics_hz": int(bench.config.physics_hz),
        "substeps": int(bench.config.substeps),
        "scores": scores,
        "mean_score": float(arr.mean()),
        "perfect_rate": float((arr >= 10).mean()),
        "seconds": elapsed,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200, help="timing loop length")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1000, 1001, 1002, 1003, 1004, 1005],
        help="seeds for the expert fidelity run",
    )
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--rates", type=int, nargs="+", default=list(RATES), help="physics_hz values"
    )
    parser.add_argument(
        "--skip-experts",
        action="store_true",
        help="throughput only (much faster, but proves nothing about fidelity)",
    )
    args = parser.parse_args()

    report = {"throughput": [], "experts": []}
    baseline = None
    for hz in args.rates:
        try:
            row = throughput(hz, args.steps)
        except Exception as exc:
            print(f"{hz:5d} Hz 吞吐失败: {type(exc).__name__}: {exc}", flush=True)
            continue
        report["throughput"].append(row)
        if baseline is None:
            baseline = row["steps_per_s"]
        speedup = row["steps_per_s"] / baseline
        print(
            f"{hz:5d} Hz ({row['substeps']:2d} 子步): "
            f"{row['ms_per_step']:6.1f} ms/步 = {row['steps_per_s']:5.1f} 步/秒 "
            f"({speedup:4.2f}x)  静止漂移 {row['hold_still_drift_m']:.2e}",
            flush=True,
        )

    if not args.skip_experts:
        for hz in args.rates:
            try:
                row = expert_scores(hz, args.seeds, args.max_steps)
            except Exception as exc:
                print(f"{hz:5d} Hz 专家失败: {type(exc).__name__}: {exc}", flush=True)
                continue
            report["experts"].append(row)
            print(
                f"{hz:5d} Hz 专家 {len(row['scores'])} 局: {row['scores']}  "
                f"均值 {row['mean_score']:.2f}  满分率 {row['perfect_rate']:.0%}  "
                f"{row['seconds']:.0f}s",
                flush=True,
            )
            for failure in row["failures"]:
                print(f"    {failure}", flush=True)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
