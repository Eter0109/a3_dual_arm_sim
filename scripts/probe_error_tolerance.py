"""At what per-step error does the expert itself stop succeeding?

Teacher forcing measured the fine-tuned policies' error at 0.030 rad per step, against
0.018 rad of motion in each expert step -- so the model's error is larger than the
expert's own move, and over 540 steps with no correction that has to accumulate.  That
is a strong argument that the cookie task fails on per-step precision.  It is not a
proof, because the argument assumes the failure is smooth: the expert might have margin
that absorbs the error, or might not.

This measures the margin directly and without a policy in the way.  It runs the scene's
own scripted expert -- which succeeds 3 of 3 -- and adds zero-mean noise of a controlled
size to every action it issues.  Sweeping the scale gives the error level at which the
task breaks.  Compare that with the model's measured 0.030 rad and the question "is
precision the binding constraint" has an answer that does not depend on the model at all.

The noise is applied as ``action + s * eps`` with ``eps`` drawn from the same shape the
teacher-forcing probe measured: independent per joint, zero-mean, with a standard
deviation of 0.030 rad on the arm joints scaled to each joint's share of the total.  The
grippers are left alone -- the probe's per-joint table showed their error is the same
size as the arm's, but the scene clips them to a different rate and a gripper command
that misses is a binary outcome rather than a positional one.

Two scenes are swept on the same axis, because the whole question is whether the cookie
task's precision requirement is what makes it hard:

* ``grasp`` -- a 70 mm cube, no clearances, about 190 steps.
* ``cookie`` -- five 50x25x6.3 mm pieces into 2.5 mm gaps, about 540 steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from a3_dual_arm_sim.collection import registered_scenes  # noqa: E402
from a3_dual_arm_sim.grasp import A3GraspEnv, GraspTaskConfig  # noqa: E402
from a3_dual_arm_sim.runner import EpisodeRunner  # noqa: E402

#: Measured by ``scripts/probe_pi05_teacher_forcing.py``: mean over frames of the
#: per-frame max-over-joints error, chunks averaged.  The unit the sweep is expressed in.
MEASURED_ERROR_RAD = 0.030
ARM_JOINTS = list(range(7)) + list(range(8, 15))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", choices=("grasp", "cookie"), default="grasp")
    parser.add_argument(
        "--scales",
        default="0,0.25,0.5,1,2",
        help="Noise as a multiple of the measured 0.030 rad; 0 is the undoped expert",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--max-steps", type=int, default=700)
    return parser.parse_args()


class NoisyExpert:
    """The scene's expert, with controlled error added to every command it issues.

    Wraps rather than modifies, so the expert's own behaviour is untouched and the
    zero-scale case is exactly the undoped expert -- which is what makes the sweep's
    own origin a control rather than an assumption.
    """

    def __init__(self, expert, scale: float, rng: np.random.Generator) -> None:
        self._expert = expert
        self._scale = scale
        self._rng = rng
        self.action_mode = expert.action_mode

    def reset(self, context) -> None:
        self._expert.reset(context)

    def act(self, observation, task: str) -> np.ndarray:
        action = np.asarray(self._expert.act(observation, task), dtype=np.float64).copy()
        if self._scale > 0:
            noise = self._rng.normal(0.0, self._scale * MEASURED_ERROR_RAD, size=len(action))
            action[ARM_JOINTS] = action[ARM_JOINTS] + noise[ARM_JOINTS]
        return action

    def close(self) -> None:
        self._expert.close()

    def __getattr__(self, name):
        # ``metrics``, ``discard_requested`` and the rest are read off the policy by
        # the runner; they belong to the expert and pass straight through.
        return getattr(self._expert, name)


def build(scene: str):
    if scene == "grasp":
        env = A3GraspEnv(
            task_config=GraspTaskConfig(position_noise_m=0.0),
            action_mode="joint_position",
            render_cameras=False,
        )
        from a3_dual_arm_sim.expert import A3GraspExpert

        return env, A3GraspExpert(env), "pick up the red cube"

    collection = registered_scenes()["a3_cookie_same_column"]()
    env = collection.build_env(0.0, 0.0)
    return env, collection.build_policy(env), collection.task.prompt


def main() -> int:
    args = parse_args()
    scales = [float(s) for s in args.scales.split(",")]

    print(f"scene: {args.scene}   episodes per scale: {args.episodes}")
    print(f"noise unit: {MEASURED_ERROR_RAD} rad  (the fine-tuned models' measured error)")
    print()
    print(f"  {'scale':>6s}  {'noise (rad)':>11s}  {'success':>9s}  {'mean steps':>10s}  {'progress (final)':>16s}")
    print("  " + "-" * 62)

    for scale in scales:
        rng = np.random.default_rng(1234)
        results = []
        for offset in range(args.episodes):
            env, expert, task = build(args.scene)
            try:
                runner = EpisodeRunner(
                    env, NoisyExpert(expert, scale, rng), task=task, realtime=False
                )
                try:
                    episode = runner.run(seed=args.seed + offset, max_steps=args.max_steps)
                finally:
                    runner.close()
                results.append(episode)
            finally:
                env.close()

        successes = sum(1 for r in results if r.success)
        steps = np.mean([r.steps for r in results])
        progress = np.mean(
            [
                float(r.final_info.get("cookies_in_target", r.final_info.get("lift_m", 0.0)))
                for r in results
            ]
        )
        name = "cookies in target" if args.scene == "cookie" else "lift (m)"
        print(
            f"  {scale:6.2f}  {scale * MEASURED_ERROR_RAD:11.5f}  "
            f"{successes:4d}/{args.episodes:<4d}  {steps:10.0f}  {name} {progress:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
