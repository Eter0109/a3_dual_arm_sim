"""Own the LIBERO rollout loop: reproduce the published score, then instrument it.

Why this exists
---------------
`lerobot-eval` turns this checkpoint into a number -- libero_spatial 100/100 against the
paper's 97.0 -- and reports nothing else: not what the model commanded, not how far the
arm was from the demonstration, not how much of the episode it spent standing still.  The
A3 investigation needs exactly those, and it needs them from a run where success is not in
doubt, so that "what a working rollout looks like" is measured rather than assumed.

So this script keeps the *setup* -- the same environments, the same observation
formatting, the same 8-wide state vector, the same relative control mode, the same
processor pipelines, the same action queue -- and replaces the loop.  If it does not
reproduce the published score, the setup is not equivalent and nothing it measures about
the A3 run can be trusted; that is why the score is printed first.

The one number it exists to produce
-----------------------------------
The A3 cookie demonstrations dwell.  On **57.2%** of frames the expert commands the arm to
stay exactly where it is (`|action - state| < 0.010 rad`), and over the whole dataset
`std(action - state) / std(action)` is only **4.4-8.8%** -- the identity component is
92-96% of the signal, and under MSE the conditional mean of a distribution that is 57%
"do not move" is very close to "do not move".

LIBERO is the control: this checkpoint places the object 97.5% of the time, so whatever
its commands look like, they are learnable.  LIBERO's action is an end-effector **delta**,
so "standing still" is literally `action == 0` and the dwell fraction is readable off the
commands directly, with no state subtraction and no unit conversion.  That makes the two
comparable as fractions even though one is radians and the other is normalised controller
units.

Usage::

    MUJOCO_GL=egl python scripts/probe_libero_reference.py --suite libero_spatial --tasks 0-9 --episodes 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

# Before robosuite/mujoco are imported anywhere: the headless context has to be chosen
# before the first GL call, and the default is to open a window that does not exist here.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# The tokenizer is the gated PaliGemma one; offline, it resolves through the local cache.
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# The policy holds ~8.7 GiB and every extra robosuite environment takes another EGL
# context, so the large-block allocator runs out of contiguous space before it runs out of
# memory.  Expandable segments let the allocator grow an existing block instead of
# demanding a new one.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]

#: Deltas below this are "do not move".  LIBERO's arm action is a normalised Cartesian
#: delta in [-1, 1] whose scale is the controller's, not a physical unit, so the threshold
#: is chosen to be small relative to the *observed* distribution and the same fraction is
#: also reported at two other thresholds -- a dwell number that depends on the cutoff is
#: not a finding.
DWELL_THRESHOLDS = (0.01, 0.05, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default="0-9",
                        help="task ids, e.g. '0-9' or '0,3,7'")
    parser.add_argument("--episodes", type=int, default=10,
                        help="rollouts per task; one environment each, stepped in parallel")
    parser.add_argument("--ckpt", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--n-action-steps", type=int, default=10,
                        help="the official command uses 10: the queue replays this many rows "
                             "of one predicted chunk before re-planning")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="override the policy's chunk_size at inference (0 = keep the "
                             "trained 50).  The same override the A3 probe uses, applied "
                             "here on a task this checkpoint is known to solve: it decides "
                             "whether 'chunk=1 collapses the policy' is a property of "
                             "pi0.5 or a property of the A3 task.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("/root/autodl-tmp/logs/libero_reference.json"))
    return parser.parse_args()


def parse_tasks(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-")
            ids.extend(range(int(low), int(high) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def summarise(actions: np.ndarray, replans: np.ndarray) -> dict:
    """Dwell and magnitude statistics for one task's rollouts.

    ``actions`` is every action the robot was executed on; ``replans`` is the subset at a
    queue boundary, i.e. the model's own predictions rather than the replay of them.  Both
    are reported because the queue makes them different numbers: with ``n_action_steps=10``
    nine of every ten executed commands are a copy of the row before, so a dwell fraction
    over executed steps measures the *schedule* as much as the model.
    """

    def block(name: str, values: np.ndarray) -> dict:
        if len(values) == 0:
            return {"n": 0}
        arm = values[:, :6]
        magnitude = np.abs(arm).max(axis=1)
        out = {
            "n": int(len(values)),
            "magnitude_mean": float(magnitude.mean()),
            "magnitude_median": float(np.median(magnitude)),
            "magnitude_p90": float(np.percentile(magnitude, 90)),
            "magnitude_max": float(magnitude.max()),
            # The gripper is commanded on the same [-1, 1] scale but is not a delta: -1 is
            # "hold closed" and +1 "hold open", so it is excluded from the arm statistic
            # and reported separately.
            "gripper_open_fraction": float((values[:, 6] > 0).mean()),
            # Per-dimension medians, because a single `max` over six dimensions can be
            # lifted by one of them.  LIBERO's six are three translations and three
            # rotations, which the OSC controller scales differently, so "the arm is
            # commanded at 81% of full scale" is only interpretable next to which
            # dimension is at 81%.
            "per_dim_median": np.median(np.abs(arm), axis=0).round(4).tolist(),
            "per_dim_p90": np.percentile(np.abs(arm), 90, axis=0).round(4).tolist(),
        }
        for threshold in DWELL_THRESHOLDS:
            # `:g`, because `0.20` formats as `0.2` and a key that quietly differs from
            # the one the pooled report looks up is a statistic that never gets printed.
            out[f"dwell_fraction_lt_{threshold:g}"] = float((magnitude < threshold).mean())
        return out

    return {"executed": block("executed", actions), "replans": block("replans", replans)}


def main() -> int:
    args = parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    task_ids = parse_tasks(args.tasks)
    print(f"suite {args.suite}  tasks {task_ids}  episodes/task {args.episodes}")
    print(f"ckpt  {args.ckpt}  n_action_steps {args.n_action_steps}")

    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    if args.chunk_size:
        if args.n_action_steps > args.chunk_size:
            raise SystemExit(
                f"n_action_steps ({args.n_action_steps}) cannot exceed chunk_size "
                f"({args.chunk_size}); the queue would ask for more rows than exist"
            )
        policy_cfg.chunk_size = args.chunk_size
    policy_cfg.n_action_steps = args.n_action_steps
    policy_cfg.device = args.device
    print(f"chunk_size {policy_cfg.chunk_size}  n_action_steps {policy_cfg.n_action_steps}")

    env_cfg = LiberoEnvConfig(task=args.suite, task_ids=task_ids)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    report: dict = {"suite": args.suite, "episodes_per_task": args.episodes, "per_task": []}

    grand_successes = 0
    grand_episodes = 0
    for task_id in task_ids:
        # One task at a time.  `make_env` builds a vector environment for **every** entry
        # in `task_ids` up front -- ten tasks of ten envs is a hundred robosuite
        # instances, each with its own EGL context, and that exhausted the 5090 before a
        # single step ran (`Tried to allocate 20.00 MiB ... 57.94 MiB is free`).  The
        # official runner builds them all too, and gets away with it only because ten
        # tasks of ten is near the edge of what fits rather than over it.
        task_cfg = LiberoEnvConfig(task=args.suite, task_ids=[task_id])
        envs = make_env(task_cfg, n_envs=args.episodes)
        vec = envs[args.suite][task_id]
        try:
            max_steps = int(vec.call("_max_episode_steps")[0])
            policy.reset()

            observation, _ = vec.reset(seed=list(range(args.episodes)))
            done = np.zeros(args.episodes, dtype=bool)
            #: Whether each environment ever reported success, not whether it was still
            #: successful at the last step.  LIBERO terminates on success and the wrapper
            #: resets the env inside `step`, so the final `is_success` is not readable
            #: afterwards -- `info["final_info"]` is the only place it survives.
            successes = np.zeros(args.episodes, dtype=bool)
            steps_taken = np.zeros(args.episodes, dtype=np.int64)
            actions: list[np.ndarray] = []
            replans: list[np.ndarray] = []

            for step in range(max_steps):
                if bool(done.all()):
                    break
                batch = preprocess_observation(observation)
                batch = add_envs_task(vec, batch)
                batch = env_pre(batch)
                batch = preprocessor(batch)
                with torch.inference_mode():
                    action = policy.select_action(batch)
                action = postprocessor(action)
                action = env_post({ACTION: action})[ACTION]
                action_np = action.to("cpu").numpy()
                if step % args.n_action_steps == 0:
                    replans.append(action_np.copy())
                actions.append(action_np.copy())

                observation, _, terminated, truncated, info = vec.step(action_np)
                # A queue boundary is where `select_action` had an empty queue and really
                # ran the network; the other nine steps replay its rows.
                if "final_info" in info:
                    final = info["final_info"]
                    if isinstance(final, dict) and "is_success" in final:
                        successes |= np.asarray(final["is_success"], dtype=bool)
                newly_done = np.asarray(terminated) | np.asarray(truncated)
                steps_taken[newly_done & ~done] = step + 1
                done = done | newly_done

            steps_taken[done & (steps_taken == 0)] = max_steps
            executed = np.concatenate(actions, axis=0) if actions else np.zeros((0, 7))
            replan_rows = np.concatenate(replans, axis=0) if replans else np.zeros((0, 7))
            stats = summarise(executed, replan_rows)
            grand_successes += int(successes.sum())
            grand_episodes += args.episodes
            report["per_task"].append(
                {
                    "task_id": task_id,
                    "episodes": args.episodes,
                    "successes": int(successes.sum()),
                    "steps": steps_taken.tolist(),
                    "statistics": stats,
                }
            )
            print(
                f"  task {task_id:>2}: {int(successes.sum()):>2}/{args.episodes} success  "
                f"steps median {int(np.median(steps_taken)):>3}   "
                f"dwell(replans) {stats['replans'].get('dwell_fraction_lt_0.2', float('nan')):.3f}  "
                f"|a| median {stats['replans'].get('magnitude_median', float('nan')):.4f}",
                flush=True,
            )
        finally:
            vec.close()
            # The EGL contexts are the expensive part and they are not released by
            # dropping the reference alone; without this the next task inherits the
            # previous one's GPU footprint and the run dies a few tasks in.
            del envs
            gc.collect()
            torch.cuda.empty_cache()

    report["successes"] = grand_successes
    report["episodes"] = grand_episodes
    report["success_rate"] = grand_successes / grand_episodes if grand_episodes else 0.0
    print()
    print(f"=== {args.suite}: {grand_successes}/{grand_episodes} = {report['success_rate']*100:.1f}% ===")
    print("（官方公布 97.0 / Lerobot 实测 100.0；若这里明显更低，说明本脚本的装配不等价）")

    # Pooled dwell, which is the number the A3 comparison is against.
    print()
    print("池化统计（按 task 平均）：")
    keys = ["magnitude_mean", "magnitude_median", "gripper_open_fraction"]
    keys += [f"dwell_fraction_lt_{threshold:g}" for threshold in DWELL_THRESHOLDS]
    for key in keys:
        values = [t["statistics"]["replans"].get(key) for t in report["per_task"]]
        values = [v for v in values if v is not None]
        if values:
            print(f"  replans {key:<26} {np.mean(values):.4f}")
    # Lists are pooled element-wise; they are the per-dimension breakdown and averaging
    # them into a scalar would throw away the thing they exist to show.
    for key in ("per_dim_median", "per_dim_p90"):
        values = [t["statistics"]["replans"].get(key) for t in report["per_task"]]
        values = [v for v in values if v is not None]
        if values:
            pooled = np.mean(np.asarray(values), axis=0)
            print(f"  replans {key:<26} " + " ".join(f"{v:.3f}" for v in pooled)
                  + "   (x y z | rx ry rz)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
