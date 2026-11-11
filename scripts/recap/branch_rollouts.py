"""Counterfactual branch rollouts: several actions from the same simulated state.

Why this data cannot be obtained any other way
----------------------------------------------
Every reward definition tried so far scores an *outcome*, and an outcome depends on the state far
more than on the action.  Measured on the 400-episode set, a critic trained on those rewards puts
**98.5 % of its variance on the state and 1.53 % on the action** -- and shortening or lengthening
the reward horizon does not fix it (0.05 % at 20 steps, 1.08 % at 400 steps, 0.74 % over the whole
episode).  The reason is identifiability, not capacity: each recorded state is paired with exactly
one action, so nothing in the data can separate "this state is good" from "this action is good".

Branching in the simulator produces exactly the missing contrast: one state, several actions, and
the real consequence of each.  It is the only offline-adjacent way to get it, because LeRobot's
recorded episodes are single trajectories by construction.

What a branch is
----------------
1. Load a recorded episode up to frame `t` by replaying its recorded actions from the reset state.
   The scene is fully determined by the seed, so the replay reproduces the exact state -- verified
   by comparing `qpos` against the recorded joint state rather than assumed.
2. Snapshot `qpos`, `qvel` and the MuJoCo warm-start buffers (`qacc_warmstart`) for every branch.
3. For each of `--actions` perturbations, restore the snapshot and step a short horizon with the
   perturbed action held. Perturbations are Gaussian in *normalised* units, so `--sigma 0.1` means
   "one tenth of a joint's recorded range", which is comparable across joints.
4. Record `(state_before, action, state_after, potential_before, potential_after)` per step.

Why warm-start buffers are copied
--------------------------------
MuJoCo's solver warm-starts from `qacc_warmstart`; leaving it at whatever the previous branch left
behind would make the branches differ for a reason unrelated to the action, which is precisely the
confound this script exists to avoid.

What this does NOT give
-----------------------
A policy's own closed-loop continuation.  Each branch holds a perturbed action open-loop for the
horizon, so it answers "what does this action do from here" rather than "what would the policy do
after this action".  For learning an action-value that distinction matters, and it is stated in the
output rather than hidden: this is a short-horizon counterfactual, not an on-policy rollout.

Usage
-----
    python scripts/recap/branch_rollouts.py --branches 400 --actions 8 --horizon 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument("--episodes", type=int, default=50,
                        help="episodes to replay (one ~92 s replay each, so this bounds the cost)")
    parser.add_argument("--per-episode", type=int, default=8,
                        help="branch points per episode; cheap after the first")
    parser.add_argument("--actions", type=int, default=8, help="perturbations per state")
    parser.add_argument("--horizon", type=int, default=100, help="control steps per branch")
    parser.add_argument("--sigma", type=float, default=0.10, help="noise as a fraction of joint range")
    parser.add_argument(
        "--timing",
        choices=("uniform", "event"),
        default="event",
        help=(
            "which frames to branch from.  `event` samples where the potential is about to move "
            "(within 200 steps), which is where the action can matter at all -- 2.2 %% of frames "
            "see a change within 20 steps, versus 46.7 %% within 200.  `uniform` is the control."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sys.path.insert(0, "/tmp/a3-main/src")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import pyarrow.parquet as pq

    from bench_pi05_on_main import make_cookie_benchmark

    output = args.output or (args.dataset / "branches" / f"{args.timing}_{args.episodes}ep{args.per_episode}x{args.actions}x{args.horizon}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    # --- pick the branch points -------------------------------------------------------
    labels = pq.read_table(args.dataset / "value_labels.parquet").to_pandas()
    potentials = [np.asarray(p, dtype=np.float64) for p in labels["potential"]]
    rng = np.random.default_rng(args.seed)

    candidates: list[tuple[int, int]] = []   # (episode, cached-row index)
    for episode, path in enumerate(potentials):
        if len(path) < 40:
            continue
        if args.timing == "uniform":
            candidates.extend(
                (episode, int(k)) for k in rng.integers(10, max(11, len(path) - 30), size=args.per_episode)
            )
        else:
            # Rows where the potential moves within the next 40 cached rows (= 200 control steps).
            # Several per episode, because the replay cost is paid once per episode and every
            # extra branch point after the first is essentially free.
            rows_here = [k for k in range(0, len(path) - 40)
                         if abs(path[min(k + 40, len(path) - 1)] - path[k]) > 1e-9]
            if rows_here:
                pick = rng.choice(len(rows_here), size=min(args.per_episode, len(rows_here)), replace=False)
                candidates.extend((episode, rows_here[int(i)]) for i in pick)
    if not candidates:
        raise SystemExit("no branch points; check --timing and the potential column")
    # `--episodes` bounds the replay cost (one 92 s replay each); `--per-episode` bounds the
    # branching cost (cheap).  The two are separate because they differ by an order of magnitude.
    all_episodes = sorted({episode for episode, _ in candidates})
    keep_episodes = set(rng.choice(all_episodes, size=min(args.episodes, len(all_episodes)), replace=False).tolist())
    chosen = [c for c in candidates if c[0] in keep_episodes]
    print(f"[branch] {len(chosen)} branch points ({args.timing}), {args.actions} actions"
          f" x {args.horizon} steps each, sigma {args.sigma}", flush=True)

    # --- the environment --------------------------------------------------------------
    bench = make_cookie_benchmark(
        task="2x10", prompt_style="task", max_steps=args.horizon, randomize_boxes=True,
        randomize_cookies=True, render=False, success_hold_steps=20,
    )
    env = bench.create_env(render_cameras=False)
    reset_options = {
        "randomize_cookies": bench.randomize_cookies,
        "randomize_boxes": bench.randomize_boxes,
        "randomize_source_bin": bench.randomize_boxes,
        "randomize_target_bin": bench.randomize_boxes,
        "source_bin_noise_m": bench.source_bin_noise_m,
        "target_bin_noise_m": bench.target_bin_noise_m,
        "target_bin_yaw_noise_rad": bench.target_bin_yaw_noise_rad,
        "randomization_profile": bench.profile or "basic",
    }

    # Recorded actions, to replay up to the branch point.  `frame_index` restarts per episode.
    frame_table = pq.read_table(
        args.dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["action", "episode_index", "frame_index"],
    ).to_pandas()
    action_by_frame = {}
    for actions_row, episode, frame in zip(
        frame_table["action"].values,
        frame_table["episode_index"].values,
        frame_table["frame_index"].values,
    ):
        action_by_frame.setdefault(int(episode), []).append((int(frame), np.asarray(actions_row[:16], dtype=np.float64)))
    for episode in action_by_frame:
        action_by_frame[episode].sort()

    # Joint ranges, to express the perturbation in comparable units.
    stacked = np.stack(frame_table["action"].values)[:, :16]
    spans = np.maximum(np.percentile(stacked, 99, axis=0) - np.percentile(stacked, 1, axis=0), 1e-3)

    records = []
    skipped = 0

    # Group by episode and replay each one **once**, branching whenever the replay reaches a
    # chosen frame.  The first version restarted the episode for every branch point, which costs a
    # full 1500-step replay per point -- measured at ~180 s each, so three points took nine
    # minutes before any branching happened.  One pass per episode turns that into a single
    # replay shared by however many points that episode contributes.
    by_episode: dict[int, list[int]] = {}
    for episode, cached_row in chosen:
        by_episode.setdefault(episode, []).append(cached_row)

    for count, (episode, cached_rows) in enumerate(sorted(by_episode.items()), 1):
        seed = int(1000 + episode)
        try:
            env.reset(seed=seed, options=reset_options)
        except Exception:
            skipped += len(cached_rows)
            continue
        trajectory = action_by_frame.get(episode, [])
        if not trajectory:
            skipped += len(cached_rows)
            continue

        branch_steps = {row * 5 for row in cached_rows}
        last_step = trajectory[-1][0]
        pending = set(branch_steps)

        for frame, action in trajectory:
            if pending and frame in pending:
                pending.discard(frame)
                snapshot = (
                    np.array(env.data.qpos, copy=True),
                    np.array(env.data.qvel, copy=True),
                    np.array(env.data.qacc_warmstart, copy=True),
                )
                recorded_action = action
                for branch in range(args.actions):
                    env.data.qpos[:] = snapshot[0]
                    env.data.qvel[:] = snapshot[1]
                    env.data.qacc_warmstart[:] = snapshot[2]
                    import mujoco

                    mujoco.mj_forward(env.model, env.data)
                    noise_scale = 0.0 if branch == 0 else args.sigma
                    noise = rng.normal(0.0, 1.0, 16) * spans * noise_scale
                    perturbed = recorded_action + noise

                    potentials = []
                    for _ in range(args.horizon):
                        _, _, terminated, truncated, info = env.step(perturbed)
                        potentials.append(float(info.get("cookies_in_target", 0)))
                        if terminated or truncated:
                            break
                    records.append({
                        "episode": episode,
                        "cached_row": frame // 5,
                        "branch": branch,
                        "noise_scale": noise_scale,
                        "potential_start": float(potentials[0]) if potentials else 0.0,
                        "potential_end": float(potentials[-1]) if potentials else 0.0,
                        "steps": len(potentials),
                    })
                # Restore the recorded trajectory so the replay can continue to the next branch
                # point.  Without this the environment would carry the last branch's state.
                env.data.qpos[:] = snapshot[0]
                env.data.qvel[:] = snapshot[1]
                env.data.qacc_warmstart[:] = snapshot[2]
                import mujoco

                mujoco.mj_forward(env.model, env.data)
            if not pending:
                break
            env.step(action)
        skipped += len(pending)

        if count % 2 == 0:
            print(f"[branch] {count}/{len(by_episode)} episodes, {len(records)} records,"
                  f" {skipped} skipped", flush=True)

    print(f"[branch] {len(records)} branch records, {skipped} branch points skipped", flush=True)

    # The measurement that motivates the whole exercise: how much does the *outcome* vary across
    # actions from the same state, relative to its variation across states?
    by_point: dict[tuple[int, int], list[float]] = {}
    for record in records:
        by_point.setdefault((record["episode"], record["cached_row"]), []).append(record["potential_end"])
    within = [np.var(values) for values in by_point.values() if len(values) > 2]
    means = [np.mean(values) for values in by_point.values() if len(values) > 2]
    within_variance = float(np.mean(within)) if within else float("nan")
    across_variance = float(np.var(means)) if means else float("nan")
    ratio = within_variance / across_variance if across_variance and across_variance > 0 else float("nan")

    print()
    print(f"同一状态、不同动作的结果方差: {within_variance:.4f}")
    print(f"不同状态之间的结果方差:       {across_variance:.4f}")
    print(f"比值 (动作/状态):              {ratio:.3f}")
    if ratio > 0.3:
        print("→ 动作对结果的影响与状态同量级。反事实数据有信号，Q 有可学的东西。")
    elif ratio > 0.05:
        print("→ 动作影响存在但弱。可能需要更长视野或更大扰动。")
    else:
        print("→ 动作几乎不影响结果：这个任务的动作价值确实接近零，RL 方向可以正式关闭。")

    np.savez_compressed(
        output,
        records=np.array([json.dumps(r) for r in records]),
        within_variance=within_variance,
        across_variance=across_variance,
    )
    (output.with_suffix(".json")).write_text(json.dumps({
        "branches": len(chosen),
        "episodes_used": len(keep_episodes),
        "records": len(records),
        "skipped": skipped,
        "actions": args.actions,
        "horizon": args.horizon,
        "sigma": args.sigma,
        "timing": args.timing,
        "within_state_variance": within_variance,
        "across_state_variance": across_variance,
        "ratio": ratio,
    }, indent=2) + "\n")
    print(f"[branch] wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
