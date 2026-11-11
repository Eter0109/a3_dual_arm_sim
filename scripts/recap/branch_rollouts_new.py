"""Counterfactual branch rollouts: several actions from the same simulated state.

Why this data cannot be obtained any other way
----------------------------------------------
Every reward definition tried so far scores an *outcome*, and an outcome depends on the state far
more than on the action.  Measured on the 400-episode set, a critic trained on those rewards puts
**99.99 % of its variance on the state and 0.0128 % on the action** -- and shortening or
lengthening the reward horizon does not fix it (0.055 % at 20 steps, 1.08 % at 400 steps, 0.745 %
over the whole episode).  The reason is identifiability, not capacity: each recorded state is
paired with exactly one action, so nothing in the data can separate "this state is good" from
"this action is good".

A stronger measurement says the same thing more sharply.  The trained critic's gradient with
respect to the action is **identical to five decimal places at five different states** whose Q
spans 1.18 to 5.19 (`gnorm` 0.09940 throughout), which means it has learned
`Q(s, a) = f(s) + c . a`: a global action preference that does not depend on the state.  Steering
a sampler with that pushes every state's noise in the same fixed direction, which is not steering.

Branching in the simulator produces exactly the missing contrast: one state, several actions, and
the real consequence of each.  It is the only offline-adjacent way to get it, because LeRobot's
recorded episodes are single trajectories by construction.

Five problems in the first version, four of them bugs
-----------------------------------------------------
That version was written once before and killed three times for being too slow.  It would not have
produced usable data even if it had been allowed to finish:

1. **`branch_steps = {row * 5}`** multiplied a *control-step* index by the feature `stride`.
   `value_labels.parquet`'s `potential` has 1500 entries per episode (one per control step -- see
   the `potential` indexing bug in the training history), so the index is already a control step.
   Multiplying by 5 pushed every branch point past the end of the episode, `pending.discard(frame)`
   never matched, and **no branch ever fired**.  The run was a 1500-step replay producing nothing.
2. **`seed = 1000 + episode_index`** is not the seed rule.  The collected set is 1000, 1008, 1016,
   ... (stride 8) with breaks between tracks, so the replay started from a *different scene* every
   time.  The seed now comes from `value_labels.parquet`'s own `seed` column.
3. **The snapshot was three arrays** (`qpos`, `qvel`, `qacc_warmstart`).  That is not the simulator
   state: it misses `ctrl`, `act`, `time`, `qfrc_applied`, `xfrc_applied`, `eq_active`, `mocap`,
   `userdata` and the plugin state.  `mujoco.mj_getState(..., mjSTATE_INTEGRATION)` is the
   supported full snapshot and is what is used now.
4. **Three Python fields were not restored either**, and two of them change the dynamics.
   `env._last_applied_action` is the reference for the rate limit inside `_limit_joint_action`, so
   without it the next branch's commands are clipped against the *previous branch's* last command;
   `_step_count` and `_success_hold_count` drift across branches and leak `truncated`/`success`.
5. **The perturbed action was held for the whole horizon.**  A joint-position command repeated for
   100 steps drives the arm off the demonstrated manifold, so the branch measured "what happens if
   the robot does something absurd" rather than "what if the chunk is slightly different".  The
   policy executes 16 of its 50-step chunk, so the default now perturbs **the first
   `--perturb-steps` (16) steps** and then follows the recorded actions.

What a branch is
----------------
1. Reset the scene with the episode's own seed and replay its recorded actions.
2. At each branch point `t`, snapshot the full simulator state *and* the Python fields above.
3. For each perturbation: restore, apply `recorded_action[t] + sigma * span * noise` for
   `--perturb-steps`, then the recorded actions for the rest of `--horizon`.
4. Record `cookies_in_target` at the start, at every step, and at the end.
5. Restore again and let the outer replay continue to the next branch point.

Two gates, and the script refuses to report a verdict if either fails
--------------------------------------------------------------------
* **Gate A -- replay fidelity.**  At every branch point, the replay's `observation.state` must equal
  the recorded `observation.state` at the same frame.  If it does not, the reset or the scene does
  not match the collection, and every branch would be comparing different scenes.
* **Gate B -- restore fidelity.**  The `sigma = 0` control restores the snapshot and replays the
  recorded actions unchanged, so it must reproduce the replay's own continuation exactly.  Its final
  `qpos` is compared against the outer replay's `qpos` at the same step, after the replay gets
  there.  This is the gate that catches an incomplete snapshot: problems 3 and 4 above show up
  here as a nonzero error and nowhere else.

Both are reported as distributions rather than pass/fail, because "1e-9" and "1e-3" mean different
things and the second is a warning even under a threshold.

What this still does NOT give
-----------------------------
A policy's own closed-loop continuation.  After `--perturb-steps` the branch follows the *recorded*
actions, which is the recorded policy's future, not the future of a policy that would have
re-planned from the perturbed state.  It answers "what does a different action chunk do from here",
which is what DSRL's noise-space actor needs, not "what would the policy do after this action".

Usage
-----
    # smoke test
    python scripts/recap/branch_rollouts.py --episodes 2 --per-episode 2 --horizon 20 --actions 3

    # the real run (~5 h)
    python scripts/recap/branch_rollouts.py --episodes 20 --per-episode 12 --horizon 100
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

#: Gripper apertures, not arm joints.  Perturbing a gripper by a Gaussian is not a meaningful
#: "different decision" -- it is a degree of opening whose whole valid range is already in the data.
GRIPPER_DIMS = (7, 15)


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """Everything needed to put the environment back exactly where it was."""

    mj_state: np.ndarray
    last_applied_action: np.ndarray
    step_count: int
    success_hold_count: int
    safety_stop: bool
    safety_reason: str | None


def take_snapshot(env, mujoco) -> Snapshot:
    """Full MuJoCo state plus the Python fields that feed back into the dynamics."""
    size = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return Snapshot(
        mj_state=state,
        last_applied_action=np.array(env._last_applied_action, copy=True),  # noqa: SLF001
        step_count=int(env._step_count),  # noqa: SLF001
        success_hold_count=int(env._success_hold_count),  # noqa: SLF001
        safety_stop=bool(env._safety_stop),  # noqa: SLF001
        safety_reason=getattr(env, "_safety_reason", None),  # noqa: SLF001
    )


def restore_snapshot(env, snapshot: Snapshot, mujoco) -> None:
    mujoco.mj_setState(env.model, env.data, snapshot.mj_state,
                       mujoco.mjtState.mjSTATE_INTEGRATION)
    env._last_applied_action = np.array(snapshot.last_applied_action, copy=True)  # noqa: SLF001
    env._step_count = snapshot.step_count  # noqa: SLF001
    env._success_hold_count = snapshot.success_hold_count  # noqa: SLF001
    env._safety_stop = snapshot.safety_stop  # noqa: SLF001
    env._safety_reason = snapshot.safety_reason  # noqa: SLF001
    # `xpos`, `site_xpos`, contact and constraint state are stale after `mj_setState`;
    # `mj_forward` rebuilds the derived quantities without integrating and without touching
    # `qacc_warmstart`, which is part of the restored state.
    mujoco.mj_forward(env.model, env.data)


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - one linear script
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path,
                        default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument("--episodes", type=int, default=20,
                        help="episodes to replay; one replay is ~115 s at 76.7 ms/step")
    parser.add_argument("--per-episode", type=int, default=6, help="branch points per episode")
    parser.add_argument("--actions", type=int, default=8,
                        help="perturbations per state for each sigma > 0")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.10],
                        help="0.0 is always run: it is the restore-fidelity control")
    parser.add_argument("--horizon", type=int, default=300,
                        help="control steps per branch.  Has to be long enough for the count to "
                             "actually move: at the default `--max-branch-step 150` no Cookie is "
                             "in the box yet, so a short horizon measures a constant and reports "
                             "it as 'the action does not matter'")
    parser.add_argument("--perturb-steps", type=int, default=16,
                        help="leading steps that get the perturbed action; `execute` is 16")
    parser.add_argument("--lookahead", type=int, default=0,
                        help="keep a branch point only if the potential moves within this many "
                             "steps.  Defaults to `--horizon`, and that default matters: a branch "
                             "whose horizon is shorter than the time-to-next-event cannot see a "
                             "change no matter what the action is, which would understate the "
                             "action's effect and look like 'no signal'.")
    parser.add_argument("--max-branch-step", type=int, default=150,
                        help="skip branch points later than this control step.  **Measured**: "
                             "replaying the recorded actions reproduces the recording's Cookie "
                             "count exactly on 6/6 episodes at steps 0-100, on 5/6 at step 150, "
                             "and on 3/6 at step 200, after which contact chaos has made it a "
                             "different episode.  The default therefore sits inside the faithful "
                             "window; raising it needs a fresh measurement, not an opinion")
    parser.add_argument("--shard", type=int, default=0, help="this process's shard index")
    parser.add_argument("--shards", type=int, default=1,
                        help="split episodes across processes; each replay is single-core-bound "
                             "so N processes give close to N times the throughput")
    parser.add_argument("--progress-every", type=int, default=200,
                        help="print replay progress every this many control steps")
    parser.add_argument("--lead", type=int, default=15,
                        help="place the branch point this many steps before a Cookie-count "
                             "increase.  Measured: consecutive placements are 3-25 steps apart "
                             "(median 5), so a lead much larger than this branches in a stretch "
                             "with no event to change")
    parser.add_argument("--select-on", choices=("counts", "potential"), default="counts",
                        help="which trajectory decides where to branch.  Default `counts` because "
                             "that is what the script measures: selecting on `potential` picks "
                             "points where a *slot* moves, which happens without any change in "
                             "the Cookie count and produced a smoke run with zero variance in "
                             "every branch -- a probe with no opportunity, indistinguishable in "
                             "the output from a probe that measured no effect")
    parser.add_argument("--timing", choices=("event", "uniform"), default="event",
                        help="`uniform` is the control: no regard for whether the potential moves")
    parser.add_argument("--min-gap", type=int, default=40,
                        help="minimum spacing between branch points within one episode")
    parser.add_argument("--gate-tol", type=float, default=1e-3,
                        help="max |qpos| error a restore may show before the run is refused")
    parser.add_argument("--allow-gate-failure", action="store_true",
                        help="write the data even if a gate fails (debugging the gates only)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1500)
    args = parser.parse_args(argv)

    sys.path.insert(0, "/tmp/a3-main/src")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import mujoco
    import pyarrow.parquet as pq

    from bench_pi05_on_main import make_cookie_benchmark

    started = time.time()

    # --- 1. per-episode metadata, from the file that actually knows it -----------------
    # `value_labels.parquet` carries `seed` (problem 2) and `potential` per control step
    # (problem 1).  Reading either from anywhere else is what caused those two bugs.
    labels = pq.read_table(args.dataset / "value_labels.parquet").to_pandas()
    potentials = [np.asarray(p, dtype=np.float64) for p in labels["potential"]]
    counts_path = [np.asarray(c, dtype=np.int64) for c in labels["counts"]]
    seeds = labels["seed"].to_numpy(dtype=np.int64)
    episode_indices = labels["episode_index"].to_numpy(dtype=np.int64)
    print(f"[branch] {len(labels)} labelled episodes; potential length {len(potentials[0])} "
          f"(control steps, not cached frames)", flush=True)

    # A `sigma = 0` control is not optional: it produces the reference every other branch is
    # compared against, the Gate B fidelity check, and the opportunity rate.  Silently running
    # without one would report "0 % changed" and look like a result.
    if not any(sigma == 0.0 for sigma in args.sigmas):
        raise SystemExit("--sigmas must include 0.0 (the control branch)")
    args.sigmas = [0.0] + [s for s in args.sigmas if s > 0]

    rng = np.random.default_rng(args.seed)
    lookahead = args.lookahead if args.lookahead > 0 else args.horizon

    # --- 2. branch points -------------------------------------------------------------
    candidates: list[tuple[int, int]] = []          # (row in `labels`, control step)
    select_paths = counts_path if args.select_on == "counts" else potentials
    print(f"[branch] selecting branch points {args.lead} steps before a Cookie-count increase "
          f"(`{args.select_on}`, horizon {args.horizon})", flush=True)
    for row in range(len(labels)):
        path = select_paths[row]
        usable = min(len(path) - lookahead - args.horizon - 1, args.max_branch_step)
        if usable <= args.min_gap:
            continue
        if args.timing == "uniform":
            picks = [int(x) for x in rng.integers(args.min_gap, usable, size=args.per_episode)]
        else:
            # Pick the step *just before* an event, not an arbitrary step that happens to have
            # an event somewhere later.  Measured on 400 episodes: the first Cookie is never
            # placed before step 179 (median 226), and consecutive placements are 3-25 steps
            # apart, so a branch point must sit a short distance ahead of an event for any action
            # to have a chance of changing it.  `--lead` is that distance.
            events = np.flatnonzero(np.diff(path) > 1e-9) + 1
            limit = len(path) - args.horizon - 1
            candidates_here = [
                int(t) - args.lead for t in events
                if args.min_gap <= int(t) - args.lead <= min(limit, args.max_branch_step)
            ]
            if not candidates_here:
                continue
            if len(candidates_here) > args.per_episode:
                index = rng.choice(len(candidates_here), size=args.per_episode, replace=False)
                candidates_here = [candidates_here[int(k)] for k in index]
            picks = candidates_here
        # Enforce a minimum spacing so two branch points do not share nearly the same state.
        used: list[int] = []
        for t in sorted(picks):
            if all(abs(t - u) >= args.min_gap for u in used):
                used.append(t)
                candidates.append((row, t))

    if not candidates:
        raise SystemExit(
            "no branch points -- with --timing event this means no episode's potential moves "
            "within --lookahead steps; try a larger --lookahead"
        )

    all_rows = sorted({row for row, _ in candidates})
    if args.shards > 1:
        # Shard over the *episode list*, not the branch points: the replay is the fixed cost and
        # splitting it is the only split that scales.  `run_shards.sh` launches one process per
        # shard with the same arguments and a different `--shard`.
        all_rows = all_rows[args.shard :: args.shards]
    if not all_rows:
        raise SystemExit(f"shard {args.shard}/{args.shards} has no episodes")
    keep_rows = sorted(
        rng.choice(all_rows, size=min(args.episodes, len(all_rows)), replace=False).tolist()
    )
    by_row: dict[int, list[int]] = {}
    for row, t in candidates:
        if row in keep_rows:
            by_row.setdefault(row, []).append(t)
    branches_per_point = 1 + args.actions * sum(1 for s in args.sigmas if s > 0)
    points_total = sum(len(v) for v in by_row.values())
    est_steps = points_total * branches_per_point * args.horizon
    # 0.20 s/step, not the 0.077 s measured over the first 200 steps of an episode: the branch
    # horizons run to 500+, where the arm is in contact and the solver costs three times as much.
    # An earlier estimate used the low figure and understated the run by 3x.
    print(f"[branch] {points_total} points over {len(by_row)} episodes "
          f"({points_total / max(len(by_row), 1):.1f} points/episode; `--max-branch-step` and "
          f"`--min-gap` cap this, so raising `--per-episode` alone does nothing)", flush=True)
    print(f"[branch] {branches_per_point} branches/point x {args.horizon} steps = "
          f"{est_steps:,} branch steps, ~{est_steps * 0.20 / 3600:.1f} h at a conservative "
          f"0.20 s/step (contact-heavy horizons run 0.15-0.32 s)", flush=True)

    # --- 3. recorded actions and states, kept episodes only ----------------------------
    # Filter in Arrow before materialising anything.  `to_pandas()` on all 552,789 rows takes
    # 0.3 s and is harmless, but the per-episode pandas boolean mask over that frame is the part
    # that does not scale when several episodes are kept.
    import pyarrow as pa
    import pyarrow.compute as pc

    table = pq.read_table(
        args.dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["action", "observation.state", "episode_index"],
    )
    actions_by_row: dict[int, np.ndarray] = {}
    states_by_row: dict[int, np.ndarray] = {}
    for row in keep_rows:
        episode_index = int(episode_indices[row])
        # The file stores frames in order within each episode, so no sort is needed; the length
        # check below is what would catch a reordering, because the action and state columns are
        # read in the same pass and must stay aligned.
        subset = table.filter(pc.equal(table["episode_index"], pa.scalar(episode_index)))
        if subset.num_rows == 0:
            print(f"[branch] skip row {row}: no frames for episode_index {episode_index}", flush=True)
            continue
        actions_by_row[row] = np.stack(subset["action"].to_pylist()).astype(np.float64)[:, :16]
        states_by_row[row] = np.stack(
            subset["observation.state"].to_pylist()
        ).astype(np.float64)[:, :16]
    del table
    by_row = {row: steps for row, steps in by_row.items() if row in actions_by_row}
    if not by_row:
        raise SystemExit("no episode resolved to frames; check --dataset")

    # Perturbation scale per joint.  Computed from *every* episode, not only the kept ones: in any
    # single episode most joints are idle, so a per-episode percentile range collapses to the
    # floor and the perturbation becomes a no-op.  Percentile range rather than min/max so one
    # outlier frame does not set the scale.
    all_actions = np.stack(
        pq.read_table(args.dataset / "data" / "chunk-000" / "file-000.parquet",
                      columns=["action"])["action"].to_pylist()
    ).astype(np.float64)[:, :16]
    spans = np.maximum(np.percentile(all_actions, 99, axis=0)
                       - np.percentile(all_actions, 1, axis=0), 1e-3)
    spans[list(GRIPPER_DIMS)] = 0.0        # never perturb an aperture
    del all_actions
    perturbable = int((spans > 1e-3).sum())
    print(f"[branch] perturbation scale from all episodes: median "
          f"{np.median(spans[spans > 0]):.3f} rad over {perturbable}/14 arm joints "
          f"(grippers excluded)", flush=True)
    for sigma in args.sigmas:
        if sigma > 0:
            print(f"    sigma {sigma:.2f} -> per-joint sd "
                  f"{sigma * np.median(spans[spans > 0]) * 1000:.1f} mrad", flush=True)

    # --- 4. the environment ------------------------------------------------------------
    bench = make_cookie_benchmark(
        task="2x10", prompt_style="task", max_steps=args.max_episode_steps,
        randomize_boxes=True, randomize_cookies=True, render=False, success_hold_steps=20,
    )
    env = bench.create_env(render_cameras=False)
    # Mirror `run_episode`'s reset options -- including `appearance_seed`, which the collector
    # passes and the first version of this script omitted.  Gate A is what tells us whether this
    # reproduces the collection's scenes.
    reset_options = {
        "randomize_cookies": bench.randomize_cookies,
        "randomize_boxes": bench.randomize_boxes,
        "randomize_source_bin": bench.randomize_boxes,
        "randomize_target_bin": bench.randomize_boxes,
        "source_bin_noise_m": bench.source_bin_noise_m,
        "target_bin_noise_m": bench.target_bin_noise_m,
        "target_bin_yaw_noise_rad": bench.target_bin_yaw_noise_rad,
        "randomization_profile": bench.profile or "basic",
        "randomization_settings": getattr(bench, "randomization_settings", None),
    }

    # --- 5. replay, branching at each point --------------------------------------------
    points: list[dict] = []
    branches: list[dict] = []
    gate_a0: list[float] = []   # reset vs recorded frame 0: tests scene + seed, must be ~0
    gate_a1: list[float] = []   # divergence growth along the replay: expected, reported, watched
    gate_a2: list[int] = []     # |replayed count - recorded count| at each branch point
    gate_b: list[float] = []
    gate_opportunity: list[float] = []   # did the *control* branch see the count move at all?
    skipped = 0

    for done, (row, steps) in enumerate(sorted(by_row.items()), 1):
        episode_index = int(episode_indices[row])
        seed = int(seeds[row])
        actions = actions_by_row[row]
        recorded_states = states_by_row[row]
        wanted = sorted(set(steps))
        last_needed = min(max(wanted) + args.horizon, len(actions) - 1, args.max_episode_steps)

        try:
            observation, info = env.reset(
                seed=seed, options={**reset_options, "appearance_seed": seed}
            )
        except Exception as error:  # noqa: BLE001 -- an unreachable layout is a skip, not a crash
            print(f"[branch] row {row} episode {episode_index} reset failed: {error}", flush=True)
            skipped += len(wanted)
            continue

        # Gate A0: the very first frame must match bit-for-bit.  If it does not, the scene or the
        # seed does not reproduce the collection and nothing downstream is a counterfactual.
        gate_a0.append(float(np.max(np.abs(
            np.asarray(observation["observation.state"], dtype=np.float64) - recorded_states[0]
        ))))

        point_id_of: dict[int, int] = {}
        for t in wanted:
            point_id_of[t] = len(points)
            points.append({
                "row": row, "episode_index": episode_index, "seed": seed,
                "control_step": t, "potential_start": float(potentials[row][t]),
                "state": None, "valid": False,
            })

        replay_qpos: dict[int, np.ndarray] = {}
        episode_started = time.time()
        branch_steps = 0
        # Deferred Gate B: the control branch's final state can only be compared once the outer
        # replay has reached that step, so the comparison is queued and resolved after the loop.
        pending_gate_b: list[tuple[int, np.ndarray]] = []

        for step in range(last_needed + 1):
            point_id = point_id_of.get(step)
            if point_id is not None:
                state_now = np.asarray(observation["observation.state"], dtype=np.float64)
                points[point_id]["state"] = state_now.astype(np.float32)
                points[point_id]["valid"] = True
                # Gate A1: joint divergence from the recording.  This is NOT expected to be zero
                # over a long replay -- the task is contact-rich and chaotic, and measured drift is
                # 1e-2 rad by step 200 -- so it is reported and watched for growth rather than
                # required to vanish.
                if step < len(recorded_states):
                    gate_a1.append(float(np.max(np.abs(state_now - recorded_states[step]))))

                snapshot = take_snapshot(env, mujoco)
                # Gate A2: does the replay still agree with the recording on *what has happened*?
                # Joint drift of 1e-2 rad is meaningless if the arm is in the same behavioural
                # state; it is meaningful if the Cookie count has diverged.  This reads the
                # replay's own count via the last `info` rather than the recording's.
                replay_count = int(info.get("cookies_in_target", 0))
                recorded_count = int(counts_path[row][step])
                # Reported, not gated.  A branch is a valid counterfactual as long as all its
                # branches leave from the *same* state, which Gate B verifies exactly; whether
                # that state is the one the recording reached is a different question and not a
                # requirement.  See "Why divergence from the recording is not a failure" above.
                gate_a2.append(abs(replay_count - recorded_count))
                count_start = replay_count    # the replay's own count, not the recording's

                for sigma in args.sigmas:
                    repeats = args.actions if sigma > 0 else 1
                    for repeat in range(repeats):
                        restore_snapshot(env, snapshot, mujoco)
                        # One noise vector per branch, applied to every step of the chunk.  That
                        # is deliberate: DSRL's actor chooses a single flow-matching latent, and
                        # that latent perturbs the whole chunk coherently.  Per-step independent
                        # noise would instead test "jitter the trajectory", which is not the
                        # action space the actor has.
                        noise = (rng.normal(0.0, 1.0, 16) * spans * sigma
                                 if sigma > 0 else np.zeros(16))
                        counts: list[int] = []
                        for offset in range(args.horizon):
                            index = step + offset
                            if index >= len(actions):
                                break
                            action = (actions[index] + noise
                                      if (sigma > 0 and offset < args.perturb_steps)
                                      else actions[index])
                            _, _, terminated, truncated, info = env.step(action)
                            counts.append(int(info.get("cookies_in_target", 0)))
                            branch_steps += 1
                            if terminated or truncated:
                                break

                        branches.append({
                            "point": point_id,
                            "sigma": float(sigma),
                            "repeat": int(repeat),
                            "count_start": int(count_start),
                            "count_end": int(counts[-1]) if counts else count_start,
                            "count_max": int(max(counts)) if counts else count_start,
                            "steps": len(counts),
                            "noise": noise.astype(np.float32),
                        })
                        if sigma == 0.0:
                            control_moved = int(max(counts) if counts else count_start) != int(count_start)
                            gate_opportunity.append(float(control_moved))
                        if sigma == 0.0:
                            pending_gate_b.append(
                                (step + len(counts),
                                 np.array(env.data.qpos, copy=True))
                            )

                restore_snapshot(env, snapshot, mujoco)

            observation, _, terminated, truncated, info = env.step(actions[step])
            replay_qpos[step + 1] = np.array(env.data.qpos, copy=True)
            # Replay is the fixed cost of an episode and it is not uniform: measured 84 ms/step
            # over the first 200 and 324 ms/step by step 1000, because a failed rollout grinds
            # the arm against the box wall.  Without this line a 10-minute replay is silent.
            if args.progress_every and step and step % args.progress_every == 0:
                elapsed = time.time() - episode_started
                # Report replay and branch cost separately.  Dividing the *total* elapsed by the
                # replay step count produced "11,582 ms/step" in an earlier run, which read as a
                # 40x slowdown and was actually the branch rollouts in between: one branch point
                # is 7 x 350 = 2,450 simulated steps, so a single point costs an order of
                # magnitude more than the 100 steps of replay that precede it.
                print(f"    replay {step:5d}/{last_needed}  wall {elapsed:6.0f}s"
                      f"  branches {len(branches)} ({branch_steps} steps,"
                      f" {1000 * branch_steps / max(elapsed, 1e-9):.1f} steps/s equivalent)",
                      flush=True)
            if terminated or truncated:
                break

        for target_step, control_qpos in pending_gate_b:
            reference = replay_qpos.get(target_step)
            if reference is not None:
                gate_b.append(float(np.max(np.abs(control_qpos - reference))))

        print(f"[branch] {done}/{len(by_row)} episodes, {len(branches)} branches,"
              f" {skipped} skipped points, {time.time() - started:.0f}s", flush=True)

    env.close()

    # --- 6. gates first, verdict second ------------------------------------------------
    a0 = np.array(gate_a0, dtype=np.float64)
    a1 = np.array(gate_a1, dtype=np.float64)
    a2 = np.array(gate_a2, dtype=np.int64)
    b_arr = np.array(gate_b, dtype=np.float64)

    def describe(values: np.ndarray, fmt: str = ".3e") -> str:
        if not values.size:
            return "no samples"
        return (f"n={values.size}  median {np.median(values):{fmt}}  "
                f"p95 {np.percentile(values, 95):{fmt}}  max {values.max():{fmt}}")

    print()
    print("=" * 78)
    print("GATE A0 -- reset vs recorded frame 0 (scene + seed reproduce the collection?)")
    print("  " + describe(a0))
    print("GATE A1 -- joint divergence along the replay (expected; contact is chaotic)")
    print("  " + describe(a1))
    print("GATE A2 -- |replayed Cookies - recorded Cookies| at each branch point")
    print("  " + describe(a2, ".2f") + f"  exact {100 * np.mean(a2 == 0):.1f} %")
    print("GATE B  -- restore fidelity: sigma=0 control vs the replay's continuation")
    print("  " + describe(b_arr))

    gate_failed = False
    # A0 is a hard gate: it tests plumbing, and plumbing must be exact.
    if a0.size and a0.max() > 1e-6:
        print("  FAIL A0: the reset does not reproduce the recorded first frame. The scene, the "
              "seed or the reset options differ from the collection.")
        gate_failed = True
    # A1 is reported, not gated: 1e-2 rad drift over 200 contact-rich steps is normal.  It is
    # printed because a value in the units rather than the 1e-2 range would mean something else
    # is wrong.
    if a1.size and a1.max() > 0.5:
        print("  WARN A1: joint divergence above 0.5 rad -- the replay has left the recorded "
              "trajectory entirely; treat the branch points as a different scene.")
        gate_failed = True
    # A2 is reported, not gated.  Divergence from the recording is expected and harmless: the
    # counterfactual needs one state and several actions from it, not a state that the recording
    # also reached.  This number is kept because it says how *late* the branch points are, and
    # late points are the interesting ones -- the recorded rollouts diverge precisely where the
    # arm starts grinding the pile.
    if b_arr.size and np.percentile(b_arr, 95) > args.gate_tol:
        print(f"  FAIL B: p95 above {args.gate_tol:.0e}. The snapshot is incomplete, so the "
              "branches are not true counterfactuals.")
        gate_failed = True
    if not gate_failed:
        print("  OK -- every gate passes")

    # --- 7. primary result: does the action change the outcome, causally? --------------
    by_point: dict[int, list[dict]] = {}
    for record in branches:
        by_point.setdefault(record["point"], []).append(record)
    usable = {
        point: records for point, records in by_point.items()
        if sum(1 for r in records if r["sigma"] > 0) >= 2
    }

    within_variances, across_means, changed = [], [], []
    for point, records in usable.items():
        ends = [r["count_end"] for r in records if r["sigma"] > 0]
        control = next((r["count_end"] for r in records if r["sigma"] == 0.0), None)
        within_variances.append(float(np.var(ends)))
        across_means.append(float(np.mean(ends)))
        if control is not None:
            changed.append(float(np.mean([e != control for e in ends])))

    within = float(np.mean(within_variances)) if within_variances else float("nan")
    across = float(np.var(across_means)) if across_means else float("nan")
    ratio = within / across if across and across > 0 else float("nan")
    changed_fraction = float(np.mean(changed)) if changed else float("nan")

    print()
    print("=" * 78)
    print("Primary result -- same state, different actions: did the outcome move?")
    print(f"  states probed:                        {len(usable)}")
    print(f"  within-state variance of count_end:   {within:.4f}")
    print(f"  across-state variance of count_end:   {across:.4f}")
    print(f"  ratio (action / state):               {ratio:.3f}")
    print(f"  branches whose outcome differs from control: {100 * changed_fraction:.1f} %")

    # Dose response.  A real causal effect must grow with the perturbation size; an effect that is
    # flat across sigma is chaos (or a broken control), not action sensitivity.
    print()
    print("Dose response -- a bigger perturbation should change the outcome more often")
    by_sigma: dict[float, list[float]] = {}
    for point, records in by_point.items():
        control = next((r["count_end"] for r in records if r["sigma"] == 0.0), None)
        if control is None:
            continue
        for record in records:
            if record["sigma"] > 0:
                by_sigma.setdefault(record["sigma"], []).append(
                    float(record["count_end"] != control)
                )
    for sigma in sorted(by_sigma):
        values = by_sigma[sigma]
        print(f"  sigma {sigma:4.2f}: {100 * np.mean(values):5.1f} % changed  (n={len(values)})")

    # The opportunity rate decides whether a zero is a result.  If the control branch never sees
    # the count move inside the horizon, then no action could have changed it either, and a zero
    # variance says nothing about the action -- only that the horizon was too short.  This is the
    # exact failure the smoke run exhibited, and it is reported before any verdict.
    opportunity = float(np.mean(gate_opportunity)) if gate_opportunity else float("nan")
    print()
    print(f"Opportunity -- the control branch saw the count move inside {args.horizon} steps on "
          f"{100 * opportunity:.1f} % of points")
    if opportunity < 0.5:
        print("  A horizon that short cannot resolve the question: raise --horizon and "
              "--lookahead together (they default to the same value) before reading the ratio.")

    print()
    if gate_failed:
        print("-> A gate failed. The numbers above cannot be used to judge DSRL.")
    elif not usable:
        print(f"-> INCONCLUSIVE: no state had {args.actions} perturbation branches, so there is "
              "no within-state comparison to make. Raise --actions (needs >= 3).")
    elif not (opportunity >= 0.5):
        print("-> INCONCLUSIVE: too few branch points had the opportunity to change the outcome. "
              "This is not evidence that the action does not matter.")
    elif len(usable) < 10:
        print(f"-> PRELIMINARY ONLY: {len(usable)} states. The direction is readable but the "
              "number is not; this is not enough to close the question either way.")
    elif ratio > 0.30:
        print("-> Activity is comparable to state. The counterfactual data has signal.")
    elif ratio > 0.05:
        print("-> The action matters but weakly. Check whether the dose response is monotone "
              "before lengthening the horizon.")
    else:
        print("-> The action barely moves the outcome: the action value really is near zero for "
              "this task, and the RL direction can be closed on evidence.")

    # --- 8. write ----------------------------------------------------------------------
    output = args.output or (
        args.dataset / "branches"
        / f"{args.timing}_{len(by_row)}ep{args.per_episode}x{args.actions}x{args.horizon}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points=np.array([
            [p["row"], p["episode_index"], p["seed"], p["control_step"],
             p["potential_start"], float(p["valid"])]
            for p in points
        ], dtype=np.float64).reshape(-1, 6),
        point_states=np.array(
            [p["state"] if p["state"] is not None else np.zeros(16, dtype=np.float32)
             for p in points], dtype=np.float32,
        ).reshape(-1, 16),
        branch_point=np.array([b["point"] for b in branches], dtype=np.int64),
        branch_sigma=np.array([b["sigma"] for b in branches], dtype=np.float64),
        branch_repeat=np.array([b["repeat"] for b in branches], dtype=np.int64),
        branch_count_start=np.array([b["count_start"] for b in branches], dtype=np.int64),
        branch_count_end=np.array([b["count_end"] for b in branches], dtype=np.int64),
        branch_count_max=np.array([b["count_max"] for b in branches], dtype=np.int64),
        branch_steps=np.array([b["steps"] for b in branches], dtype=np.int64),
        branch_noise=np.array([b["noise"] for b in branches], dtype=np.float32).reshape(-1, 16),
        gate_a0=a0, gate_a1=a1, gate_a2=a2, gate_b=b_arr,
    )
    (output.with_suffix(".json")).write_text(json.dumps({
        "points_probed": len(usable),
        "points_selected": points_total,
        "branches": len(branches),
        "episodes_used": len(by_row),
        "skipped_points": skipped,
        "actions": args.actions,
        "sigmas": args.sigmas,
        "horizon": args.horizon,
        "perturb_steps": args.perturb_steps,
        "timing": args.timing,
        "gate_a0_max": float(a0.max()) if a0.size else None,
        "gate_a1_median": float(np.median(a1)) if a1.size else None,
        "gate_a1_max": float(a1.max()) if a1.size else None,
        "gate_a2_exact_fraction": float(np.mean(a2 == 0)) if a2.size else None,
        "gate_a2_median": float(np.median(a2)) if a2.size else None,
        "gate_b_median": float(np.median(b_arr)) if b_arr.size else None,
        "gate_b_p95": float(np.percentile(b_arr, 95)) if b_arr.size else None,
        "gate_failed": gate_failed,
        "within_state_variance": within,
        "across_state_variance": across,
        "ratio": ratio,
        "changed_fraction": changed_fraction,
        "opportunity_rate": opportunity,
        "select_on": args.select_on,
        "changed_by_sigma": {f"{k:.3f}": float(np.mean(v)) for k, v in by_sigma.items()},
        "seconds": time.time() - started,
    }, indent=2) + "\n")
    print(f"\n[branch] wrote {output}")

    if gate_failed and not args.allow_gate_failure:
        print("[branch] no verdict reported: a gate failed (use --allow-gate-failure only while "
              "debugging the gates)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
