"""Autonomous DAgger-style corrective collection (v0.1 — official-harness architecture).

Runs the model through the OFFICIAL bench.run_episode path (identical env
construction/reset/randomization as all dev20/acceptance evaluations — layout
identity by construction, fixing the v0 custom-loop confound).

TakeoverPolicy wraps the SmolVLA adapter; on hover-stall (gripper open AND EEF
near-stationary for a window, AND zero cookies delivered so far) it re-enters
the scripted expert IN-PLACE on the model's own reached state (no env reset —
the corrective states are the model's own error distribution, which is the
source of DAgger's value).

GatedRecorder makes InterventionsOnly true by construction: the real LeRobot
recorder only starts at the takeover frame, so unlabeled model-prefix actions
never enter the dataset.

State-consistency crux: qpos+qvel+mocap triple snapshot around expert
construction+reset; any delta > 1e-9 aborts the takeover (expert init must be
physics-read-only).

v0.2 trigger (budget, data-driven): the v8-generation failure profile is slow
grinding with a near-always-closed gripper (pilot 10/10: open_frac <= 0.024,
zero hover-stall triggers), so v0's open-hover streak can never fire. Budget
trigger: step >= budget_T0 (default 250) with zero cookies delivered — expert
pace delivers the first batch by ~215 steps, so T==0 at budget is decisively
off-pace. If the gripper is closed at takeover, a short normalization phase
(hold position + open gripper, recorded as valid corrective labels) precedes
the expert hand-off — fixing the closed-gripper batch-timeout failure mode
seen in dry-run 4b seed 1015.

v0.2 scope: takeover only at T==0 (clean target-slot state). Partial-transfer
episodes (T>0 at budget) are logged and run out (slot reconciliation = v1).

LAYOUT-IDENTITY WARNING (2026-09-26, measured): a hand-rolled env.reset loop
does NOT reproduce the official harness layout — same seed yielded target_bin_pos
differing by ~4mm (source_bin matched bit-for-bit). The target-box randomization
has a harness-side dependency that was not traced. This once faked a "0→10
cookies variance bomb" that was actually a different task instance. NEVER build
a custom reset loop for data that must be distribution-identical to dev20 /
acceptance evaluations — always go through bench.run_episode like this script.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from a3_dual_arm_sim.controllers.expert import CookiePhase
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.data.recording import LeRobotV3Recorder
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark, SmolVLAPolicyAdapter

TASK = "transfer 10 cookies into target box"
RAW_PURE = dict(
    ema_alpha=0,
    anchor_right_arm=False,
    gripper_sharpening=False,
    align_vertical=False,
    clamp_z=False,
    n_action_steps=8,
    num_steps=25,
)
STALL_DEFAULTS = {
    "trigger": "budget",  # "budget" (v0.2) | "hover" (v0 legacy) | force_at overrides both
    "budget_T0": 250,  # budget mode: takeover when step >= this and delivered == 0
    "normalize_steps": 12,  # gripper-open normalization steps before expert hand-off
    "min_steps": 100,
    "window": 120,
    "gripper_open_min": 0.6,
    "max_eef_speed": 0.005,
}


class GateState:
    def __init__(self):
        self.taken = False
        self.takeover_step = None
        self.skip_reason = None
        self.reinit_deltas = None
        self.aborted = False  # expert construction rejected the disturbed state
        self.last_T = 0  # delivered count, polled per act() after takeover


class GatedRecorder:
    """Forwards to a real LeRobotV3Recorder only after takeover fires.

    Three-tier ingestion (peer refinement 2026-09-26): full AND partial
    recoveries are kept — expert action labels on model-reached states stay
    valid even when a later lift fails; only recovered_none / aborted_reinit
    are discarded. Kept partials carry honest success=False metadata (the v9
    prepare path bypasses the all-success a3 audit for corrective sets).
    """

    def __init__(self, inner, gate):
        self.inner = inner
        self.gate = gate
        self.context = None
        self.started = False
        self.pending_discard = False

    def start_episode(self, context, controller_type):
        # Defensive: an unresolved pending discard (e.g. crashed episode) is truly
        # discarded before the next episode starts.
        if self.pending_discard:
            self.inner.discard_episode()
            self.pending_discard = False
        self.context = context
        self.started = False

    def add_frame(self, observation, action):
        if self.gate.taken:
            if not self.started:
                self.inner.start_episode(self.context, "expert_recovery")
                self.started = True
            self.inner.add_frame(observation, action)

    def finish_episode(self, success=None):
        if self.started:
            if success:
                self.inner.finish_episode(success=True)
            else:
                self.pending_discard = True  # resolve(final_T) decides

    def discard_episode(self):
        # run_episode calls discard (never finish(False)) for non-success episodes.
        # Defer the three-tier decision to resolve(): gate.last_T is a PRE-step
        # count and can miss last-moment settling (seed 5016 batch-2 lost a
        # T=3 partial to this race); the post-step final count is authoritative.
        if self.started:
            self.pending_discard = True

    def resolve(self, final_target_count):
        """Called by main() with the post-step cookies_in_target from EpisodeScore."""
        if self.pending_discard:
            if final_target_count > 0:
                self.inner.finish_episode(success=False)  # partial: keep, honest label
            else:
                self.inner.discard_episode()
            self.pending_discard = False

    def close(self):
        if self.pending_discard:
            self.inner.discard_episode()
            self.pending_discard = False
        self.inner.close()


class TakeoverPolicy:
    """Model prefix with hover-stall detection; in-place expert re-entry."""

    action_mode = "joint_position"
    name = "dagger_recovery"

    def __init__(self, model, gate, stall_cfg, fps):
        self.model = model
        self.gate = gate
        self.cfg = stall_cfg
        self.fps = fps
        self.env = None
        self.expert = None
        self._reset_counters()

    def _reset_counters(self):
        self.step = 0
        self.open_streak = 0
        self.slow_streak = 0
        self.prev_eef = None
        self.expert = None
        self.normalize_remaining = 0
        # behavior profile counters for the stall-type x outcome crosstab
        self.model_steps_run = 0
        self.open_steps = 0
        self.gripper_transitions = 0
        self._prev_gripper_open = None

    def _normalize_action(self, observation):
        """Hold position, command left gripper fully open (rate-limited by env)."""
        a = np.asarray(observation["observation.state"], dtype=float).ravel()[:16].copy()
        a[7] = 1.0
        return a

    def bind_env(self, env):
        self.env = env
        self.model.bind_env(env)

    def reset(self, context=None):
        self._reset_counters()
        self.gate.taken = False
        self.gate.takeover_step = None
        self.gate.skip_reason = None
        self.gate.reinit_deltas = None
        self.gate.aborted = False
        if hasattr(self.model, "plugin") and context is not None:
            self.model.plugin.inference_seed = context.seed
        self.model.reset(context)

    def _delivered(self):
        env = self.env
        return sum(
            1 for i in range(env.task_config.cookie_count) if env.privileged_cookie_in_target(i)
        )

    @property
    def finished(self):
        if self.gate.aborted:
            return True  # expert rejected this state; don't burn the step budget
        return (
            self.gate.taken
            and self.expert is not None
            and self.expert.phase in (CookiePhase.DONE, CookiePhase.FAILED)
        )

    def act(self, observation, task=TASK):
        self.step += 1
        if self.gate.taken:
            self.gate.last_T = self._delivered()  # for three-tier ingestion
            if self.normalize_remaining > 0:
                # Gripper-open normalization before expert hand-off; these frames
                # are themselves valid corrective labels and ARE recorded.
                self.normalize_remaining -= 1
                return self._normalize_action(observation)
            if self.expert.phase in (CookiePhase.DONE, CookiePhase.FAILED):
                # One harmless hold step; run_episode breaks on `finished`.
                return np.asarray(observation["observation.state"], dtype=float).ravel()[:16].copy()
            return self.expert.act(observation, task)

        state = np.asarray(observation["observation.state"], dtype=float).ravel()
        eef = np.asarray(observation["observation.eef_pose"], dtype=float).ravel()[:3]
        speed = (
            0.0 if self.prev_eef is None else float(np.linalg.norm(eef - self.prev_eef) * self.fps)
        )
        self.prev_eef = eef
        gripper_open = state[7] > self.cfg["gripper_open_min"]
        self.open_streak = self.open_streak + 1 if gripper_open else 0
        self.slow_streak = self.slow_streak + 1 if speed < self.cfg["max_eef_speed"] else 0
        self.model_steps_run += 1
        self.open_steps += int(gripper_open)
        if self._prev_gripper_open is not None and gripper_open != self._prev_gripper_open:
            self.gripper_transitions += 1
        self._prev_gripper_open = gripper_open

        mode = self.cfg.get("trigger", "budget")
        force_at = self.cfg.get("force_at")
        if force_at is not None:
            trigger = self.step == int(force_at)
        elif mode == "budget":
            # v0.2: the v8-generation failure profile is slow grinding with a
            # near-always-closed gripper (open_frac <= 0.024, pilot 10/10) — the
            # v0 open-hover streak can never fire. Budget trigger: expert pace
            # delivers the first batch by ~215 steps; zero delivered at budget
            # is decisively off-pace. Gripper state irrelevant (normalization
            # below handles the closed case).
            trigger = self.step >= int(self.cfg["budget_T0"]) and self._delivered() == 0
        else:  # legacy hover criteria
            trigger = (
                self.step >= self.cfg["min_steps"]
                and self.open_streak >= self.cfg["window"]
                and self.slow_streak >= self.cfg["window"]
            )
        if trigger:
            delivered = self._delivered()
            if delivered > 0:
                self.gate.skip_reason = f"stall_with_partial_transfer_T{delivered}"
                self.cfg = dict(self.cfg, window=10**9, budget_T0=10**9)
            else:
                data = self.env.data
                qpos0, qvel0 = data.qpos.copy(), data.qvel.copy()
                mocap0 = (
                    (data.mocap_pos.copy(), data.mocap_quat.copy())
                    if self.env.model.nmocap > 0
                    else None
                )
                try:
                    expert = A3SameColumnBatchExpert(self.env)
                    expert.reset()
                except RuntimeError as exc:
                    # The expert's own workspace validation can reject the
                    # model-disturbed state (e.g. mm-shifted target box, seed
                    # 5020: 2.32mm/2.06deg over tolerance). Skip the seed —
                    # never repair physics state (DAgger purity + crux assert).
                    self.gate.skip_reason = f"expert_reinit_rejected: {exc}"
                    self.gate.aborted = True
                    self.cfg = dict(self.cfg, window=10**9, budget_T0=10**9)
                    return self.model.act(observation, task)
                deltas = (
                    float(np.max(np.abs(data.qpos - qpos0))),
                    float(np.max(np.abs(data.qvel - qvel0))),
                    max(
                        float(np.max(np.abs(data.mocap_pos - mocap0[0]))),
                        float(np.max(np.abs(data.mocap_quat - mocap0[1]))),
                    )
                    if mocap0 is not None
                    else 0.0,
                )
                self.gate.reinit_deltas = deltas
                if max(deltas) <= 1e-9:
                    self.expert = expert
                    self.gate.taken = True
                    self.gate.takeover_step = self.step
                    if not gripper_open:
                        self.normalize_remaining = int(self.cfg.get("normalize_steps", 12))
                        self.normalize_remaining -= 1
                        return self._normalize_action(observation)
                    return self.expert.act(observation, task)
                self.gate.skip_reason = "reinit_moved_physics"
                self.cfg = dict(self.cfg, window=10**9, budget_T0=10**9)
        return self.model.act(observation, task)

    def close(self):
        self.model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--expert-max-steps-note",
        action="store_true",
        help="(informational) run_episode max_steps bounds the whole episode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2200,
        help="episode step budget: model prefix + expert completion",
    )
    parser.add_argument(
        "--stall-config",
        type=str,
        default=None,
        help=f"JSON overrides merged into {STALL_DEFAULTS}",
    )
    parser.add_argument("--dry-run", action="store_true", help="no recording; mechanism check only")
    args = parser.parse_args()
    for seed in args.seeds:
        if 3000 <= seed <= 3019:
            parser.error(f"seed {seed}: acceptance seeds are reserved")
    stall_cfg = dict(STALL_DEFAULTS)
    if args.stall_config:
        stall_cfg.update(json.loads(args.stall_config))

    bench = CookieBatchBenchmark(max_steps=args.max_steps)
    fps = getattr(bench.config, "control_hz", 20)
    args.root.mkdir(parents=True, exist_ok=True)
    inner = None
    if not args.dry_run:
        inner = LeRobotV3Recorder(
            args.root,
            repo_id="local/a3-dagger-recovery-v0",
            fps=fps,
            image_height=bench.config.image_height,
            image_width=bench.config.image_width,
        )
    model = SmolVLAPolicyAdapter(
        args.checkpoint,
        dataset_root=Path("datasets/a3_front_close_left_100"),
        device="cuda",
        **RAW_PURE,
    )
    attempts_path = args.root / "dagger_attempts.jsonl"
    records = []
    try:
        for seed in args.seeds:
            gate = GateState()
            policy = TakeoverPolicy(model, gate, dict(stall_cfg), fps)
            recorder = GatedRecorder(inner, gate) if inner is not None else None
            try:
                score = bench.run_episode(policy, seed, recorder=recorder)
            except Exception as exc:  # 10h-run resilience: log and continue
                record = dict(
                    seed=seed,
                    takeover=gate.taken,
                    outcome="crashed",
                    stall_type="crashed",
                    ingested=False,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    result={},
                )
                records.append(record)
                attempts_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
                print(json.dumps(record), flush=True)
                continue
            result = score.to_dict()
            if recorder is not None:
                recorder.resolve(int(result["cookies_in_target"]))
            expert_phase = None
            if policy.expert is not None:
                expert_phase = policy.expert.phase.name
            open_frac = (
                policy.open_steps / policy.model_steps_run if policy.model_steps_run else 0.0
            )
            final_t = int(result["cookies_in_target"])
            if gate.taken:
                stall_type = "takeover_" + stall_cfg.get("trigger", "budget")
            elif gate.aborted:
                stall_type = "budget_T0_expert_rejected"
            elif gate.skip_reason and gate.skip_reason.startswith("stall_with_partial"):
                stall_type = "stall_partial_T>0"
            elif result["success"]:
                stall_type = "model_success"
            elif final_t > 0:
                stall_type = "progressing_partial"
            elif open_frac < 0.2:
                stall_type = "closed_fiddling"  # v0 blind spot: grasp-drop cycles
            else:
                stall_type = "open_no_trigger"
            if gate.taken:
                outcome = (
                    "recovered_full"
                    if result["success"]
                    else "recovered_partial"
                    if final_t > 0
                    else "recovered_none"
                )
            elif gate.skip_reason == "reinit_moved_physics":
                outcome = "aborted_reinit"
            else:
                outcome = "not_taken"
            record = dict(
                seed=seed,
                takeover=gate.taken,
                takeover_step=gate.takeover_step,
                skip_reason=gate.skip_reason,
                reinit_deltas=gate.reinit_deltas,
                expert_phase=expert_phase,
                stall_type=stall_type,
                outcome=outcome,
                ingested=outcome in ("recovered_full", "recovered_partial"),
                model_profile=dict(
                    model_steps=policy.model_steps_run,
                    open_frac=round(open_frac, 3),
                    gripper_transitions=policy.gripper_transitions,
                ),
                result=result,
            )
            records.append(record)
            attempts_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "takeover": gate.taken,
                        "at": gate.takeover_step,
                        "skip": gate.skip_reason,
                        "deltas": gate.reinit_deltas,
                        "expert_phase": expert_phase,
                        "target": result["cookies_in_target"],
                        "success": result["success"],
                        "steps": result["steps"],
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
        if inner is not None:
            inner.close()

    taken = [r for r in records if r["takeover"]]
    crosstab = {}
    for r in records:
        crosstab.setdefault(r["stall_type"], {}).setdefault(r["outcome"], 0)
        crosstab[r["stall_type"]][r["outcome"]] += 1
    summary = dict(
        seeds=len(records),
        takeovers=len(taken),
        takeover_successes=sum(1 for r in taken if r["result"].get("success")),
        ingested=sum(1 for r in records if r.get("ingested")),
        crashes=sum(1 for r in records if r.get("outcome") == "crashed"),
        skips=sorted({r["skip_reason"] for r in records if r.get("skip_reason")}),
        crosstab_stalltype_x_outcome=crosstab,
    )
    (args.root / "dagger_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
