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

#: The field-of-view grid, as ``(name, scope, degrees)``.  One family, measured as a dose
#: response rather than as a single point, because "how far can the framing move before the
#: policy breaks" and "which camera carries the constraint" are separate questions and a
#: single symmetric pair answers neither.
#:
#: The shipped values are **45 for the agentview and 75 for the wrist camera**, so 45 is the
#: agentview's own value (a no-op on it when ``scope='front'``) and a 30-degree narrowing
#: when ``scope='eye'``.  The eye column brackets the shipped 75 on both sides -- 60/45
#: narrow, 90 widens -- so an asymmetric failure is visible as one.
_FOVY_CASES: tuple[tuple[str, str, float], ...] = (
    ("fovy_30", "both", 30.0),
    ("fovy_60", "both", 60.0),
    ("fovy_front_30", "front", 30.0),
    ("fovy_front_60", "front", 60.0),
    ("fovy_eye_45", "eye", 45.0),
    ("fovy_eye_60", "eye", 60.0),
    ("fovy_eye_90", "eye", 90.0),
)
_FOVY: dict[str, tuple[str, float]] = {n: (s, v) for n, s, v in _FOVY_CASES}
_FOVY_CAMERAS: dict[str, tuple[str, ...]] = {
    "front": ("agentview",),
    "eye": ("robot0_eye_in_hand",),
    "both": ("agentview", "robot0_eye_in_hand"),
}

#: Everything except the rendering is held fixed in a perturbation run: same checkpoint,
#: same seed, same episodes, same action interface.  Each name changes one axis.
PERTURBATIONS = (
    "none",
    *(name for name, _, _ in _FOVY_CASES),
    "cam_shift_10",
    "no_lights",
    "grey",
    "no_headlight",
)


def _robosuite_env(sub: Any) -> Any:
    """The robosuite environment inside libero's wrapper.

    `libero.libero.envs.OffScreenRenderEnv` is a thin object holding `self.env`, the
    robosuite `MujocoEnv`.  `sim`, `robots` and `check_success` are proxied, so a wrapper
    that reaches for `hard_reset` finds nothing, sets a *new* attribute, and disables
    nothing -- while every later call to `env.sim` quietly resolves to the inner object.
    """

    return getattr(sub._env, "env", sub._env)  # noqa: SLF001


def _change(model: Any, name: str) -> Any:
    """Apply one perturbation to a raw `MjModel`; return a predicate that it is in force.

    Written against `mujoco.MjModel` rather than robosuite's binding wrapper because some
    of the wrapper's attributes are views and some are copies, and a copy swallows the
    assignment without raising.  The returned predicate takes a model so that it can be
    re-asked after a reset may have replaced it, and it is the only thing that separates
    "this changed nothing" from "this changed nothing measurable".
    """

    import mujoco

    def camera_id(which: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, which))

    if name == "none":
        return lambda _model: True
    if name in _FOVY:
        # The suffix names the cameras the change applies to and the number is the new
        # field of view in degrees.  `front` touches only the agentview, `eye` only the
        # wrist camera, neither suffix touches both -- which is the distinction the first
        # sweep could not make when both were moved together.
        scope, value = _FOVY[name]
        cameras = _FOVY_CAMERAS[scope]
        for camera in cameras:
            model.cam_fovy[camera_id(camera)] = value
        return lambda m: all(
            m.cam_fovy[int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c))] == value
            for c in cameras
        )
    if name == "cam_shift_10":
        camera = camera_id("agentview")
        before = float(model.cam_pos[camera, 1])
        model.cam_pos[camera, 1] = before + 0.10
        return lambda m: abs(float(m.cam_pos[int(mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")), 1]) - before - 0.10) < 1e-6
    if name == "no_lights":
        # The scene's two lights off, leaving only the camera headlight.  (There is no
        # "shadows off" variant: `light_castshadow` is `[0, 0]` on this scene already, so
        # such a run would measure the absence of shadows in the baseline and report it as
        # robustness -- which is exactly the failure the image check below catches.)
        model.light_diffuse[:] = 0.0
        model.light_specular[:] = 0.0
        return lambda m: not m.light_diffuse.any()
    if name == "grey":
        # The extreme one: every material flattened to the same mid-grey and every texture
        # reference dropped, so the policy sees shapes and nothing else.  It is the closest
        # single-axis stand-in for "a different renderer".
        model.mat_rgba[:] = (0.5, 0.5, 0.5, 1.0)
        model.mat_texid[:] = -1
        return lambda m: bool(
            np.allclose(m.mat_rgba, (0.5, 0.5, 0.5, 1.0)) and (m.mat_texid == -1).all()
        )
    if name == "no_headlight":
        model.vis.headlight.diffuse[:] = 0.0
        model.vis.headlight.ambient[:] = 0.0
        return lambda m: not m.vis.headlight.diffuse.any()
    raise AssertionError(name)  # pragma: no cover - PERTURBATIONS is checked upstream


def apply_perturbation(vec: Any, name: str) -> list[tuple[Any, Any]]:
    """Change one rendering axis of a freshly built LIBERO vector environment.

    This is where the cameras and materials of a benchmark scene can be reached without
    editing the arena XML that ships with LIBERO.  Everything else -- the robot, the
    objects, the task, the state vector, the action scaling -- is untouched, so a drop in
    the success rate is attributable to the axis named in `name` and nothing else.

    `hard_reset` has to be turned off on the robosuite env, not on libero's wrapper (see
    `_robosuite_env`): robosuite's `reset` calls `_load_model()` and re-parses the MJCF
    **from disk** when it is set, so a camera written here is back to its original value
    the first time an episode ends -- which on a task that terminates on success is almost
    all of them.  With it off, `reset` is no more than `sim.reset()` and the episode's
    state still comes from `set_init_state`.  `none` turns it off too, so the baseline is
    measured under the same reset semantics.

    Returns the (environment, predicate) pairs a caller re-checks after its first reset,
    because whether the edit survived is a property of robosuite rather than of this code.
    """

    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {name!r}; known: {list(PERTURBATIONS)}")

    pairs = []
    for sub in vec.envs:
        inner = _robosuite_env(sub)
        inner.hard_reset = False
        cameras = ("agentview", "robot0_eye_in_hand")
        before = {
            name: np.asarray(inner.sim.render(width=256, height=256, camera_name=name,
                                              depth=False)) for name in cameras
        }
        check = _change(inner.sim.model._model, name)
        inner.sim.forward()
        after = {
            name: np.asarray(inner.sim.render(width=256, height=256, camera_name=name,
                                              depth=False)) for name in cameras
        }
        if name != "none":
            # The strongest statement available without a policy: this perturbation changes
            # what the cameras see.  A field that reads back as changed, and a render that
            # comes out byte-identical, is a perturbation that does nothing -- and a run
            # under it would be reported as robustness to it.
            delta = max(
                float(np.abs(after[c].astype(int) - before[c].astype(int)).mean())
                for c in cameras
            )
            if delta < 0.5:
                raise RuntimeError(
                    f"perturbation {name!r} leaves both cameras' images unchanged "
                    f"(largest mean difference {delta:.4f}).  It is a no-op on this scene, "
                    "and measuring it would report the unperturbed policy as robust."
                )
        pairs.append((sub, check))
    verify_perturbation(pairs, name)
    return pairs


def verify_perturbation(pairs: list[tuple[Any, Any]], name: str) -> None:
    """Fail loudly if a perturbation is not in force on the environment as it stands now."""

    for sub, check in pairs:
        model = _robosuite_env(sub).sim.model._model
        if not check(model):
            raise RuntimeError(
                f"perturbation {name!r} is not in force on the environment anymore.  A "
                "reset that reloads the model from disk will have reverted it, and the "
                "run would then report the unperturbed scene as robustness."
            )


def parse_perturbations(spec: str) -> list[str]:
    names = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [name for name in names if name not in PERTURBATIONS]
    if unknown:
        raise SystemExit(f"unknown perturbation(s) {unknown}; known: {list(PERTURBATIONS)}")
    return names


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
    parser.add_argument("--perturb", default="none",
                        help="comma-separated single-axis rendering changes, applied to a "
                             "freshly built environment: " + ", ".join(PERTURBATIONS))
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

    perturbations = parse_perturbations(args.perturb)
    report: dict = {"suite": args.suite, "episodes_per_task": args.episodes, "runs": []}

    def run_one(perturbation: str, task_id: int) -> tuple[int, np.ndarray, dict]:
        """One task's episodes under one perturbation.  The environment is new each time.

        One task at a time: `make_env` builds a vector environment for **every** entry in
        `task_ids` up front -- ten tasks of ten envs is a hundred robosuite instances, each
        with its own EGL context, and that exhausted the 5090 before a single step ran
        (`Tried to allocate 20.00 MiB ... 57.94 MiB is free`).  The official runner builds
        them all too, and gets away with it only because ten tasks of ten is near the edge
        of what fits rather than over it.
        """

        task_cfg = LiberoEnvConfig(task=args.suite, task_ids=[task_id])
        envs = make_env(task_cfg, n_envs=args.episodes)
        vec = envs[args.suite][task_id]
        try:
            # After construction, before the first step: robosuite's cameras and materials
            # are only reachable through the compiled model.
            pairs = apply_perturbation(vec, perturbation)
            max_steps = int(vec.call("_max_episode_steps")[0])
            policy.reset()

            observation, _ = vec.reset(seed=list(range(args.episodes)))
            # And again after the first reset, because whether the edit survived one is a
            # property of robosuite's `hard_reset` rather than of anything written above.
            verify_perturbation(pairs, perturbation)
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

            # Zero means "never terminated", not "terminated at step zero": LIBERO's
            # wrapper returns `truncated=False` unconditionally, so an episode that fails
            # simply runs out the `range(max_steps)` and never sets `terminated`.  Reading
            # the raw array as a step count put a 0 in the failure column, where it looked
            # like the environment had ended instantly.
            steps_taken[steps_taken == 0] = max_steps
            executed = np.concatenate(actions, axis=0) if actions else np.zeros((0, 7))
            replan_rows = np.concatenate(replans, axis=0) if replans else np.zeros((0, 7))
            return int(successes.sum()), steps_taken, summarise(executed, replan_rows)
        finally:
            vec.close()
            # The EGL contexts are the expensive part and they are not released by
            # dropping the reference alone; without this the next task inherits the
            # previous one's GPU footprint and the run dies a few tasks in.
            del envs
            gc.collect()
            torch.cuda.empty_cache()

    for perturbation in perturbations:
        successes = 0
        episodes = 0
        dwells: list[float] = []
        lost = 0
        per_task: list[int] = []
        for task_id in task_ids:
            task_successes, steps_taken, stats = run_one(perturbation, task_id)
            successes += task_successes
            episodes += args.episodes
            lost += int((steps_taken >= 200).sum())
            per_task.append(task_successes)
            value = stats["replans"].get("dwell_fraction_lt_0.2")
            if value is not None:
                dwells.append(value)
            print(
                f"  [{perturbation}] task {task_id:>2}: {task_successes:>2}/{args.episodes}  "
                f"steps median {int(np.median(steps_taken)):>3}  "
                f"dwell {value if value is not None else float('nan'):.3f}",
                flush=True,
            )
        report["runs"].append(
            {
                "perturbation": perturbation,
                "successes": successes,
                "episodes": episodes,
                "per_task_successes": per_task,
                "dwell_fraction_lt_0.2": float(np.mean(dwells)) if dwells else None,
                "timed_out": lost,
            }
        )
        print(f"** {perturbation:<14} {successes}/{episodes} = "
              f"{100.0 * successes / episodes:.0f}%", flush=True)
        print()

    baseline = next((run for run in report["runs"] if run["perturbation"] == "none"), None)
    print("=== 单轴渲染扰动 ===")
    print(f"{'perturbation':<16} {'success':>9} {'rate':>7} {'vs none':>9} "
          f"{'dwell<20%':>10} {'timeouts':>9}")
    for run in report["runs"]:
        rate = 100.0 * run["successes"] / run["episodes"]
        delta = "" if baseline is None or run is baseline else f"{rate - 100.0 * baseline['successes'] / baseline['episodes']:+.0f} pt"
        dwell = "  -  " if run["dwell_fraction_lt_0.2"] is None else f"{run['dwell_fraction_lt_0.2']:.3f}"
        print(f"{run['perturbation']:<16} {run['successes']:>4}/{run['episodes']:<4} "
              f"{rate:>6.0f}% {delta:>9} {dwell:>10} {run['timed_out']:>9}")
    if baseline is not None and baseline["successes"] == baseline["episodes"]:
        print()
        print("（`none` 是 100%：官方公布 97.0、本机官方脚本 100.0，所以装配等价；"
              "其他行的下降才是扰动的效应）")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
