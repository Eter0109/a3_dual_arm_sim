"""Drive the 20k pi0.5 checkpoint through the MAIN-branch CookieBatchBenchmark.

Staged 2026-10-15, run when the GPU is free.  Not committed anywhere: main is untouched,
this file only *uses* it.

What it answers: how the 20k model scores under the benchmark that ships on main -- the
original single-box 2x5 task with the strict judging (15 deg upright, exact slots, wall
contact, `success_hold_steps=5`), with box/cookie randomization (boxes +-1 cm, cookies
0.3 mm / 0.9 deg), 20 episodes, up to 1000 steps each.  Everything except the *weights* is
main's own code: this driver imports main's benchmark from the worktree at /tmp/a3-main
(git worktree of ssh-origin/main @ 29fe90e) and talks to the model through the same shared
bridge every other 20k measurement used, so the inference path is identical to the sweeps.

The bridge (started separately, ~4 min to load, 10 GB):

    cd /root/autodl-tmp/a3-collect && \
    MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 \
    .venv/bin/python /root/a3_dual_arm_sim/scripts/a3_bridge_server.py \
        --adapter adapters/pi05_a3.py \
        --listen unix:///tmp/bench-main.sock \
        --config ckpt=/root/autodl-tmp/pi05-runs/a3_b2_unfreeze_20k/checkpoints/020000/pretrained_model

Then:

    MUJOCO_GL=egl python /tmp/bench_main_20k.py --socket /tmp/bench-main.sock \
        --episodes 20 > /root/autodl-tmp/logs/bench_main_20k.log 2>&1

    # integration check without any bridge (hold policy, 2 episodes, CPU+EGL only):
    MUJOCO_GL=egl python /tmp/bench_main_20k.py --selftest

Time (measured 2026-10-15, env side on this machine): 98.8 ms/step with camera obs; the
bridge adds ~320 ms per 50-step chunk executed as 16 (~20 ms/step amortized).  So one
1000-step episode is ~2 min and 20 episodes ~45 min serial.

Known limitation (this is main's, and it is the optimization target): `evaluate()` refuses
any loaded policy with workers>1 ("Policy instances require workers=1"), so the 20 episodes
run serially against one env.  See the optimization plan in the report that stages this file.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scene_compat lives next to it

import scene_compat  # noqa: E402

#: This file's own module name, so the spawned workers of main's process pool can import it
#: and call the factory below.  Requires the directory holding this file on `sys.path`, which
#: is true when it is run as a script (`sys.path[0]` is its directory).
FACTORY_MODULE = "bench_main_20k"

#: The dataset every checkpoint in this line was fine-tuned on; its `meta/tasks.parquet`
#: holds the one instruction they were trained with.
TRAINING_DATASET = os.environ.get("A3_TRAINING_DATASET", "/root/autodl-tmp/datasets/a3_b2_merged")


def training_prompt() -> str:
    """The instruction the checkpoint was trained on, read from the dataset rather than
    copied here -- a literal would drift the moment the dataset is re-collected."""
    import pyarrow.parquet as pq

    rows = pq.read_table(f"{TRAINING_DATASET}/meta/tasks.parquet").to_pylist()
    if not rows or "task" not in rows[0]:
        raise SystemExit(f"no task string in {TRAINING_DATASET}/meta/tasks.parquet")
    return str(rows[0]["task"])

# main's benchmark -- first on sys.path so `a3_dual_arm_sim` resolves to the worktree, never
# to the feature-branch checkout that this file's protocol comes from.
MAIN_ROOT = os.environ.get("A3_MAIN_ROOT", "/tmp/a3-main")
MAIN_SRC = os.environ.get("A3_MAIN_SRC", os.path.join(MAIN_ROOT, "src"))
sys.path.insert(0, MAIN_SRC)

from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark  # noqa: E402

#: Set to a scene fingerprint (see `scene_compat`) to rebuild the training scene inside main's
#: environment: camera mounts, arm start pose, both bins and all 80 Cookies.  Exported to the
#: environment because the spawned episode workers import this module fresh and must install
#: the same shim, and the fingerprint itself travels as a path rather than as an argument.
SCENE_ENV_VAR = "BENCH_TRAINING_SCENE"
#: "full" or "cameras"; travels beside the fingerprint for the same reason.
SCENE_MODE_ENV_VAR = "BENCH_SCENE_MODE"


def install_scene_compat(fingerprint_path: str, mode: str = "full") -> dict:
    """Wrap main's env factory so every environment rebuilds the training scene.

    A wrapper rather than a config file, for two reasons found the hard way: main places both
    bins from code, and this branch's `source_bin` is a mocap body, so no YAML key can express
    where the training scene's bins sit.  `create_env` is patched on the *class*, so a
    benchmark instance built before the patch still picks it up, and `reset` is wrapped per
    environment because the layout is re-derived on every reset.

    `mode="cameras"` keeps only the camera block of the override config and leaves the layout
    alone.  That is the apples-to-apples regime: this branch's environment draws a **different
    layout every episode seed** (20 unique start states) while the full shim pins main's 20
    seeds onto one layout (1 unique start state), so the two were never comparable -- frozen
    single-scene averages and randomised averages are different quantities.
    """
    from a3_dual_arm_sim.workflows import benchmark as benchmark_module

    fingerprint = scene_compat.load_fingerprint(fingerprint_path)
    if mode == "cameras":
        return fingerprint
    for class_name in ("CookieBatchBenchmark", "Cookie2x10Benchmark"):
        cls = getattr(benchmark_module, class_name, None)
        if cls is None:
            continue
        original = cls.create_env
        if getattr(original, "_rebuilds_training_scene", False):
            continue

        def create_env(self, *args, _original=original, **kwargs):
            env = _original(self, *args, **kwargs)
            scene_compat.wrap_reset(env, fingerprint)
            return env

        create_env._rebuilds_training_scene = True  # type: ignore[attr-defined]
        cls.create_env = create_env  # type: ignore[method-assign]
    return fingerprint


_INSTALLED_FINGERPRINT = None
if os.environ.get(SCENE_ENV_VAR):
    # Only true in a spawned worker (or a parent that set the variable itself).
    _INSTALLED_FINGERPRINT = install_scene_compat(
        os.environ[SCENE_ENV_VAR], os.environ.get(SCENE_MODE_ENV_VAR, "full")
    )

# The bridge protocol, loaded standalone by file path so the two branches' packages never
# collide: main's tree stays `a3_dual_arm_sim`, and the client/wire code comes from the
# feature branch without importing it as a package.
_PROTOCOL_PATH = str(
    Path(__file__).resolve().parents[1] / "src" / "a3_dual_arm_sim" / "bridge" / "protocol.py"
)
_spec = importlib.util.spec_from_file_location("a3bridge_protocol", _PROTOCOL_PATH)
proto = importlib.util.module_from_spec(_spec)
# Registered *before* exec: dataclasses introspects sys.modules[cls.__module__], and the
# ActionChunk dataclass is defined at import time -- an unregistered module means that
# lookup returns None and the import dies inside the stdlib.
sys.modules["a3bridge_protocol"] = proto
_spec.loader.exec_module(proto)

try:
    import cv2  # the feature branch's observation.py resizes with INTER_AREA; use the same
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"cv2 unavailable ({exc}); the camera resize needs it") from None

VIEWS = {
    "front_view": "observation.images.front",
    "left_wrist_view": "observation.images.left_wrist",
    "right_wrist_view": "observation.images.right_wrist",
}
STATE_KEY = "observation.state"


class BridgeChunkPolicy:
    """main's BenchmarkPolicy protocol over the shared bridge, with chunk execution.

    `act()` holds a queue of `execute` actions and only forwards the observation once per
    `execute` steps: the same chunk semantics (chunk 50 / execute 16) every 20k measurement
    used.  Forwarding every step instead would cost ~219 ms/step rather than ~14 ms and
    would be a different (and slower) evaluation protocol, not a more accurate one.

    `task_prompt` overrides the instruction the harness hands in.  It has to be overridable:
    this checkpoint was fine-tuned on one sentence -- the dataset's own task string -- and
    main's benchmark sends `"transfer 10 cookies into target box"` by default, which the
    model has never seen.  Run with the harness prompt it scores 0/10 and moves no cookie at
    all; that is an out-of-distribution instruction, not a measurement of the policy.
    """

    name = "pi05_b2_20k"

    def __init__(self, sock_path: str, *, chunk: int = 50, execute: int = 16,
                 task_prompt: str | None = None):
        self.action_mode = "joint_position"
        self.chunk = chunk
        self.execute = execute
        self.task_prompt = task_prompt
        self._sock_path = sock_path
        self.sock = None
        self._reader = None
        self._writer = None
        self._episode = 0
        self._queue: list[np.ndarray] = []
        self._step = 0
        self.forwards = 0
        self.forward_seconds = 0.0

    def _ensure_connected(self) -> None:
        """Connect on first use, never in `__init__`.

        `evaluate(workers=N)` only reads `name` and `action_mode` from the parent's policy
        object, yet a connection made in `__init__` is still a client the bridge has to
        serve.  The bridge answers one request at a time, so an *idle* parent connection
        competes with the N workers for the same server -- connect lazily and the parent
        never connects at all.
        """
        if self.sock is not None:
            return
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._sock_path)
        # `write_frame` / `read_frame` are stream-shaped -- they call `.write()` and `.read()`
        # -- so the socket goes through a pair of file objects, which is the same two lines
        # the feature branch's `SocketTransport.connect` uses.  Handing them the socket
        # itself fails on the very first frame with "'socket' object has no attribute
        # 'write'", which reads like a protocol bug rather than a wrapping mistake.
        self._reader = self.sock.makefile("rb")
        self._writer = self.sock.makefile("wb")
        # The server speaks first: every connection gets a `ready` frame saying the model is
        # loaded.  A client that writes its first request without consuming it reads the
        # handshake back as the answer, and the symptom is a policy that appears to return
        # no actions ("bridge replied 'ready' with keys []").
        kind, _fields, _ = proto.decode_message(proto.read_frame(self._reader))
        if kind != "ready":
            raise RuntimeError(f"expected a 'ready' frame at startup, got {kind!r}")

    def reset(self, context=None) -> None:
        self._episode += 1
        self._queue.clear()
        self._step = 0

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        self._ensure_connected()
        if not self._queue:
            arrays = {}
            for name, key in VIEWS.items():
                image = np.asarray(observation[key])
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                if image.shape[:2] != (224, 224):
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                arrays[name] = image
            arrays[STATE_KEY] = np.asarray(observation[STATE_KEY], dtype=np.float32).reshape(16)
            payload = proto.encode_request(
                episode_id=f"mainbench-{self._episode}",
                step=self._step,
                task=self.task_prompt or task or "transfer 10 cookies into target box",
                arrays=arrays,
                chunk=self.chunk,
            )
            started = time.monotonic()
            proto.write_frame(self._writer, payload)
            kind, fields, req_arrays = proto.decode_message(proto.read_frame(self._reader))
            if kind != "actions" or "actions" not in req_arrays:
                raise RuntimeError(f"bridge replied {kind!r} with keys {sorted(req_arrays)}")
            self.forwards += 1
            self.forward_seconds += time.monotonic() - started
            actions = np.asarray(req_arrays["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != 16:
                raise RuntimeError(f"bridge returned actions shaped {actions.shape}")
            # Execute the first `execute` of the chunk: the manifest's interface for this
            # checkpoint is chunk 50 / execute 16 / dt 0.05, and control_hz 20 matches.
            self._queue = list(actions[: self.execute])
        self._step += 1
        return self._queue.pop(0)

    def close(self) -> None:
        if self.sock is None:  # never connected: the parent's policy is only a name holder
            return
        self._reader.close()
        self._writer.close()
        self.sock.close()


def make_bridge_policy(sock_path: str, *, chunk: int = 50, execute: int = 16,
                       task_prompt: str | None = None):
    """`module:factory` entry point for the workers of main's process pool.

    The pool is started with `spawn`, so this is imported by module name in each child; the
    child then opens its own connection to the same bridge, whose model was loaded once.  A
    `BridgeChunkPolicy` cannot be sent to a worker -- it holds a socket -- which is the whole
    reason `worker_policy_spec` exists.
    """
    return BridgeChunkPolicy(sock_path, chunk=chunk, execute=execute, task_prompt=task_prompt)


class HoldPolicy:
    """Integration self-test: keep the arms at their reset state, no bridge."""

    name = "hold"
    action_mode = "joint_position"

    def __init__(self):
        self._action = None

    def reset(self, context=None) -> None:
        self._action = None

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        if self._action is None:
            state = np.asarray(observation.get(STATE_KEY), dtype=float).reshape(-1)
            self._action = np.zeros(16)
            n = min(16, state.shape[0])
            self._action[:n] = state[:n]
        return self._action


def verify_scene(bench, fingerprint: dict) -> int:
    """Render one reset of the patched benchmark and compare it with training.

    The receipt that matters before a long run: the front camera alone cannot tell you the
    scene is right (35.9 -> 2.7 is a visible improvement, but the wrist cameras sat at 28/255
    until the start pose was matched too, and 28/255 is the difference between an episode that
    grasps and one that hovers).
    """
    import mujoco

    ref_dir = os.environ.get("A3_BENCH_SHOTS", "/tmp/bench_shots")
    env = bench.create_env(render_cameras=True)
    try:
        env.reset(seed=0)
        model, data = env.model, env.data
        for name, pose in fingerprint["bodies"].items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            delta = np.abs(np.asarray(data.xpos[body_id]) - np.asarray(pose["xpos"]))
            print(f"[verify] {name:11s} now={np.round(data.xpos[body_id], 4).tolist()} "
                  f"training={np.round(pose['xpos'], 4).tolist()} "
                  f"max|d|={1000 * float(delta.max()):.2f} mm")
        cookies = np.asarray(env.cookie_positions)
        training_cookies = np.asarray(fingerprint["cookie_positions"])
        print(f"[verify] cookies     now={np.round(cookies[:, :2].mean(axis=0), 4).tolist()} "
              f"training={np.round(training_cookies[:, :2].mean(axis=0), 4).tolist()}")
        for name in ("front", "left_wrist", "right_wrist"):
            path = f"{ref_dir}/refT_{name}.npy"
            if not os.path.exists(path):
                continue
            reference = np.load(path).astype(float)
            image = np.asarray(env._render_camera(name), dtype=float)
            print(f"[verify] {name:11s} MAE vs training render = "
                  f"{np.abs(image - reference).mean():.3f} / 255")
    finally:
        env.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", default="/tmp/bench-main.sock")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--scene-mode",
        choices=("full", "cameras"),
        default="full",
        help=(
            "full: cameras + arm pose + bins + Cookies restored after every reset (one layout "
            "for every seed).  cameras: only the camera block, leaving main's own per-seed "
            "randomisation intact -- the regime this branch's sweeps actually measure."
        ),
    )
    parser.add_argument(
        "--training-scene",
        default=None,
        help=(
            "Fingerprint written by scene_fingerprint.py.  When given, every environment "
            "rebuilds the training scene (cameras, arm start pose, bins, Cookies) after each "
            "reset, and the camera block of the override config is used.  Without it the "
            "checkpoint is shown an observation distribution it has never seen and scores 0."
        ),
    )
    parser.add_argument(
        "--verify-scene",
        action="store_true",
        help=(
            "Build one environment, print the restored layout and the per-camera difference "
            "against this branch's own renders (/tmp/bench_shots/refT_*.npy), then exit -- a "
            "receipt that the observation matches training before spending hours on episodes."
        ),
    )
    parser.add_argument("--execute", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Episodes to run in parallel.  Each worker opens its own socket to the same "
            "bridge and rebuilds the policy from the `module:factory` spec -- which is what "
            "`worker_policy_spec` exists for (see the bench:rebuild-an-external-policy "
            "change on the feature branch).  One worker per episode is the useful ceiling: "
            "the environment is ~90 %% of a step and the bridge serialises inference anyway."
        ),
    )
    parser.add_argument("--output", default=None, help="Write the result JSON here")
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help=(
            "Freeze the layout, which is what every previous measurement of these checkpoints "
            "did: main's benchmark randomises boxes by +-1 cm and cookies by 0.3 mm, and the "
            "a3_b2 dataset is a single fixed arrangement, so the randomised version is a "
            "different (and out-of-distribution) task."
        ),
    )
    parser.add_argument(
        "--task-prompt",
        default=None,
        help=(
            "Instruction to send instead of the benchmark's own.  Defaults to the task string "
            "in the training dataset's meta/tasks.parquet, because a fine-tuned checkpoint "
            "only knows the sentence it was trained on; with main's default prompt this one "
            "scores 0/10 and moves nothing."
        ),
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-steps", type=int, default=120)
    args = parser.parse_args()

    fingerprint = None
    config_path = None
    if args.training_scene:
        # Exported as well as installed: a spawned worker imports this module fresh and only
        # sees the environment, not our parsed arguments.
        os.environ[SCENE_ENV_VAR] = args.training_scene
        os.environ[SCENE_MODE_ENV_VAR] = args.scene_mode
        fingerprint = install_scene_compat(args.training_scene, args.scene_mode)
        config_path = scene_compat.write_override_config(fingerprint)
        print(
            f"[bench] training scene: {args.training_scene} mode={args.scene_mode} "
            f"({len(fingerprint['joints'])} joints, {len(fingerprint['cookie_positions'])} "
            f"cookies) | cameras from {config_path}",
            flush=True,
        )

    task_prompt = args.task_prompt if args.task_prompt is not None else training_prompt()
    print(f"[bench] instruction: {task_prompt!r}", flush=True)
    # make_cookie_benchmark defaults: task 2x5, prompt_style "task", randomize_boxes=True
    # (+-1 cm), randomize_cookies=True (0.3 mm / 0.9 deg), hold 5, slots (15, 5) mm,
    # wall 30 mm.  These are main's benchmark parameters unless overridden here.
    bench = make_cookie_benchmark(
        task="2x5",
        max_steps=args.max_steps,
        randomize_boxes=not args.no_randomize,
        randomize_cookies=not args.no_randomize,
        config_path=config_path,
    )

    if args.verify_scene:
        return verify_scene(bench, fingerprint)

    if args.selftest:
        bench.max_steps = args.selftest_steps
        result = bench.evaluate(HoldPolicy(), num_episodes=2, seed_start=100)
        print(f"[selftest] {result}")
        return 0

    if args.workers > 1:
        # The point of the parallel path: `policy` is only read for its name and action_mode
        # in the parent; every worker builds its own client, so no socket changes hands.
        spec = f"{FACTORY_MODULE}:make_bridge_policy"
        importlib.import_module(FACTORY_MODULE)
        policy = BridgeChunkPolicy(args.socket, execute=args.execute)
        t0 = time.monotonic()
        result = bench.evaluate(
            policy,
            num_episodes=args.episodes,
            seed_start=args.seed_start,
            workers=args.workers,
            worker_policy_spec=spec,
            worker_policy_kwargs={"sock_path": args.socket, "execute": args.execute,
                                  "task_prompt": task_prompt},
        )
        wall = time.monotonic() - t0
        print(f"[bench] {args.workers} workers | wall {wall / 60:.1f} min | {result}")
        policy.close()
    else:
        policy = BridgeChunkPolicy(args.socket, execute=args.execute, task_prompt=task_prompt)
        t0 = time.monotonic()
        result = bench.evaluate(policy, num_episodes=args.episodes, seed_start=args.seed_start)
        wall = time.monotonic() - t0
        per_forward = 1000 * policy.forward_seconds / max(policy.forwards, 1)
        print(f"[bench] wall {wall / 60:.1f} min | bridge forwards {policy.forwards} "
              f"({per_forward:.0f} ms each) | {result}")
        policy.close()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)
        print(f"[bench] result written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
