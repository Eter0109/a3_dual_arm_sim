"""Attribute one number: is the low `ret` in the online DSRL loop the prompt, or the noise?

The confound
------------
Every online DSRL run (run4, 7, 8, 9, 10) was launched without `--task-prompt` and without
`A3_TRAINING_DATASET`, so `training_prompt()` fell back to `DEFAULT_TRAINING_PROMPT` -- a sentence
from `a3_b2_merged` that does **not appear** in the v9 training set:

    used   "grasp five yellow blocks together ... the 2 by 5 box ... fill all ten slots"
    v9     11 instructions, none of them that one

while the scene was 2x10 (`required_cookies = 20`, `target_rows = 10`).  The documented baseline of
13.18/20 was measured with *main's own instructions*, i.e. a v9 sentence.

Meanwhile `ret` in those runs was 4.57-10.0 and `greedy` **started at 19.00**.  Those numbers
differ in two ways at once, so none of them can be attributed:

    condition         prompt     noise
    13.18 baseline    correct    a fresh N(0,1) draw -- how the flow policy samples normally
    ret               wrong      tanh(N(0,1)), std ~0.63, one w repeated across all 50 chunks
    greedy            wrong      zero -- `tanh(mean)`, and at init mean = 0

What this measures
------------------
Both prompts against three noise conditions on the **same frozen checkpoint, no training**: a 2x3
grid that separates the two effects instead of guessing at their sum.

    noise=raw     w ~ N(0,1)          what the 13.18 baseline used
    noise=tanh    w = tanh(N(0,1))    what DSRL's actor produces at init (std 0.63, bounded by 1)
    noise=zero    w = 0               what `greedy` produces at init (mean starts at 0)

Only the noise tensor varies besides the prompt.  Nothing here depends on the actor: `w` is drawn
independently of the observation, so there is no projection, no `SteeringActor` and no actor
checkpoint involved -- which is why this runs today, against the frozen VLA alone.

Usage
-----
    python scripts/recap/probe_prompt_noise.py --episodes 8

Eight episodes per cell because this project's own note on the 20000-step checkpoint records three
rollouts scoring 6, 1 and 1 out of 10.  Seeds are identical across all six cells, so every
comparison is paired.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_dsrl_online as trainer  # noqa: E402
from steering import encode_prefix, repeat_noise, sample_with_prefix  # noqa: E402

V9_TASKS = Path("/root/autodl-tmp/datasets/a3_clean_v9/meta/tasks.parquet")
NOISE_MODES = ("raw", "tanh", "zero")

#: Verbatim from `bench_pi05_on_main.DEFAULT_TRAINING_PROMPT`, which is what every DSRL run used
#: by accident.  Copied rather than imported because importing that module inserts main's `src` on
#: `sys.path` and redirects `a3_dual_arm_sim` to a different worktree; and because this string is
#: the thing under test, so it should be visible here.
B2_PROMPT = (
    "grasp five yellow blocks together from the source bin, insert them upright into the "
    "marked slots of the 2 by 5 box, then return to the source bin, grasp five more, and "
    "fill all ten slots"
)


def v9_prompt(plan: int) -> str:
    """The v9 instruction whose 2x10 plan is `plan`, read from the dataset.

    Read rather than pasted: the whole point of this probe is that a pasted prompt went stale, and
    the v9 set carries ten 2x10 plan sentences plus one 2x5 sentence.
    """
    import pyarrow.parquet as pq

    tasks = [str(r["task"]) for r in pq.read_table(V9_TASKS).to_pylist()]
    for task in tasks:
        if f"Grasp {plan} cookies first" in task:
            return task
    raise SystemExit(f"no v9 instruction with `Grasp {plan} cookies first`; have {len(tasks)}")


def make_noise(mode: str, batch: int, action_dim: int, device: str) -> torch.Tensor:
    """One `(batch, action_dim)` noise block -- the only thing varying between cells besides the
    prompt."""
    if mode == "zero":
        return torch.zeros(batch, action_dim, device=device)
    draw = torch.randn(batch, action_dim, device=device)
    return torch.tanh(draw) if mode == "tanh" else draw


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=8,
                        help="scenes stepped concurrently.  8 matches the training loop;"
                             " a serial loop left the GPU at 0 %% and cost 7x per step.")
    parser.add_argument("--max-control-steps", type=int, default=1500)
    parser.add_argument("--execute", type=int, default=16)
    parser.add_argument("--plan", type=int, default=5, help="which v9 2x10 plan sentence to use")
    parser.add_argument("--seed-offset", type=int, default=51000,
                        help="the online runs' default, so the layouts are the ones training saw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=Path("/root/autodl-tmp/prompt_noise_grid.json"))
    parser.add_argument("--cells", default="",
                        help="comma-separated subset of `prompt/noise`, e.g. `v9/raw`.  Empty means"
                             " the full 2x3 grid.  Exists because 48 episodes x 1500 steps is 1.6 h"
                             " and a one-cell sanity check should cost 2 min.")
    args = parser.parse_args()

    device = args.device
    import train_pi05_launcher as launcher  # noqa: PLC0415
    from bench_pi05_on_main import make_cookie_benchmark  # noqa: PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.pi05 import PI05Policy  # noqa: PLC0415

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001
    policy = PI05Policy.from_pretrained(str(args.checkpoint)).to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, postprocess = make_pre_post_processors(
        policy.config, str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}})
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.max_action_dim)
    original_dim = int(policy.config.output_features["action"].shape[0])
    num_steps = int(policy.config.num_inference_steps)
    print(f"[grid] chunk {chunk_size}, action_dim {action_dim}, arm dims {original_dim},"
          f" {num_steps} sampler steps", flush=True)

    # The same scene the online loop builds: 2x10, `required_cookies = 20`.
    bench = make_cookie_benchmark(task="2x10", prompt_style="short",
                                  max_steps=args.max_control_steps)
    print(f"[grid] scene required_cookies = {bench.required_cookies}", flush=True)

    prompts = {"v9": v9_prompt(args.plan), "b2": B2_PROMPT}
    for name, text in prompts.items():
        print(f"[grid] prompt {name}: {text[:92]}", flush=True)

    wanted = {cell.strip() for cell in args.cells.split(",") if cell.strip()}
    if wanted:
        unknown = wanted - {f"{p}/{n}" for p in prompts for n in NOISE_MODES}
        if unknown:
            raise SystemExit(f"unknown cell(s) {sorted(unknown)}; have"
                             f" {sorted(f'{p}/{n}' for p in prompts for n in NOISE_MODES)}")
        print(f"[grid] restricted to {sorted(wanted)}", flush=True)
    else:
        cell_count = len(prompts) * len(NOISE_MODES)
        print(f"[grid] {cell_count} cells x {args.episodes} episodes x {args.max_control_steps}"
              f" control steps", flush=True)

    results: dict[str, list[float]] = {}
    pool = ThreadPoolExecutor(max_workers=args.parallel)
    for prompt_name, prompt in prompts.items():
        for noise_mode in NOISE_MODES:
            key = f"{prompt_name}/{noise_mode}"
            if wanted and key not in wanted:
                continue
            scores: list[float] = []
            started = time.monotonic()
            for group_start in range(0, args.episodes, args.parallel):
                group = list(range(group_start,
                                   min(group_start + args.parallel, args.episodes)))
                scenes: list[trainer.Scene] = []
                for episode in group:
                    seed = args.seed_offset + episode
                    # Fresh environment per episode, matching `--fresh-env-per-episode`: a
                    # persistent one does not reproduce its own first reset, so reusing it would
                    # measure a scene distribution nothing else uses.
                    env = bench.create_env(render_cameras=False)
                    env.reset(seed=seed, options=trainer.reset_options(bench, seed))
                    scenes.append(trainer.Scene(
                        index=len(scenes), env=env,
                        renderer=trainer._make_renderer(env.model),  # noqa: SLF001
                        camera_ids=dict(env._camera_ids),  # noqa: SLF001
                        seed=seed, required_cookies=int(bench.required_cookies),
                        max_control_steps=args.max_control_steps))
                try:
                    # Batched exactly like the training loop: one physics call per scene through a
                    # thread pool (MuJoCo releases the GIL), one forward for the whole batch, then
                    # wait for all of them.  An earlier version submitted a play() per active scene
                    # inside the loop and left 89 threads parked on `futex_wait`; the trainer submits
                    # the whole round and joins once, which is what this now does.
                    while any(s.control_steps < args.max_control_steps for s in scenes):
                        frames = [trainer.render_frames(s.renderer, s) for s in scenes]
                        observations = [s.env._observation() for s in scenes]  # noqa: SLF001
                        for s, observation in zip(scenes, observations):
                            s.seed_history(observation)
                        raw = trainer.raw_batch(scenes, frames, observations, prompt, device)
                        prepared = preprocess(raw)
                        prefix = encode_prefix(policy, prepared, keep_embeddings=True)
                        with torch.no_grad():
                            noise = make_noise(noise_mode, len(scenes), action_dim, device)
                            flat = repeat_noise(noise, chunk_size, action_dim, "single")
                            chunk = sample_with_prefix(policy, prefix, flat, num_steps, device)
                        executed = trainer.postprocess_actions(
                            postprocess, chunk[:, :, :original_dim])
                        futures = [
                            pool.submit(s.play, np.asarray(executed[i], dtype=np.float64),
                                        args.execute, "increment", False, 1.0)
                            for i, s in enumerate(scenes)
                            if s.control_steps < args.max_control_steps]
                        for future in futures:
                            future.result()
                finally:
                    for s in scenes:
                        s.env.close()
                for episode, s in zip(group, scenes):
                    scores.append(float(s.episode_return))
                    print(f"[grid] {key:<12} ep{episode} seed{s.seed} ->"
                          f" {s.episode_return:5.0f}  ({s.control_steps} control steps)",
                          flush=True)
            results[key] = scores
            print(f"[grid] {key:<12} mean {np.mean(scores):6.2f}"
                  f" +/- {np.std(scores) / max(1, np.sqrt(len(scores))):.2f}"
                  f"  ({(time.monotonic() - started) / 60:.1f} min)", flush=True)
    pool.shutdown(wait=False)

    print("\n[grid] === summary (cookies out of %d) ===" % bench.required_cookies, flush=True)
    for key, scores in results.items():
        print(f"[grid] {key:<14} {np.mean(scores):7.2f}"
              f" {np.std(scores) / max(1, np.sqrt(len(scores))):6.2f}"
              f"  {[int(s) for s in scores]}", flush=True)

    needed = ("v9/raw", "b2/raw", "v9/zero", "b2/zero", "v9/tanh")
    if all(k in results for k in needed):
        def mean(key: str) -> float:
            return float(np.mean(results[key]))

        print("", flush=True)
        print(f"[grid] prompt cost, noise=raw   (v9 - b2)    {mean('v9/raw') - mean('b2/raw'):+7.2f}",
              flush=True)
        print(f"[grid] prompt cost, noise=zero  (v9 - b2)    {mean('v9/zero') - mean('b2/zero'):+7.2f}",
              flush=True)
        print(f"[grid] zero-noise gain, prompt=v9 (zero - raw) {mean('v9/zero') - mean('v9/raw'):+7.2f}",
              flush=True)
        print(f"[grid] tanh-noise cost, prompt=v9 (tanh - raw) {mean('v9/tanh') - mean('v9/raw'):+7.2f}",
              flush=True)
        print("[grid] `greedy` in the online runs was b2/zero and `ret` was b2/tanh;"
              " the 13.18 baseline is v9/raw.", flush=True)

    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[grid] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
