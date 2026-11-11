"""DSRL-NA on this stack: steer the frozen π0.5 flow sampler by learning its initial noise.

What it does
------------
The policy is frozen. A small actor maps an observation to the **noise** the flow sampler starts
from, and the sampler turns that noise into an action chunk. The actor is trained with SAC to
maximise a critic's estimate of the chunk's value.

Three critics, which is the part that is easy to get wrong
---------------------------------------------------------
Reading `stable_baselines3/dsrl/dsrl.py` (the reference, vendored under /root/refs) shows DSRL keeps
two critics and uses them for different things:

* `critic` -- `Q(s, a)` on **real actions**, trained by TD from the replay buffer. This is what
  estimates value.
* `critic_noise` -- `Q_noise(s, epsilon)` on **noise**, trained by distillation from `critic`. This
  is what the actor maximises.

The split exists because the actor's action is the noise, and taking `dQ/da * da/depsilon` through
the frozen sampler on every actor step is expensive and contractive; distilling into a noise-space
critic instead lets `clip(epsilon, ...)`-style SAC machinery apply directly. The measured cost of
the alternative is in `measure_dsrl_feasibility.py`: 0.63 s per batch-4 backward through the
sampler, which would be paid on every actor update.

Why the offline variant is the only one available here
------------------------------------------------------
DSRL's reference implementation is on-policy SAC in a gym environment. This repository has no
online loop and each simulated episode costs ~12 minutes, so training runs **offline** on recorded
transitions. That is a real departure and it is stated in the output rather than hidden: the critic
sees only the state distribution the recording policy produced, so the actor's improvements are
extrapolations beyond it.

What the paper actually feeds the noise policy, and why no VLA is in the loop
----------------------------------------------------------------------------
Appendix C.5 of the paper is explicit about the pi_0 experiments: the noise policy consumes
"the proprioceptive state, the final token's last hidden feature from pi_0's VLM backbone (a
2,048-dimensional vector), and visual features".  Every one of those is a function of the *frame*,
not of the noise, so all of it is computed once.  Two things follow:

* the actor and both critics are plain MLPs on cached features -- no image, no SigLIP, no
  PaliGemma in the loop;
* the only quantity that genuinely depends on the noise is the noise critic's distillation target,
  `critic(s, pi(s, epsilon))`.  It needs the denoiser, so `transitions x distill_per_frame`
  `(epsilon, chunk)` pairs are generated in a single batched pass and the loop draws from that
  table.

That is the same decomposition the paper uses when it runs "pi_0 inference on a remote policy
server" and "DSRL training locally on a ... RTX 3070 GPU with 8GB of VRAM": the policy side is a
batched offline pass, the RL side never touches it.

The cost of the departure is stated rather than hidden: the noise critic only ever sees
`--distill-per-frame` noises per state, where the paper draws a fresh noise every gradient step.
That number is printed, and it is the knob that buys the coverage back.

A bug this rewrite fixes
------------------------
`obs_by_frame` used to be built by a loop whose source was `cached_rows - offset` and whose
destination was `cached_rows`; every destination already held a value, so the `np.where` kept it
and the loop did nothing.  `features_v9` is written on every 5th row, so **80 % of the observation
table was all zero** and every earlier run trained its actor and critics on it.  Fixed by
nearest-preceding-row.

Usage
-----
    python scripts/recap/train_dsrl.py --epochs 2000 --batch 8 --utd 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn


# The networks and the sampler live in `steering.py`, because the *inference* path needs the
# very same code.  Two copies of `encode_prefix` would drift, and a drift here is invisible:
# it shows up as a slightly worse score in an experiment that is trying to detect a small
# effect.  `train_dsrl.py` keeps its own data handling and training loop; the model and the
# sampler are imported.
from steering import (  # noqa: E402
    MLP,
    NoiseActor,
    QHead,
    build_observation as build_observation_shared,
    encode_prefix,
    repeat_noise as repeat_noise_shared,
    sample_with_prefix,
    soft_update,
    twin_critic,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model"),
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--batch", type=int, default=8,
        help=(
            "transitions per update.  No longer bounded by VRAM: the VLA is not in the loop, so "
            "this only sizes three MLPs and a table lookup."
        ),
    )
    parser.add_argument(
        "--hidden", type=int, default=2048,
        help=(
            "width of the actor and both critics.  2048 is what the reference uses "
            "(`cfg/robomimic/dsrl_can.yaml: layer_size: 2048`) and what appendix B recommends "
            "('layer size up to 2048').  512 was a memory decision from when the VLA was still in "
            "the loop; it is not needed now that an update is three MLP passes."
        ),
    )
    parser.add_argument(
        "--transitions", type=int, default=512,
        help=(
            "how many adjacent-row pairs to cache.  Each transition needs its successor's "
            "features too, so 2 x this many frames are decoded once (measured ~45 ms per frame).  "
            "This is the critic's whole state distribution, so it is the first number to raise "
            "for a real run; the cache is written to disk and reused."
        ),
    )
    parser.add_argument(
        "--distill-per-frame", type=int, default=8,
        help=(
            "precomputed `(epsilon, pi(s, epsilon))` pairs per transition, used to distil the "
            "noise critic.  The paper draws a fresh epsilon every gradient step, which needs the "
            "denoiser; this is the finite set that replaces it.  Generation costs ~129 ms per pair "
            "per batch of eight, once."
        ),
    )
    parser.add_argument(
        "--critic-target", choices=["td", "mc"], default="mc",
        help=(
            "what `critic` regresses.  `td` is the reference's bootstrapped target and needs a "
            "*contiguous* pool to propagate value (see `--segment-length`).  `mc` regresses the "
            "Monte-Carlo return computed from the episode's own potential trace, which needs no "
            "propagation and no contiguity.  Measured on this data: MC reaches held-out R^2 +0.55 "
            "while TD on the same rows and features reaches +0.06 (anchored) or -1.17 (scattered). "
            "See `probe_value_ceiling.py`."
        ),
    )
    parser.add_argument(
        "--segment-length", type=int, default=0,
        help=(
            "0 keeps the historical behaviour: transitions drawn independently, which makes the "
            "pool a scatter of isolated rows.  Measured: a 4096-row scatter has 3687 contiguous "
            "runs with a mean length of 1.11 and a maximum of 5, while gamma=0.999 needs ~693 "
            "steps of propagation -- so TD is structurally unable to carry value back from the "
            "reward.  Set this to N to draw contiguous N-step segments instead (each segment "
            "must also fit the feature cache budget).  Ignored when `--critic-target mc`."
        ),
    )
    parser.add_argument(
        "--noise-mode", choices=["single", "full"], default="single",
        help=(
            "`single` is appendix C.5: a 32-dim single-step noise repeated across the chunk axis, "
            "so the actor and noise critic live in R^32 instead of R^1600.  `full` is the "
            "1,600-dim version every earlier run in this repository used."
        ),
    )
    parser.add_argument("--obs-proprio", action=argparse.BooleanOptionalAction, default=True,
                        help="include `observation.state` (normalised) in the observation")
    parser.add_argument("--obs-vlm", action=argparse.BooleanOptionalAction, default=True,
                        help="include the 2,048-dim VLM final-token hidden state (paper C.5)")
    parser.add_argument("--obs-visual", action=argparse.BooleanOptionalAction, default=True,
                        help="include the 256-dim frozen SigLIP summary from features_v9")
    parser.add_argument(
        "--feature-batch", type=int, default=8,
        help="frames per VLA pass while building the cache; 8 keeps the KV cache near 120 MB",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="where the feature/distillation caches live (default: <dataset>/dsrl_cache)",
    )
    parser.add_argument("--regenerate-cache", action="store_true",
                        help="ignore an existing cache and rebuild it")
    parser.add_argument(
        "--utd", type=int, default=20,
        help=(
            "actor/critic gradient steps per epoch.  Appendix B of the paper: UTD >= 20.  "
            "Affordable now that an update is three MLP passes and a table lookup."
        ),
    )
    parser.add_argument(
        "--noise-critic-steps", type=int, default=10,
        help=(
            "gradient steps per epoch on `critic_noise` ALONE, each on a **fresh batch**.  The "
            "reference (`dsrl.py:326-335`) runs this as a separate loop with its own "
            "`replay_buffer.sample`, not as part of the main SAC loop -- so the distillation "
            "critic sees 10 independent batches per environment step.  An earlier version here "
            "updated it 1:1 inside the `--utd` loop, which is ~30x fewer distinct batches over the "
            "same wall clock (10 vs 20 updates, but always on the same 128 states).  It matters "
            "because the actor's objective is `critic_noise`, so its coverage is the actor's "
            "coverage."
        ),
    )
    parser.add_argument(
        "--action-magnitude", type=float, default=1.5,
        help=(
            "the noise fed to the sampler is `--action-magnitude * tanh(.)`, while `critic_noise` "
            "receives the tanh output directly.  This is `scale_action` in the reference: the "
            "actor's distribution stays unit-scale so the entropy has a fixed meaning, and the "
            "bound lives outside it.  1.5 is the reference's default for robomimic."
        ),
    )
    parser.add_argument(
        "--target-entropy", type=float, default=0.0,
        help=(
            "SAC target entropy, per the whole action vector (not per dimension).  The reference "
            "sets `target_ent: 0.0` for every task, which is only meaningful with a tanh-squashed "
            "action: the squashed density satisfies log_prob <= 0, so 0 is the deterministic "
            "limit and the entropy controller drives sigma down.  An earlier version used +45.4 "
            "(the differential entropy of N(0,1) in 32 dims) against an *unbounded* actor, where "
            "that equilibrium was unreachable and `alpha` was inert."
        ),
    )
    parser.add_argument(
        "--entropy-in-target", action="store_true",
        help=(
            "include the reference's `- ent_coef * next_log_prob` in the TD target "
            "(`dsrl.py:286`).  **Off by default, and measured to diverge here.**\n\n"
            "The term is a per-bootstrap bonus, so at discount gamma it compounds to "
            "`alpha * |log pi| / (1 - gamma)`.  Measured in this setup: `alpha` starts at 1.0, "
            "`log pi` is about -20.9 (32 dims of a tanh-squashed unit Gaussian), and "
            "gamma = 0.99 -- so the series sums to about 2090 against a true return of 9.56.  "
            "Q went to 131 and the critic loss to 611 within 250 updates before this was found.\n\n"
            "The reference can afford it because its reward scale is the same order as the term: "
            "robomimic pays -1 per step over ~75 steps, so a return of about -75 against a bonus "
            "of 20.  Fixing this properly needs the reward rescaled, not the term removed."
        ),
    )

    parser.add_argument(
        "--check-every",
        type=int,
        default=10,
        help=(
            "epochs between log lines.  A long overnight run needs this small enough to show a "
            "curve and large enough not to bury it: the earlier 1000-epoch probe logged every 10 "
            "and produced 100 lines for one card of information."
        ),
    )
    # The value target is a *discounted* return, and the discount has to be chosen from the
    # episode length rather than copied from a gym config.  Measured on this dataset: episodes run
    # 1382 steps (median 1500) at 20 Hz, the first placement lands at step 226 (median) and the
    # last at 834.
    #
    #     gamma=0.99   half-life   69 steps ( 3 s)   discounted return  0.87  (vs 22.76 undiscounted)
    #     gamma=0.999  half-life  693 steps (35 s)   discounted return 13.83
    #
    # At 0.99 the first placement is worth `0.99^226 = 10 %` of its face value at the start of the
    # episode, and the whole return collapses to 0.87 -- the same order as a randomly initialised
    # Q head, so TD carries no signal and Q just decays from its initialisation.  Observed directly:
    # critic loss fell to 2e-4 within 250 updates with `Q = -0.98`.  0.999 keeps 80 % of that first
    # placement.
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help=(
            "discount, applied at `--step-stride` granularity.  The reference uses 0.99 for "
            "robomimic can, where `max_episode_steps=300` and `act_steps=4` make an episode 75 "
            "RL steps -- so the 69-step half-life covers ~90 % of the episode.  With the default "
            "stride of 16 this dataset's 1382-step episodes become 86 RL steps: the same regime."
        ),
    )
    parser.add_argument(
        "--step-stride", type=int, default=16,
        help=(
            "control steps per RL transition.  This is the single most important number in the "
            "file, and getting it wrong is what made every earlier run fail.\n\n"
            "The reference gets it right structurally: `ActionChunkWrapper.step` advances "
            "`act_steps` environment steps per call, so one RL step IS one action chunk.  "
            "Measured here, consecutive control steps differ by 0.0021 in action while steps 16 "
            "apart differ by 0.0250 -- 11.8x more -- because all 16 slices come from one policy "
            "call on one noise vector.  Treating them as 16 independent (s, a) pairs invents 16 "
            "decisions where the policy made one, and stretches the horizon value must propagate "
            "over by 16x: at stride 1 the 1382-step episode needed a 693-step propagation "
            "distance, which no pool could supply.  At stride 16 it is 69 steps over an 86-step "
            "episode, and gamma 0.99 is correct."
        ),
    )
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sys.path.insert(0, "/tmp/a3-main/src")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import train_pi05_launcher as launcher  # noqa: F401

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device

    # --- frozen policy ----------------------------------------------------------------
    print(f"[dsrl] loading and freezing {args.checkpoint.name}", flush=True)
    policy = PI05Policy.from_pretrained(str(args.checkpoint)).to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    model = policy.model
    preprocess, _ = make_pre_post_processors(
        policy.config, str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    num_steps = int(policy.config.num_inference_steps)
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.max_action_dim)
    noise_dim = chunk_size * action_dim
    # Appendix C.5: the noise space is `(R^d)^C` with C = 50 and d = 32, i.e. 1,600 dims, and
    # "naively training the actor and critic on this high dimensionality is challenging".  The
    # paper's answer is to train a *single-step* actor and critic on R^d and to repeat the sampled
    # noise across the chunk axis when querying the policy.  `full` keeps the 1,600-dim version
    # for comparison; it is what every earlier run here used.
    actor_dim = action_dim if args.noise_mode == "single" else noise_dim

    def repeat_noise(noise: torch.Tensor) -> torch.Tensor:
        """R^d -> (R^d)^C, via the shared implementation used at inference too."""
        return repeat_noise_shared(noise, chunk_size, action_dim, args.noise_mode)

    print(f"[dsrl] {num_steps} steps, chunk {chunk_size} x {action_dim} = {noise_dim} noise dims; "
          f"actor/critic on R^{actor_dim} ({args.noise_mode})", flush=True)

    # --- the prefix, separated from the sampler -----------------------------------------
    # The prefix (three 224x224 PaliGemma encodes plus the language tokens) depends only on the
    # *observation*, never on the noise.  Measured: 155 ms per call, against 127 ms for the ten
    # denoise steps that follow, so recomputing it per sample throws away more than half the work.
    #
    # Splitting it is what makes `--utd > 1` affordable at all: `utd` updates on one batch reuse a
    # single prefix and only pay the 127 ms denoise each time.
    # The prefix and the sampler are the shared implementations from `steering.py`, so the training
    # cache and the deployed inference path cannot disagree about what a noise means.  The wrappers
    # exist only to bind `policy`, `num_steps` and `device`.
    def encode_prefix(prepared: dict):
        return encode_prefix_shared(policy, prepared, keep_embeddings=False)

    def sample_with_prefix(prefix, noise):
        return sample_with_prefix_shared(policy, prefix, noise, num_steps, device)

    # --- offline transitions ------------------------------------------------------------
    dataset = LeRobotDataset(repo_id="dsrl", root=args.dataset, video_backend="pyav")
    labels = pq.read_table(args.dataset / "value_labels.parquet").to_pandas()
    potentials = [np.asarray(p, dtype=np.float64) for p in labels["potential"]]

    frame_table = pq.read_table(
        args.dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "frame_index"],
    ).to_pandas()
    episodes = frame_table["episode_index"].to_numpy(dtype=np.int64)
    frames = frame_table["frame_index"].to_numpy(dtype=np.int64)

    # Reward: the layered potential's own increment. Non-zero rarely, which is the sparse-reward
    # condition DSRL is meant to cope with, so it is reported rather than tuned away.
    rewards = np.zeros(len(episodes), dtype=np.float32)
    dones = np.zeros(len(episodes), dtype=np.float32)
    # `episode_index` in the label file restarts at 1 inside each part, so it is *not* a global
    # dataset index -- the same trap `build_value_labels.py` documents. The row order is the
    # dataset order, so enumerate rather than keying on the column.
    bounds = {}
    offset = 0
    for position, length in enumerate(labels["frames"]):
        bounds[position] = (offset, offset + int(length))
        offset += int(length)
    for episode in range(len(labels)):
        start, stop = bounds[episode]
        path = potentials[episode]
        increments = np.diff(np.concatenate([[0.0], path[: stop - start]]))
        rewards[start:stop] = increments[: stop - start]
        if stop - 1 < len(dones):
            dones[stop - 1] = 1.0
    non_zero = float((rewards != 0).mean())
    print(f"[dsrl] offline transitions: {len(episodes)} rows, reward non-zero on"
          f" {100 * non_zero:.2f} % of frames", flush=True)

    # --- one RL step = one executed action chunk -----------------------------------------
    # `--step-stride` control steps.  The reward of an RL step is the sum of the potential's
    # increments over those control steps; the successor state is the row `stride` later.  Doing
    # this on the control-step table instead is what stretched the horizon 16x and made TD's
    # bootstrap unusable, so the two tables are kept distinct rather than aliased.
    stride = max(1, args.step_stride)
    step_rewards = np.zeros(len(episodes), dtype=np.float32)
    step_dones = np.zeros(len(episodes), dtype=np.float32)
    headroom = np.zeros(len(episodes), dtype=bool)   # is row+stride inside the same episode?
    for episode in range(len(labels)):
        start, stop = bounds[episode]
        for row in range(start, stop):
            end = min(row + stride, stop)
            step_rewards[row] = rewards[row:end].sum()
            if row + stride <= stop - 1:
                headroom[row] = True
            else:
                step_dones[row] = 1.0
    rl_lengths = np.array([len(range(s, t, stride)) for s, t in
                           (bounds[i] for i in range(len(labels)))])
    print(f"[dsrl] RL step = {stride} control steps: episode "
          f"{np.mean(rl_lengths):.0f} RL steps (reference robomimic can: 300/4 = 75), "
          f"{100 * float((step_rewards != 0).mean()):.1f} % of RL steps carry a reward",
          flush=True)
    # Per episode, not over the concatenated array: a single discount ramp across the whole table
    # would multiply across episode boundaries and report a meaningless number.
    per_episode = []
    for episode in range(len(labels)):
        start, stop = bounds[episode]
        path = step_rewards[start:stop:stride]
        per_episode.append(float(sum(path * args.gamma ** np.arange(len(path)))))
    print(f"[dsrl] gamma {args.gamma}: return/episode {np.mean(per_episode):.2f} discounted"
          f" ({100 * np.mean(per_episode) / max(np.mean([r.sum() for r in (rewards[s:t] for s, t in (bounds[i] for i in range(len(labels))))]), 1e-9):.0f} %"
          f" of the {np.mean([r.sum() for r in (rewards[s:t] for s, t in (bounds[i] for i in range(len(labels))))]):.2f}"
          f" undiscounted); half-life {np.log(0.5) / np.log(args.gamma):.0f} RL steps of a"
          f" {np.mean(rl_lengths):.0f}-step episode", flush=True)
    # That number is the return *from the first row*.  The critic is trained on a pool spread
    # across whole episodes, so the pool's own mean return is far lower and is the number to
    # compare any `Q` against -- comparing `Q` to the episode-start return would read as a large
    # underestimate when it is not.  Printed after the pool is chosen, below.

    # The Monte-Carlo return, `G(s) = r + gamma * G(s')` accumulated backwards over the full trace
    # and reset at every episode end.  This is the `--critic-target mc` target.
    #
    # It exists because TD cannot work on a scattered pool, and that is a fact about the pool, not
    # about the code: `mujoco` rows drawn independently produce 3,687 contiguous runs of mean
    # length 1.11 in a 4,096-row pool, so `Q(s)` can only ever be shaped by `Q(s+1)`, which is in
    # turn unanchored.  Carrying a reward back ~693 steps (the 0.999 half-life) needs a chain, and
    # a scatter has none.  `probe_value_ceiling.py` measures the consequence on identical features:
    # MC R^2 +0.55, TD R^2 -1.17 on the scattered pool.
    #
    # The cost of using MC is that it is only unbiased for the *behaviour* policy: it evaluates
    # `pi_dp + actor_noise` as it was recorded, so it cannot credit an action nobody took.  That is
    # acceptable here for the same reason the whole offline plan is: the actor searches the noise
    # space of a policy that already solves the task, and the recorded noise *is* standard normal.
    return_all = np.zeros(len(episodes), dtype=np.float64)
    # Computed at **every row**, not on the `range(start, stop, stride)` grid.
    #
    # The grid version had a phase bug that was invisible in the logs and cost this run its
    # accuracy: the pool draws rows from `adjacent`, which only requires `frame_index` to advance
    # by `stride`, so a pool row can sit at any phase relative to its episode's first frame.  A
    # backward recursion over `range(start, stop, stride)` only fills the rows congruent to
    # `start`, so **15 of every 16 rows read as `G = 0`**.  Since `G` is exactly what the segment
    # tails regress on, 63 of the 64 anchors were told the answer was zero.  Measured on the
    # resulting pool: true `G` had median 0.00 and p90 0.00 while the segment interiors plainly
    # contained rewards (4 in one case) -- an inconsistency that only showed up when the pool and
    # the return table were compared directly.
    #
    # Row-wise it is unambiguous: `G[r] = r_step + gamma * G[r + stride]` while `r + stride` is in
    # the same episode, else `G[r] = r_step`.  `step_rewards[r]` is the sum of the potential's
    # increments over rows `r .. r+stride-1`, which is the reward of the decision taken at `r`.
    episode_end = np.zeros(len(episodes), dtype=np.int64)
    for episode in range(len(labels)):
        episode_end[episode] = bounds[episode][1]
    for row in range(len(episodes) - 1, -1, -1):
        successor = row + stride
        if successor < episode_end[episodes[row]]:
            return_all[row] = float(step_rewards[row]) + args.gamma * return_all[successor]
        else:
            return_all[row] = float(step_rewards[row])
    non_trivial = float((return_all != 0).mean())
    print(f"[dsrl] Monte-Carlo return defined on every row; non-zero on"
          f" {100 * non_trivial:.1f} % of them (mean {return_all.mean():.2f})", flush=True)

    # The visual half of the observation: the same frozen SigLIP summary the value head used,
    # reduced so the actor and critics stay small.  The paper's "visual features" are a shallow
    # convolutional encoder over the three cameras; this is the frozen summary that already exists
    # for every frame, which keeps the sensor path out of the loop entirely.
    features_dir = args.dataset / "features_v9"
    written = np.load(features_dir / "written.npy")
    cached_rows = np.flatnonzero(written)
    projection_dim = 256
    projection = rng.normal(0.0, 1.0 / np.sqrt(projection_dim), (8192, projection_dim))
    # Deliberately *not* called `stride`: that name is the RL step length above, and an earlier
    # version reassigned it here, silently turning every `row + stride` into `row + 5`.
    feature_stride = int(json.loads((features_dir / "index.json").read_text())["stride"])
    cache = np.load(features_dir / "features.npy", mmap_mode="r")[cached_rows].astype(np.float32) @ projection

    # Fill every row from its nearest *preceding* cached row.
    #
    # The previous version read
    #     obs_by_frame[cached_rows[valid]] = np.where(obs_by_frame[cached_rows[valid]] ...)
    # -- `cached_rows` as both source and destination -- so every destination already held a
    # value and the `np.where` kept it.  The loop was a no-op, and because `features_v9` is
    # written every `feature_stride` rows, **80 % of the observation table stayed exactly zero**.
    # Every earlier DSRL run in this repository trained its actor and both critics on a zero
    # observation on four sampled transitions out of five.  `searchsorted` is what the comment
    # above the old loop always claimed the code did.
    nearest = np.searchsorted(cached_rows, np.arange(len(written)), side="right") - 1
    nearest = np.clip(nearest, 0, len(cached_rows) - 1)
    obs_by_frame = cache[nearest]
    print(f"[dsrl] visual features {projection_dim} dims; "
          f"{np.count_nonzero(obs_by_frame.any(axis=1))}/{len(written)} rows non-zero "
          f"(feature stride {feature_stride}, filled forward)", flush=True)

    # --- transitions: adjacent row pairs ------------------------------------------------
    # A transition needs its successor's features as well.  Sampling rows independently and then
    # looking up `row + stride` would need a second, disjoint decode of the successors; drawing the
    # pool as *pairs* gives every successor for the same cost.  `frames` is the `frame_index`
    # column, so a pair is usable exactly when it lands `stride` rows later in the same episode.
    lead = stride
    if lead == 1:
        adjacent = np.flatnonzero((frames[1:] == frames[:-1] + 1) & (episodes[1:] == episodes[:-1]))
    else:
        same_episode = episodes[lead:] == episodes[:-lead]
        contiguous = frames[lead:] == frames[:-lead] + lead
        adjacent = np.flatnonzero(same_episode & contiguous)
    adjacent_set = set(adjacent.tolist())

    # Every reward-carrying transition is in the pool, and the rest of the budget is filled at
    # random.  This is not a detail: a uniformly drawn pool of 512 transitions holds almost none
    # of them, and TD then has nothing to anchor to -- measured, and it is why the critic loss once
    # fell to 2e-4 with Q pinned near its initialisation.
    reward_rows = np.intersect1d(adjacent, np.flatnonzero(step_rewards != 0))
    budget = int(min(args.transitions, len(adjacent)))
    if args.critic_target == "td" and args.segment_length > 1:
        # Contiguous segments, measured in **RL steps**: a segment of `L` RL steps spans
        # `(L-1)*stride + 1` rows.  Earlier this used rows directly, so a "128-step segment" at
        # stride 16 would have covered only 8 RL steps -- 1/9 of the 69-step half-life.
        length = int(args.segment_length)              # in RL steps
        span = (length - 1) * stride + 1               # in rows
        usable = adjacent[adjacent + span - 1 < len(episodes)]
        usable = usable[[int(s) + span - 2 in adjacent_set for s in usable]]
        # Anchored segments first, then random fill.  An earlier version computed the anchored set,
        # *printed its size*, and then shuffled `union(anchored, usable)` -- which is just
        # `usable`, so no segment was actually preferred and the reported count described a pool
        # that was never built.
        anchored = np.intersect1d(usable, reward_rows - (length - 1) * stride)
        rng.shuffle(anchored)
        chosen: list[int] = []
        for start in anchored:
            chosen.extend(range(int(start), int(start) + span, stride))
            if len(chosen) >= budget:
                break
        if len(chosen) < budget:
            spare = np.setdiff1d(usable, anchored, assume_unique=False)
            rng.shuffle(spare)
            for start in spare:
                chosen.extend(range(int(start), int(start) + span, stride))
                if len(chosen) >= budget:
                    break
        starts = np.array(sorted(set(chosen)), dtype=np.int64)
        print(f"[dsrl] pool: {len(starts)} transitions in {len(chosen) // length} contiguous"
              f" segments of {length} RL steps ({span} rows) each, all anchored at a reward row"
              f" ({len(anchored)} anchor candidates of {len(usable)} starts);"
              f" half-life {np.log(0.5) / np.log(args.gamma):.0f} RL steps", flush=True)
    elif len(reward_rows) > budget:
        print(f"[dsrl] {len(reward_rows)} reward transitions exceed the pool budget {budget};"
              f" keeping a random {budget}", flush=True)
        starts = np.sort(rng.choice(reward_rows, size=budget, replace=False))
        anchors = budget
    else:
        anchors = len(reward_rows)
        rest = np.setdiff1d(adjacent, reward_rows, assume_unique=False)
        fill = np.sort(rng.choice(rest, size=min(budget - anchors, len(rest)), replace=False))
        starts = np.sort(np.concatenate([reward_rows, fill]))
        print(f"[dsrl] pool: {anchors} reward-carrying + {len(fill)} random = {len(starts)}"
              f" of {len(adjacent)} adjacent pairs"
              f" ({anchors / max(len(starts), 1):.1%} anchors vs {100 * non_zero:.2f} % base rate)",
              flush=True)
    transitions = int(len(starts))
    rows_cur = starts
    rows_next = starts + stride

    # Which pool rows are segment *tails*, i.e. their successor is not itself a training row.
    #
    # Those are the rows whose bootstrapped target has nothing to anchor it: `Q(s')` there is an
    # extrapolation, and TD is free to place it anywhere.  Measured on a scattered pool, that
    # freedom is not benign -- Q climbed 1.8 -> 25.0 with `minQ` at 49.7 against a true maximum of
    # 26.9, and the actor loss fell -46 -> -96 chasing it.  On an all-anchored 32x128 segment pool
    # the opposite failure appeared: Q settled at -1.6, the fixed point of `Q = 0.999 Q` on the
    # reward-free rows that dominate each chain's interior.
    #
    # The fix is to make the tail's target exact instead of bootstrapped: `Q(tail) = G(tail)`,
    # where G is the Monte-Carlo return we already have for every row.  Inside a segment TD still
    # propagates normally, so the critic keeps its action dependence (the thing MC alone loses);
    # at the boundary it inherits the return, so it has nothing to invent.
    pool_set = set(rows_cur.tolist())
    tail_mask = np.array([int(r) + stride not in pool_set for r in rows_cur], dtype=bool)

    # The actions the behaviour policy actually took, padded to the model's `max_action_dim`
    # exactly as `PI05Pytorch.prepare_action` does.  `chunk_size` rows per transition are taken
    # from the episode's own future, clamped at the episode end so a chunk never spans episodes.
    # Only the cached transitions are materialised: building this for all 552 k rows cost 3.5 GB
    # of device memory and the loop never looked at any of it.
    action_values = np.stack(
        pq.read_table(
            args.dataset / "data" / "chunk-000" / "file-000.parquet", columns=["action"]
        ).to_pandas()["action"].values
    )[:, :16].astype(np.float32)

    def recorded_chunks(rows: np.ndarray) -> torch.Tensor:
        out = np.zeros((len(rows), chunk_size, action_dim), dtype=np.float32)
        for position, row in enumerate(rows):
            row = int(row)
            _, stop = bounds[int(episodes[row])]
            for offset in range(chunk_size):
                out[position, offset, :16] = action_values[min(row + offset, stop - 1)]
        return torch.from_numpy(out)

    action_flat = recorded_chunks(rows_cur).to(device).reshape(transitions, -1)
    action_flat_next = recorded_chunks(rows_next).to(device).reshape(transitions, -1)

    # --- the two caches -----------------------------------------------------------------
    # Everything the noise policy consumes is a per-frame constant, so the whole VLA side of this
    # script is one batched pass at startup.  Both files are keyed by the checkpoint and by the
    # pool geometry so a rerun reuses them; `--regenerate-cache` forces a rebuild.
    ckpt_key = "_".join(args.checkpoint.parts[-4:-1])
    # The pool's *geometry* belongs in the key, not just its size.  Measured the hard way: a run
    # with `--segment-length 128` produced 4,096 rows exactly like the scattered run, reused
    # `obs_..._t4096_k4_s0.pt` because the signature matched, and then trained on the scattered
    # pool's features while indexing the segment pool's rewards and actions -- a silent mismatch
    # between three tables that each looked right in isolation.  The row set is hashed in so two
    # different pools can never share a file again.
    pool_hash = zlib.crc32(np.ascontiguousarray(rows_cur, dtype=np.int64).tobytes()) & 0xFFFFFFFF
    signature = (
        f"{ckpt_key}_{args.noise_mode}_t{transitions}_k{args.distill_per_frame}"
        f"_s{args.seed}_p{pool_hash:08x}"
    )
    cache_dir = args.cache_dir if args.cache_dir is not None else args.dataset / "dsrl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_file = cache_dir / f"obs_{signature}.pt"
    distill_file = cache_dir / f"distill_{signature}.pt"

    def raw_batch(rows: np.ndarray) -> dict:
        """The dict `preprocess` expects, for a batch of dataset rows."""
        items = [dataset[int(row)] for row in rows]
        raw = {key: torch.stack([item[key] for item in items])
               for key in dataset.meta.features if key.startswith("observation.images.")}
        raw["observation.state"] = torch.stack([item["observation.state"] for item in items])
        if "observation.history" in items[0]:
            raw["observation.history"] = torch.stack(
                [item["observation.history"] for item in items]
            )
        raw["task"] = [item["task"] for item in items]
        return raw

    def featurise(rows: np.ndarray, per_row: int):
        """One batched VLA pass over `rows`.

        Returns the paper's 2,048-dim VLM feature, the normalised proprioception, and -- when
        `per_row > 0` -- `per_row` `(epsilon, pi(s, epsilon))` pairs per row, generated against the
        prefix that is already in hand.  Running the sampler in a second pass over the images
        would pay the 132 ms prefix forward twice.
        """
        probes, proprios, all_chunks, all_noises = [], [], [], []
        started_pass = time.monotonic()
        for start in range(0, len(rows), args.feature_batch):
            index = rows[start:start + args.feature_batch]
            prepared = preprocess(raw_batch(index))
            prefix = encode_prefix(prepared)
            probes.append(prefix.hidden_last.detach().cpu())
            proprios.append(
                prepared.get("observation.state", torch.zeros(len(index), 16)).detach().cpu().float()
            )
            if per_row:
                batch_chunks, batch_noises = [], []
                for _ in range(per_row):
                    epsilon = torch.randn(len(index), actor_dim, device=device)
                    chunk = sample_with_prefix(prefix, repeat_noise(epsilon))
                    batch_chunks.append(chunk.reshape(len(index), -1).float().cpu())
                    batch_noises.append(epsilon.cpu())
                all_chunks.append(torch.stack(batch_chunks, dim=1))   # (B, K, C*d)
                all_noises.append(torch.stack(batch_noises, dim=1))   # (B, K, d)
            del prefix
            done = min(start + args.feature_batch, len(rows))
            if start // args.feature_batch % 16 == 0:
                elapsed = time.monotonic() - started_pass
                rate = elapsed / max(done, 1)
                print(f"    {done}/{len(rows)} rows  {elapsed:.0f}s  "
                      f"(eta {rate * (len(rows) - done):.0f}s)", flush=True)
        return (
            torch.cat(probes, dim=0),
            torch.cat(proprios, dim=0),
            torch.cat(all_chunks, dim=0) if all_chunks else None,
            torch.cat(all_noises, dim=0) if all_noises else None,
        )

    if feature_file.exists() and distill_file.exists() and not args.regenerate_cache:
        print(f"[dsrl] reusing {feature_file.name}", flush=True)
        cached = torch.load(feature_file, weights_only=True)
        noise_table, chunk_table = torch.load(distill_file, weights_only=True)
        probe_cur, proprio_cur = cached["probe_cur"], cached["proprio_cur"]
        probe_next, proprio_next = cached["probe_next"], cached["proprio_next"]
        del cached
    else:
        frames_to_decode = 2 * transitions
        print(f"[dsrl] one VLA pass over {frames_to_decode} frames "
              f"(decode ~45 ms each, then never again)", flush=True)
        probe_cur, proprio_cur, chunk_table, noise_table = featurise(
            rows_cur, args.distill_per_frame
        )
        probe_next, proprio_next, _, _ = featurise(rows_next, 0)
        torch.save(
            {
                "probe_cur": probe_cur, "proprio_cur": proprio_cur,
                "probe_next": probe_next, "proprio_next": proprio_next,
                # As tensors, not numpy, so the file loads under `weights_only=True`.
                "rows_cur": torch.from_numpy(rows_cur).long(),
                "rows_next": torch.from_numpy(rows_next).long(),
            },
            feature_file,
        )
        torch.save((noise_table, chunk_table), distill_file)
        print(f"[dsrl] cache written to {cache_dir}", flush=True)

    noise_table = noise_table.to(device)
    chunk_table = chunk_table.to(device)

    observation_parts = {
        "proprio": args.obs_proprio, "vlm": args.obs_vlm, "visual": args.obs_visual,
    }

    def build_observation(probe: torch.Tensor, proprio: torch.Tensor, rows: np.ndarray):
        """Delegates to `steering.build_observation`, which fixes the concatenation order.

        The order has to match the checkpoint exactly: the actor's first `Linear` has a fixed
        width, so a re-ordered pair of blocks is a same-shape, wrong-meaning input -- the failure
        mode with no error message.
        """
        return build_observation_shared(
            proprio.float(), probe.float(), torch.from_numpy(obs_by_frame[rows]),
            observation_parts,
        ).to(device)

    obs_cur = build_observation(probe_cur, proprio_cur, rows_cur)
    obs_next = build_observation(probe_next, proprio_next, rows_next)
    del probe_cur, probe_next, proprio_cur, proprio_next
    observation_dim = obs_cur.shape[1]
    print(f"[dsrl] observation {observation_dim} dims "
          f"(proprio {args.obs_proprio}, vlm {args.obs_vlm}, visual {args.obs_visual}); "
          f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated", flush=True)

    # `step_rewards`/`step_dones` are the RL-step tables (`stride` control steps summed), not the
    # per-control-step ones: using the latter here would leave 15 of every 16 rewards unclaimed.
    reward_tensor = torch.from_numpy(step_rewards[rows_cur]).to(device)
    done_tensor = torch.from_numpy(step_dones[rows_cur]).to(device)
    return_tensor = torch.from_numpy(return_all[rows_cur].astype(np.float32)).to(device)
    tail_tensor = torch.from_numpy(tail_mask).to(device)
    print(f"[dsrl] pool has {int(tail_mask.sum())} segment tails"
          f" ({100 * tail_mask.mean():.0f} %); those regress on the exact return,"
          f" the rest bootstrap", flush=True)
    # Self-checks on the return table, because three separate bugs in this file were only visible
    # by comparing the pool against the anchor it regresses on:
    #   * the phase bug left 15/16 of the anchors reading G = 0;
    #   * a stray `stride` reassignment turned `row + stride` into `row + 5`;
    #   * a partial tail mask anchored only a fraction of the segments.
    # All three produce a curve that looks like learning.  These two lines make them loud.
    pool_return = return_all[rows_cur]
    tail_return = return_all[rows_cur[tail_mask]] if tail_mask.any() else np.zeros(1)
    print(f"[dsrl] pool return: mean {pool_return.mean():.2f}  median"
          f" {np.median(pool_return):.2f}  p90 {np.percentile(pool_return, 90):.2f}"
          f"  non-zero {100 * float((pool_return != 0).mean()):.0f} %", flush=True)
    print(f"[dsrl] anchor return (the tails): mean {tail_return.mean():.2f}"
          f"  median {np.median(tail_return):.2f}  non-zero"
          f" {100 * float((tail_return != 0).mean()):.0f} %  <- must not be ~0", flush=True)
    if tail_mask.any() and float((tail_return != 0).mean()) < 0.5:
        print("[dsrl] WARNING: most segment tails have a zero return.  Either the segments are not"
              " anchored on reward rows, or the return table is misaligned with the pool.  Q will"
              " converge to a truncated value and the actor will optimise that.", flush=True)
    if args.critic_target == "mc":
        print(f"[dsrl] critic target: Monte-Carlo (mean {float(return_tensor.mean()):.2f},"
              f" std {float(return_tensor.std()):.2f}, max {float(return_tensor.max()):.2f})",
              flush=True)
    else:
        print(f"[dsrl] critic target: TD, gamma {args.gamma}", flush=True)

    # --- networks -----------------------------------------------------------------------
    # `critic` is Q(s, a) on the *real* 1,600-dim action chunk; `critic_noise` is Q(s, u) in the
    # actor's unit-scale u = tanh(.) space.  The asymmetry is the point: the TD critic has to see
    # real actions, the actor's critic has to live where the actor lives.
    actor = NoiseActor(observation_dim, actor_dim, hidden=args.hidden).to(device)
    critic = twin_critic(observation_dim, noise_dim, hidden=args.hidden).to(device)
    critic_target = twin_critic(observation_dim, noise_dim, hidden=args.hidden).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_noise = twin_critic(observation_dim, actor_dim, hidden=args.hidden).to(device)
    print(f"[dsrl] actor/critic width {args.hidden}, "
          f"actor {sum(p.numel() for p in actor.parameters()) / 1e6:.2f}M params, "
          f"critic {sum(p.numel() for p in critic.parameters()) / 1e6:.2f}M x2 heads", flush=True)

    # `Adam`, not `AdamW`.  The reference inherits SB3's `th.optim.Adam` (`dsrl.py:175,199`), and
    # the difference is not cosmetic over a long run: `AdamW` decouples a weight-decay term that
    # `Adam` does not apply at all (SB3 passes no `weight_decay`, so it is zero).  With 8000 epochs
    # of updates that is a persistent pull toward zero that the reference never has.
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
    noise_critic_optimizer = torch.optim.Adam(critic_noise.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.lr)

    # The reference's `target_ent: 0.0`, for every task.  With a tanh-squashed action the density
    # satisfies `log_prob <= 0` (a density on a bounded interval cannot exceed the uniform one),
    # so 0 is the deterministic limit and the controller drives sigma down.  Comparing against an
    # *unbounded* actor is what made an earlier version's target meaningless: there `log_prob` is
    # `-H`, the equilibrium is at `log_prob = +45.4`, and `alpha` marched to zero instead.
    target_entropy = float(args.target_entropy)

    # --- training loop ------------------------------------------------------------------
    # Nothing below touches the VLA: `obs_cur`/`obs_next` are the cached paper features,
    # `action_flat*` are the recorded chunks, and `noise_table`/`chunk_table` are the cached
    # `(epsilon, pi(s, epsilon))` pairs.  An update is therefore three MLP passes and a gather.
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        # Drawn with replacement: the table is small enough that without replacement would make
        # consecutive batches share nothing but also make each batch a permutation of the last.
        index = torch.randint(0, transitions, (args.batch,), device=device)
        state = obs_cur[index]
        next_state = obs_next[index]
        reward_now = reward_tensor[index]
        done_now = done_tensor[index]
        recorded_flat = action_flat[index]

        with torch.no_grad():
            if args.critic_target == "mc":
                # A plain regression target: no bootstrap, no target network, no chain to
                # propagate along.  `critic_target` is kept in step anyway so the two modes share
                # the rest of the loop and so `min(.)` over two heads still applies.
                target_q = return_tensor[index].unsqueeze(-1)
            else:
                # Bootstrap on the *next* recorded action rather than on a re-sampled one: this is
                # the offline setting and the next state's action is known.  Resampling would
                # inject the actor's current opinion into its own target, which is how Q-learning
                # diverges without ever being wrong on its own loss.  (The reference can resample
                # because it is online; §5.3 is the offline-to-online case and reports that the
                # naive offline baselines "are not able to learn at all" there.)
                next_q = torch.min(
                    torch.stack(
                        [head(next_state, action_flat_next[index]) for head in critic_target]
                    ),
                    dim=0,
                ).values
                # The reference's target is a **soft** value: `dsrl.py:286` subtracts
                # `ent_coef * next_log_prob` before the bootstrapping multiply, so Q estimates the
                # return *plus* the entropy of the policy it is bootstrapping from.  Omitting that
                # term -- which an earlier version did -- makes Q a hard value and silently changes
                # what the actor is climbing toward.
                #
                # What is still not identical: the reference evaluates Q on
                # `diffusion_policy(s', actor_noise(s'))`, i.e. the action the *current* actor
                # would take.  This offline version evaluates it on the recorded action, and the
                # entropy is taken from the actor at `s'`.  So the term is the reference's, applied
                # to a slightly different action -- a documented residual, not an oversight.
                if args.entropy_in_target:
                    _, next_log_prob = actor.sample(next_state)
                    next_q = next_q - log_alpha.exp().detach() * next_log_prob
                target_q = (reward_now.unsqueeze(-1)
                            + args.gamma * (1.0 - done_now.unsqueeze(-1)) * next_q)
                # At a segment tail the bootstrap is replaced by the exact return.  `torch.where`
                # rather than an index assignment so the mask stays on the same device and the
                # whole target is one expression that can be read at once.
                tail_now = tail_tensor[index].unsqueeze(-1)
                target_q = torch.where(tail_now, return_tensor[index].unsqueeze(-1), target_q)

        # --- `--utd` updates on this batch ----------------------------------------------
        # Appendix B: UTD >= 20, i.e. update the actor and both critics once per parallel-env
        # step.  This used to cost a 129 ms denoise per update because the distillation target
        # had to be sampled; now it is a lookup, so the paper's setting is the default rather
        # than an aspiration.
        critic_loss = noise_critic_loss = actor_loss = None
        for _update in range(max(1, args.utd)):
            # --- critic: TD on the actions actually recorded -----------------------------
            # The reference (`dsrl.py:292`) fits `critic` to `replay_data.actions` -- the action the
            # behaviour policy really took.  An earlier version of this file fed it
            # `sampler(actor_noise)` instead, which made the critic's training distribution depend
            # on the actor it is supposed to evaluate: a moving target with no fixed reference.
            q_values = torch.stack([head(state, recorded_flat) for head in critic])
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in q_values)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            critic_optimizer.step()

            # --- noise critic: distilled against the cached sampler outputs --------------
            # `dsrl.py:349-363` draws an independent `noise_actions = randn(...)`, evaluates
            # `critic(obs, sampler(noise_actions))`, and fits `critic_noise(obs, noise_actions)`
            # to it.  The point is coverage: a critic that only sees the actor's current noise
            # cannot tell the actor which direction to move.  Building `sampler(noise)` needs the
            # denoiser, so those pairs were generated in `featurise` and are gathered here; each
            # update draws a different one of the `--distill-per-frame` cached noises.
            #
            # `critic_noise` takes the *unit-scale* value: the cache holds the raw standard-normal
            # epsilon that went into the sampler, and `scale_action` in the reference divides by
            # the magnitude before the critic sees it.  Same here.
            pick = torch.randint(0, args.distill_per_frame, (args.batch,), device=device)
            probe_noise = noise_table[index, pick] / args.action_magnitude   # (batch, R^d)
            probe_action = chunk_table[index, pick]         # (batch, C*d) real action chunks
            with torch.no_grad():
                probe_q = torch.stack([head(state, probe_action) for head in critic]).detach()
            noise_q = torch.stack([head(state, probe_noise) for head in critic_noise])
            noise_critic_loss = 0.5 * sum(
                F.mse_loss(value, target) for value, target in zip(noise_q, probe_q)
            )
            noise_critic_optimizer.zero_grad()
            noise_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_noise.parameters(), 10.0)
            noise_critic_optimizer.step()

            # --- actor: maximise the noise critic, with SAC entropy -----------------------
            # The actor never needs gradients through the sampler: `critic_noise` takes the
            # noise directly.  That is the reason DSRL keeps a second critic, and the reason the
            # 0.63 s/step measured in `measure_dsrl_feasibility.py` does not have to be paid.
            #
            # `sampled` is `tanh(gaussian)` in [-1, 1] and `log_prob` is the density of *that*
            # variable, Jacobian included (see `NoiseActor.sample`).  The sampler receives
            # `--action-magnitude * sampled`; `critic_noise` sees `sampled`.  This split is
            # `scale_action`, and it is what lets `--target-entropy 0` mean something.
            sampled, log_prob = actor.sample(state)
            alpha = log_alpha.exp().detach()
            min_q = torch.min(
                torch.stack([head(state, sampled) for head in critic_noise]), dim=0
            ).values
            actor_loss = (alpha * log_prob - min_q).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            # --- target network, **inside** the loop -----------------------------------
            # The reference does this on every gradient step (`dsrl.py:323`, with
            # `target_update_interval = 1`), i.e. `utd` times per environment step.  An earlier
            # version here updated once per epoch, which is `utd = 20` times slower: the effective
            # rate per epoch was 0.005 against the reference's `1 - 0.995^20 = 0.095`.
            #
            # That discrepancy is very likely why `Q` needed thousands of epochs to climb even
            # after the return table was fixed, and why the climb was attributed to "the chain is
            # too long".  The chain length was real, but the target network was moving 19x slower
            # than the one that produced the published results.
            #
            # With a Monte-Carlo target nothing reads `critic_target`; the update is kept anyway so
            # the two modes stay structurally identical and a mid-run switch is meaningful.
            soft_update(critic_target, critic, args.tau)

        # --- noise critic on its own, 10 independent batches -----------------------------
        # The reference runs this as a *separate* loop after the SAC steps
        # (`dsrl.py:326-335`), each iteration calling `replay_buffer.sample` again.  Doing it
        # inside the loop above -- the earlier version -- gave it the same `--utd` states every
        # time, so over one epoch it saw 128 states 20 times instead of 1280 states once.  The
        # actor's objective is this critic, so its coverage is the actor's coverage.
        for _ in range(max(1, args.noise_critic_steps)):
            fresh = torch.randint(0, transitions, (args.batch,), device=device)
            pick = torch.randint(0, args.distill_per_frame, (args.batch,), device=device)
            with torch.no_grad():
                target = torch.stack(
                    [head(obs_cur[fresh], chunk_table[fresh, pick]) for head in critic]
                ).detach()
            value = torch.stack(
                [head(obs_cur[fresh], noise_table[fresh, pick] / args.action_magnitude)
                 for head in critic_noise]
            )
            noise_critic_loss = 0.5 * sum(
                F.mse_loss(v, t) for v, t in zip(value, target)
            )
            noise_critic_optimizer.zero_grad()
            noise_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_noise.parameters(), 10.0)
            noise_critic_optimizer.step()


        if epoch % args.check_every == 0 or epoch == 1:
            elapsed = time.monotonic() - started
            # `|u|` and `sigma` replace the old `|mean|`: the actor's output is now `tanh` of a
            # Gaussian mean, so `|u| -> 1` is the saturation signal that used to be
            # `|mean| -> clip`, and `sigma` is learned rather than fixed.
            sigma = float(actor.log_std.detach().clamp(-20.0, 2.0).exp().mean())
            print(f"[dsrl] {epoch:4d}  critic {float(critic_loss.detach()):8.4f}"
                  f"  noise_critic {float(noise_critic_loss.detach()):8.4f}"
                  f"  actor {float(actor_loss.detach()):8.4f}  alpha {float(alpha):.4f}"
                  f"  Q {float(q_values.mean()):7.3f}"
                  f"  logp {float(log_prob.mean()):+8.1f}"
                  f"  minQ {float(min_q.mean()):7.3f}"
                  f"  |u| {float(sampled.abs().mean()):.3f}  sigma {sigma:.3f}"
                  f"  ({1000 * elapsed / epoch:.1f} ms/epoch)", flush=True)
        history.append({
            "epoch": epoch,
            "critic_loss": float(critic_loss.detach()),
            "noise_critic_loss": float(noise_critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "q_mean": float(q_values.mean().detach()),
            "alpha": float(alpha),
        })

    print()
    print(f"[dsrl] done in {(time.monotonic() - started) / 60:.1f} min")
    print(f"[dsrl] Q went {history[0]['q_mean']:+.3f} -> {history[-1]['q_mean']:+.3f}")
    print(f"[dsrl] actor loss {history[0]['actor_loss']:+.3f} -> {history[-1]['actor_loss']:+.3f}")
    print()
    print("判读：")
    print("  actor loss 下降 且 Q 上升 = actor 找到了噪声空间里的高价值区域。")
    print("  actor loss 下降 但 Q 不升 = critic 平坦，actor 在拟合噪声（预期结果，见 9.85）。")
    print("  Q 持续上升而 actor loss 不动 = critic 在自我放大，蒸馏链断裂。")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # The noise mode and the observation layout are part of the actor's contract: an actor
        # trained on `single` noise emits R^32 that has to be repeated across the chunk axis
        # before it reaches the sampler, and one trained with `--no-vlm` cannot be evaluated
        # without the same feature layout.  Saving them with the weights is cheaper than
        # rediscovering the mismatch at inference time.
        torch.save({
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "critic_noise": critic_noise.state_dict(),
            "projection": projection,
            "noise_mode": args.noise_mode,
            "actor_dim": actor_dim,
            "chunk_size": chunk_size,
            "action_dim": action_dim,
            "observation_dim": observation_dim,
            "hidden": args.hidden,
            "observation_parts": {
                "proprio": args.obs_proprio,
                "vlm": args.obs_vlm,
                "visual": args.obs_visual,
            },
        }, args.output)
        (args.output.parent / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
