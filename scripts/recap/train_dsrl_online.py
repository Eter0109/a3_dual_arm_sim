#!/usr/bin/env python3
"""DSRL as the paper does it: online, on the environment, with the VLA in the loop.

Why this file replaces the offline trainer
------------------------------------------
`train_dsrl.py` learned the noise policy from a fixed pool of recorded transitions.  That pool has
**one action per state**, so the critic could only ever fit that one action: it can be numerically
accurate and still be worthless at `argmax`.  `probe_critic_ranking.py` measured exactly that --
within-state pairwise accuracy 0.508 (v5) and 0.508 (v6), both indistinguishable from chance on 823
discordant pairs, bootstrapped over 45 points.

Chance is the correct answer for that experiment, and not a bug to be fixed: ordering two actions at
one state requires having seen two actions at that state, and offline data never does.  The paper
does not train this way either -- §5.1 is fully online, and §5.3 says the naive offline baselines
are "not able to learn at all".  This file is the pipeline that can.

What made online look expensive, and why it is not
--------------------------------------------------
Measured on this stack before writing anything:

* `mj_step` is 1.12 ms, but a control step runs **50 substeps** (`timestep=0.001`, control 50 ms),
  so one control step is ~61 ms of physics.  That is the benchmark's own configuration and cannot
  be shortened without changing the dynamics being trained on.
* MuJoCo **releases the GIL**: four scenes stepping in threads run 7.94x faster than sequentially
  and produce **bit-identical** trajectories.  So physics parallelises.
* Rendering **cannot** go in a worker thread: MuJoCo keeps one global EGL context and a second
  thread rendering at once dies with `EGL_BAD_ACCESS`.  So the loop steps physics with
  `render_cameras=False` and renders on the main thread, which was verified to reproduce the
  environment's own images exactly (max|diff| = 0 on all three cameras).
* Batching the policy across scenes is strongly sublinear: 183 ms at batch 1, 299 ms at 4, **437 ms
  at 8**.  So eight scenes cost roughly one forward.

Throughput, measured and not guessed
------------------------------------
The physics is the whole budget, and it is not the constant the first measurement suggested.  One
control step cost 61 ms in a *synthetic* rollout that stayed near the start pose, and 61 ms is what
the first round of the real loop cost too -- 1,164 ms of physics for eight scenes, i.e. perfect
parallelism.  By mid-episode the same 16 steps cost 4-9 s.

`probe_step_slowdown.py` settled the cause by replaying saved `(qpos, qvel, ctrl)` states: the
slowest six steps cost 570.9 ms in `mj_step` alone against 580.7 ms originally, while the fastest six
cost 57.3 against 60.3.  So 95-98 % of a control step is the solver, and it is the solver that gets
slower.  `nefc` rises only 1,980 -> 2,379 (+20 %) for that 10x, so the row count is not the driver;
Newton with `iterations=100` simply does not converge as fast once the gripper is dragging a heap of
80 free bodies.

Consequences, all of which the earlier estimate got wrong:

* **Parallelism works, but it does not multiply the work.**  Eight scenes stepping concurrently
  cost about what *one* scene's own 16 steps cost -- which is why the first round of a fresh episode
  reports `physics 1.2 s` for all eight and the eighth round reports 4-9 s.  Seven to eight times
  the throughput of one scene, and no more.
* **A round costs 1.2 s at the start of an episode, 10-13 s mid-episode, and ~30 s on the round**
  **where episodes end**, because ending an episode closes eight environments and builds eight
  fresh ones (1.36 s each, serial) so training sees the same fresh-scene distribution the
  evaluation does. Pulled together: **11.2 s per round on an otherwise idle machine**, i.e. 11.4
  scene-steps/s and roughly **320 episodes/hour** at `--scenes 8`. An episode is 17.5 minutes, so a
  340-minute run reaches ~19 episode-equivalents (~155 episodes across the eight scenes).

  The first version of this docstring said 18.3 s per round and 135 episodes/hour. That was
  measured while three probes were running on the same machine, competing for the same sixteen
  cores, so it was a measurement of contention rather than of the loop. The rule this project keeps
  relearning: a number is meaningless without stating what else was running when it was taken.
* **Gradient steps are free.**  At `--utd 20 --batch 256` an update batch costs 120-400 ms, all of it
  hidden inside the physics wait.  `--no-train` and the default differ in Q's accuracy, never in
  wall time, so there is no reason to lower `--utd`.

What is deliberately identical to `train_dsrl.py`
------------------------------------------------
The update rules are not re-derived here.  Both critics, the actor objective, the learned `alpha`,
`soft_update` inside the uTD loop, `Adam`, width 2048, `Tanh`, the `repeat_noise` chunk mapping and
the `scale_action` split all come from the offline trainer, which was itself aligned against
`refs/sb3-dsrl`.  Only the *source of data* changes.

Diff against `refs/sb3-dsrl` (`DSRL.train` / `DSRL.update_noise_critic`), checked line by line
-------------------------------------------------------------------------------------------
Read against `stable_baselines3/dsrl/dsrl.py:230-365` and `refs/dsrl/cfg/robomimic/dsrl_can.yaml`.

| # | reference | here | verdict |
|---|---|---|---|
| 1 | `update_noise_critic` draws **fresh `randn`** and denoises it: `w ~ N(0,1)`, `a = D(s, w)`, fits `Q_ns(s, scale(w))` to `Q(s, a)` | fits `Q_ns(s, w_buffer)` to `Q(s, chunk_buffer/1.5)`, both from the buffer | **SUBSTANTIVE GAP.**  Coverage: the reference covers the whole prior, this covers only where the behaviour policy has been.  As `sigma` shrinks the coverage shrinks with it, so the actor's `argmax` can leave the region `Q_ns` was fit on.  The reference pays one denoise per noise-critic gradient step (10 per RL step) for that coverage; at 50x32x256 that is unaffordable here, which is why this was done, but it is a real difference and not a rounding error. |
| 2 | TD bootstrap: `next_a = D(s', actor(s'))` with the **current** actor, denoised every gradient step | `chunk_next` from the buffer, i.e. the sample the **behaviour** actor took at collection time | **SUBSTANTIVE GAP.**  The reference's target is the current policy's value; this one is the behaviour policy's.  Off-policy correctness prefers the reference.  Cost: one denoise per gradient step, again unaffordable. |
| 3 | `ent_coef` updated **before** the actor, from this step's `log_prob` | updated **after** the actor step | negligible; both use the same `log_prob`. |
| 4 | actor width = `act_steps * action_dim` (28 for robomimic can), **independent noise per chunk position** | actor width = `action_dim` (32), **repeated across the 50 chunk positions** | intentional.  `act_steps=4` makes the full space small enough for the reference to use it directly; at `C=50, d=32` the full space is 1,600 dims, which is exactly the case paper §C.5 addresses by repeating `w_single`.  Follows the paper, not the repo. |
| 5 | `target_update_interval = 1`, polyak every gradient step | `soft_update` inside the uTD loop | identical |
| 6 | `actor_gradient_steps = -1` -> actor updated on every gradient step | actor updated every uTD step | identical |
| 7 | `target_entropy = 0.0`, `ent_coef = 'auto'` init 1.0 | same | identical |
| 8 | `batch_size 256`, `utd 20`, `noise_critic_grad_steps 10`, `tau 0.005`, `gamma 0.99`, `layer_size 2048`, `num_layers 3`, `action_magnitude 1.5`, `n_critics 2`, min-combine | same | identical |
| 9 | `learning_starts = 1` after `init_rollout_steps = 1501` | `--learning-starts 256` after `--warmup-rounds 8` | scale-adapted; an episode here is 1,382 control steps, not 300. |
| 10 | env reward: robomimic's `-1` per step | `count[t+1] - count[t]` cookies | necessary: see `--reward`.  This environment's own reward is a *level*, which integrates to ~1500 over an episode. |

Gaps 1 and 2 both come from the same cause: **a denoise costs 40 ms per sample here**, and the
reference calls one on every gradient step.  Fixing them properly means paying that cost, or
pre-generating noise/action pairs as `train_dsrl.py` did -- which is what the offline trainer's
`--distill-per-frame` cache was for.  Until then, `min_q - q` in the log is expected to grow as the
actor moves away from the behaviour distribution, and its growth is a measure of this gap rather
than of learning.

Reward
------
The environment's reward is `min(cookies_in_target / required, 1.0)` per control step.  `--reward`
sums it over the executed chunk (default, matching the reference's `ActionChunkWrapper`, which
accumulates `act_steps` rewards into one RL step), averages it, or takes the last value.  The scale
matters and is always printed next to `Q`: a sum over 16 control steps reaches 16, so a return runs
to ~1400 against the offline pool's 9.56.

Usage
-----
    python scripts/recap/train_dsrl_online.py --minutes 240 --output /root/autodl-tmp/dsrl_online.pt

    # plumbing check: 3 rounds, no gradient steps
    python scripts/recap/train_dsrl_online.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
os.environ.setdefault("A3_MAIN_ROOT", "/tmp/a3-main")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from steering import (  # noqa: E402
    NoiseActor,
    build_observation,
    encode_prefix,
    repeat_noise,
    sample_with_prefix,
    soft_update,
    twin_critic,
    visual_feature,
)

CAMERAS = ("front", "left_wrist", "right_wrist")
VIEW_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)
STATE = "observation.state"
VELOCITY = "observation.velocity"
FRAME_DIM = 48            # [q(16), q_dot(16), applied_action(16)]
HISTORY_FRAMES = 10       # v9's history encoder width; frames x 48 = 480
HISTORY_DIM = HISTORY_FRAMES * FRAME_DIM
PROJECTION_DIM = 256
VISUAL_RAW_DIM = 8192

DEFAULT_CHECKPOINT = Path(
    "/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model")


# =================================================================================================
# environment side
# =================================================================================================
@dataclass
class Scene:
    """One environment plus the observation bookkeeping the v9 checkpoint requires.

    The history window is a *per control step* thing.  `BridgeChunkPolicy._history_vector` records
    the bug that comes from getting this wrong: appending only when a chunk is exhausted makes ten
    frames span 160 control steps instead of 10, at a fifth of the rate the encoder was trained to
    read.  So a frame is appended inside the stepping loop, in the worker thread, and the window
    that reaches the policy ends at the current step.
    """

    index: int
    env: Any
    renderer: Any
    camera_ids: dict[str, int]
    seed: int
    episode: int = 0
    history: list[np.ndarray] = field(default_factory=list)
    applied: np.ndarray | None = None
    control_steps: int = 0
    episode_return: float = 0.0
    cookies: int = 0
    required_cookies: int = 20
    max_control_steps: int = 1500
    step_times: list[float] = field(default_factory=list)
    step_nefc: list[int] = field(default_factory=list)

    def frame(self, observation: dict) -> np.ndarray:
        """One `[q, q_dot, a_previous]` frame, the layout the history encoder was trained on."""
        state = np.asarray(observation[STATE], dtype=np.float32).reshape(-1)[:16]
        velocity = np.asarray(observation.get(VELOCITY, np.zeros(16)),
                              dtype=np.float32).reshape(-1)[:16]
        previous = (np.zeros(16, dtype=np.float32) if self.applied is None
                    else np.asarray(self.applied, dtype=np.float32).reshape(-1)[:16])
        return np.concatenate([state, velocity, previous]).astype(np.float32)

    def history_vector(self) -> np.ndarray:
        """Ten frames, oldest first, leading frames repeated rather than zero-filled."""
        frames = list(self.history)
        if not frames:
            return np.zeros(HISTORY_DIM, dtype=np.float32)
        while len(frames) < HISTORY_FRAMES:
            frames.insert(0, frames[0])
        return np.concatenate(frames[-HISTORY_FRAMES:])

    def seed_history(self, observation: dict) -> bool:
        """Append the current frame if the window is empty.  Returns whether it did.

        Evaluation builds its window differently from a plain "append after each step", and the
        difference only shows up on the **first chunk of every episode**:

        * `BridgeChunkPolicy.act` calls `_advance_history(observation)` *before* deciding whether to
          forward, so the window that reaches the policy at step 0 already contains the current
          frame, and `_history_vector` then repeats it ten times.
        * A loop that only appends after `env.step` has an empty window at that moment, and
          `history_vector` returns zeros.

        So without this, the first chunk of every training episode is conditioned on an all-zero
        history while the same chunk at evaluation is conditioned on a repeated first frame -- a
        same-shape, wrong-meaning input, and exactly the failure mode this file's docstring warns
        about.  It is one chunk in ninety-four, which is why it is fixed here rather than treated as
        urgent, but it is a *systematic* bias on the first decision of every episode.
        """
        if self.history:
            return False
        self.history.append(self.frame(observation))
        return True

    def reset_counters(self) -> None:
        self.history.clear()
        self.applied = None
        self.control_steps = 0
        self.episode_return = 0.0
        self.cookies = 0

    def play(self, chunk: np.ndarray, execute: int, reward_mode: str,
             profile: bool = False, scale: float = 1.0,
             reward_offset: float = 1.0) -> tuple[float, bool, bool]:
        """Execute `execute` actions of `chunk`.  Runs in a worker thread -- no rendering here.

        Returns `(reward, terminated, truncated)`; the caller treats either flag as `done`.

        **Why a time-limit cut counts as `done`, and why an earlier version of this file was
        wrong to "fix" that.**  Textbook TD keeps the bootstrap at a truncation, because the world
        continues.  Getting that right here would mean keeping the final observation of the
        truncated episode as `s'` -- and the loop cannot, because by the time the next observation
        is read the finished scene has already been closed and replaced.  So the choice is
        between two errors:

        * cut the bootstrap (error = the value at the final state),
        * keep it on the *next episode's first state* (error = the value at a start state).

        Those are not comparable.  With `gamma = 0.99` over a 94-step episode, `V(s_93)` is one
        step of reward, `6.00 / 94 = 0.064`; `V(s_0)` is `r * (1 - gamma^94) / (1 - gamma) =
        3.90`.  The docstring here used to claim the cut cost "0.39 * 4 = 1.6 cookies" -- that
        multiplied the *start-state* value by the discount, which is the wrong state's value, and
        overstated the correct figure by 25x.  Keeping the bootstrap then cost `0.99 * 3.84 =
        3.80`, sixty times more than the thing it was fixing.

        The reference agrees, incidentally: `ActionChunkWrapper.step` sets `done = True` when
        `count >= max_episode_steps` and always returns `truncated=False`, so SB3's
        `reward + (1 - done) * gamma * next_q` cuts the bootstrap at the time limit too.
        """
        reward_sum = 0.0
        reward_last = 0.0
        level_sum = 0.0
        steps = min(execute, len(chunk))
        terminated = False
        truncated = False
        cookies_before = self.cookies
        # `step_index`, not `offset`: `reward_offset` is a parameter of this function and the loop
        # variable used to shadow it, which silently made `reward = level_sum - reward_offset * n`
        # use the loop counter instead of the offset.
        for step_index in range(steps):
            started = time.monotonic() if profile else 0.0
            observation, reward, term, trunc, info = self.env.step(chunk[step_index])
            if profile:
                self.step_times.append(time.monotonic() - started)
                self.step_nefc.append(int(self.env.data.nefc))
            self.history.append(self.frame(observation))
            if len(self.history) > HISTORY_FRAMES:
                del self.history[0]
            self.applied = np.asarray(info.get("applied_action", chunk[step_index]),
                                      dtype=np.float32).reshape(-1)[:16].copy()
            reward_sum += float(reward)
            level_sum += float(reward)
            reward_last = float(reward)
            self.control_steps += 1
            self.cookies = int(info.get("cookies_in_target", self.cookies))
            if term:
                terminated = True
                break
            # The environment's own horizon comes from the config and is 6000 for the cookie
            # scenes; the benchmark caps an episode at `max_steps` by counting in `run_episode`,
            # which is not code this loop runs.  Without this counter a training episode would run
            # four times longer than the episode the evaluation measures, and `--max-control-steps`
            # would silently do nothing -- which is exactly what happened: 640 control steps with
            # `--max-control-steps 400` reported `ep 0`.
            if trunc or self.control_steps >= self.max_control_steps:
                truncated = True
                break
        executed = max(1, min(execute, len(chunk)))
        if reward_mode == "paper":
            # **The reference's formulation, transplanted.**
            #
            # robomimic's wrapper (`refs/dsrl/env_utils.py:85`) does `reward = reward - reward_offset`
            # with `reward_offset = 1`, on an environment whose own reward is 0 until success and 1
            # after.  So every control step pays **-1** until the task is done, and
            # `ActionChunkWrapper.step` sums `act_steps` of those into one RL step: a constant -4,
            # identical on every step of every episode, regardless of what the arm is doing.
            #
            # That is a *timer*, not a progress signal: it says nothing about where the arm is, it
            # only charges for time that has not yet produced success.  Its properties are what make
            # everything downstream work:
            #
            #   * dense   -- 100 % of transitions carry signal, so TD is never a bare contraction;
            #   * large   -- |r| = 4 against a twin-critic `min` bias of ~0.56 sigma, so the bias
            #                cannot dominate the target (that is what collapsed Q to -3.5 in §9.107);
            #   * bounded -- returns sit in [-300, 0].
            #
            # The cookie task's own reward is already a level in [0, 1]
            # (`min(cookies_in_target / required, 1.0)`), so the same construction applies directly:
            # subtract 1 and sum over the chunk.  Early in an episode that is -16 per RL step; with
            # 5 of 20 cookies placed it is about -12.  Dense, large, bounded, and deliberately *not*
            # a progress signal -- the reference's semantics exactly.
            reward = level_sum - reward_offset * executed
        elif reward_mode == "chunk_mean":
            reward = reward_sum / executed
        elif reward_mode == "level_last":
            reward = reward_last
        elif reward_mode == "increment":
            reward = float(self.cookies - cookies_before)
        else:
            reward = reward_sum
        # `episode_return` is reported in the units the *evaluation* scores in, which for `paper`
        # means the cookie count, not the shaped level.  Mixing the two would make the logged return
        # incomparable with the 13.18 baseline.
        if reward_mode == "paper":
            self.episode_return += float(self.cookies - cookies_before)
        else:
            self.episode_return += reward
        return reward * scale, terminated, truncated


def render_frames(renderer, scene: Scene) -> dict[str, np.ndarray]:
    """Three 224x224 uint8 frames, from **this** thread.  Verified identical to the env's own."""
    out = {}
    for name in CAMERAS:
        renderer.update_scene(scene.env.data, camera=scene.camera_ids[name])
        image = np.ascontiguousarray(renderer.render(), dtype=np.uint8)
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        out[name] = image
    return out


# =================================================================================================
# replay
# =================================================================================================
class NoiseBank:
    """`(s, w, sampler(s, w))` for **fresh** `w ~ N(0,1)` -- what the reference distils to.

    Why this exists, and why the previous version was wrong
    ------------------------------------------------------
    `refs/sb3-dsrl/.../dsrl.py:349-363` (`update_noise_critic`) does this on every noise-critic
    gradient step:

        w  = randn(batch, C, d)                     # fresh, from the prior
        a  = diffusion_policy(s, w)                 # denoise
        Q  = critic(s, a)                           # TD critic, calibrated on real returns
        loss = mse(Q_noise(s, scale(w)), Q)         # fit the distilled critic to it

    Note that `scale(w)` is fed to `Q_noise` -- i.e. the distilled critic is fitted on the
    **standard normal** noise, the whole support of the prior.

    The first online version here instead fitted `Q_noise(s, w_buffer)` to `Q(s, chunk_buffer)`,
    where both came from the replay buffer.  That covers only the noise the *behaviour* policy
    happened to sample at the states it happened to visit.  As `sigma` anneals the behaviour
    distribution narrows, so the covered region narrows with it, and the actor's `argmax` walks
    straight out of it.

    Measured consequence: on a checkpoint trained 808 rounds, `max_w Q_ns(s,w) - min_w Q_ns(s,w)`
    was **14.5 cookies** -- larger than the whole episode score (13.18).  A real value function
    cannot have that range in the noise direction: changing a 32-dim noise vector is worth a
    fraction of an action, and one action is worth a fraction of a cookie.  So the surface the
    actor was climbing was almost entirely extrapolation.

    Cost
    ----
    One denoise per fresh noise vector.  Measured on this stack: 183 ms for batch 1, 437 ms for
    batch 8 -- strongly sublinear, so `--fresh-noise 4 --scenes 8` is one 32-row denoise, about
    0.7 s, on a round that costs 11 s.  Six percent, for a calibrated objective.
    """

    def __init__(self, capacity: int, chunk_dim: int, noise_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, 0, device=device)
        self.chunk = torch.zeros(capacity, chunk_dim, device=device)
        self.noise = torch.zeros(capacity, noise_dim, device=device)
        self.obs = None
        self.position = 0
        self.size = 0

    def add(self, obs: torch.Tensor, chunk: torch.Tensor, noise: torch.Tensor) -> None:
        if self.obs is None:
            self.obs = torch.zeros(self.capacity, obs.shape[1], device=self.device)
        count = obs.shape[0]
        # Write in at most two contiguous spans so a wrap-around does not become a Python loop.
        first = min(count, self.capacity - self.position)
        self.obs[self.position:self.position + first] = obs[:first]
        self.chunk[self.position:self.position + first] = chunk[:first]
        self.noise[self.position:self.position + first] = noise[:first]
        if count > first:
            rest = count - first
            self.obs[:rest] = obs[first:]
            self.chunk[:rest] = chunk[first:]
            self.noise[:rest] = noise[first:]
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch: int) -> dict[str, torch.Tensor] | None:
        if self.obs is None or self.size == 0:
            return None
        index = torch.randint(0, self.size, (batch,), device=self.device)
        return {"obs": self.obs[index], "chunk": self.chunk[index],
                "noise": self.noise[index]}


class Replay:
    """A GPU ring buffer of RL transitions, plus the raw inputs needed to re-encode a state.

    Every row carries the pair the policy actually produced -- `noise` (the actor's 32-dim sample)
    and `chunk` (the 1,600-dim action chunk the sampler made from it).

    It also carries **images and the history window**, which are not used for training at all.  They
    exist so `BootstrapSet` can re-run the encoder and recompute `chunk_next` with the *current*
    actor; a `Prefix` cannot be stored (it is a key/value cache), so the only way back to a state is
    through its pixels.  Cost, measured: 451 KB per row (3 x 224 x 224 x 3, uint8), i.e. 0.9 GB for
    the default 2,048 rows.
    """

    def __init__(self, capacity: int, obs_dim: int, noise_dim: int, chunk_dim: int,
                 device: torch.device, keep_inputs: bool = True, image_hw: int = 224,
                 history_dim: int = HISTORY_DIM, state_dim: int = 16):
        self.capacity = capacity
        self.device = device
        self.keep_inputs = keep_inputs
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.obs_next = torch.zeros(capacity, obs_dim, device=device)
        self.chunk = torch.zeros(capacity, chunk_dim, device=device)
        self.chunk_next = torch.zeros(capacity, chunk_dim, device=device)
        self.noise = torch.zeros(capacity, noise_dim, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.done = torch.zeros(capacity, device=device)
        # Inputs for the refresh, kept on the host: they are only read by `BootstrapSet`.
        # Images are stored **HWC**, which is what MuJoCo's renderer returns and what
        # `render_frames` hands back; the permute to channels-first happens at refresh time, in the
        # same place `raw_batch` does it.  Storing them CHW here and forgetting the permute there
        # fails at the encoder with a 224-vs-3 shape error, which reads like a model bug.
        self.next_images = (torch.zeros(capacity, len(CAMERAS), image_hw, image_hw, 3,
                                       dtype=torch.uint8) if keep_inputs else None)
        self.next_state = (torch.zeros(capacity, state_dim) if keep_inputs else None)
        self.next_history = (torch.zeros(capacity, history_dim) if keep_inputs else None)
        self.next_task: list[str] = []
        self.position = 0
        self.size = 0

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (
            self.obs, self.obs_next, self.chunk, self.chunk_next, self.noise,
            self.reward, self.done))

    def add(self, obs, chunk, noise, reward, done, obs_next, chunk_next,
            next_inputs: dict | None = None) -> None:
        index = self.position
        for target, value in (
            (self.obs, obs), (self.chunk, chunk), (self.noise, noise),
            (self.reward, reward), (self.done, done),
            (self.obs_next, obs_next), (self.chunk_next, chunk_next),
        ):
            target[index] = value
        if next_inputs is not None and self.keep_inputs:
            self.next_images[index] = next_inputs["images"]
            self.next_state[index] = next_inputs["state"]
            self.next_history[index] = next_inputs["history"]
            self.next_task.append(next_inputs["task"])
            del self.next_task[:-self.capacity]
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int) -> dict[str, torch.Tensor]:
        index = torch.randint(0, self.size, (batch,), device=self.device)
        return {
            "obs": self.obs[index],
            "chunk": self.chunk[index],
            "noise": self.noise[index],
            "reward": self.reward[index],
            "done": self.done[index],
            "obs_next": self.obs_next[index],
            "chunk_next": self.chunk_next[index],
            "index": index,
        }

    def obs_next_of(self, index: torch.Tensor) -> torch.Tensor:
        return self.obs_next[index]


class BootstrapSet:
    """Recompute `chunk_next` in place with the **current** actor, for a rotating slice of the replay.

    The gap this closes
    -------------------
    The TD target is `r + gamma * (1 - done) * min_i Q_target(s', a')`, and `a'` has to be what the
    *current* policy would do there -- that is what makes the fixed point of the Bellman operator
    the value of the policy being improved.  `dsrl.py:274-278` does exactly that: it calls
    `self.actor.action_log_prob(next_observations)` and denoises inside the gradient loop.

    Storing `chunk_next` at collection time instead freezes `a'` at whatever the actor did back
    then.  While the actor barely moves that is harmless, which is why the first 100 rounds of run4
    tracked the truth to within 0.26 cookies.  But `alpha` anneals 0.9 -> 0.003 over the same run,
    so the actor moves a lot, and the frozen target becomes the value of a policy that no longer
    exists.  Measured on run4: `q` fell from 2.53 to -0.19 while the true discounted return *rose*
    from 2.28 to 4.23 -- a bias that grew monotonically to **-4.3 cookies**, ending with a negative
    Q for a task whose reward is a non-negative cookie count.

    Why it is a rotating slice and not every gradient step
    ------------------------------------------------------
    Recomputing needs a `Prefix` for `s'`, and a `Prefix` is a PaliGemma key/value cache -- not
    storable per row.  So it needs the pixels, re-encoded.  Measured with
    `probe_bootstrap_cost.py`, `encode_prefix` is **linear** at ~40 ms/row (313 ms at batch 8,
    2,555 ms at batch 64; batch 128 does not fit in 31 GB), unlike the denoise which is sublinear.
    The reference re-encodes 256 rows on every one of 20 gradient steps; that is over ten seconds a
    round here, more than the round itself.

    So this refreshes `--bootstrap-refresh` rows per round (32 by default, ~1.6 s, +15 %).  A replay
    of 2,048 rows and 32 rows a round means every row is refreshed every 64 rounds; the actor moves
    by 20 x 64 = 1,280 gradient steps in between.  That is not the reference's ``every step'', and
    it is stated here rather than left to be discovered.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.cursor = 0

    def refresh(self, args, replay: "Replay", policy, preprocess, actor, projection, parts,
                chunk_flat_dim: int) -> int:
        """Recompute `chunk_next` for a slice of rows.  Returns how many were updated.

        The index arithmetic is the part that was wrong, so it is spelled out.  The first version
        did

            index = torch.arange(cursor, cursor + count) % replay.capacity
            cursor = (cursor + count) % replay.capacity
            index = index[index < replay.size]          # <-- kills everything

        which is only correct while `cursor` stays below `size`.  It does not: `cursor` advances by
        `count` (32) per round while `size` grows by `scenes` (8) per round, so from about round 11
        the whole slice lies past the end of the buffer, the filter empties it, and `refresh` returns
        0 forever.  Measured on run5: a refresh happened on rounds 8, 9 and 10 and never again, so
        the fix this class exists for was effectively off for the entire run -- which is why `q`
        still drifted there.

        Rotating inside `[0, size)` instead is monotone in the right variable and always fills the
        slice: while `size < count` it repeats rows (harmless, the set is small), and once
        `size >= count` it walks the whole live buffer exactly once every `size / count` rounds.
        """
        if not replay.keep_inputs or replay.size == 0:
            return 0
        count = min(args.bootstrap_refresh, replay.size)
        # Host index: `next_images` / `next_state` / `next_history` are host tensors (a 224x224x3
        # uint8 frame per camera per row is 451 KB, which belongs in RAM, not VRAM).  Building the
        # index on `replay.device` and using it here raises "indices should be either on cpu or on
        # the same device as the indexed tensor (cpu)".
        index = torch.arange(self.cursor, self.cursor + count) % replay.size
        self.cursor = (self.cursor + count) % replay.size
        if index.numel() == 0:
            return 0

        # (n, cameras, H, W, 3) uint8 -> (n, 3, H, W) float in [0, 1] per camera key, which is the
        # LeRobot convention `preprocess` expects.
        images = replay.next_images[index].to(self.device, non_blocking=True).float().div_(255.0)
        batch = {f"observation.images.{name}": images[:, i].permute(0, 3, 1, 2)
                 for i, name in enumerate(CAMERAS)}
        batch[STATE] = replay.next_state[index].to(self.device)
        batch["observation.history"] = replay.next_history[index].to(self.device)
        task = replay.next_task[-1] if replay.next_task else ""
        batch["task"] = [task] * index.numel()

        with torch.no_grad():
            prepared = preprocess(batch)
            prefix = encode_prefix(policy, prepared, keep_embeddings=True)
            observation = actor_observation(policy, prepared, prefix, projection, parts)
            noise, _ = actor.sample(observation)
            flat = repeat_noise(noise, args.chunk_size, args.action_dim, "single")
            chunk = sample_with_prefix(policy, prefix, flat, args.num_steps, self.device)
            padded = torch.zeros(chunk.shape[0], args.chunk_size, args.action_dim,
                                 device=self.device)
            padded[:, :, :args.original_action_dim] = chunk[:, :, :args.original_action_dim]
            divisor = args.action_magnitude if args.action_magnitude > 0 else 1.0
            replay.chunk_next[index] = (padded.reshape(chunk.shape[0], -1) / divisor).float()
        return int(index.numel())


# =================================================================================================
# observation assembly
# =================================================================================================
def raw_batch(scenes: list[Scene], frames: list[dict[str, np.ndarray]],
              observations: list[dict], task: str, device: str) -> dict:
    """The dict `preprocess` expects.

    Images arrive as `(B, H, W, C)` uint8 and are handed over as `(B, C, H, W)` float in [0, 1].
    That is not cosmetic: the LeRobot convention is channels-first in [0, 1] and
    `_preprocess_images` maps it to [-1, 1] for SigLIP.  Passing HWC is not rejected at the
    boundary -- it reaches the first conv as 224 channels and fails there, which reads like a model
    bug rather than an input bug.
    """
    stacks = {key: [] for key in VIEW_KEYS}
    for scene_frames in frames:
        for name in CAMERAS:
            stacks[f"observation.images.{name}"].append(scene_frames[name])
    raw = {key: torch.as_tensor(np.stack(value), device=device)
           .permute(0, 3, 1, 2).float().div_(255.0)
           for key, value in stacks.items()}
    raw[STATE] = torch.as_tensor(
        np.stack([np.asarray(o[STATE], dtype=np.float32) for o in observations]), device=device)
    raw["observation.history"] = torch.as_tensor(
        np.stack([scene.history_vector() for scene in scenes]), device=device)
    raw["task"] = [task] * len(scenes)
    return raw


def actor_observation(policy, prepared: dict, prefix, projection: torch.Tensor,
                      parts: dict) -> torch.Tensor:
    """`[proprio, vlm, visual]` in the order the checkpoint was trained with.

    Same construction as `SteeringActor.observe`, but for a *trainable* actor: the offline trainer
    built this vector once per dataset row and stored it in `dsrl_cache`; here it is rebuilt every
    round, because the states are new every round.
    """
    proprio = prepared.get(STATE)
    if proprio is None and parts.get("proprio", True):
        raise ValueError("the checkpoint expects proprioception but the batch has none")
    visual = None
    if parts.get("visual", True):
        visual = visual_feature(policy, prefix) @ projection
    vlm = prefix.hidden_last if parts.get("vlm", True) else None
    return build_observation(proprio.float() if proprio is not None else None,
                             vlm, visual, parts).detach()


# =================================================================================================
# training
# =================================================================================================
def gradient_steps(args, actor, critic, critic_target, critic_noise, log_alpha, optimizers,
                   replay: Replay, noise_bank: "NoiseBank | None" = None) -> dict[str, float]:
    """One `--utd` batch of updates plus the reference's separate noise-critic loop.

    Structurally copied from `train_dsrl.py`; see that file for why each choice is what it is (TD
    on the executed action, `soft_update` **inside** the loop, Adam, the `scale_action` split).
    The only changes are where the data comes from.
    """
    actor_optimizer, critic_optimizer, noise_critic_optimizer, alpha_optimizer = optimizers
    stats: dict[str, float] = {}
    for _ in range(max(1, args.utd)):
        data = replay.sample(args.batch)
        state = data["obs"]
        # `chunk_next` is refreshed in place by `BootstrapSet` with the current actor, so reading it
        # here is equivalent to the reference's `D(s', actor_noise(s'))`.  When
        # `--bootstrap-refresh 0` it is the collection-time chunk and the target is frozen on the
        # behaviour policy -- the configuration that biased Q down by 4.3 cookies in run4.
        with torch.no_grad():
            next_q = torch.min(
                torch.stack([head(data["obs_next"], data["chunk_next"]) for head in critic_target]),
                dim=0).values
            if args.entropy_in_target:
                _, next_log_prob = actor.sample(data["obs_next"])
                next_q = next_q - log_alpha.exp().detach() * next_log_prob
            target_q = (data["reward"].unsqueeze(-1)
                        + args.gamma * (1.0 - data["done"].unsqueeze(-1)) * next_q)

        q_values = torch.stack([head(state, data["chunk"]) for head in critic])
        critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in q_values)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        # One call, not two.  `clip_grad_norm_` **returns** the norm and only clips when it is asked
        # to; calling it twice in a row re-clips the already-clipped gradients against the same
        # threshold, which is idempotent and therefore harmless -- but the second call was written as
        # `grader = ...` for logging while the first was already doing the clipping, i.e. the same
        # "two names for one effect" shape as the bugs in §9.106.  Kept as one call plus a read.
        grader = torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
        critic_optimizer.step()
        # Diagnostics for the TD fixed point.  `q` was observed to settle at about -3.5 on runs whose
        # episodes return +2.75 cookies, and `probe_td_fixed_point.py` showed the update rules alone
        # converge upward to the correct value on synthetic data.  So the negative value must come
        # from what the replay feeds them, and these four numbers say which part:
        #
        #   reward      mean reward in the batch.  If this is positive, the only way for the target
        #               to be negative is the bootstrap term.
        #   boot        mean of `gamma * (1 - done) * min Q_target`.  Compare with `target`.
        #   done_frac   fraction of the batch whose bootstrap was cut.  A truncation-heavy batch
        #               pushes `boot` toward zero, which pulls `target` up, not down.
        #   q_gap       mean `target - q`, signed.  A persistent negative gap with a positive reward
        #               means Q is *above* target and being pushed down.
        target_now = target_q.detach()
        boot_term = target_now - data["reward"].unsqueeze(-1)
        stats.update({
            "reward_b": float(data["reward"].mean()),
            "boot_b": float(boot_term.mean()),
            "done_frac": float(data["done"].mean()),
            "q_gap": float((target_now - q_values.detach().mean(dim=0, keepdim=True)).mean()),
            "grad_norm": float(grader),
        })

        # NOTE: no noise-critic update here.  The reference runs `critic_noise` in a **separate**
        # loop after all `utd` gradient steps (`dsrl.py:326-335`), `noise_critic_grad_steps` times,
        # each drawing a fresh replay batch.  An earlier version of this function also fitted it
        # once per `utd` step, which made it 20 + 10 = 30 updates per round against the reference's
        # 10 -- three times the intended rate on the network the *actor* maximises.  The offline
        # trainer had the same structure, so this was inherited rather than introduced.

        sampled, log_prob = actor.sample(state)
        alpha = log_alpha.exp().detach()
        min_q = torch.min(torch.stack([head(state, sampled) for head in critic_noise]),
                          dim=0).values
        # The actor's objective is divided by `--reward-scale`.
        #
        # Scaling the reward is what stops the value collapsing (see `--reward-scale`), but it also
        # multiplies Q by the same factor, and the actor maximises Q directly.  Left alone, the
        # actor's gradient grows with the scale: run6 (`--reward-scale 30`) had `min_q` climb
        # 3.77 -> 32.73 while `alpha` annealed on its original schedule, `actor_loss` went
        # -23 -> -39, the placement rate fell from 5.5 % to 0.5 % of RL steps, and the episode
        # return dropped 9 % against run4.  Dividing by the scale restores the gradient magnitude
        # the actor saw at scale 1, while the critic keeps the smoother TD target.
        #
        # This is not a change of objective, only of units: the argmax over `sampled` is unaffected.
        actor_loss = (alpha * log_prob - min_q).mean() / max(1e-9, args.reward_scale)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        alpha_loss = -(log_alpha * (log_prob + args.target_entropy).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        soft_update(critic_target, critic, args.tau)
        # `.detach()` on every statistic: these tensors still carry the graph, and `float()` on
        # one that requires grad raises a warning that reads like an autograd bug.
        stats = {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(alpha),
            "q": float(q_values.detach().mean()),
            "min_q": float(min_q.detach().mean()),
            "log_prob": float(log_prob.detach().mean()),
            "u_abs": float(sampled.detach().abs().mean()),
            "sigma": float(actor.log_std.detach().clamp(-20.0, 2.0).exp().mean()),
            "target_q": float(target_q.detach().mean()),
        }

    # --- the reference's separate noise-critic loop, `noise_critic_steps` fresh batches ------
    noise_critic_loss = torch.zeros((), device=log_alpha.device)
    for _ in range(max(0, args.noise_critic_steps)):
        data = replay.sample(args.batch)
        distilled = noise_bank.sample(args.batch) if noise_bank is not None else None
        source = distilled if distilled is not None else data
        with torch.no_grad():
            target = torch.stack([head(source["obs"], source["chunk"]) for head in critic]).detach()
        value = torch.stack([head(source["obs"], source["noise"]) for head in critic_noise])
        noise_critic_loss = 0.5 * sum(F.mse_loss(v, t) for v, t in zip(value, target))
        noise_critic_optimizer.zero_grad()
        noise_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic_noise.parameters(), args.grad_clip)
        noise_critic_optimizer.step()
    stats["noise_critic_loss"] = float(noise_critic_loss.detach())
    return stats


def save_checkpoint(path: Path, actor, critic, critic_noise, projection, args,
                    extra: dict) -> None:
    """The format `SteeringActor` loads: weights, architecture, and feature layout together.

    `projection` travels with the weights because retraining with a different seed produces a
    different projection and an incompatible actor, so regenerating it at inference is the easiest
    way to silently evaluate noise.  `hidden` is written even though `SteeringActor` can infer it
    from the first layer, because a checkpoint should be readable by hand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "critic_noise": critic_noise.state_dict(),
        "projection": projection.detach().cpu().numpy(),
        "noise_mode": "single",
        "actor_dim": args.actor_dim,
        "chunk_size": args.chunk_size,
        "action_dim": args.action_dim,
        "observation_dim": args.observation_dim,
        "hidden": args.hidden,
        "observation_parts": {"proprio": True, "vlm": True, "visual": True},
        "online": extra,
    }, path)
    print(f"[online] wrote {path}", flush=True)


# =================================================================================================
# helpers that need the benchmark or the postprocessor
# =================================================================================================
def reset_options(bench, seed: int) -> dict:
    """The options every evaluation and rollout uses, so the scene distribution matches."""
    return {
        "randomize_cookies": bench.randomize_cookies,
        "randomize_boxes": bench.randomize_boxes,
        "randomize_source_bin": bench.randomize_boxes,
        "randomize_target_bin": bench.randomize_boxes,
        "source_bin_noise_m": bench.source_bin_noise_m,
        "target_bin_noise_m": bench.target_bin_noise_m,
        "target_bin_yaw_noise_rad": bench.target_bin_yaw_noise_rad,
        "randomization_profile": bench.profile or "basic",
        "appearance_seed": seed,
        "randomization_settings": bench.randomization_settings,
    }


def postprocess_actions(postprocess, chunk: torch.Tensor) -> np.ndarray:
    """Normalised chunk -> physical joint targets, one array per scene.

    The batch postprocessor unnormalises with the statistics the checkpoint was trained under, so
    the output is in the same units as the dataset's `action` column -- which is what the
    environment expects and what the history frame's `a_previous` carries.
    """
    processed = postprocess(chunk)
    if hasattr(processed, "detach"):
        processed = processed.detach().float().cpu()
    return np.asarray(processed, dtype=np.float64)


def _make_renderer(model):
    """A renderer owned by the main thread.

    `create_env(render_cameras=False)` leaves `_renderer` as None and `_render_camera` returns
    zeros, so the loop builds its own.  That is only acceptable because it was verified: on this
    stack the manual renderer reproduces the environment's own images bit for bit on all three
    cameras, and 3.9 ms for the set.
    """
    import mujoco  # noqa: PLC0415

    height = width = 256
    return mujoco.Renderer(model, height=height, width=width)


# =================================================================================================
# main
# =================================================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/dsrl_online.pt"))
    parser.add_argument("--task", default="2x10", choices=("2x5", "2x10"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    # --- rollout -------------------------------------------------------------------------
    parser.add_argument("--scenes", type=int, default=8,
                        help="parallel environments.  One thread per scene; physics measured 7.94x"
                             " faster at 4 threads with bit-identical results, and the batched"
                             " forward is sublinear to 8 (437 ms for eight against 183 ms for one).")
    parser.add_argument("--execute", type=int, default=16,
                        help="control steps per RL step, fixed by the checkpoint's manifest (chunk"
                             " 50, execute 16).  Not a tuning knob: one RL step IS one executed"
                             " chunk, which is what makes gamma 0.99 correct over 87-step"
                             " episodes.")
    parser.add_argument("--max-control-steps", type=int, default=1500,
                        help="control steps per episode.  Must match what the evaluation uses"
                             " (`run_episode(max_steps=1500)`), because a training episode that"
                             " runs longer than the evaluated one trains the critic on a horizon"
                             " nothing is measured over.  Enforced by counting in `Scene.play`:"
                             " the environment's own horizon is 6000 from the config and does"
                             " not know about the benchmark's cap.")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="added to every scene seed, so successive runs see fresh layouts")
    parser.add_argument("--fresh-env-per-episode", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="build a new environment instead of resetting the old one.  On by"
                             " default because a *persistent* environment does not reproduce its"
                             " first reset: with one seed it gave layout A on reset 1 and a"
                             " different, stable layout B on resets 2-4, while two fresh"
                             " environments with that seed both gave A.  Evaluation builds fresh"
                             " environments, so training on B would train one scene distribution"
                             " and evaluate another.  The cost is 1.36 s per episode, about 8%%.")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="multiplier on the reward.\n\n"
                             "Not a tuning knob for its own sake: the TD target is"
                             " `r + gamma * min Q_target(s', a')`, so when the expected reward is small"
                             " next to the bootstrap term, the bootstrap wins and the value collapses."
                             "  Measured on this task with `--max-control-steps 320`: mean reward"
                             " 0.078 per RL step (a cookie every ~28 steps, value 1), and Q fell to"
                             " -3.58 and stayed there -- a stable negative fixed point, reproduced"
                             " from scratch in `probe_q_init_drift.py` with only the reward changed"
                             " (replacing it with a constant +1 sent Q to +7.62 instead).\n\n"
                             "  The reference never meets this because robomimic pays -1 per step:"
                             " dense, and large enough to anchor the bootstrap.  Scaling the cookie"
                             " count by ~25-30 puts the expected reward near 1, which is the same"
                             " regime.  It changes the *unit* of Q, not the ordering, so the actor's"
                             " argmax is unaffected.\n\n"
                             "  Default 30, from a three-point sweep with everything else fixed"
                             " (26 rounds, 8 scenes): `q` at round 20 was -3.39 at scale 1, -2.68 at 10,"
                             " and **+3.27 at 30**.  This is not only a matter of the reward being"
                             " small: `probe_min_bias.py` shows that with a sparse reward the twin"
                             " critics' `min` aggregation is itself biased low by E[min] = mu -"
                             " 0.56 sigma, and with scale 1 that bias dominates the target.  Under the"
                             " same sparse reward, `mean` aggregation stayed at +0.67 where `min` went"
                             " to -0.63.")
    parser.add_argument("--reward", default="paper",
                        choices=("paper", "increment", "chunk_sum", "chunk_mean", "level_last"),
                        help="what one RL step's reward is.\n\n"
                             "`paper` copies the reference: robomimic subtracts `reward_offset = 1`"
                             " from a 0/1 success reward, so every control step pays -1 and`"
                             " ActionChunkWrapper` sums `act_steps` of them into a constant -4 per RL"
                             " step.  This environment's reward is already a level in [0, 1], so the"
                             " same construction gives `level_sum - offset * execute`, i.e. -16 early"
                             " in an episode and about -12 once five cookies are in.\n\n"
                             "  That is a *timer*, not a progress signal -- and it is what makes the"
                             " reference work: 100 % of transitions carry signal, |r| = 4 dwarfs the"
                             " twin-critic `min` bias of ~0.56 sigma, and returns are bounded.  The"
                             " `increment` mode used before this was 96 % zeros, which is what let the"
                             " `min` bias dominate and collapse Q to -3.5 (see §9.107).")
    parser.add_argument("--reward-offset", type=float, default=1.0,
                        help="subtracted from the level reward in `--reward paper`.  The reference"
                             " uses 1.0, which is exactly its `env.reward_offset` in"
                             " `cfg/robomimic/dsrl_can.yaml`.  1.0 makes the reward strictly negative"
                             " until the task is solved, i.e. a pure timer; 0 leaves it as the raw"
                             " level, which is dense but rewards holding cookies rather than"
                             " delivering them.")

    # --- algorithm -----------------------------------------------------------------------
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--utd", type=int, default=20,
                        help="gradient steps per RL step (paper §B)")
    parser.add_argument("--noise-critic-steps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-entropy", type=float, default=0.0)
    parser.add_argument("--entropy-in-target", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="the reference subtracts `ent_coef * next_log_prob` from the TD target"
                             " (`dsrl.py:286`), and `train_dsrl.py` has it off by default.  Off"
                             " here for the same reason, and the arithmetic is the same: the term"
                             " is a per-bootstrap bonus, so at discount gamma it compounds to"
                             " `alpha * |log pi| / (1 - gamma)`.  Measured on this stack alpha"
                             " starts at 1.0 and `log pi` is about -21, so the bonus sums to about"
                             " 2100 against a true return of a few dozen cookies.  Observed"
                             " directly: with it on, Q climbed 20.4 -> 35.5 -> 41.3 while the"
                             " mean reward was 0.006, i.e. two orders of magnitude of pure"
                             " offset.  It does not break the *actor* -- a state-dependent"
                             " constant does not change the argmax over actions at a fixed"
                             " state -- but it makes every Q in the log unreadable, which is"
                             " what the rest of this file is trying to keep honest.")
    parser.add_argument("--action-magnitude", type=float, default=1.5,
                        help="scale_action divisor.\n\n"
                             "Read `dsrl.py` carefully, because this number's role is narrower than"
                             " it looks.  The reference's `critic` is fed `replay_data.actions`,"
                             " which the rollout stores as the **raw denoised output** -- it"
                             " overwrites `buffer_action` with `diffusion_policy(...)` right before"
                             " returning, so the `scale_action` dance above it is a no-op round trip."
                             "  Only `critic_noise` sees a scaled value, via"
                             " `scale_action(randn)` inside `update_noise_critic`.\n\n"
                             " With `algorithm: dsrl_na` (the config that produced the published can"
                             " numbers) the action space is robomimic's `Box(-1, 1)`, so that"
                             " scaling is the identity and `critic_noise` is fitted on std-1 noise"
                             " while the actor poses arguments at std ~0.63.  With `dsrl_sac` the"
                             " space is `Box(-1.5, 1.5)` and it becomes `w / 1.5`, std 0.667, which"
                             " does line the two up.\n\n"
                             " 0 disables the divisor entirely and matches `dsrl_na` exactly.")
    parser.add_argument("--bootstrap-refresh", type=int, default=32,
                        help="replay rows per round whose `chunk_next` is recomputed with the"
                             " **current** actor, ~40 ms/row (measured; `encode_prefix` is linear,"
                             " unlike the denoise).\n\n"
                             "This closes the difference from `dsrl.py:274-278`, which recomputes"
                             " the bootstrap action on every gradient step.  Storing it at"
                             " collection time instead froze the target on the behaviour policy and"
                             " biased `q` from +0.26 to -4.3 cookies over 1,000 rounds in run4,"
                             " ending negative on a task whose reward cannot be negative.\n\n"
                             "0 disables it and restores that bias; kept for A/B comparison."
                             " 32 costs ~1.6 s on an 11 s round (+15 %) and refreshes a 2,048-row"
                             " replay every 64 rounds.")
    parser.add_argument("--grad-clip", type=float, default=10.0,
                        help="gradient-norm clip for both critics.\n\n"
                             "The reference inherits SB3's default of 10.0, which is fine when the TD"
                             " target is order 1.  With `--reward paper` the target is `-offset *"
                             " execute / (1 - gamma)` in magnitude -- about -978 for a 94-step episode"
                             " at -16 per RL step -- so a clip of 10 would squash essentially every"
                             " gradient to the same direction and destroy the learning rate.  Raise"
                             " this in proportion to the target scale, or leave the reward small.")
    parser.add_argument("--replay-capacity", type=int, default=20000)
    parser.add_argument("--fresh-noise", type=int, default=4,
                        help="fresh `N(0,1)` noise vectors denoised per scene per round, to train the"
                             " distilled critic on the prior rather than only on behaviour noise.\n\n"
                             "The reference denoises a whole batch of fresh noise on **every**"
                             " noise-critic gradient step (10 per RL step); that is unaffordable"
                             " here at 50x32x10 denoise steps.  Doing it once per round for a few"
                             " vectors per scene buys the same coverage more slowly.  Measured"
                             " cost: one 32-row denoise is ~0.7 s against an 11 s round.\n\n"
                             "0 restores the old behaviour (distil against the buffer's own"
                             " noise).  That is what produced a `Q_ns` surface with a 14.5-cookie"
                             " range, so it is kept only for A/B comparison.")
    parser.add_argument("--noise-bank-capacity", type=int, default=8192)
    parser.add_argument("--learning-starts", type=int, default=256,
                        help="transitions before any gradient step")
    parser.add_argument("--warmup-rounds", type=int, default=8,
                        help="rounds of pure collection first, so the first batch is not 256 copies"
                             " of a handful of states")

    # --- budget and logging --------------------------------------------------------------
    parser.add_argument("--minutes", type=float, default=240.0)
    parser.add_argument("--rounds", type=int, default=0, help="overrides `--minutes`")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every-minutes", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="3 rounds, no gradient steps: checks the plumbing")
    parser.add_argument("--no-train", action="store_true",
                        help="roll out only; measures loop throughput without the MLP cost")
    parser.add_argument("--profile-physics", action="store_true",
                        help="record per-control-step cost and `nefc` per scene.  Answers whether"
                             " the physics share grows because the solver is working harder on"
                             " more contacts, and therefore whether throughput should be budgeted"
                             " from the worst case or the average.")
    args = parser.parse_args(argv)

    if args.dry_run:
        args.rounds = 3
        args.no_train = True
        args.warmup_rounds = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device

    # --- frozen VLA ----------------------------------------------------------------------
    import train_pi05_launcher as launcher  # noqa: PLC0415
    from bench_pi05_on_main import make_cookie_benchmark, training_prompt  # noqa: PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.pi05 import PI05Policy  # noqa: PLC0415

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    print(f"[online] loading {args.checkpoint.parent.parent.parent.name}", flush=True)
    policy = PI05Policy.from_pretrained(str(args.checkpoint)).to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, postprocess = make_pre_post_processors(
        policy.config, str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    args.chunk_size = int(policy.config.chunk_size)
    args.action_dim = int(policy.config.max_action_dim)
    args.original_action_dim = int(policy.config.output_features["action"].shape[0])
    args.num_steps = int(policy.config.num_inference_steps)
    args.actor_dim = args.action_dim          # the actor's variable is a single-step R^d
    chunk_flat_dim = args.chunk_size * args.action_dim
    args.observation_dim = 16 + 2048 + PROJECTION_DIM

    projection = torch.from_numpy(
        rng.normal(0.0, 1.0 / math.sqrt(PROJECTION_DIM),
                   (VISUAL_RAW_DIM, PROJECTION_DIM)).astype(np.float32)).to(device)
    parts = {"proprio": True, "vlm": True, "visual": True}

    # --- networks ------------------------------------------------------------------------
    actor = NoiseActor(args.observation_dim, args.actor_dim, hidden=args.hidden).to(device)
    critic = twin_critic(args.observation_dim, chunk_flat_dim, hidden=args.hidden).to(device)
    critic_target = twin_critic(args.observation_dim, chunk_flat_dim, hidden=args.hidden).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_noise = twin_critic(args.observation_dim, args.actor_dim, hidden=args.hidden).to(device)
    print(f"[online] actor {sum(p.numel() for p in actor.parameters()) / 1e6:.2f}M,"
          f" critic {sum(p.numel() for p in critic.parameters()) / 1e6:.2f}M,"
          f" noise critic {sum(p.numel() for p in critic_noise.parameters()) / 1e6:.2f}M,"
          f" observation {args.observation_dim}, chunk {chunk_flat_dim},"
          f" noise {args.actor_dim}", flush=True)

    alpha = torch.zeros(1, requires_grad=True, device=device)
    optimizers = (
        torch.optim.Adam(actor.parameters(), lr=args.lr),
        torch.optim.Adam(critic.parameters(), lr=args.lr),
        torch.optim.Adam(critic_noise.parameters(), lr=args.lr),
        torch.optim.Adam([alpha], lr=args.lr),
    )
    log_alpha = alpha

    replay = Replay(args.replay_capacity, args.observation_dim, args.actor_dim, chunk_flat_dim,
                    torch.device(device), keep_inputs=args.bootstrap_refresh > 0)
    noise_bank = (NoiseBank(args.noise_bank_capacity, chunk_flat_dim, args.actor_dim,
                            torch.device(device))
                  if args.fresh_noise > 0 else None)
    bootstrap = BootstrapSet(torch.device(device)) if args.bootstrap_refresh > 0 else None
    print(f"[online] replay {args.replay_capacity} rows, {replay.bytes() / 1024 ** 3:.2f} GB on"
          f" {device}", flush=True)
    print(f"[online] noise-critic target: "
          + (f"{args.fresh_noise} fresh N(0,1) x {args.scenes} scenes per round, denoised, "
             f"bank {args.noise_bank_capacity}" if noise_bank is not None
             else "WARNING: buffer noise only -- `Q_ns` will extrapolate off-distribution"),
          flush=True)
    print(f"[online] bootstrap action: "
          + (f"current actor, {args.bootstrap_refresh} rows refreshed per round"
             if bootstrap is not None
             else "WARNING: buffer `chunk_next` -- the target is frozen on the behaviour policy"
                  " and Q will be biased downward as alpha anneals"), flush=True)
    if args.action_magnitude > 0:
        print(f"[online] NOTE: critic input divided by action_magnitude={args.action_magnitude};"
              f" the reference's `dsrl_na` path feeds the raw denoised chunk, so 0 matches it"
              f" exactly.  Kept at the historical default so runs stay comparable.", flush=True)

    # --- environments --------------------------------------------------------------------
    bench = make_cookie_benchmark(task=args.task, prompt_style="short",
                                  max_steps=args.max_control_steps)
    task = training_prompt()

    def make_scene(index: int, episode: int) -> Scene:
        seed = args.seed_offset + index * 1000 + episode
        env = bench.create_env(render_cameras=False)
        env.reset(seed=seed, options=reset_options(bench, seed))
        return Scene(index=index, env=env, renderer=_make_renderer(env.model),
                     camera_ids=dict(env._camera_ids), seed=seed, episode=episode,  # noqa: SLF001
                     required_cookies=int(bench.required_cookies),
                     max_control_steps=args.max_control_steps)

    scenes = [make_scene(index, 0) for index in range(args.scenes)]
    pool = ThreadPoolExecutor(max_workers=args.scenes)
    print(f"[online] {args.scenes} scenes up; instruction {task!r}", flush=True)

    # The reward is the one quantity every other number is measured against, so it is checked
    # before any gradient step rather than inferred from Q later.  A zero reward here produces a
    # perfectly finite loss curve that means nothing -- the failure mode this project has hit
    # repeatedly, and the reason the offline trainer prints the pool return and the anchor return
    # at startup.
    print(f"[online] reward '{args.reward}': scene cookies at reset"
          f" {[scene.cookies for scene in scenes]}", flush=True)
    print(f"[online] reward scale x{args.reward_scale:g}"
          + ("  (the actor's objective is divided by the same factor -- see `--reward-scale`)"
             if args.reward_scale != 1.0 else ""), flush=True)

    def observe_all(clock: dict[str, float] | None = None
                    ) -> tuple[list[dict], list[dict[str, np.ndarray]]]:
        """Render and read the observation for every scene, on the main thread."""
        clock = clock if clock is not None else {}
        started = time.monotonic()
        frames, observations = [], []
        for scene in scenes:
            frames.append(render_frames(scene.renderer, scene))
            observation = scene.env._observation()  # noqa: SLF001
            # The first observation of an episode gets a one-frame window that `history_vector`
            # repeats, which is what evaluation does -- see `Scene.seed_history`.
            scene.seed_history(observation)
            observations.append(observation)
        clock["render"] = time.monotonic() - started
        return observations, frames

    def forward_all(observations, frames, clock: dict[str, float] | None = None) -> dict[str, Any]:
        """One batched pass: prefix, actor observation, noise, and the chunk to execute."""
        clock = clock if clock is not None else {}
        started = time.monotonic()
        raw = raw_batch(scenes, frames, observations, task, device)
        prepared = preprocess(raw)
        clock["prepare"] = time.monotonic() - started
        started = time.monotonic()
        prefix = encode_prefix(policy, prepared, keep_embeddings=True)
        clock["prefix"] = time.monotonic() - started
        started = time.monotonic()
        with torch.no_grad():
            actor_obs = actor_observation(policy, prepared, prefix, projection, parts)
            noise, log_prob = actor.sample(actor_obs)
            flat = repeat_noise(noise, args.chunk_size, args.action_dim, "single")
            chunk = sample_with_prefix(policy, prefix, flat, args.num_steps, device)
        clock["sample"] = time.monotonic() - started
        # The critic sees the executed chunk at unit scale, which is what `scale_action` defines.
        # Columns past `original_action_dim` are the sampler's own padding and are held at zero, so
        # the critic's 1,600-dim input carries exactly the 800 numbers that reach the arm.
        padded = torch.zeros(chunk.shape[0], args.chunk_size, args.action_dim, device=device)
        padded[:, :, :args.original_action_dim] = chunk[:, :, :args.original_action_dim]
        divisor = args.action_magnitude if args.action_magnitude > 0 else 1.0
        scaled = (padded.reshape(chunk.shape[0], -1) / divisor).float()
        # Fresh-noise pass for the distilled critic.  `torch.randn` here is the *prior*, which is
        # what the reference feeds `Q_noise`; the behaviour noise already in the buffer is a
        # narrower, actor-dependent set.
        #
        # One denoise per fresh-noise index, each batched over all scenes: `prefix` is a single
        # `Prefix` whose batch is `args.scenes`, so a noise tensor of any other row count would not
        # line up with its key/value cache.  Cost measured on this stack: batch 8 is 437 ms, so
        # four indices is ~1.7 s against an 11 s round.
        fresh: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        if noise_bank is not None and args.fresh_noise > 0:
            started = time.monotonic()
            with torch.no_grad():
                for _ in range(args.fresh_noise):
                    prior = torch.randn(args.scenes, args.actor_dim, device=device)
                    flat_prior = repeat_noise(prior, args.chunk_size, args.action_dim, "single")
                    made = sample_with_prefix(policy, prefix, flat_prior, args.num_steps, device)
                    padded_prior = torch.zeros(made.shape[0], args.chunk_size, args.action_dim,
                                               device=device)
                    padded_prior[:, :, :args.original_action_dim] = \
                        made[:, :, :args.original_action_dim]
                    fresh.append((actor_obs, prior,
                                  (padded_prior.reshape(made.shape[0], -1)
                                   / divisor).float()))
            clock["fresh"] = time.monotonic() - started
        return {
            "actor_obs": actor_obs,
            "noise": noise.detach(),
            "chunk_scaled": scaled,
            "fresh": fresh,
            "executed": postprocess_actions(
                postprocess, chunk[:, :, :args.original_action_dim]),
        }

    # --- warm start: every scene needs a chunk before the first step ---------------------
    observations, frames = observe_all()
    state = forward_all(observations, frames)

    started = time.monotonic()
    deadline = started + args.minutes * 60.0
    round_limit = args.rounds if args.rounds else 10 ** 9
    history: list[dict] = []
    transitions = 0
    episodes = 0
    episode_returns: list[float] = []
    recent_rewards: list[float] = []
    #: `(q, theoretical discounted value, horizon)` for finished episodes.  This is the acceptance
    #: criterion for the bootstrap fix: `q` must track the theory instead of diverging from it.
    #: run4 (bootstrap frozen on the behaviour policy) went from +0.26 to **-4.27** over 1,000 rounds.
    calibration: list[tuple[float, float, int]] = []
    last_save = time.monotonic()
    step_stats: dict[str, float] = {}
    rounds_done = 0

    for round_index in range(round_limit):
        if not args.rounds and time.monotonic() > deadline:
            print(f"[online] time budget reached after {round_index} rounds", flush=True)
            break

        # Stage timers.  The first version of this loop ran at 78 control steps/s for one round and
        # 22 for the next ten, and without a split there was no way to tell whether that was the
        # physics, the forward, or something never measured at all.
        clock = {"physics": 0.0, "render": 0.0, "prepare": 0.0, "prefix": 0.0,
                 "sample": 0.0, "fresh": 0.0, "update": 0.0, "bootstrap": 0.0}

        chunks = state["executed"]
        submitted = time.monotonic()
        futures = [
            pool.submit(scene.play, np.asarray(chunks[i], dtype=np.float64),
                        args.execute, args.reward, args.profile_physics, args.reward_scale,
                        args.reward_offset)
            for i, scene in enumerate(scenes)
        ]

        # The gradient steps run *while* the scenes step.  Physics is ~0.99 s per round and the MLP
        # updates are the only other thing on the critical path, so overlapping them is the
        # difference between the measured 81 control steps/s and roughly half of it.
        if (not args.no_train and replay.size >= args.learning_starts
                and round_index >= args.warmup_rounds):
            step_stats = gradient_steps(args, actor, critic, critic_target, critic_noise,
                                        log_alpha, optimizers, replay, noise_bank)
            # Refresh after the updates, so the bootstrap target is the actor that will act next
            # round rather than the one that just acted.  This is where the reference's per-step
            # `D(s', actor_noise(s'))` lands, at round granularity -- see `BootstrapSet`.
            if bootstrap is not None:
                boot_started = time.monotonic()
                refreshed = bootstrap.refresh(args, replay, policy, preprocess, actor,
                                              projection, parts, chunk_flat_dim)
                clock["bootstrap"] = time.monotonic() - boot_started
                step_stats["bootstrap_rows"] = float(refreshed)
        clock["update"] = time.monotonic() - submitted

        results = [future.result() for future in futures]
        clock["physics"] = time.monotonic() - submitted
        rewards = [float(row[0]) for row in results]
        # `done` is `terminated or truncated`, and it gates two things at once: the TD mask and
        # whether the scene is rebuilt.  They were briefly separated here, so that a truncation kept
        # its bootstrap -- which is textbook-correct and was still wrong, because the `s'` available
        # at that point belongs to the *next* episode.  `Scene.play` documents the arithmetic:
        # cutting the bootstrap costs 0.064 cookies, bootstrapping on the wrong state costs 3.80.
        dones = [bool(row[1] or row[2]) for row in results]
        recent_rewards.extend(rewards)
        del recent_rewards[:-200]

        # Episode bookkeeping happens before the next observation is read, so a finished scene is
        # rebuilt and contributes its *new* first observation as `s'`.  That row is masked out by
        # `done`, so the new episode's state is never used as a bootstrap target -- which is the
        # whole reason `done` has to include truncation.
        for i, scene in enumerate(scenes):
            if not dones[i]:
                continue
            episode_returns.append(scene.episode_return)
            episodes += 1
            # Record the calibration pair now, using the `q` from this round's updates.
            #
            # Getting the reference value right matters more than it looks.  `q` is the mean of
            # `Q(s_t, a_t)` over a batch sampled **uniformly over the replay**, i.e. over states
            # spread through the episode -- so it must be compared against the value of an average
            # state, not of a start state.  For a constant per-step reward `r` over `T` steps:
            #
            #     V(start) = r * (1 - gamma^T) / (1 - gamma)
            #     mean_t V(s_t) = r / T * sum_t (1 - gamma^(T-t)) / (1 - gamma)
            #
            # and with gamma = 0.99, T = 94 those differ by a factor of 1.72 (3.90 vs 2.27).  The
            # first version of this block used the start-state value and therefore reported a
            # bias of about -1.63 that was pure bookkeeping.
            if step_stats.get("q") is not None and scene.control_steps > 0:
                horizon = max(1, int(scene.control_steps / max(1, args.execute)))
                per_step = scene.episode_return / horizon
                if args.gamma < 1.0:
                    mean_value = per_step * sum(
                        1.0 - args.gamma ** (horizon - t) for t in range(horizon)
                    ) / horizon / (1.0 - args.gamma)
                else:
                    mean_value = per_step * (horizon + 1) / 2.0
                calibration.append((float(step_stats["q"]), float(mean_value), horizon))
                del calibration[:-40]
            if args.fresh_env_per_episode:
                scene.env.close()
                scenes[i] = make_scene(i, scene.episode + 1)
            else:
                seed = args.seed_offset + i * 1000 + scene.episode + 1
                scene.env.reset(seed=seed, options=reset_options(bench, seed))
                scene.episode += 1
                scene.reset_counters()

        clock["render"], clock["prepare"], clock["prefix"], clock["sample"] = 0.0, 0.0, 0.0, 0.0
        clock["fresh"] = 0.0
        observation_next, frames_next = observe_all(clock)
        state_next = forward_all(observation_next, frames_next, clock)
        for i in range(args.scenes):
            replay.add(
                obs=state["actor_obs"][i],
                chunk=state["chunk_scaled"][i],
                noise=state["noise"][i],
                reward=torch.tensor(rewards[i], device=device),
                done=torch.tensor(float(dones[i]), device=device),
                obs_next=state_next["actor_obs"][i],
                chunk_next=state_next["chunk_scaled"][i],
                next_inputs=(
                    {"images": torch.as_tensor(
                        np.stack([frames_next[i][name] for name in CAMERAS])),
                     "state": torch.as_tensor(
                        np.asarray(observation_next[i][STATE], dtype=np.float32)),
                     "history": torch.as_tensor(scenes[i].history_vector()),
                     "task": task}
                    if replay.keep_inputs else None),
            )
        transitions += args.scenes
        for observation_vector, prior, prior_chunk in state["fresh"]:
            noise_bank.add(observation_vector, prior_chunk, prior)
        state = state_next
        rounds_done = round_index + 1

        if rounds_done % args.log_every == 0 or round_index == 0:
            wall = time.monotonic() - started
            control = rounds_done * args.execute * args.scenes
            recent = episode_returns[-20:]
            line = (f"[online] r{rounds_done:5d}  steps {transitions:6d}  ep {episodes:4d}"
                    f"  ret {np.mean(recent) if recent else float('nan'):8.2f}"
                    f"  rew {np.mean(recent_rewards):7.3f}"
                    f"  nz {(np.asarray(recent_rewards) != 0).mean() * 100:5.1f}%"
                    f"  ctrl/s {control / wall:6.1f}")
            for key in ("critic_loss", "noise_critic_loss", "actor_loss", "alpha", "q",
                        "min_q", "log_prob", "u_abs", "sigma", "target_q"):
                if key in step_stats:
                    line += f"  {key} {step_stats[key]:.4f}"
            for key in ("reward_b", "boot_b", "done_frac", "q_gap", "grad_norm"):
                if key in step_stats:
                    line += f"  {key} {step_stats[key]:+.4f}"
            if "bootstrap_rows" in step_stats:
                line += f"  boot_rows {step_stats['bootstrap_rows']:.0f}"
            if calibration:
                recent_cal = calibration[-8:]
                bias = float(np.mean([q - theory for q, theory, _ in recent_cal]))
                scale = float(np.mean([abs(theory) for _, theory, _ in recent_cal]))
                line += f"  q_bias {bias:+.3f}"
                # Relative form as well: `--reward-scale` changes Q's units, so an absolute bias
                # is not comparable across settings.  `q_rel` is the bias divided by the mean
                # magnitude of the reference value, and is the number to compare across runs.
                if scale > 1e-9:
                    line += f"  q_rel {bias / scale:+.3f}"
            print(line, flush=True)
            print("[online]   round cost: " + "  ".join(
                f"{name} {value * 1000:.0f}ms" for name, value in clock.items())
                + f"  (update overlaps physics: {min(clock['update'], clock['physics']) * 1000:.0f}ms"
                f" hidden)", flush=True)
            if noise_bank is not None:
                print(f"[online]   noise bank {noise_bank.size}/{noise_bank.capacity}", flush=True)
            history.append({
                "round": rounds_done, "transitions": transitions, "episodes": episodes,
                "mean_return_20": float(np.mean(recent)) if recent else None,
                "control_per_s": control / wall, **step_stats,
                "q_bias": (float(np.mean([q - t for q, t, _ in calibration[-8:]]))
                           if calibration else None),
            })

        if time.monotonic() - last_save > args.save_every_minutes * 60:
            last_save = time.monotonic()
            save_checkpoint(args.output, actor, critic, critic_noise, projection, args,
                            {"rounds": rounds_done, "transitions": transitions,
                             "episodes": episodes})

    save_checkpoint(args.output, actor, critic, critic_noise, projection, args,
                    {"rounds": rounds_done, "transitions": transitions, "episodes": episodes})
    (args.output.parent / "online_history.json").write_text(json.dumps(history, indent=2) + "\n")

    wall = time.monotonic() - started
    print(f"\n[online] {rounds_done} rounds, {transitions} transitions, {episodes} episodes,"
          f" {wall / 60:.1f} min", flush=True)
    if args.profile_physics:
        times = np.concatenate([np.asarray(s.step_times) for s in scenes]) if scenes else []
        nefcs = np.concatenate([np.asarray(s.step_nefc) for s in scenes]) if scenes else []
        if len(times):
            print(f"[online] control step over the whole run: mean {times.mean() * 1000:.1f} ms"
                  f"  median {np.median(times) * 1000:.1f}"
                  f"  p90 {np.percentile(times, 90) * 1000:.1f}"
                  f"  max {times.max() * 1000:.1f} (n={len(times)})", flush=True)
            order = np.argsort(nefcs)
            print(f"[online] nefc {nefcs.min()}..{nefcs.max()}", flush=True)
            for label, mask in (("first 10%", slice(0, len(times) // 10)),
                                ("last 10%", slice(-len(times) // 10, None))):
                print(f"[online]   {label}: {times[mask].mean() * 1000:.1f} ms/step,"
                      f" nefc {nefcs[mask].mean():.0f}", flush=True)
            # The relationship that decides the answer: cost per constraint row.
            if nefcs.max() > nefcs.min():
                slope = np.polyfit(nefcs, times * 1000, 1)
                print(f"[online]   ms ~ {slope[0]:.5f} * nefc + {slope[1]:.2f}", flush=True)
            del order
    if episode_returns:
        print(f"[online] episode return: mean {np.mean(episode_returns):.2f}"
              f" median {np.median(episode_returns):.2f}"
              f" max {np.max(episode_returns):.2f} (n={len(episode_returns)})", flush=True)
    if recent_rewards:
        values = np.asarray(recent_rewards)
        # Reported in **cookies**, the unit the evaluation scores in and the unit the 13.18 baseline
        # is quoted in.  `recent_rewards` holds the scaled values that went to the critic, so it has
        # to be divided back; comparing scaled rewards against a cookie baseline would read 30x off
        # and fire a warning that means nothing.
        divisor = args.reward_scale if args.reward_scale else 1.0
        cookies = values / divisor
        per_step = cookies.mean()
        print(f"[online] RL-step reward over the last {len(values)}: mean {per_step:+.4f} cookies"
              f"  non-zero {(values != 0).mean() * 100:.1f}%"
              f"  max {cookies.max():+.4f}"
              + (f"  (scaled x{divisor:g} -> {values.mean():+.3f} for the critic)"
                 if divisor != 1.0 else ""), flush=True)
        # The check that means something is a comparison against the *baseline policy on the same
        # task*, not a threshold invented on the spot.  `a3_clean_v9_30000` scores 13.18 cookies per
        # 1500-control-step episode, i.e. 13.18 / 1500 = 0.00879 cookies per control step, or
        # 0.1406 per RL step at `execute=16`.  An untrained noise policy IS that baseline (its
        # noise is the sampler's own), so anything near 0.14 means the reward plumbing is right and
        # anything near 0 means it is not -- which a fixed percentage cannot tell you.
        baseline = 13.18 / 1500 * args.execute
        ratio = per_step / baseline if baseline else float("nan")
        print(f"[online]   against the v9 baseline ({baseline:+.4f} cookies per RL step, from 13.18"
              f" cookies per 1500 control steps): {ratio * 100:.0f}%", flush=True)
        if ratio < 0.3:
            print("[online] WARNING: the reward rate is far below the baseline's.  With `--reward"
                  " increment` a step that delivers nothing pays nothing, so a collapsed rate makes"
                  " almost every reward zero and Q fits noise.  Check that the policy moves at all"
                  " before reading the critic curve.", flush=True)
    if step_stats:
        q = step_stats.get("q")
        divisor = args.reward_scale if args.reward_scale else 1.0
        expected = float(np.mean(recent_rewards)) / divisor
        print(f"[online] Q {q:.3f} against a reward mean of {expected:+.4f} cookies and gamma"
              f" {args.gamma}: at gamma 0.99 over 87 RL steps the discounted sum of a constant"
              f" {expected:+.4f} is {expected / (1 - args.gamma):+.2f} (x{divisor:g} scale"
              f" = {expected / (1 - args.gamma) * divisor:+.1f}), which is the scale Q should be on"
              " if the reward is right.", flush=True)
    pool.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
