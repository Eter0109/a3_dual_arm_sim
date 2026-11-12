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
| 1 | `update_noise_critic` draws **fresh `randn`** and denoises it: `w ~ N(0,1)`, `a = D(s, w)`, fits `Q_ns(s, scale(w))` to `Q(s, a)` | **closed in form** by `NoiseBank`: it stores `(s, w ~ N(0,1), D(s, w) / 1.5)` for `--fresh-noise` fresh priors per scene per round, and the noise critic is fitted on those rows | **residual is sample count, not structure.**  Reference: 10 noise-critic steps x 256 batch = **2,560 fresh samples per RL step**. Here: 8 x 8 = **64 per round**, i.e. 40x fewer. Cost decision, stated rather than hidden. |
| 2 | TD bootstrap: `next_a = D(s', actor(s'))` with the **current** actor, denoised every gradient step | **closed in form** by `BootstrapSet`: `--bootstrap-refresh` rows per round have `chunk_next` recomputed with the current actor | **residual is granularity**: 32 rows per round against every row of every gradient step. A 2,048-row replay is fully refreshed every 64 rounds. |
| 3 | `ent_coef` updated **before** the actor, from this step's `log_prob` | updated **after** the actor step | negligible; both use the same `log_prob`. |
| 4 | actor width = `act_steps * action_dim` (28 for robomimic can), **independent noise per chunk position** | actor width = `action_dim` (32), **repeated across the 50 chunk positions** | intentional.  `act_steps=4` makes the full space small enough for the reference to use it directly; at `C=50, d=32` the full space is 1,600 dims, which is exactly the case paper §C.5 addresses by repeating `w_single`.  Follows the paper, not the repo. |
| 5 | `target_update_interval = 1`, polyak every gradient step | `soft_update` inside the uTD loop | identical |
| 6 | `actor_gradient_steps = -1` -> actor updated on every gradient step | actor updated every uTD step | identical |
| 7 | `target_entropy = 0.0`, `ent_coef = 'auto'` init 1.0 | same | identical as written.  But see `--eval-scenes`: the reference's 28 independent dims can keep `log_prob` negative, while 32 dims repeated over 50 chunk positions makes tanh saturation count as density, and run8 drove `log_prob` to **+2.41**.  At `target_entropy = 0` that flips `alpha`'s gradient sign. |
| 8 | `batch_size 256`, `utd 20`, `noise_critic_grad_steps 10`, `tau 0.005`, `gamma 0.99`, `layer_size 2048`, `num_layers 3`, `action_magnitude 1.5`, `n_critics 2`, min-combine | same | identical |
| 9 | `learning_starts = 1` after `init_rollout_steps = 1501` | `--learning-starts 256` after `--warmup-rounds 8` | scale-adapted; an episode here is 1,382 control steps, not 300. |
| 10 | env reward: robomimic's `-1` per step | `--reward paper`: `level_sum - offset * execute` | necessary, and now the same construction rather than an approximation of it.  This environment's own reward is a *level*; subtracting 1 per control step reproduces robomimic's timer exactly. |
| 11 | **no gradient clipping** (`clip_grad_norm_` is in SB3's `a2c`/`dqn`/`ppo`, absent from `sac.py` and `dsrl.py`) | `--grad-clip 1000` from run13 on | **deliberate deviation, and it covers the actor too** (the old code clipped only the two critics).  Not aligned any more, and the threshold is not SB3's 10.0: under `--reward paper` the value scale is `-execute/(1-gamma)`, so 10 would rescale every update.  run12 measured critic norms from 29 to 57,239 at `--execute 20` with 1.0 M parameters; 1000 bounds the tail and leaves the rest. |
| 12 | `ent_coef * next_log_prob` subtracted from the TD target (`dsrl.py:286`) | `--entropy-in-target`, **off** | deliberate deviation, on evidence.  The term compounds to `alpha*|log pi|/(1-gamma)`, which at initialisation is 2143 against a true return of about -978.  run9 turned it on and measured the cost: at matched rounds it returned 3.70 where run8 (off) returned 6.30, with the same `q`.  The reference absorbs the same transient over ~250k RL steps; this loop has 2,166. |
| 13 | evaluates with `deterministic_eval` | `--eval-scenes 1`: 1 of 8 collection scenes uses a **seeded N(0,1)**, not `tanh(mean)` | earlier versions used zero noise, which the reference implementation showed is out of distribution (7 %% against 47 %% at w ~ N(0,1)).  The fixed noise is a fresh draw per round and identical across runs. |
| 15 | **`DSRL-SAC`** for the pi0 experiments (paper §C.5) | `--algorithm sac` (default): critic sees obs + **32-dim noise**, no distilled critic | the paper's pi0 setting.  `--algorithm na` keeps the previous `dsrl_na` path for comparison; it needs a critic over the 1600-dim chunk plus a 26 M noise critic and its distillation. |
| 16 | `hidden size 128`, `actor lr 1e-4`, `critic lr 3e-4` (Table 11, pi0 columns) | `--hidden 128`, `--actor-lr 1e-4`, `--lr 3e-4` | aligned.  Was `hidden 2048` with one shared 3e-4, i.e. 35x the parameters and 3x the actor step. |
| 17 | `target_entropy = -d/2 = -16` (Table 11) | `--target-entropy 21` | **deliberate deviation.**  The equilibrium is `log_prob = -target_entropy`; this checkpoint's measured initial `log_prob` is about -21, so 21 pins `alpha` at its initial value.  Measured on the reference: with `target_entropy 0` alpha decayed 10x and `log_prob` drifted -18.4 -> -0.6, and setting the positive value held `log_prob` at -18 for the whole run.  Also requires `--entropy-in-target` off, or the pinned alpha keeps the entropy bonus permanently on. |
| 14 | critic output starts at 0 (SB3 default init) | `--q-init` auto: output bias at the analytic batch-mean value, about -566 | deliberate, and the largest structural difference in this table.  The value is not a knob being tuned: it is where TD would put the head anyway, and run9 showed the cost of starting 566 units away (650 rounds to travel 391).  A constant on the output is invisible to the actor's gradient, which sees `dQ/da` rather than `Q`, so the objective is unchanged. |
| 18 | truncation is `done` (`ActionChunkWrapper` sets `done=True` at `max_episode_steps`, `truncated` always `False` -- their issue #6) | same: truncation cuts the bootstrap | **same defect, and it is a real one.**  `observation.state` is 16 joint/gripper values and carries **no episode time**, so aliased states receive one value.  With only `1/horizon` of the batch terminalised, the bootstrapped fixed point is `r/(1-gamma*(1-p))`, not the batch mean TD would otherwise reach.  At `--execute 20` those are -862 against -602 (1.43x); at `--execute 5` they are -376 against -343 (1.10x).  run12 was measured sliding -712 -> -760 toward the predicted -907.6.  Shorter `--execute` therefore *reduces* the bias as a side effect of the longer RL horizon. |

Rows 1 and 2 were both recorded as **open** substantive gaps for several runs after they had in fact
been closed, which is its own kind of bug: a stale alignment table sends the next diagnosis to the
wrong place.  What remains in both cases is a cost limit, and it is written down as a number
(40x fewer fresh samples, 32 rows per round) rather than as a verdict.

Reward
------
The environment's own reward is `min(cookies_in_target / required, 1.0)` per control step -- a
*level*, not an increment.  `--reward` decides what one RL step pays:

* `paper` (default) -- `level_sum - offset * execute`, i.e. the reference's construction.  robomimic
  subtracts `reward_offset = 1` from a 0/1 success reward, so every control step pays -1 and
  `ActionChunkWrapper` sums four of them into a constant -4 per RL step.  The same operation on this
  level gives -16 early in an episode and about -12 once five cookies are in: dense, large, bounded,
  and deliberately *not* a progress signal.
* `chunk_sum` -- the raw level summed over the chunk, up to +16 per RL step, so a return runs to
  ~1400.  Dense, but it pays for holding cookies rather than for delivering them.
* `chunk_mean`, `level_last`, `increment` -- kept for comparison.  `increment` is the one that was
  used first and it is 96 % zeros, which is what let the twin-critic `min` bias dominate the target
  and collapse `q`.

Two returns are tracked per episode and they are in different units, on purpose: `episode_return` in
**cookies** (comparable to the 13.18-cookie baseline) and `reward_return` in **reward units** (what
the calibration block needs to compare against `q`).

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
    #: The same episode accumulated in **reward units** rather than cookies.  Two accumulators
    #: because the calibration block needs reward units and every human-facing number is in cookies;
    #: deriving one from the other is exactly how a 400x unit mismatch got printed as a "bias".
    reward_return: float = 0.0
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
        self.reward_return = 0.0
        self.cookies = 0

    def play(self, chunk: np.ndarray, execute: int, reward_mode: str,
             profile: bool = False,
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
        #
        # `reward_return` is the same episode accumulated in **reward units**, and it exists because
        # the calibration block needs it.  It used to compute its reference value from
        # `episode_return`, which under `--reward paper` is cookies: it compared a Q of about -300
        # (reward units, -16 per RL step) against a theoretical value of about 1.5 (cookies) and
        # reported the difference, -300, as a "bias".  Every `q_bias` and `q_rel` number printed
        # before this fix was that unit mismatch, not a bias in the critic.
        if reward_mode == "paper":
            self.episode_return += float(self.cookies - cookies_before)
        else:
            self.episode_return += reward
        self.reward_return += reward
        return reward, terminated, truncated


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
        #: The noise the *next* state's actor would draw.  Only `--algorithm sac` reads it: there
        #: the noise IS the action, so `a'` must come from the current actor exactly as in SAC,
        #: and `BootstrapSet` has nothing to refresh because the replay row is already in the
        #: action space.
        self.noise_next = torch.zeros(capacity, noise_dim, device=device)
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
        # The next state's noise, needed only by `sac`, where the TD target's bootstrap action is
        # the noise itself rather than a denoised chunk.  Allocated unconditionally: it costs
        # `capacity x 32` floats (2.5 MB at 20000 rows) and keeping the layout uniform means `add`
        # does not need to know which algorithm is running.
        self.noise_next = torch.zeros(capacity, noise_dim, device=device)
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
            self.noise_next, self.reward, self.done))

    def add(self, obs, chunk, noise, reward, done, obs_next, chunk_next,
            noise_next, next_inputs: dict | None = None) -> None:
        index = self.position
        for target, value in (
            (self.obs, obs), (self.chunk, chunk), (self.noise, noise),
            (self.reward, reward), (self.done, done),
            (self.obs_next, obs_next), (self.chunk_next, chunk_next),
            (self.noise_next, noise_next),
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
            "noise_next": self.noise_next[index],
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
def clip_gradients(module: torch.nn.Module, limit: float) -> float:
    """Gradient-norm clip, returning the norm.  `limit <= 0` means **do not clip**.

    Called on the **actor** and on both critics; it used to be called on the critics only, which
    left the actor unbounded and made the logged `grad_norm` ambiguous.

    The reference does not clip at all.  `clip_grad_norm_` appears in this vendored SB3 in `a2c`,
    `dqn` and `ppo` -- and **nowhere in `sac.py`, `off_policy_algorithm.py` or `dsrl.py`**.  The
    help text for `--grad-clip` used to say "the reference inherits SB3's default of 10.0", which is
    true of A2C and PPO and false of the algorithm this file reproduces.  So `--grad-clip 1000` in
    run8 was already behaving like the reference while being *documented* as a deviation, and a clip
    of 10 would have been an undocumented deviation presented as alignment.

    **The threshold is a property of the value scale, not a constant.**  `--reward paper` fixes
    `Q` near `-offset*execute/(1-gamma)` -- about -600 at `--execute 20`, -340 at `--execute 5` --
    so the same clip means something completely different at the two settings.  run12 logged critic
    norms from 29 to 57,239 at `--execute 20`; run13 starts at `--grad-clip 1000`, which leaves the
    sub-1,000 majority alone and bounds only the runaway tail.

    Clipping to `inf` is a no-op that still returns the norm, which is why one code path serves both.
    """
    return float(torch.nn.utils.clip_grad_norm_(
        module.parameters(), limit if limit > 0 else float("inf")))


def combine_heads(stacked: torch.Tensor, how: str) -> torch.Tensor:
    """Aggregate the twin critics' outputs, `stacked` shaped `(heads, batch, 1)`.

    `min` is what the reference uses (`critic_backup_combine_type: min`), and it is biased **low**:
    for two roughly Gaussian heads, `E[min(a, b)] = mu - sigma/sqrt(pi)`, i.e. about `0.56 sigma`
    below the mean.  At a TD fixed point that bias is not a one-off -- it is applied at every
    bootstrap, so it compounds by `1/(1-gamma)`.  `probe_min_bias.py` measured it directly under a
    sparse reward: `min` settled at -0.631 where `mean` settled at +0.668 and a single head at
    +0.564.

    Under `--reward paper` the compounding is what makes it visible rather than marginal.  With
    `r = -16` per RL step, `gamma = 0.99` and a 93-step episode, the finite-horizon factor is 35.4,
    so an offset of `delta` per step moves `q` by `35.4 * delta`.  run10's `q` reached -755 where
    the reward bounds allow no value below -566, so the offset there was at least 5.3 per step --
    larger than a third of the reward itself.

    `mean` has no such bias but loses the conservatism that makes `min` a variance penalty in the
    overestimation sense.  Which one is right here is an empirical question, so both are logged
    every round (`boot_min`, `boot_mean`, `head_gap`) and this flag only chooses which one the
    target uses.
    """
    if how == "mean":
        return stacked.mean(dim=0)
    return stacked.min(dim=0).values


def analytic_value(args) -> float:
    """The value a uniformly sampled replay state should have, from `args` alone.

    Used to initialise the critics' output bias.  The reasoning is that the critic starts at **0**
    and has to travel to a value that is fixed by the reward: with `--reward paper` that is
    `-offset * execute` per RL step, and an episode is `--max-control-steps / execute` RL steps.
    run9 spent 650 rounds moving `q` from 0 to -391 without reaching the target, and the actor was
    taking gradients against a critic that was wrong by hundreds of units the whole time.  Starting
    the bias at the answer removes that transient without changing the objective: a constant added
    to a network's output is invisible to the actor's gradient, which sees `dQ/da`, not `Q`.

    **Which average, and why this one.**  Replay samples are uniform over stored transitions, so
    `q` in the log is the mean over states spread through the episode.  For a constant per-step
    reward `r` over `T` steps:

        V(start)       = r * (1 - gamma^T) / (1 - gamma)
        mean_t V(s_t)  = r / T * sum_t (1 - gamma^(T-t)) / (1 - gamma)

    and at gamma 0.99, T 94 those differ by 1.72x (`-978` against `-569`).  This file has already
    been wrong in exactly this way once -- the calibration block's docstring records a 25x error
    from using the start-state value -- so the initialiser must use the batch mean, which is the
    quantity `q` measures.

    The horizon is `--max-control-steps / execute`, matching the `horizon` the calibration block
    computes from `scene.control_steps`.  A truncated episode therefore matches; an episode that
    terminates early had a shorter horizon and its true value is closer to zero, so this initial
    bias is on the low side for those.  That is a second-order error in a starting guess, and TD
    corrects it.
    """
    if args.reward != "paper":
        return 0.0
    per_step = -args.reward_offset * args.execute
    horizon = max(1, int(args.max_control_steps / max(1, args.execute)))
    if args.gamma >= 1.0:
        return per_step * (horizon + 1) / 2.0
    total = sum(1.0 - args.gamma ** (horizon - t) for t in range(horizon))
    return per_step * total / horizon / (1.0 - args.gamma)


def set_output_bias(module: torch.nn.Module, value: float) -> int:
    """Set every sub-head's final `Linear` bias to `value`.  Returns how many were touched.

    Located structurally rather than by index so a change to `MLP` cannot silently make this a
    no-op -- a probe that cannot fire is the failure mode this project keeps hitting, and an
    initialiser that quietly does nothing would be indistinguishable from one that does not help.
    """
    touched = 0
    for head in module:                       # the twin critics are `ModuleList`s of `QHead`
        last_linear = [m for m in head.net.net if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            last_linear.bias.fill_(value)
        touched += 1
    return touched


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
        action_now = data["noise"] if args.algorithm == "sac" else data["chunk"]
        action_next = data["noise_next"] if args.algorithm == "sac" else data["chunk_next"]
        with torch.no_grad():
            stacked_next = torch.stack(
                [head(data["obs_next"], action_next) for head in critic_target])
            next_q = combine_heads(stacked_next, args.critic_combine)
            # Both aggregates are recorded every round, whatever `--critic-combine` says, because
            # their difference **is** the bias being studied: `E[min(a,b)] = mu - sigma/sqrt(pi)` for
            # two heads, so `mean - min` estimates `sigma/sqrt(pi)` directly and needs no theory.
            # Without this column the choice between them is untestable except by two 12-hour runs.
            boot_min = stacked_next.min(dim=0).values
            boot_mean = stacked_next.mean(dim=0)
            if args.entropy_in_target:
                _, next_log_prob = actor.sample(data["obs_next"])
                next_q = next_q - log_alpha.exp().detach() * next_log_prob
            target_q = (data["reward"].unsqueeze(-1)
                        + args.gamma * (1.0 - data["done"].unsqueeze(-1)) * next_q)

        # The critic's action input is the executed chunk under `na` and the noise under `sac`
        # (see `action_now` / `action_next` above).  This is the whole point of the SAC variant:
        # `a` and `a'` are the 32-dim latent the actor actually controls, not a 1,600-dim
        # denoised artefact of it.
        q_values = torch.stack([head(state, action_now) for head in critic])
        critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in q_values)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        # One call, not two.  `clip_grad_norm_` **returns** the norm and only clips when it is asked
        # to; calling it twice in a row re-clips the already-clipped gradients against the same
        # threshold, which is idempotent and therefore harmless -- but the second call was written as
        # `grader = ...` for logging while the first was already doing the clipping, i.e. the same
        # "two names for one effect" shape as the bugs in §9.106.  Kept as one call plus a read.
        grader = clip_gradients(critic, args.grad_clip)
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
        # `na` maximises the distilled noise critic; `sac` maximises the critic it just trained,
        # which is why the SAC variant needs no distillation at all.
        pi_critic = critic if args.algorithm == "sac" else critic_noise
        stacked_pi = torch.stack([head(state, sampled) for head in pi_critic])
        min_q = combine_heads(stacked_pi, args.critic_combine)
        # The actor's objective is **not** rescaled here.
        #
        # It used to be divided by `--reward-scale`, a knob that existed to make the sparse reward
        # large enough for TD to see.  `--reward paper` supplies that scale from the reward itself
        # (-16 per RL step is the reference's construction, not a tuning choice), so the knob had
        # become an identity: run8 passed `--reward-scale 1.0`, i.e. multiply by 1 and divide by 1.
        # Removing it removes the possibility of the two halves drifting apart again -- which is what
        # happened in run6, where `--reward-scale 30` multiplied Q by 30 while the actor divided by
        # 30 and `actor_loss` doubled to -39.
        actor_loss = (alpha * log_prob - min_q).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        # `--grad-clip` covers the actor too, which it did **not** before: `clip_gradients` was
        # called on `critic` and `critic_noise` only, so the logged `grad_norm` was the critic's and
        # the actor had no bound at all.  That gap mattered once the critic's value scale grew (run12
        # reached `q` -760 against the reference's -50), because `actor_loss = alpha*log_prob - min_q`
        # then carries a term of the same size and its gradient grows with it.  Reporting the two
        # norms separately is the point: one number cannot say which network moved.
        actor_grader = clip_gradients(actor, args.grad_clip)
        actor_optimizer.step()

        alpha_loss = -(log_alpha * (log_prob + args.target_entropy).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        soft_update(critic_target, critic, args.tau)
        # `.detach()` on every statistic: these tensors still carry the graph, and `float()` on
        # one that requires grad raises a warning that reads like an autograd bug.
        #
        # `update`, not `=`.  This was an assignment from the first commit, so it wiped the five
        # diagnostics above on every uTD step and none of them ever reached the log -- run4 through
        # run9 all printed without `reward_b`, `boot_b`, `done_frac`, `q_gap` or `grad_norm`.  The
        # numbers were computed, then discarded, then missed: several diagnoses in this session
        # were made without the four quantities that had been added specifically to make them.
        stats.update({
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "actor_grad_norm": float(actor_grader),
            "alpha": float(alpha),
            "q": float(q_values.detach().mean()),
            "min_q": float(min_q.detach().mean()),
            "log_prob": float(log_prob.detach().mean()),
            "u_abs": float(sampled.detach().abs().mean()),
            "sigma": float(actor.log_std.detach().clamp(-20.0, 2.0).exp().mean()),
            "target_q": float(target_q.detach().mean()),
            "boot_min": float(boot_min.mean()),
            "boot_mean": float(boot_mean.mean()),
            "head_gap": float((boot_mean - boot_min).mean()),
        })

    # --- the reference's separate noise-critic loop, `noise_critic_steps` fresh batches ------
    noise_critic_loss = torch.zeros((), device=log_alpha.device)
    # `sac` has no distilled critic: the actor maximises the critic trained on the noise action
    # space directly, so this whole loop is `na`-only.
    for _ in range(0 if critic_noise is None else max(0, args.noise_critic_steps)):
        data = replay.sample(args.batch)
        distilled = noise_bank.sample(args.batch) if noise_bank is not None else None
        source = distilled if distilled is not None else data
        with torch.no_grad():
            target = torch.stack([head(source["obs"], source["chunk"]) for head in critic]).detach()
        value = torch.stack([head(source["obs"], source["noise"]) for head in critic_noise])
        noise_critic_loss = 0.5 * sum(F.mse_loss(v, t) for v, t in zip(value, target))
        noise_critic_optimizer.zero_grad()
        noise_critic_loss.backward()
        clip_gradients(critic_noise, args.grad_clip)
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
    # Write to a temporary name and `os.replace` it into place.  `torch.save` straight onto `path`
    # truncates the previous checkpoint before it writes the new one, so a termination inside the
    # ~2 s a 289 MB save takes leaves **no** usable checkpoint -- the old one is gone and the new
    # one is half a file.  That is not hypothetical: a 480-minute run is routinely stopped from
    # outside (run8 was, at round 641), and the only reason it survived is that the kill happened
    # not to land in that window.  `os.replace` is atomic on POSIX, so the file is either the old
    # checkpoint or the new one.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "critic_noise": None if critic_noise is None else critic_noise.state_dict(),
        "projection": projection.detach().cpu().numpy(),
        "noise_mode": "single",
        "actor_dim": args.actor_dim,
        "chunk_size": args.chunk_size,
        "action_dim": args.action_dim,
        "observation_dim": args.observation_dim,
        "hidden": args.hidden,
        "observation_parts": {"proprio": True, "vlm": True, "visual": True},
        "online": extra,
    }, temporary)
    os.replace(temporary, path)
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
    parser.add_argument("--execute", type=int, default=20,
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
    # `--reward-scale` used to live here and is gone, not deprecated.  It multiplied the reward to
    # lift a *sparse* per-step reward (mean 0.078, `q` collapsing to -3.58) into TD's field of view,
    # and the actor's objective was divided by the same factor to keep its gradient honest.  Two
    # knobs that had to agree, and run6 showed what happens when they do not.  `--reward paper`
    # replaces both with the reference's own construction: `level_sum - offset * execute`, which is
    # dense and large by definition.  With that reward the multiplier was an identity in run8
    # (scale 1.0), so keeping it could only mislead the next reader into thinking it still matters.
    parser.add_argument("--eval-scenes", type=int, default=1,
                        help="how many of the `--scenes` scenes draw a **fixed, seeded N(0,1)**"
                             " instead of the actor's own sample.\n\n"
                             "This exists because `ret` is measured *with the actor's exploration"
                             " noise*, so a policy that improved while its noise widened looks"
                             " identical to one that got worse.  run8 fell 9.05 -> 6.30 cookies"
                             " with `alpha` rising and `log_prob` going positive, and there was no"
                             " way to say which of the two it was.\n\n"
                             "**It used to use `tanh(mean)`, i.e. zero noise, and that was wrong.**"
                             "  Measured on the reference implementation (docs/dsrl-repro-reference"
                             ".md): a converged diffusion policy scores **7 %%** at w = 0 against"
                             " **47 %%** at w ~ N(0,1) and 70 %% at std 0.5.  The sampler expects a"
                             " unit Gaussian, so zero noise is an input the policy never saw -- the"
                             " `greedy` column was measuring out-of-distribution behaviour, and the"
                             " run8-run11 reading of \"greedy fell while ret held\" was that"
                             " artefact rather than policy damage.\n\n"
                             "The noise is seeded per (round, scene) rather than frozen, so it is"
                             " a fresh draw every round but the same draw across runs.\n\n"
                             "Costs no wall time: same envs, same control steps.  1/8 of the"
                             " transitions are then off-policy for the actor's noise, which is the"
                             " price of the measurement.\n\n"
                             "0 disables it and restores all-stochastic collection.")
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
                             " reference work: 100 %% of transitions carry signal, |r| = 4 dwarfs the"
                             " twin-critic `min` bias of ~0.56 sigma, and returns are bounded.  The"
                             " `increment` mode used before this was 96 %% zeros, which is what let the"
                             " `min` bias dominate and collapse Q to -3.5 (see §9.107).")
    parser.add_argument("--reward-offset", type=float, default=1.0,
                        help="subtracted from the level reward in `--reward paper`.  The reference"
                             " uses 1.0, which is exactly its `env.reward_offset` in"
                             " `cfg/robomimic/dsrl_can.yaml`.  1.0 makes the reward strictly negative"
                             " until the task is solved, i.e. a pure timer; 0 leaves it as the raw"
                             " level, which is dense but rewards holding cookies rather than"
                             " delivering them.")

    # --- algorithm -----------------------------------------------------------------------
    parser.add_argument("--q-init", type=float, default=float("nan"),
                        help="initial bias of every Q head's output layer.  NaN (the default) means"
                             " compute it: `-reward_offset * execute` per RL step, discounted over"
                             " `max_control_steps / execute` RL steps, **averaged over states** the"
                             " way `q` in the log is.  That is about -569, not the -978 a start-state"
                             " computation gives; see `analytic_value` for why the distinction"
                             " matters and why this file has got it wrong before.\n\n"
                             "Why initialise at all: the critic starts at 0 and has to travel to a"
                             " value fixed by the reward.  run9 spent 650 rounds going 0 -> -391"
                             " without arriving, and the actor was taking gradients against a"
                             " critic that was wrong by hundreds of units throughout.  A constant"
                             " on the output is invisible to the actor's gradient (`dQ/da`, not"
                             " `Q`), so this removes the transient without changing the objective."
                             "\n\n"
                             "0 reproduces the old behaviour and is the A/B arm.")
    parser.add_argument("--algorithm", choices=("na", "sac"), default="sac",
                        help="which DSRL variant to run.  **`sac` is the paper's pi0 setting.**\n\n"
                             "`na` (noise-aliased) learns a critic on the **executed chunk**"
                             " (1,600 dims) and distils a second critic on the noise (32 dims);"
                             " the actor maximises the distilled one.  `sac` puts the noise itself"
                             " in the action space, so there is one critic, its input is"
                             " `obs + 32` instead of `obs + 1600`, and there is nothing to"
                             " distil.\n\n"
                             "Paper C.5: \"We use the DSRL-SAC variant of our method for these"
                             " experiments\" (the pi0 / Libero / Aloha ones).  Table 4's robomimic"
                             " runs use NA.  Our setup is a pi0-class VLA with a 50x32 chunk, so"
                             " SAC is the aligned choice.  It also removes `--fresh-noise`,"
                             " `--bootstrap-refresh` and the noise critic entirely: with the noise"
                             " as the action, the bootstrap action is `actor(s')` and the replay"
                             " row is already in the right space.")
    parser.add_argument("--critic-combine", choices=("min", "mean"), default="min",
                        help="how the twin critics are aggregated for the TD target and the actor's"
                             " objective.  The reference uses `min`.\n\n"
                             "`min` is biased low by `E[min(a,b)] = mu - sigma/sqrt(pi)` for two"
                             " heads, and a TD fixed point applies that at **every** bootstrap, so it"
                             " compounds by `1/(1-gamma)`.  Measured under sparse reward in"
                             " `probe_min_bias.py`: `min` -0.631, `mean` +0.668, single head"
                             " +0.564.\n\n"
                             "run10 made the compounding visible: with `--reward paper` the"
                             " finite-horizon factor is 35.4, and `q` reached -755 where the reward"
                             " bounds imply no value below -566.  That is impossible for a correct"
                             " estimate at any policy, so at least 5.3 per RL step was bias -- a"
                             " third of the reward itself.\n\n"
                             "`boot_min`, `boot_mean` and `head_gap` are logged every round"
                             " regardless, so `head_gap * 35.4` estimates the distortion without"
                             " trusting either the theory or this flag.")
    parser.add_argument("--hidden", type=int, default=128,
                        help="MLP width for the actor and both critics.\n\n"
                             "Paper Table 11 (pi0 column) says 128; Table 3 (the robomimic"
                             " experiments) says 2048, and 2048 is what the reference repo"
                             " defaults to.  The pi0 column is the one that matches this setup --"
                             " a 1600-dim chunk and a VLA observation -- so 128 is the aligned"
                             " value.  Parameter counts: 72.5 M at 2048 against 2.07 M at 128.")
    parser.add_argument("--utd", type=int, default=20,
                        help="gradient steps per RL step (paper §B)")
    parser.add_argument("--noise-critic-steps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="critic learning rate.  Paper Table 11 (pi0 column) gives 3e-4 for"
                             " the critic and 1e-4 for the actor, so these are separate flags"
                             " here.  They used to share one value at 3e-4, which is 3x the"
                             " paper's actor rate -- and the recurring symptom across run8-run11"
                             " was the actor leaving a good w=0 start and degrading the policy"
                             " while the critic stayed fitted.")
    parser.add_argument("--actor-lr", type=float, default=1e-4,
                        help="actor learning rate.  Paper Table 11: 0.0001.  See `--lr`.")
    parser.add_argument("--target-entropy", type=float, default=21.0,
                        help="the `log_prob` the learned `alpha` steers towards:"
                             " `d(alpha_loss)/d(log_alpha) = -(log_prob + target_entropy)`, so the"
                             " equilibrium is `log_prob = -target_entropy`.\n\n"
                             "0 (the reference's value) puts that equilibrium at `log_prob = 0`, i.e."
                             " a policy with no entropy.  A 32-dim unit Gaussian has"
                             " `log_prob = -29.4`, and the tanh Jacobian can carry the measured"
                             " `log_prob` **positive** once actions saturate -- run8 reached +2.41 and"
                             " run9 +8.09.  Positive `log_prob` means saturation *raises* the"
                             " objective, so the actor is rewarded for pinning its noise at +-1.\n\n"
                             "**`-dim` would make this worse, not better.**  SB3's default is `-dim`;"
                             " here that is -32, which puts the equilibrium at `log_prob = +32`, i.e."
                             " more saturation still.  A proposal earlier in this session to set -32"
                             " had the sign backwards.  The direction that suppresses saturation is"
                             " **positive** `target_entropy` (+29.4 pins `log_prob` at the unit-"
                             "Gaussian value, holding `sigma` near 1), but it also pins `alpha` near"
                             " 1, which with `--entropy-in-target` on would turn a transient entropy"
                             " bonus into a permanent one.  The two knobs interact.  Left at the"
                             " reference's 0 for run10.")
    parser.add_argument("--entropy-in-target", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="subtract `ent_coef * next_log_prob` from the TD target, as the"
                             " reference does (`dsrl.py:286`).  **Off, on evidence.**\n\n"
                             "The term compounds to `alpha * |log pi| / (1 - gamma)`.  At"
                             " initialisation that is `1.0 * 21.65 / 0.01 = 2143` against a true"
                             " return of about -978, which is why it was off while the reward was"
                             " sparse.  I turned it on for run9 on the argument that `--reward paper`"
                             " had made it small (3.6 %%).  **That argument used run8's late"
                             " `log_prob` of +2.41, not the initialisation value.**  At"
                             " initialisation `log_prob` is -21.65, the term is +19.25 per RL step"
                             " against a reward of -16, and the effective reward is **+3.25: positive,"
                             " i.e. the critic learns that stalling pays.**  It decays (to +-0.1 by"
                             " r450), so it is a transient rather than a permanent offset -- but at"
                             " matched rounds run9 (on) returned **3.70 at r475** where run8 (off)"
                             " returned **6.30**, with the same `q` of about -220.\n\n"
                             "This is a deliberate deviation from the reference, recorded as one.  The"
                             " reference survives the same transient because it runs on the order of"
                             " 250k RL steps, not 2,166.")
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
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="gradient-norm clip for the **actor and both critics**.  **0 means do"
                             " not clip, which is what the reference does.**\n\n"
                             "This help text used to claim \"the reference inherits SB3's default of"
                             " 10.0\".  It does not: `clip_grad_norm_` appears in this vendored SB3"
                             " in `a2c`, `dqn` and `ppo`, and nowhere in `sac.py`,"
                             " `off_policy_algorithm.py` or `dsrl.py`.  The claim was true of A2C and"
                             " PPO and false of the algorithm being reproduced.\n\n"
                             "A positive value is therefore a deliberate deviation.  The value has to"
                             " be chosen against the **value scale**, not copied from SB3's 10.0: under"
                             " `--reward paper` the critic settles near `-offset*execute/(1-gamma)`,"
                             " which is about -600 at `--execute 20` but only -340 at `--execute 5`, and"
                             " the observed critic norm at `--execute 20` ranged from 29 to 57,000.  A"
                             " clip of 10 would rescale essentially every update and silently cut the"
                             " learning rate by orders of magnitude; a clip near the top of the"
                             " observed range bounds only the outlier batches that would otherwise"
                             " poison Adam's second moment.  The actor's norm is logged as"
                             " `actor_grad_norm` so the two scales can be compared before choosing.")
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
                             " 32 costs ~1.6 s on an 11 s round (+15 %%) and refreshes a 2,048-row"
                             " replay every 64 rounds.")
    parser.add_argument("--replay-capacity", type=int, default=20000)
    parser.add_argument("--fresh-noise", type=int, default=8,
                        help="fresh `N(0,1)` noise vectors denoised per scene per round, to train the"
                             " distilled critic on the prior rather than only on behaviour noise.\n\n"
                             "The reference denoises a whole batch of fresh noise on **every**"
                             " noise-critic gradient step (10 per RL step).  At batch 256 that is"
                             " 2,560 fresh samples per RL step; `--fresh-noise 8` over 8 scenes is 64"
                             " per round, i.e. **40x below the reference**.  This closes the gap in"
                             " *form* -- `Q_noise` is fitted on the prior, not on the buffer's own"
                             " noise -- but not in sample count, and the residual is a cost decision,"
                             " not a rounding error: one denoise is ~40 ms per row and the reference"
                             " pays 2,560 of them per step.\n\n"
                             "Measured cost of the default: 8 indices x 437 ms is ~1.7 s against an"
                             " 11 s round.")
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
    # `sac`: the action is the noise, so the critic sees `obs + actor_dim`.
    # `na`:  the action is the executed chunk, so it sees `obs + chunk_flat_dim`, and a second
    #        critic is distilled onto the noise space.
    critic_action_dim = args.actor_dim if args.algorithm == "sac" else chunk_flat_dim
    critic = twin_critic(args.observation_dim, critic_action_dim, hidden=args.hidden).to(device)
    critic_target = twin_critic(args.observation_dim, critic_action_dim,
                                hidden=args.hidden).to(device)
    critic_noise = (None if args.algorithm == "sac"
                    else twin_critic(args.observation_dim, args.actor_dim,
                                     hidden=args.hidden).to(device))
    # Start every Q head at the value TD would drive it to (`--q-init`).  `critic_target` is copied
    # *after* this so the target network agrees from step 1; `critic_noise` gets the same bias
    # because it is distilled from `critic` and an untrained student at 0 would need to re-climb the
    # whole distance on the network the actor actually maximises.
    q_init = analytic_value(args) if math.isnan(args.q_init) else args.q_init
    if q_init:
        set_output_bias(critic, q_init)
        if critic_noise is not None:
            set_output_bias(critic_noise, q_init)
    critic_target.load_state_dict(critic.state_dict())
    if q_init:
        horizon = max(1, int(args.max_control_steps / max(1, args.execute)))
        start_value = (-args.reward_offset * args.execute
                       * (1.0 - args.gamma ** horizon) / (1.0 - args.gamma))
        print(f"[online] Q init {q_init:+.1f}: reward/step {-args.reward_offset * args.execute:+.0f}"
              f" x {horizon} RL steps, batch mean; the start-state value would be"
              f" {start_value:+.1f} and using it is the 1.72x error this file records having made"
              " once already", flush=True)
    noise_params = (0.0 if critic_noise is None
                    else sum(p.numel() for p in critic_noise.parameters()) / 1e6)
    print(f"[online] algorithm {args.algorithm}"
          f" (critic action dim {critic_action_dim},"
          f" {'no noise critic' if critic_noise is None else 'distilled noise critic'})",
          flush=True)
    print(f"[online] actor {sum(p.numel() for p in actor.parameters()) / 1e6:.2f}M,"
          f" critic {sum(p.numel() for p in critic.parameters()) / 1e6:.2f}M,"
          f" noise critic {noise_params:.2f}M,"
          f" observation {args.observation_dim}, chunk {chunk_flat_dim},"
          f" noise {args.actor_dim}", flush=True)

    alpha = torch.zeros(1, requires_grad=True, device=device)
    optimizers = (
        torch.optim.Adam(actor.parameters(), lr=args.actor_lr),
        torch.optim.Adam(critic.parameters(), lr=args.lr),
        # `None` under `sac`, where there is no distilled critic.  The loop that would use this is
        # already skipped (`range(0 if critic_noise is None ...)`), so nothing dereferences it; the
        # guard exists because `None.parameters()` would raise during setup, before any of that.
        None if critic_noise is None else torch.optim.Adam(critic_noise.parameters(), lr=args.lr),
        torch.optim.Adam([alpha], lr=args.lr),
    )
    log_alpha = alpha

    replay = Replay(args.replay_capacity, args.observation_dim, args.actor_dim, chunk_flat_dim,
                    torch.device(device), keep_inputs=args.bootstrap_refresh > 0)
    noise_bank = (None if args.algorithm == "sac" else
                  NoiseBank(args.noise_bank_capacity, chunk_flat_dim, args.actor_dim,
                            torch.device(device))
                  if args.fresh_noise > 0 else None)
    # `BootstrapSet` exists to re-denoise `chunk_next` with the *current* actor, which is what the
    # reference's TD target needs.  Under `sac` the bootstrap action is the **noise**, and a noise
    # vector does not depend on the policy at all -- it is the same number it was at collection
    # time.  So there is nothing to refresh and re-encoding would be pure cost.
    bootstrap = (None if args.algorithm == "sac" or args.bootstrap_refresh <= 0
                 else BootstrapSet(torch.device(device)))
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
    # Which sentence reached the VLA, said out loud and *why*, because run4-run10 all trained a
    # 2x10 scene on a 2x5 sentence and nothing in the log showed it.  `training_prompt()` reads
    # `A3_TRAINING_DATASET` when set and otherwise falls back to a hard-coded string from
    # `a3_b2_merged`; the fallback does not appear anywhere in the v9 training set, so leaving the
    # variable unset is a correctness bug rather than a default.
    if os.environ.get("A3_TRAINING_DATASET"):
        print(f"[online]   from A3_TRAINING_DATASET={os.environ['A3_TRAINING_DATASET']}",
              flush=True)
    else:
        print("[online]   WARNING: A3_TRAINING_DATASET is unset, so this is the embedded"
              " a3_b2_merged sentence, not this checkpoint's.  It will not appear in the v9"
              " training set and the policy may be answering a different task than the scene"
              " presents.  Set A3_TRAINING_DATASET or pass --task-prompt.", flush=True)
    if args.eval_scenes > 0:
        print(f"[online] scenes 0-{args.eval_scenes - 1} draw a **frame-fixed, round-fixed N(0,1)**"
              f" and are reported as `eval`; the other {args.scenes - args.eval_scenes} draw the"
              " actor's own samples and are reported as `ret`.  Both columns use the same envs and"
              " the same 1500 control steps, so `eval - ret` is the cost of the actor's noise and"
              " `eval` alone is the policy", flush=True)
    print(f"[online] entropy_in_target={args.entropy_in_target}  grad_clip={args.grad_clip:g}"
          f"  fresh_noise={args.fresh_noise}/scene/round  bootstrap_refresh={args.bootstrap_refresh}"
          "  (all four printed because three of them changed meaning while the DSRL loop was being"
          " aligned against the reference)", flush=True)

    # The reward is the one quantity every other number is measured against, so it is checked
    # before any gradient step rather than inferred from Q later.  A zero reward here produces a
    # perfectly finite loss curve that means nothing -- the failure mode this project has hit
    # repeatedly, and the reason the offline trainer prints the pool return and the anchor return
    # at startup.
    print(f"[online] reward '{args.reward}': scene cookies at reset"
          f" {[scene.cookies for scene in scenes]}", flush=True)

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

    #: Round counter the evaluation noise generator is seeded from.  A one-element list because
    #: `forward_all` is defined before the training loop and closes over this container; the loop
    #: assigns `eval_round[0] = round_index` at the top of each round.
    eval_round = [0]

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
            # Greedy rows for the deterministic evaluation (`--eval-scenes`).  `sample` returns
            # `tanh(mean + std * eps)`, so `tanh(mean)` is the mode of the same distribution: SB3's
            # `deterministic=True`, and what `SteeringActor(deterministic=True)` does at inference.
            #
            # `log_prob` stays the one from the stochastic draw, because `alpha` has to keep
            # regulating the *sampling* distribution -- feeding it the mode's (undefined) density
            # would make the temperature measure something the agent never samples from.
            if args.eval_scenes > 0:
                # The evaluation rows draw a **fresh N(0,1)**, they do not use `tanh(mean)`.
                #
                # This used to be the greedy action, on the reasoning that zero noise is the
                # policy's mode.  Measured on the reference implementation (see
                # docs/dsrl-repro-reference.md): a converged diffusion policy scores **7 %** at
                # w = 0 against **47 %** at w ~ N(0,1) and 70 % at std 0.5.  Zero noise is an
                # input the policy was never trained on -- the sampler expects a unit Gaussian --
                # so `greedy` was measuring out-of-distribution behaviour, and the run8-run11
                # readings of "greedy fell while ret held" were that artefact, not policy damage.
                #
                # Fixed seed so the column is comparable across rounds: each evaluation row uses
                # its own generator seeded by (round, row), which keeps the comparison paired
                # without freezing the noise across the whole run.
                generators = [torch.Generator(device=noise.device).manual_seed(
                    1_000_003 * int(eval_round[0]) + i) for i in range(args.eval_scenes)]
                eval_noise = torch.stack([
                    torch.randn(args.actor_dim, device=noise.device, generator=g)
                    for g in generators])
                noise = torch.cat([eval_noise, noise[args.eval_scenes:]], dim=0)
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
    #: Returns of the `--eval-scenes` fixed-noise scenes, kept apart from `episode_returns` because
    #: whole point is to compare a policy acting deterministically against one acting with
    #: exploration noise.  Both are cookie counts over full episodes, so both are comparable to the
    #: 13.18-cookie baseline; only the greedy one measures the policy rather than the noise.
    eval_returns: list[float] = []
    recent_rewards: list[float] = []
    #: `(q, theoretical discounted value, horizon)` for finished episodes.  This is the acceptance
    #: criterion for the bootstrap fix: `q` must track the theory instead of diverging from it.
    #: run4 (bootstrap frozen on the behaviour policy) went from +0.26 to **-4.27** over 1,000 rounds.
    calibration: list[tuple[float, float, int]] = []
    last_save = time.monotonic()
    step_stats: dict[str, float] = {}
    rounds_done = 0

    for round_index in range(round_limit):
        eval_round[0] = round_index
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
                        args.execute, args.reward, args.profile_physics,
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
            if i < args.eval_scenes:
                eval_returns.append(scene.episode_return)
                del eval_returns[:-200]
            else:
                episode_returns.append(scene.episode_return)
                del episode_returns[:-200]
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
            # The reference block assumes a **constant reward per RL step**, which is what
            # `--reward paper` produces; it must therefore use `reward_return`, the episode summed in
            # reward units.  It used to use `episode_return`, which under `--reward paper` is cookies:
            # that compared a Q near -300 against a reference near 1.5 and printed the difference as
            # a "bias".
            if step_stats.get("q") is not None and scene.control_steps > 0:
                horizon = max(1, int(scene.control_steps / max(1, args.execute)))
                per_step = scene.reward_return / horizon
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
                # `s'`'s noise is the actor's own draw at the next state, which is exactly what
                # SAC wants for `a'`.  Under `na` nothing reads this column.
                noise_next=state_next["noise"][i],
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
            fixed = eval_returns[-20:]
            line = (f"[online] r{rounds_done:5d}  steps {transitions:6d}  ep {episodes:4d}"
                    f"  ret {np.mean(recent) if recent else float('nan'):8.2f}"
                    f"  eval {np.mean(fixed) if fixed else float('nan'):7.2f}"
                    f" (+{len(fixed)})"
                    f"  rew {np.mean(recent_rewards):7.3f}"
                    f"  nz {(np.asarray(recent_rewards) != 0).mean() * 100:5.1f}%"
                    f"  ctrl/s {control / wall:6.1f}")
            for key in ("critic_loss", "noise_critic_loss", "actor_loss", "actor_grad_norm",
                        "alpha", "q",
                        "min_q", "log_prob", "u_abs", "sigma", "target_q"):
                if key in step_stats:
                    line += f"  {key} {step_stats[key]:.4f}"
            for key in ("reward_b", "boot_b", "done_frac", "q_gap", "grad_norm"):
                if key in step_stats:
                    line += f"  {key} {step_stats[key]:+.4f}"
            # The twin-critic bias, measured rather than assumed: `head_gap` is the mean and min of
            # two heads differing, which estimates `sigma/sqrt(pi)` directly.
            for key in ("boot_min", "boot_mean", "head_gap"):
                if key in step_stats:
                    line += f"  {key} {step_stats[key]:+.2f}"
            if "bootstrap_rows" in step_stats:
                line += f"  boot_rows {step_stats['bootstrap_rows']:.0f}"
            if calibration:
                recent_cal = calibration[-8:]
                bias = float(np.mean([q - theory for q, theory, _ in recent_cal]))
                # Relative form **per sample**, not `bias / mean(|theory|)`.  The denominator used
                # to be a rolling mean of the reference magnitude; because that magnitude scales
                # with episode length, the ratio moved 3.2x between r375 and r400 while the bias
                # moved 1.12x.  It was measuring the composition of the window, not the critic.
                relative = float(np.mean([(q - theory) / max(1e-9, abs(theory))
                                          for q, theory, _ in recent_cal]))
                line += f"  q_bias {bias:+.3f}  q_rel {relative:+.3f}"
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
    if episode_returns or eval_returns:
        if episode_returns:
            print(f"[online] episode return (stochastic, {args.scenes - args.eval_scenes} scenes):"
                  f" mean {np.mean(episode_returns):.2f}  median {np.median(episode_returns):.2f}"
                  f"  max {np.max(episode_returns):.2f} (n={len(episode_returns)})", flush=True)
        if eval_returns:
            print(f"[online] episode return (greedy, {args.eval_scenes} scenes):"
                  f" mean {np.mean(eval_returns):.2f}"
                  f"  median {np.median(eval_returns):.2f}"
                  f"  max {np.max(eval_returns):.2f} (n={len(eval_returns)})", flush=True)
            # The two together are the diagnosis: a greedy return that holds while the stochastic one
            # falls means the exploration noise widened, not that the policy got worse.  run8 could
            # not tell those apart, which is why this exists.
            print(f"[online]   greedy vs 13.18 baseline: {np.mean(eval_returns) / 13.18 * 100:.0f}%"
                  "  -- this is the number that measures the policy rather than the noise.",
                  flush=True)
    if recent_rewards:
        values = np.asarray(recent_rewards)
        # `recent_rewards` holds the **shaped** reward that went to the critic, which under
        # `--reward paper` is `level_sum - offset * execute`, not a cookie count.  It is printed as
        # it is, and the cookie comparison is made against `episode_return`, which is accumulated in
        # cookies.  Mixing the two was the bug fixed in `Scene.play`'s `reward_return`.
        per_step = values.mean()
        print(f"[online] RL-step reward over the last {len(values)}: mean {per_step:+.4f}"
              f"  non-zero {(values != 0).mean() * 100:.1f}%"
              f"  max {values.max():+.4f}"
              f"  min {values.min():+.4f}", flush=True)
        # The check that means something is the *episode* return against the baseline policy on
        # the same task, not a threshold invented on the spot.  `a3_clean_v9_30000` scores 13.18
        # cookies per 1500-control-step episode.  An untrained noise policy IS that baseline (its
        # noise is the sampler's own), so a return near 13 means the plumbing is right.  This used
        # to be computed from the shaped reward divided by `--reward-scale`, which only happened to
        # land near the right number because the sparse reward and the scale were both ~0.03.
        if episode_returns:
            baseline = 13.18
            mean_return = float(np.mean(episode_returns))
            print(f"[online]   episode return against the v9 baseline ({baseline:.2f} cookies per"
                  f" 1500 control steps): {mean_return / baseline * 100:.0f}%", flush=True)
            if mean_return / baseline < 0.3:
                print("[online] WARNING: the episode return is far below the baseline's.  Check"
                      " that the policy moves at all before reading the critic curve.", flush=True)
    if step_stats:
        q = step_stats.get("q")
        # The fixed point Q should sit at: a constant reward `r` per RL step, discounted over the
        # episode, which is `r / (1 - gamma)` for the start state and less for an average state.
        expected = float(np.mean(recent_rewards))
        print(f"[online] Q {q:.3f} against a reward mean of {expected:+.4f} per RL step and gamma"
              f" {args.gamma}: for a constant {expected:+.4f} the discounted sum over 87 RL steps is"
              f" {expected / (1 - args.gamma):+.2f} at the start state, which is the scale Q should"
              " be on if the reward is right.", flush=True)
    pool.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
